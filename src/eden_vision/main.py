import logging
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from eden_vision.api.health import router
from eden_vision.api.readiness import router as readiness_router
from eden_vision.config import Settings
from eden_vision.logging_config import configure_logging
from eden_vision.api.vision import router as vision_router
from eden_vision.api.errors import image_error_handler, validation_error_handler, http_error_handler
from eden_vision.api.upload_guard import UploadGuard
from eden_vision.classification.confidence import ConfidencePolicy
from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import load_taxonomy
from eden_vision.services.classification import ClassificationService
from eden_vision.services.image_input import ImageInputError, ImageInputService
from eden_vision.vision.adapters.dummy import DummyVisionModel
from eden_vision.vision.contracts import VisionModel
from eden_vision.vision.types import CandidatePrediction, ScoreKind
from eden_vision.vision.runtime.lifecycle import (
    RankingRuntimeFactory, RankingRuntimeSettings, initialize_ranking_state, ranking_lifespan,
)

settings = Settings()
logger = logging.getLogger("eden_vision")


def build_dummy_model() -> VisionModel:
    """Explicit Phase 3 composition: constant fixture, NOT image recognition."""
    return DummyVisionModel((CandidatePrediction("plant", 0.9, ScoreKind.COSINE_SIMILARITY),))


def create_app(config: Settings | None = None, *, model_factory: Callable[[], VisionModel] = build_dummy_model,
               ranking_runtime_factory: RankingRuntimeFactory | None = None,
               ranking_runtime_settings: RankingRuntimeSettings | None = None) -> FastAPI:
    config = config if config is not None else Settings()
    if ranking_runtime_settings is not None and ranking_runtime_factory is None:
        raise ValueError("ranking settings require an explicit ranking_runtime_factory")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.app_ready = False
        configure_logging(config.log_level)
        logger.info("service=%s host=%s port=%s", config.app_name, config.host, config.port)
        model = model_factory()
        try:
            model.load()
            application.state.classification = ClassificationService(
                model, ConfidencePolicy(HierarchyResolver(load_taxonomy())))
            application.state.image_input = ImageInputService(
                max_upload_bytes=config.max_upload_bytes, max_image_pixels=config.max_image_pixels)
            application.state.classification_lock = asyncio.Lock()
            async with ranking_lifespan(application, ranking_runtime_factory, ranking_runtime_settings):
                application.state.app_ready = True
                yield
        finally:
            application.state.app_ready = False
            model.close()

    application = FastAPI(title=config.app_name, lifespan=lifespan,
                          description="Phase 3 upload validation with dummy inference only.")
    application.state.settings = config
    application.state.app_ready = False
    initialize_ranking_state(application, ranking_runtime_factory is not None)
    application.add_middleware(UploadGuard, max_upload_bytes=config.max_upload_bytes)
    application.add_exception_handler(ImageInputError, image_error_handler)
    application.add_exception_handler(RequestValidationError, validation_error_handler)
    application.add_exception_handler(HTTPException, http_error_handler)
    application.include_router(router)
    application.include_router(readiness_router)
    application.include_router(vision_router)

    @application.get("/")
    def root() -> dict[str, str]:
        return {"service": config.app_name, "status": "ok"}

    return application


app = create_app(settings)


def run() -> None:
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
