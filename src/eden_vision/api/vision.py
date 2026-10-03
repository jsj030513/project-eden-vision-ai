"""Upload transport only. Decoding and policy are delegated to services."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Request, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from eden_vision.api.errors import error_response
from eden_vision.schemas.classification import VisionResponse, ImageInfo, response_from_result
from eden_vision.schemas.error import ErrorResponse
from eden_vision.services.image_input import ImageErrorCode, ImageInputError
from eden_vision.vision.contracts import ClassificationHandler
from eden_vision.vision.types import ImageInput, RankingResult

router = APIRouter(prefix="/v1/vision")
logger = logging.getLogger("eden_vision")


def execute_classification(state, image: ImageInput):
    """App composition selects execution; callers hold the app's existing lock."""
    if state.ranking_runtime_enabled:
        runtime = state.ranking_runtime
        if not state.ranking_runtime_ready or runtime is None or runtime.metadata.get("ready") is not True:
            raise RuntimeError("configured ranking runtime is unavailable")
        result = runtime.rank(image)
        if not isinstance(result, RankingResult):
            raise TypeError("ranking runtime must return RankingResult")
        return result
    handler: ClassificationHandler = state.classification
    return handler.classify(image)


@router.post("/classify", response_model=VisionResponse,
             responses={code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500)},
             description="App composition selects legacy classification (1.0) or opt-in raw ranking (1.1).")
async def classify(request: Request, response: Response, image: Annotated[UploadFile, File()]):
    state = request.app.state
    try:
        data = await image.read(state.settings.max_upload_bytes + 1)
        if len(data) > state.settings.max_upload_bytes:
            raise ImageInputError(ImageErrorCode.IMAGE_TOO_LARGE, "Upload exceeds byte limit")
        # Serialize the non-thread-safe model contract and limit concurrent decodes.
        async with state.classification_lock:
            decoded = await run_in_threadpool(state.image_input.decode, data, image.content_type)
            result = await run_in_threadpool(execute_classification, state, decoded)
        response.headers["X-Request-ID"] = request.state.request_id
        return response_from_result(result, request_id=request.state.request_id,
                                                   image=ImageInfo(width=decoded.width, height=decoded.height))
    except ImageInputError:
        raise
    except Exception:
        logger.exception("classification failed request_id=%s", request.state.request_id)
        return error_response(request.state.request_id, 500, "INFERENCE_FAILED", "Classification could not be completed")
    finally:
        await image.close()
