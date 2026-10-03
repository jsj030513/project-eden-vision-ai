"""Liveness remains /health; readiness reflects this app's active execution mode."""
import logging
from fastapi import APIRouter, Request, Response
from eden_vision.schemas.readiness import ReadinessResponse

router = APIRouter()
logger = logging.getLogger('eden_vision')


@router.get('/ready', response_model=ReadinessResponse, response_model_exclude_none=True,
            responses={503: {'model': ReadinessResponse}})
def ready(request: Request, response: Response) -> ReadinessResponse:
    state = request.app.state
    enabled = state.ranking_runtime_enabled
    available = getattr(state, 'app_ready', False)
    if not enabled:
        response.status_code = 200 if available else 503
        return ReadinessResponse(status='ready' if available else 'not_ready',
                                 mode='legacy_classification', ranking_runtime_enabled=False,
                                 ranking_runtime_ready=False)
    runtime = state.ranking_runtime
    if (available and state.ranking_runtime_ready and not state.ranking_runtime_startup_failed
            and runtime is not None):
        try:
            metadata = runtime.metadata
            if metadata.get('ready') is True:
                # Select only public keys, then validate the fixed production
                # identity. Unexpected strings cannot leak a filesystem path.
                return ReadinessResponse(status='ready', mode='clip_ranking',
                    ranking_runtime_enabled=True, ranking_runtime_ready=True,
                    **{key: metadata.get(key) for key in (
                        'model_id', 'model_revision', 'taxonomy_version', 'device')})
        except Exception:
            logger.exception('readiness metadata unavailable')
    response.status_code = 503
    return ReadinessResponse(status='not_ready', mode='clip_ranking',
                             ranking_runtime_enabled=True, ranking_runtime_ready=False)
