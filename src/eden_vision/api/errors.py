from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from eden_vision.schemas.error import ErrorDetail, ErrorResponse
from eden_vision.services.image_input import ImageErrorCode, ImageInputError


def error_response(request_id: str, status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(request_id=request_id, error=ErrorDetail(code=code, message=message))
    return JSONResponse(body.model_dump(), status_code=status, headers={"X-Request-ID": request_id})


async def image_error_handler(request: Request, exc: ImageInputError) -> JSONResponse:
    status = {ImageErrorCode.IMAGE_TOO_LARGE: 413, ImageErrorCode.UNSUPPORTED_IMAGE_TYPE: 415,
              ImageErrorCode.INVALID_IMAGE: 422}[exc.code]
    return error_response(request.state.request_id, status, exc.code.value, str(exc))


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(request.state.request_id, 422, "INVALID_IMAGE", "A multipart image file is required")


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return error_response(request.state.request_id, exc.status_code, "INVALID_REQUEST", "Request could not be processed")
