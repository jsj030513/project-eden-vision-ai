"""Bound request bytes BEFORE multipart spooling; no custom multipart parser."""

from uuid import uuid4

from starlette.types import ASGIApp, Receive, Scope, Send

from eden_vision.api.errors import error_response

MULTIPART_OVERHEAD_BYTES = 64 * 1024


class UploadGuard:
    def __init__(self, app: ASGIApp, max_upload_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_upload_bytes + MULTIPART_OVERHEAD_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        if scope["method"] != "POST" or scope["path"].rstrip("/") != "/v1/vision/classify":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                response = error_response(request_id, 413, "IMAGE_TOO_LARGE", "Multipart request exceeds byte limit")
                await response(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)
