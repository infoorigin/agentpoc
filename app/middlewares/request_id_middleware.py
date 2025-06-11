# middlewares/request_id_middleware.py

import uuid
import structlog
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

logger = structlog.get_logger("request_logger")

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # request_id = request.headers.get("request_id", str(uuid.uuid4()))
        request_id = (
            request.headers.get('request_id') 
            or request.headers.get('x-request-id') 
            or f'{__name__}+{uuid.uuid4()}'
        )

        structlog.threadlocal.bind_threadlocal(request_id=request_id)

        start_time = time.time()
        response = await call_next(request)  # Let exceptions propagate to handler
        duration = time.time() - start_time
        message = "HTTP request completed"
        if response.status_code < 400:
            logger.info(
                event = message,
                message = message,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration=f"{duration:.3f}s"
            )

        response.headers["request_id"] = request_id
        structlog.threadlocal.clear_threadlocal()

        return response
