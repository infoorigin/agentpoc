# exception_logging_middleware.py

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import structlog

logger = structlog.get_logger("error_logger")

class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            logger.exception("Unhandled exception occurred", error=str(e))
            raise  # Re-raise so FastAPI returns 500
