from fastapi import Request
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger("global_exception")

async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    response_code = "ERROR"
    response_message = "An unexpected error occurred."

    return JSONResponse(
        status_code=500,
        content = {
            'status_code': 500,
            'code': response_code,
            'message': response_message,
            'data': None
        }
    )
