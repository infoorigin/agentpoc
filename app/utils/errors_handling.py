# app/utils/errors_handling.py

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog
import uuid

logger = structlog.get_logger('Error Logger')

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    event = "HTTPException occurred"
    request_id = (
        request.headers.get('request_id') 
        or request.headers.get('x-request-id') 
        or f'{__name__}+{uuid.uuid4()}'
    )
    
    logger.error(
        event = event,
        message = exc.detail,
        method=request.method,
        path=request.url.path,
        status=exc.status_code,
        detail=exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
            content={
                "status": "error",
                "data": None,
                "message": exc.detail,
                'request_id': request_id
            }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    event = "Validation error occurred"
    request_id = (
        request.headers.get('request_id') 
        or request.headers.get('x-request-id') 
        or f'{__name__}+{uuid.uuid4()}'
    )
    # Extract missing/invalid keys
    missing_keys = []
    for error in exc.errors():
        loc = error.get("loc", [])
        if len(loc) > 1 and loc[0] == "body":
            missing_keys.append(str(loc[1]))
        elif len(loc) == 1:
            missing_keys.append(str(loc[0]))

    # Prepare compact error message
    if missing_keys:
        required_keys_str = ', '.join(sorted(set(missing_keys)))
        message_str = f"Missing key(s): {required_keys_str}"
    else:
        message_str = "Validation failed due to incorrect input format."

    logger.error(
        event = event,
        message = message_str,
        method=request.method,
        path=request.url.path,
        status=422,
        detail=exc.errors()
    )
    return JSONResponse(
        status_code=422,
            content={
                "status": "error",
                "data": None,
                "message": message_str,
                'request_id': request_id
            }
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    logger = structlog.get_logger('Exception Logger')

    message = "Unhandled exception occurred"
    request_id = (
        request.headers.get('request_id') 
        or request.headers.get('x-request-id') 
        or f'{__name__}+{uuid.uuid4()}'
    )
    logger.error(
        event = message,
        message = message,
        method=request.method,
        path=request.url.path,
        status=500
    )
    return JSONResponse(
        status_code=500,
            content={
                "status": "error",
                "data": None,
                "message": message,
                'request_id': request_id
            }
    )
