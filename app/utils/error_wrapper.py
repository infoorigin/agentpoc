from fastapi.responses import JSONResponse
from typing import Any, Optional

class ErrorResponse:
    @staticmethod
    def bad_request(request_id: str, message: str = "Bad Request", status_code: int = 400, data: Optional[Any] = None):
        # return ErrorResponse._build(request_id, message, 400, data)
        return ErrorResponse._build(request_id, message, status_code, data)

    @staticmethod
    def unauthorized(request_id: str, message: str = "Unauthorized", status_code: int = 401, data: Optional[Any] = None):
        # return ErrorResponse._build(message, 401, data)
        return ErrorResponse._build(request_id, message, status_code, data)

    @staticmethod
    def forbidden(request_id: str, message: str = "Forbidden", status_code: int = 401, data: Optional[Any] = None):
        # return ErrorResponse._build(message, 403, data)
        return ErrorResponse._build(request_id, message, status_code, data)

    @staticmethod
    def not_found(request_id: str, message: str = "Not Found", status_code: int = 401, data: Optional[Any] = None):
        # return ErrorResponse._build(message, 404, data)
        return ErrorResponse._build(request_id, message, status_code, data)

    @staticmethod
    def conflict(request_id: str, message: str = "Conflict", status_code: int = 409, data: Optional[Any] = None):
        # return ErrorResponse._build(message, 409, data)
        return ErrorResponse._build(request_id, message, status_code, data)

    @staticmethod
    def unprocessable(request_id: str, message: str = "Unprocessable Entity", status_code: int = 422, data: Optional[Any] = None):
        return ErrorResponse._build(request_id, message, status_code, data)

    @staticmethod
    def server_error(request_id: str, message: str = "Internal Server Error", status_code: int = 500, data: Optional[Any] = None):
        return ErrorResponse._build(request_id, message, status_code, data)

    @staticmethod
    def _build(request_id: str, message: str, status_code: int, data: Optional[Any]):
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "error",
                "data": data,
                "message": message,
                'request_id': request_id
            }
        )
