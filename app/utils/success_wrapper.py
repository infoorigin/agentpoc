# utils/success_response.py

from fastapi.responses import JSONResponse
from typing import Any, Optional

class SuccessResponse:
    @staticmethod
    # def ok(data: Any = None, message: str = "OK", status_code: int = 200):
    def ok(request_id: str, message: str = "OK", status_code: int = 200, data: Optional[Any] = None):
        return {
                    'request_id': request_id,
                    "message": message,
                    "status": "success",
                    "data": data,
                }

    @staticmethod
    def created(request_id: str, message: str = "Created", status_code = 201, data: Any = None, ):
        return SuccessResponse.ok(request_id, data = data, message = message, status_code = status_code)

    @staticmethod
    def updated(request_id: str, message: str = "Updated", status_code = 200, data: Any = None, ):
        return SuccessResponse.ok(request_id, data = data, message = message, status_code = status_code)

    @staticmethod
    def deleted(request_id: str, message: str = "Deleted", status_code = 200, data: Any = None, ):
        return SuccessResponse.ok(request_id, data = data, message = message, status_code = status_code)
