# app/middlewares/standard_response_middleware.py

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing import Any
import json

class CustomizedResponse(BaseModel):
    status_code: int
    code: str
    message: str
    data: Any

class CustomResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:

        # Exclude specific paths from middleware processing
        included_paths = ["/agentic-network/api/v1/workflows/dialogue", '/agents/apps/', "/agents/apps/warning"]
        if not request.url.path in included_paths:
            return await call_next(request)

        # Proceed with the normal response
        response = await call_next(request)

        # Initialize default response code and message
        response_code = "SUCCESS"
        response_message = "Request processed successfully."
        status_code = response.status_code

        # If there’s an error (4xx or 5xx), modify response code and message
        if status_code >= 500:
            response_code = "ERROR"
            response_message = "An error occurred."
        elif status_code >= 400:
            response_code = "WARNING"
            response_message = "Something is wrong with request"

        # Prepare the actual data from the response
        if isinstance(response, StreamingResponse.__base__): # type: ignore
            body = b"".join([chunk async for chunk in response.body_iterator]) # type: ignore
            data = json.loads(body.decode())

            # Instead of returning data directly, we can keep the response type intact
            custom_response = JSONResponse(content={
                "status_code": status_code,
                "code": response_code,
                "message": response_message,
                "data": data  # Include raw data here
            }, 
            status_code = status_code
            )
            
            return custom_response

        elif isinstance(response, JSONResponse):
            data = response  # Get the JSON object directly
            # Do not serialize again, because data is already a Python dict
        else:
            data = None  # Handle other response types if necessary

        # Construct the custom response
        custom_response = CustomizedResponse(
            status_code=status_code,
            code=response_code,
            message=response_message,
            data=data  # Here, data should be a dict or list, not a string
        )

        return JSONResponse(content=custom_response.model_dump(), status_code=status_code)  # Return the custom response
