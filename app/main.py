import structlog
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logger.log_config import configure_logging
from app.api.api_router import api_router
from app.utils.errors_handling import (
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
# from app.middlewares.standard_response_middleware import CustomResponseMiddleware
from app.middlewares.request_id_middleware import RequestIDMiddleware
from app.middlewares.exception_logging_middleware import ExceptionLoggingMiddleware

# ---------- Setup Logging ----------
configure_logging()
logger = structlog.get_logger()

# ---------- Load App Metadata ----------
metadata_path = Path("app/app_metadata.json")
app_config = json.loads(metadata_path.read_text(encoding="utf-8"))

# Extract FastAPI-specific fields only
fastapi_keys = {
    "title", "description", "version", "root_path",
    "docs_url", "redoc_url", "contact"
}
fastapi_metadata = {k: v for k, v in app_config.items() if k in fastapi_keys}

# ---------- Initialize FastAPI ----------
app = FastAPI(**fastapi_metadata)

# ---------- Middleware ----------
app.add_middleware(RequestIDMiddleware)
app.add_middleware(ExceptionLoggingMiddleware)
# app.add_middleware(CustomResponseMiddleware)

# ---------- CORS Handling ----------
allowed_origins = app_config.get("allowed_origins") or ["*"]
is_wildcard = allowed_origins == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=not is_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Exception Handlers ----------
app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore
app.add_exception_handler(Exception, unhandled_exception_handler)

# ---------- Routers ----------
app.include_router(api_router, prefix="/api/v1")

# ---------- Health Check ----------
@app.get("/")
def read_root():
    return {"message": "Welcome to Agentic Apps"}
