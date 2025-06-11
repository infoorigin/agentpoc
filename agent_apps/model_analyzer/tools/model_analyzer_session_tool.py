import os
from urllib.parse import urlparse

from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from app.cache.session_cache_factory import SessionCacheFactory
from app.storage.file_object_reader import FileObjectReader
from app.storage.s3_object_reader import S3ObjectReader
from llama_index.core.workflow.context import Context


async def create_model_analyzer_session_from_pickle(ctx: Context,file_path: str) -> str:
    """
    Create a model analysis session from a pickle file stored locally or in an S3 bucket.

    Based on the file path, this tool will:
    - Use FileObjectReader for local `.pkl` files
    - Use S3ObjectReader for `s3://` URIs

    It initializes a ModelAnalyzerSession using JoblibSessionCache and returns a unique session ID.

    Args:
        file_path (str): Path to the pickle file. Supported formats:
            - Local path (e.g., "/path/to/model.pkl")
            - S3 URI (e.g., "s3://my-bucket/models/model.pkl")

    Returns:
        str: The generated session ID for the model analyzer session.

    Raises:
        FileNotFoundError: If the local file is missing.
        ValueError: If the S3 URI is invalid.
    """
    if file_path.startswith("s3://"):
        parsed = urlparse(file_path)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 path: {file_path}")
        reader = S3ObjectReader(bucket=bucket, key=key)
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        reader = FileObjectReader(file_path)

    session_cache = SessionCacheFactory.get_cache_manager()
    session = ModelAnalyzerSession(session_cache)
    session_id = session.create_session(reader)
    await ctx.set("session_id",session_id)
    return session_id
