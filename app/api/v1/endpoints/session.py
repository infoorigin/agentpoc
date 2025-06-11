import structlog
logger = structlog.get_logger(__name__)

from fastapi import APIRouter,Depends,HTTPException, Query, Request
from pydantic import BaseModel
from app.api.security.api_key import get_api_key
from typing import List,Optional
from llama_index.core.llms import ChatMessage
from app.storage.file_object_reader import FileObjectReader
from app.storage.s3_object_reader import S3ObjectReader
from app.cache.joblib_session_cache import JoblibSessionCache
from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession

from app.utils.success_wrapper import SuccessResponse
from app.core.s3_client import get_s3_client
import uuid
# app/api/v1/endpoints/modelquery.py

session_router = APIRouter()

class ModelPerformanceRequest(BaseModel):
    file_key:str
    bucket_name: str
    chat_history: Optional[List[ChatMessage]] = [] # Added chat history. Default is an empty list
    
@session_router.post('/session_id')
async def create_session(request:Request, req:ModelPerformanceRequest, api_key:str = Depends(get_api_key) ):

    request_id = (
        request.headers.get('request_id') 
        or request.headers.get('x-request-id') 
        or f'{__name__}+{uuid.uuid4()}'
    )
    try:
        logger.info("Before reading file", request_id=request_id, bucket_name=req.bucket_name, file_key=req.file_key)
        reader = S3ObjectReader(bucket = req.bucket_name, key=req.file_key, s3_client=get_s3_client())
        # reader = FileObjectReader("tests/test_data/pateint_fullfilment_model.pkl")
        memory = JoblibSessionCache.get_instance()
        session = ModelAnalyzerSession(memory)
        session_id = session.create_session(reader)
        result = {"session_id": session_id}
        
        return SuccessResponse.ok(request_id = request_id, data = result) 
        
    except Exception as e:
        logger.exception("Error occurred in model session end point", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f'Session error: {str(e)}')

  