import structlog

from app.core.utils.agent_result_utils import AgentResultUtils
logger = structlog.get_logger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
# from fastapi

from pydantic import BaseModel
from app.api.security.api_key import get_api_key
from app.core.workflows.model_performance_analysis_workflow import ModelPerformanceWorkflow
from typing import List,Literal,Optional
from llama_index.core.llms import ChatMessage
from app.storage.file_object_reader import FileObjectReader
from app.storage.s3_object_reader import S3ObjectReader
from app.cache.joblib_session_cache import JoblibSessionCache
from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from app.llms.llm_manager import LLMManager
from app.core.agent.kernel.workflow_agent.function_agent import FunctionAgent
from agent_apps.model_analyzer.tools.shap_insight_narrative_tool import shap_insight_narrative_tool
from agent_apps.model_analyzer.tools.shap_summary_plot_tool import shap_summary_plot_image_tool
from agent_apps.model_analyzer.agent.ds_model_agents import PharmaModelAnalyzerAgent
from llama_index.core.workflow.context import Context
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    ToolCallResult
)
from agent_apps.model_analyzer.agent.ds_model_kernels import ModelAnalyzerKernel
from agent_apps.model_analyzer.tools.tool_names import ModelAnalyzerToolName

from app.utils.success_wrapper import SuccessResponse
from app.utils.error_wrapper import ErrorResponse
from app.core.config import settings
from app.core.s3_client import get_s3_client
import uuid
import dataclasses
from typing import Literal, Any
# app/api/v1/endpoints/modelquery.py

router = APIRouter()

class ModelPerformanceRequest(BaseModel):
    user_query:str
    file_key:str
    bucket_name: str
    chat_history: Optional[List[ChatMessage]] = [] # Added chat history. Default is an empty list

@dataclasses.dataclass
class OutputData:
    content_type: Literal['BASE64IMAGE','STORAGEARTIFACT','JSON','STRING','BINARY']
    conv_text: str
    content: Any | None = None


@router.post('/agent/query')
async def agent_query(request:Request, req:ModelPerformanceRequest, api_key:str = Depends(get_api_key)):
    
    request_id = (
        request.headers.get('request_id') 
        or request.headers.get('x-request-id') 
        or f'{__name__}+{uuid.uuid4()}'
    )
    
    try:
      llm = LLMManager.get_llm(model_name="40-mini")
      agent = PharmaModelAnalyzerAgent(
          tools=[shap_summary_plot_image_tool, shap_insight_narrative_tool],
          llm=llm,
      )
    except Exception as e:
        logger.exception("LLM or Agent initialization error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f'LLM or Agent initialization error: {str(e)}')
    try:
      # reader = FileObjectReader("tests/test_data/pateint_fullfilment_model.pkl")
      reader = S3ObjectReader(bucket = req.bucket_name, key=req.file_key, s3_client=get_s3_client())
    except Exception as e:
        logger.exception("File reading error", request_id=request_id, bucket_name=req.bucket_name, file_key=req.file_key, error=str(e))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail = f'File reading error: {str(e)}')
    
    try:
      memory = JoblibSessionCache.get_instance()
      session = ModelAnalyzerSession(memory)
      session_id = session.create_session(reader)
      conversation_id = str(uuid.uuid4())
      context = Context(workflow=agent) # type: ignore

      await context.set("session_id",session_id)
      await context.set("model_analyzer_session",session)
      await context.set("conversation_id",conversation_id)
    except Exception as e:
        logger.exception("Session creation error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail = f'Session creation error: {str(e)}')

    try:
      # raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail='Testing Agent run error')
      response: AgentOutput = await agent.run(user_msg = req.user_query, chat_history = req.chat_history, ctx = context, kernel_cls = ModelAnalyzerKernel) # Sample query "generate shap summary image for 5 features"
      response_content = str(response)
      savant_agent_output = await AgentResultUtils.parse_result_output(context, response)
      return SuccessResponse.ok(request_id = request_id, data = savant_agent_output)
    except Exception as e:
        logger.exception("Agent run error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail = f'Agent run error: {str(e)}')


@router.post('/query')
async def query_workflow(request:Request, req:ModelPerformanceRequest, api_key:str = Depends(get_api_key)):
    
    request_id = (
        request.headers.get('request_id') 
        or request.headers.get('x-request-id') 
        or f'{__name__}+{uuid.uuid4()}'
    )
    try:
      llm = LLMManager.get_llm(model_name="40-mini")
      agent = PharmaModelAnalyzerAgent(
          tools=[shap_summary_plot_image_tool, shap_insight_narrative_tool],
          llm=llm,
      )
    except Exception as e:
        logger.exception("LLM or Agent initialization error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f'LLM or Agent initialization error: {str(e)}')
    try:
      # reader = FileObjectReader("tests/test_data/pateint_fullfilment_model.pkl")
      reader = S3ObjectReader(bucket = req.bucket_name, key=req.file_key, s3_client=get_s3_client())
    except Exception as e:
        logger.exception("File reading error", request_id=request_id, bucket_name=req.bucket_name, file_key=req.file_key, error=str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail = f'File reading error: {str(e)}')
    
    try:
      memory = JoblibSessionCache.get_instance()
      session = ModelAnalyzerSession(memory)
      session_id = session.create_session(reader)
      conversation_id = str(uuid.uuid4())
      context = Context(workflow=agent) # type: ignore

      await context.set("session_id",session_id)
      await context.set("model_analyzer_session",session)
      await context.set("conversation_id",conversation_id)
    except Exception as e:
        logger.exception("Session creation error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail = f'Session creation error: {str(e)}')

    try:
      response: AgentOutput = await agent.run(user_msg = req.user_query, chat_history = req.chat_history, ctx = context, kernel_cls = ModelAnalyzerKernel) # Sample query "generate shap summary image for 5 features"
      savant_agent_output = await AgentResultUtils.parse_result_output(context, response)
      shap_plot_output = AgentResultUtils.get_raw_output( response.tool_calls, ModelAnalyzerToolName.SHAP_SUMMARY_PLOT) # type: ignore

      base64_img = shap_plot_output.content if shap_plot_output else None
      
    except Exception as e:
        logger.exception("Agent run error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail = f'Agent run error: {str(e)}')
    try:
      if base64_img:
          content_type = 'BASE64IMAGE'
          content = base64_img

      # Add elif logic for other content types if needed in the future

      else:
          content_type = 'STRING'
          content = None

      response_content = str(response)

      result = OutputData(
          content_type=content_type,
          content=content,
          conv_text=response_content
      )
      return SuccessResponse.ok(request_id = request_id, data = result)
    
    except Exception as e:
        logger.exception("Response content processing error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail = f'Response content processing error: {str(e)}')

# Dummy endpoint to return JSON data. Delete this endpoint in production.

@router.get('/dummy-json')
def dummy_json(request: Request):

    request_id = (
        request.headers.get('request_id') 
        or request.headers.get('x-request-id') 
        or f'{__name__}+{uuid.uuid4()}'
    )
    content_type = 'JSON'
    # Dummy Json Data
    content = [
    {
      "subject": "Mathematics",
      "date": "2023-01-15",
      "score": 85
    },
    {
      "subject": "Science",
      "date": "2023-01-15",
      "score": 90
    },
    {
      "subject": "English",
      "date": "2023-01-15",
      "score": 78
    },
    {
      "subject": "Mathematics",
      "date": "2023-02-15",
      "score": 88
    },
    {
      "subject": "Science",
      "date": "2023-02-15",
      "score": 92
    },
    {
      "subject": "English",
      "date": "2023-02-15",
      "score": 80
    },
    {
      "subject": "Mathematics",
      "date": "2023-03-15",
      "score": 90
    },
    {
      "subject": "Science",
      "date": "2023-03-15",
      "score": 95
    },
    {
      "subject": "English",
      "date": "2023-03-15",
      "score": 82
    }
  ]
    conv_text =  "The dummy json data has been generated successfully."
    result = OutputData(
        content_type = content_type,
        content = content,
        conv_text = conv_text
    )
    try:
        return SuccessResponse.ok(request_id = request_id, data = result)
    except Exception as e:
        logger.exception("Error occurred in dummy JSON endpoint", request_id = request_id, error = str(e))
        raise HTTPException(status_code = 500, detail = f'Dummy JSON error: {str(e)}')  

  