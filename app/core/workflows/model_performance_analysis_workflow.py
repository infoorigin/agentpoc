import time
from llama_index.core.workflow import(
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from pydantic import BaseModel
from app.core.config import settings
from typing import Literal,Optional,List,Dict, Any
from app.models.llm_models import LLMTokenCount
from app.tools.model_performance_query import a_queryResponse
from app.llms.open_ai import  llm_openAI
import asyncio


class ModelPerformanceEvent(Event):
    user_query: str 
    query_response : str
    tokenConsumed : int

class ModelPerformedResponse(BaseModel):
    user_query: str
    query_response : str
    tokenConsumed : int


class ModelPerformanceWorkflow(Workflow):
    llm = llm_openAI

    @step()
    async def model_performance (self, ctx: Context, ev: StartEvent) -> ModelPerformanceEvent:
        #token_count = LLMTokenCount()
        #await ctx.set("token_count", token_count)
        #step_tokens = await ctx.get("token_count")
        response, token_count = await a_queryResponse(user_query = ev.user_query)
        #step_token.add_tokens(token_count)
        return ModelPerformanceEvent(user_query=ev.user_query, query_response = response, tokenConsumed = token_count)

    @step()
    async def generate_response(self, ctx:Context, ev:ModelPerformanceEvent) -> StopEvent:
        workflowresponse = ModelPerformedResponse(user_query=ev.user_query, query_response=ev.query_response,tokenConsumed=ev.tokenConsumed)
        return StopEvent(result={'result':workflowresponse})