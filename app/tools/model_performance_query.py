from dotenv import load_dotenv
load_dotenv()
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from app.models.llm_models import LLMTokenCount
from app.llms.llm_token_program import LLMTextCompletionProgramWithToken
from app.llms.open_ai import  llm_openAI

class userMessagePromptResponse(BaseModel):
    """Data model for a User Message Response."""

    user_query: str
    valid_query: bool
    assistant_response: str
    reason: str

class PromptResponse(BaseModel):
    PromptContent:str

dialogue_handling_str = """"""


async def a_queryResponse(user_query:str) -> Tuple[str,LLMTokenCount]:
    """
    This tool must be used to response back user's message
    
        user_query (str): The user's input.
    """
    program = LLMTextCompletionProgramWithToken.from_defaults(
        output_cls=userMessagePromptResponse,
        llm=llm_openAI,
        prompt_template_str=dialogue_handling_str,
        verbose=True,
    )
    #result,token_count = await program.acall(user_query=user_query)
    result  = "Will get back with response Shortly!"
    token_count= 50
    return result,token_count


async def queryResponse(user_query:str) -> Tuple[str,LLMTokenCount]:
    """
    This tool must be used to response back user's message
    
        user_query (str): The user's input.
    """
    program = LLMTextCompletionProgramWithToken.from_defaults(
        output_cls=userMessagePromptResponse,
        llm=llm_openAI,
        prompt_template_str=dialogue_handling_str,
        verbose=True,
    )
    #result,token_count = await program.acall(user_query=user_query)
    result  = "Will get back with response Shortly!"
    token_count= 50
    return result,token_count

userMessageHandlingTool = FunctionTool.from_defaults(fn=queryResponse, async_fn =a_queryResponse)