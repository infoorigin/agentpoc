from pydantic import BaseModel
from typing import Literal,List,Dict,Any,Optional
from llama_index.core.llms import ChatMessage

from app.models.llm_models import LLMTokenCount

class DialogueRequest(BaseModel):
    usermessage: str
    chatHistory: Optional[List[Dict[str,Any]]]
