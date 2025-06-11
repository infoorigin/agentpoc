from pydantic import BaseModel, Field
from typing import Dict, Literal,Any,Optional
from llama_index.core.llms import MessageRole

class ToolImageRespose(BaseModel):
    image_key: str = Field(description="image_key of the generated image")
    session_id: str = Field(description="Model Analysis session id")
    encoded_image: str = Field(description="Base 64 encoded image")

class SavantChatMessage(BaseModel):
    """Savant Chat message."""

    role: MessageRole = MessageRole.USER
    content:Any=None

class ToolResultOutput(BaseModel):
    tool_name:str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    output_ref: Optional[str] = None
    content_type : Literal['BASE64IMAGE','STORAGEARTIFACT','JSON','STRING','BINARY']
    content:Any

class ToolResult(BaseModel):
    tool_name:str
    tool_input: Dict[str, Any]
    content: str
    tool_output:ToolResultOutput|None

class SavantAgentOutput(BaseModel):
    """Savant Agent output."""

    response: SavantChatMessage
    tools_results: list[ToolResult]  

    def __str__(self) -> str:
        return self.response.content or ""  