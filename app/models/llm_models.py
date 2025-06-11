from pydantic  import BaseModel
from typing import Optional

class LLMTokenCount(BaseModel):
    embedding_tokens : Optional[int]=0
    prompt_tokens : Optional[int]=0
    completion_tokens : Optional[int]=0
    total_tokens : Optional[int]=0

    def add_tokens(self,tokens: 'LLMTokenCount'):
        self.embedding_tokens = self.embedding_tokens
        self.prompt_tokens = self.prompt_tokens + tokens.prompt_tokens
        self.completion_tokens = self.completion_tokens +tokens.completion_tokens
        self.total_tokens = self.total_tokens + tokens.total_tokens