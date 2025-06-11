from app.models.llm_models import LLMTokenCount

def get_token_count(**additional_kwargs) -> LLMTokenCount:

    embedding_tokens = additional_kwargs.get('embedding_tokens',0)
    prompt_tokens = additional_kwargs.get('prompt_tokens',0)
    completion_tokens = additional_kwargs.get('completion_tokens',0)
    total_tokens = embedding_tokens+prompt_tokens+completion_tokens
    return LLMTokenCount(embedding_tokens=embedding_tokens, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,total_tokens=total_tokens)