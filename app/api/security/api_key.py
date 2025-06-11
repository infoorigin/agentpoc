from app.core.config import settings
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import APIKeyHeader

app = FastAPI()

#Load API key from environment variable
API_KEY = settings.AI_AGENT_API_KEY

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error= False)
async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key