from fastapi import APIRouter
from app.api.v1.agents.agents_api import agent_router
from app.api.v1.endpoints.modelquery import router
from app.api.v1.endpoints.session import session_router

api_router = APIRouter()

api_router.include_router( agent_router, prefix="/agents")
api_router.include_router( router, prefix="/modelquery")

api_router.include_router( session_router, prefix="/session")