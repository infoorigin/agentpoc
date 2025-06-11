import logging
from fastapi import APIRouter
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.utils.success_wrapper import SuccessResponse
logger = logging.getLogger(__name__)

agent_router = APIRouter()



@agent_router.post("/hello", tags=["Semantic SQL Translation"])
def hello(name:str):
    try:
        return SuccessResponse.ok(data= {"greetings": f"Hello {name}"})
       

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
