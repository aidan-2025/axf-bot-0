from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from src.services.ai_service import AIService

logger = logging.getLogger(__name__)
router = APIRouter()


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=4, max_length=4000)
    context: Optional[str] = Field(None, max_length=4000)


class GenerateStrategyRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=6000)
    constraints: Optional[Dict[str, Any]] = None


@router.post("/research")
async def research(request: ResearchRequest):
    try:
        service = AIService()
        result = service.research(request.query, request.context)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-strategy")
async def generate_strategy(request: GenerateStrategyRequest):
    try:
        service = AIService()
        result = service.generate_strategy(request.prompt, request.constraints)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Generate strategy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


