from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging

from src.services.orchestrator import Orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


class PlanRequest(BaseModel):
    instruments: List[str] = Field(..., min_items=1)
    timeframes: List[str] = Field(..., min_items=1)
    risk: str = Field("medium")


@router.post("/plan")
async def plan(req: PlanRequest):
    try:
        orch = Orchestrator()
        plan = orch.plan(req.instruments, req.timeframes, req.risk)
        return {"status": "success", "data": plan}
    except Exception as e:
        logger.error(f"Plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def run(req: PlanRequest):
    try:
        orch = Orchestrator()
        result = orch.run(req.instruments, req.timeframes, req.risk)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Run error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


