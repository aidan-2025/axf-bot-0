"""
Sentiment API routes
Simple endpoints for sentiment analysis
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/current")
async def get_current_sentiment():
    """Get current market sentiment"""
    try:
        return {
            "status": "success",
            "data": {
                "overall": 15.5,
                "news": 12.3,
                "social": 18.7,
                "technical": 15.5,
                "last_updated": "2024-01-15T12:00:00Z"
            }
        }
    except Exception as e:
        logger.error(f"Error getting sentiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sentiment")
