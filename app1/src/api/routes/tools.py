#!/usr/bin/env python3
"""
Tools API Routes
Expose individual tool adapters via REST API.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from src.services.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize tool registry
tool_registry = ToolRegistry()


class NewsRequest(BaseModel):
    instruments: List[str] = Field(..., min_items=1)
    hours_back: int = Field(24, ge=1, le=168)  # 1 hour to 1 week


class SentimentRequest(BaseModel):
    instruments: List[str] = Field(..., min_items=1)


class IndicatorsRequest(BaseModel):
    price_data: List[Dict[str, Any]] = Field(..., min_items=1)
    indicators: Optional[List[str]] = Field(None)


class EconomicCalendarRequest(BaseModel):
    days_ahead: int = Field(7, ge=1, le=30)


@router.get("/tools")
async def list_tools():
    """List all available tools."""
    try:
        tools = tool_registry.list_tools()
        return {"status": "success", "tools": tools}
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/news")
async def get_news(request: NewsRequest):
    """Get forex news for specified instruments."""
    try:
        result = tool_registry.get_news_data(
            instruments=request.instruments,
            hours_back=request.hours_back
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/economic-calendar")
async def get_economic_calendar(days_ahead: int = 7):
    """Get economic calendar events."""
    try:
        result = tool_registry.get_economic_calendar(days_ahead=days_ahead)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting economic calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/market-sentiment")
async def get_market_sentiment(request: SentimentRequest):
    """Get market sentiment for specified instruments."""
    try:
        result = tool_registry.get_market_sentiment(instruments=request.instruments)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting market sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/social-sentiment")
async def get_social_sentiment(request: SentimentRequest):
    """Get social media sentiment for specified instruments."""
    try:
        result = tool_registry.analyze_social_sentiment(instruments=request.instruments)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting social sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/fear-greed")
async def get_fear_greed_index():
    """Get fear and greed index."""
    try:
        result = tool_registry.get_fear_greed_index()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting fear/greed index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/central-bank-sentiment")
async def get_central_bank_sentiment():
    """Get central bank sentiment analysis."""
    try:
        result = tool_registry.analyze_central_bank_sentiment()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting central bank sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/technical-indicators")
async def calculate_technical_indicators(request: IndicatorsRequest):
    """Calculate technical indicators for price data."""
    try:
        result = tool_registry.calculate_technical_indicators(
            price_data=request.price_data,
            indicators=request.indicators
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/support-resistance")
async def get_support_resistance_levels(request: IndicatorsRequest):
    """Get support and resistance levels for price data."""
    try:
        result = tool_registry.get_support_resistance_levels(
            price_data=request.price_data
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting support/resistance levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/comprehensive-analysis")
async def get_comprehensive_analysis(request: SentimentRequest):
    """Get comprehensive analysis using all available tools."""
    try:
        result = tool_registry.get_comprehensive_analysis(
            instruments=request.instruments
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error getting comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

