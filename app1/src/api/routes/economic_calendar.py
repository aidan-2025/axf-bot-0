#!/usr/bin/env python3
"""
Economic Calendar API Routes
API endpoints for economic calendar functionality
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from ...economic_calendar import (
    EconomicEvent, CalendarFilter, CalendarStats, EventAnalysisResult,
    CalendarProcessor, CalendarProcessorConfig, CalendarServiceConfig, EventAnalysisConfig
)
from ...economic_calendar.models import EventImpact, EventCategory, Country, Currency

logger = logging.getLogger(__name__)

# Global calendar processor instance
calendar_processor: Optional[CalendarProcessor] = None

# Pydantic models for API
class EventFilterRequest(BaseModel):
    """Request model for event filtering"""
    countries: Optional[List[str]] = Field(None, description="Country codes to filter by")
    currencies: Optional[List[str]] = Field(None, description="Currency codes to filter by")
    categories: Optional[List[str]] = Field(None, description="Event categories to filter by")
    impacts: Optional[List[str]] = Field(None, description="Impact levels to filter by")
    start_date: Optional[datetime] = Field(None, description="Start date for events")
    end_date: Optional[datetime] = Field(None, description="End date for events")
    limit: Optional[int] = Field(None, description="Maximum number of events to return")

class UpcomingEventsRequest(BaseModel):
    """Request model for upcoming events"""
    hours_ahead: int = Field(24, description="Hours ahead to look for events")
    countries: Optional[List[str]] = Field(None, description="Country codes to filter by")
    impacts: Optional[List[str]] = Field(None, description="Impact levels to filter by")

class EventAnalysisResponse(BaseModel):
    """Response model for event analysis"""
    event: EconomicEvent
    market_impact_score: float
    volatility_expected: float
    affected_currency_pairs: List[str]
    trading_opportunities: List[str]
    risk_factors: List[str]
    confidence: float
    analysis_timestamp: datetime

class CalendarHealthResponse(BaseModel):
    """Response model for calendar health"""
    status: str
    processor_initialized: bool
    last_update: Optional[str]
    cached_analyses: int
    calendar_stats: Optional[Dict[str, Any]]

# Initialize calendar processor
async def initialize_calendar_processor():
    """Initialize the calendar processor"""
    global calendar_processor
    
    if calendar_processor is None:
        try:
            logger.info("Initializing calendar processor")
            
            # Create configuration
            calendar_service_config = CalendarServiceConfig(
                fmp_api_key=None,  # Will be set from environment
                trading_economics_api_key=None,  # Will be set from environment
                eodhd_api_key=None,  # Will be set from environment
                enable_fmp=False,  # Disable by default since we don't have API keys
                enable_trading_economics=False,
                enable_eodhd=False,
                enable_forex_factory=True,  # Enable mock data
                cache_ttl_hours=1,
                max_events_per_source=1000,
                deduplicate_events=True,
                auto_update_interval_minutes=30,
                update_on_startup=True
            )
            
            event_analyzer_config = EventAnalysisConfig(
                impact_weight=0.4,
                category_weight=0.3,
                country_weight=0.2,
                timing_weight=0.1,
                base_volatility=0.1,
                high_impact_multiplier=2.0,
                very_high_impact_multiplier=3.0
            )
            
            processor_config = CalendarProcessorConfig(
                calendar_service_config=calendar_service_config,
                event_analyzer_config=event_analyzer_config,
                auto_analyze_events=True,
                analysis_batch_size=50,
                update_interval_minutes=30,
                enable_sentiment_integration=True,
                enable_market_data_integration=True
            )
            
            calendar_processor = CalendarProcessor(processor_config)
            await calendar_processor.initialize()
            
            logger.info("Calendar processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize calendar processor: {e}")
            raise

# Get calendar processor instance
def get_calendar_processor() -> CalendarProcessor:
    """Get the calendar processor instance"""
    global calendar_processor
    if calendar_processor is None:
        raise HTTPException(status_code=503, detail="Calendar processor not initialized")
    return calendar_processor

# Create router
router = APIRouter()

@router.on_event("startup")
async def startup_calendar_processor():
    """Initialize calendar processor on startup"""
    await initialize_calendar_processor()

@router.on_event("shutdown")
async def shutdown_calendar_processor():
    """Cleanup calendar processor on shutdown"""
    global calendar_processor
    if calendar_processor:
        await calendar_processor.close()
        calendar_processor = None
    logger.info("Calendar processor shutdown")

@router.get("/health", response_model=CalendarHealthResponse)
async def get_calendar_health():
    """Get economic calendar health status"""
    try:
        processor = get_calendar_processor()
        processor_info = await processor.get_processor_info()
        
        # Get calendar stats
        calendar_stats = None
        try:
            stats = await processor.get_calendar_stats()
            calendar_stats = stats.to_dict()
        except Exception as e:
            logger.warning(f"Could not get calendar stats: {e}")
        
        return CalendarHealthResponse(
            status="healthy" if processor_info.get("status") == "initialized" else "unhealthy",
            processor_initialized=processor_info.get("status") == "initialized",
            last_update=processor_info.get("last_update"),
            cached_analyses=processor_info.get("cached_analyses", 0),
            calendar_stats=calendar_stats
        )
    except Exception as e:
        logger.error(f"Error getting calendar health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events", response_model=List[EconomicEvent])
async def get_events(
    countries: Optional[str] = Query(None, description="Comma-separated country codes"),
    currencies: Optional[str] = Query(None, description="Comma-separated currency codes"),
    categories: Optional[str] = Query(None, description="Comma-separated event categories"),
    impacts: Optional[str] = Query(None, description="Comma-separated impact levels"),
    start_date: Optional[datetime] = Query(None, description="Start date for events"),
    end_date: Optional[datetime] = Query(None, description="End date for events"),
    limit: Optional[int] = Query(None, description="Maximum number of events")
):
    """Get economic events with optional filtering"""
    try:
        processor = get_calendar_processor()
        
        # Parse filter parameters
        filter_criteria = CalendarFilter(
            countries=[Country(c.strip()) for c in countries.split(",")] if countries else None,
            currencies=[Currency(c.strip()) for c in currencies.split(",")] if currencies else None,
            categories=[EventCategory(c.strip()) for c in categories.split(",")] if categories else None,
            impacts=[EventImpact(i.strip()) for i in impacts.split(",")] if impacts else None,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        events = await processor.calendar_service.get_events(filter_criteria)
        return events
        
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/upcoming", response_model=List[EconomicEvent])
async def get_upcoming_events(
    hours_ahead: int = Query(24, description="Hours ahead to look for events"),
    countries: Optional[str] = Query(None, description="Comma-separated country codes"),
    impacts: Optional[str] = Query(None, description="Comma-separated impact levels")
):
    """Get upcoming economic events"""
    try:
        processor = get_calendar_processor()
        
        # Parse parameters
        country_list = [c.strip() for c in countries.split(",")] if countries else None
        impact_list = [i.strip() for i in impacts.split(",")] if impacts else None
        
        events = await processor.get_upcoming_analysis(
            hours_ahead=hours_ahead,
            countries=country_list,
            impacts=impact_list
        )
        
        # Return just the events (not the full analysis)
        return [result.event for result in events]
        
    except Exception as e:
        logger.error(f"Error getting upcoming events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/high-impact", response_model=List[EconomicEvent])
async def get_high_impact_events(
    hours_ahead: int = Query(24, description="Hours ahead to look for events"),
    countries: Optional[str] = Query(None, description="Comma-separated country codes")
):
    """Get high impact economic events"""
    try:
        processor = get_calendar_processor()
        
        # Parse parameters
        country_list = [c.strip() for c in countries.split(",")] if countries else None
        
        events = await processor.get_high_impact_analysis(
            hours_ahead=hours_ahead,
            countries=country_list
        )
        
        # Return just the events (not the full analysis)
        return [result.event for result in events]
        
    except Exception as e:
        logger.error(f"Error getting high impact events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis", response_model=List[EventAnalysisResponse])
async def get_event_analysis(
    hours_ahead: int = Query(24, description="Hours ahead to analyze events"),
    countries: Optional[str] = Query(None, description="Comma-separated country codes"),
    impacts: Optional[str] = Query(None, description="Comma-separated impact levels")
):
    """Get analysis for economic events"""
    try:
        processor = get_calendar_processor()
        
        # Parse parameters
        country_list = [c.strip() for c in countries.split(",")] if countries else None
        impact_list = [i.strip() for i in impacts.split(",")] if impacts else None
        
        results = await processor.get_upcoming_analysis(
            hours_ahead=hours_ahead,
            countries=country_list,
            impacts=impact_list
        )
        
        # Convert to response format
        analysis_responses = []
        for result in results:
            analysis_responses.append(EventAnalysisResponse(
                event=result.event,
                market_impact_score=result.market_impact_score,
                volatility_expected=result.volatility_expected,
                affected_currency_pairs=result.affected_currency_pairs,
                trading_opportunities=result.trading_opportunities,
                risk_factors=result.risk_factors,
                confidence=result.confidence,
                analysis_timestamp=result.analysis_timestamp
            ))
        
        return analysis_responses
        
    except Exception as e:
        logger.error(f"Error getting event analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/summary")
async def get_analysis_summary(
    hours_ahead: int = Query(24, description="Hours ahead to analyze events"),
    countries: Optional[str] = Query(None, description="Comma-separated country codes")
):
    """Get summary of event analysis"""
    try:
        processor = get_calendar_processor()
        
        # Parse parameters
        country_list = [c.strip() for c in countries.split(",")] if countries else None
        
        summary = await processor.get_analysis_summary(
            hours_ahead=hours_ahead,
            countries=country_list
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting analysis summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=CalendarStats)
async def get_calendar_stats():
    """Get calendar statistics"""
    try:
        processor = get_calendar_processor()
        stats = await processor.get_calendar_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting calendar stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update")
async def update_events():
    """Manually update events from all sources"""
    try:
        processor = get_calendar_processor()
        await processor.update_events()
        return {"message": "Events updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/simple")
async def get_events_simple(
    hours_ahead: int = Query(24, description="Hours ahead to look for events"),
    countries: Optional[str] = Query(None, description="Comma-separated country codes")
):
    """Simple endpoint for getting upcoming events"""
    try:
        processor = get_calendar_processor()
        
        # Parse parameters
        country_list = [c.strip() for c in countries.split(",")] if countries else None
        
        events = await processor.get_upcoming_analysis(
            hours_ahead=hours_ahead,
            countries=country_list
        )
        
        # Return simplified format
        return {
            "events": [
                {
                    "id": result.event.event_id,
                    "title": result.event.title,
                    "country": result.event.country.value,
                    "currency": result.event.currency.value,
                    "category": result.event.category.value,
                    "impact": result.event.impact.value,
                    "event_time": result.event.event_time.isoformat(),
                    "actual": result.event.actual,
                    "forecast": result.event.forecast,
                    "previous": result.event.previous,
                    "market_impact_score": result.market_impact_score,
                    "volatility_expected": result.volatility_expected,
                    "affected_pairs": result.affected_currency_pairs,
                    "trading_opportunities": result.trading_opportunities,
                    "risk_factors": result.risk_factors,
                    "confidence": result.confidence
                }
                for result in events
            ],
            "total_events": len(events),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting simple events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

