#!/usr/bin/env python3
"""
News and Sentiment API Routes
API endpoints for news and sentiment data
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from ...news_sentiment.models import NewsArticle, EconomicEvent, NewsSource, SentimentLabel
from ...news_sentiment.news_ingestion_service import NewsIngestionService, NewsIngestionConfig

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/news", tags=["news"])

# Global news service instance
news_service: Optional[NewsIngestionService] = None

class NewsResponse(BaseModel):
    """News response model"""
    articles: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_more: bool

class EventResponse(BaseModel):
    """Event response model"""
    events: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_more: bool

class SentimentResponse(BaseModel):
    """Sentiment response model"""
    currency_pair: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    positive_count: int
    negative_count: int
    neutral_count: int
    total_articles: int
    last_updated: str

class HealthResponse(BaseModel):
    """Health response model"""
    status: str
    service_running: bool
    clients: Dict[str, Any]
    statistics: Dict[str, Any]

def get_news_service() -> NewsIngestionService:
    """Get news service instance"""
    global news_service
    if news_service is None:
        # Initialize with default config
        config = NewsIngestionConfig()
        news_service = NewsIngestionService(config)
    return news_service

@router.on_event("startup")
async def startup_news_service():
    """Start news service on startup"""
    global news_service
    if news_service is None:
        config = NewsIngestionConfig()
        news_service = NewsIngestionService(config)
        await news_service.start()
        logger.info("News service started")

@router.on_event("shutdown")
async def shutdown_news_service():
    """Stop news service on shutdown"""
    global news_service
    if news_service:
        await news_service.stop()
        logger.info("News service stopped")

@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Get news service health status"""
    try:
        service = get_news_service()
        health_data = await service.get_health_status()
        
        return HealthResponse(
            status="healthy" if health_data['service_running'] else "unhealthy",
            service_running=health_data['service_running'],
            clients=health_data['clients'],
            statistics=health_data['statistics']
        )
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/articles", response_model=NewsResponse)
async def get_news_articles(
    limit: int = Query(100, ge=1, le=1000, description="Number of articles to return"),
    page: int = Query(1, ge=1, description="Page number"),
    source: Optional[str] = Query(None, description="Filter by news source"),
    min_relevance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum relevance score"),
    hours: int = Query(24, ge=1, le=168, description="Hours back to fetch articles"),
    currency_pair: Optional[str] = Query(None, description="Filter by currency pair")
):
    """Get news articles"""
    try:
        service = get_news_service()
        
        # Calculate time range
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get articles
        articles = await service.get_recent_news(
            limit=limit,
            source=NewsSource(source) if source else None,
            min_relevance=min_relevance
        )
        
        # Filter by currency pair if specified
        if currency_pair:
            articles = [
                article for article in articles
                if currency_pair.upper() in article.currency_pairs
            ]
        
        # Apply pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_articles = articles[start_idx:end_idx]
        
        # Convert to dict
        articles_data = [article.to_dict() for article in paginated_articles]
        
        return NewsResponse(
            articles=articles_data,
            total=len(articles),
            page=page,
            page_size=limit,
            has_more=end_idx < len(articles)
        )
        
    except Exception as e:
        logger.error(f"Error getting news articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events", response_model=EventResponse)
async def get_economic_events(
    limit: int = Query(100, ge=1, le=1000, description="Number of events to return"),
    page: int = Query(1, ge=1, description="Page number"),
    hours_ahead: int = Query(24, ge=1, le=168, description="Hours ahead to fetch events"),
    currency: Optional[str] = Query(None, description="Filter by currency")
):
    """Get economic events"""
    try:
        service = get_news_service()
        
        # Get events
        events = await service.get_upcoming_events(
            hours_ahead=hours_ahead,
            currency=currency
        )
        
        # Apply pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_events = events[start_idx:end_idx]
        
        # Convert to dict
        events_data = [event.to_dict() for event in paginated_events]
        
        return EventResponse(
            events=events_data,
            total=len(events),
            page=page,
            page_size=limit,
            has_more=end_idx < len(events)
        )
        
    except Exception as e:
        logger.error(f"Error getting economic events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/{currency_pair}", response_model=SentimentResponse)
async def get_sentiment(
    currency_pair: str,
    hours: int = Query(24, ge=1, le=168, description="Hours back to analyze sentiment")
):
    """Get sentiment analysis for a currency pair"""
    try:
        service = get_news_service()
        
        # Get sentiment summary
        sentiment_data = await service.sentiment_analyzer.get_sentiment_summary(
            currency_pairs=[currency_pair.upper()],
            hours=hours
        )
        
        return SentimentResponse(
            currency_pair=currency_pair.upper(),
            sentiment_score=sentiment_data.get('sentiment_score', 0.0),
            sentiment_label=sentiment_data.get('overall_sentiment', 'neutral'),
            confidence=sentiment_data.get('confidence', 0.5),
            positive_count=sentiment_data.get('positive_count', 0),
            negative_count=sentiment_data.get('negative_count', 0),
            neutral_count=sentiment_data.get('neutral_count', 0),
            total_articles=sentiment_data.get('total_articles', 0),
            last_updated=sentiment_data.get('last_updated', datetime.utcnow().isoformat())
        )
        
    except Exception as e:
        logger.error(f"Error getting sentiment for {currency_pair}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment", response_model=List[SentimentResponse])
async def get_all_sentiment(
    hours: int = Query(24, ge=1, le=168, description="Hours back to analyze sentiment")
):
    """Get sentiment analysis for all currency pairs"""
    try:
        service = get_news_service()
        
        # Major currency pairs
        major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'CHFJPY', 'CADJPY', 'AUDJPY'
        ]
        
        sentiment_responses = []
        
        for pair in major_pairs:
            try:
                sentiment_data = await service.sentiment_analyzer.get_sentiment_summary(
                    currency_pairs=[pair],
                    hours=hours
                )
                
                sentiment_responses.append(SentimentResponse(
                    currency_pair=pair,
                    sentiment_score=sentiment_data.get('sentiment_score', 0.0),
                    sentiment_label=sentiment_data.get('overall_sentiment', 'neutral'),
                    confidence=sentiment_data.get('confidence', 0.5),
                    positive_count=sentiment_data.get('positive_count', 0),
                    negative_count=sentiment_data.get('negative_count', 0),
                    neutral_count=sentiment_data.get('neutral_count', 0),
                    total_articles=sentiment_data.get('total_articles', 0),
                    last_updated=sentiment_data.get('last_updated', datetime.utcnow().isoformat())
                ))
                
            except Exception as e:
                logger.error(f"Error getting sentiment for {pair}: {e}")
                continue
        
        return sentiment_responses
        
    except Exception as e:
        logger.error(f"Error getting all sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refresh")
async def refresh_news_data():
    """Force refresh of all news data"""
    try:
        service = get_news_service()
        await service.force_refresh()
        
        return {"message": "News data refresh initiated", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Error refreshing news data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_statistics():
    """Get news ingestion statistics"""
    try:
        service = get_news_service()
        stats = service.get_statistics()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sources")
async def get_news_sources():
    """Get available news sources"""
    try:
        service = get_news_service()
        health_data = await service.get_health_status()
        
        sources = []
        for source_name, client_data in health_data['clients'].items():
            sources.append({
                'name': source_name,
                'status': client_data.get('status', 'unknown'),
                'last_request': client_data.get('last_request'),
                'total_requests': client_data.get('total_requests', 0),
                'success_rate': client_data.get('success_rate', 0.0),
                'rate_limit_hits': client_data.get('rate_limit_hits', 0)
            })
        
        return {"sources": sources}
        
    except Exception as e:
        logger.error(f"Error getting news sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

