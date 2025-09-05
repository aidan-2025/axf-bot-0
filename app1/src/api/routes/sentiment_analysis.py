#!/usr/bin/env python3
"""
Sentiment Analysis API Routes
API endpoints for sentiment analysis functionality
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from ...sentiment_analysis import (
    SentimentProcessor, SentimentProcessorConfig,
    SentimentResult, SentimentBatchResult, SentimentTrend,
    EnsembleConfig, BERTConfig, LexiconConfig, TrendConfig
)

logger = logging.getLogger(__name__)

# Global sentiment processor instance
sentiment_processor: Optional[SentimentProcessor] = None

# Pydantic models for API
class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis"""
    text: str = Field(..., description="Text to analyze")
    language: str = Field("en", description="Language of the text")
    currency_pairs: Optional[List[str]] = Field(None, description="Relevant currency pairs")
    use_cache: bool = Field(True, description="Use cached results if available")

class BatchSentimentAnalysisRequest(BaseModel):
    """Request model for batch sentiment analysis"""
    texts: List[str] = Field(..., description="Texts to analyze")
    languages: Optional[List[str]] = Field(None, description="Languages of the texts")
    currency_pairs_list: Optional[List[List[str]]] = Field(None, description="Currency pairs for each text")
    use_cache: bool = Field(True, description="Use cached results if available")

class SentimentTrendRequest(BaseModel):
    """Request model for sentiment trend analysis"""
    currency_pairs: List[str] = Field(..., description="Currency pairs to analyze")
    timeframe: str = Field("1h", description="Timeframe for trend analysis")
    hours_back: int = Field(24, description="Hours to look back")

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    result: SentimentResult
    success: bool = True
    processing_time_ms: float

class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis"""
    results: SentimentBatchResult
    success: bool = True

class SentimentTrendResponse(BaseModel):
    """Response model for sentiment trends"""
    trends: List[SentimentTrend]
    success: bool = True

class SentimentHealthResponse(BaseModel):
    """Response model for sentiment analysis health"""
    status: str
    processor_initialized: bool
    ensemble_info: Dict[str, Any]
    trend_analyzer_info: Dict[str, Any]

# Initialize sentiment processor
async def initialize_sentiment_processor():
    """Initialize the sentiment processor"""
    global sentiment_processor
    
    if sentiment_processor is None:
        try:
            logger.info("Initializing sentiment processor")
            
            # Create configuration
            config = SentimentProcessorConfig(
                ensemble_config=EnsembleConfig(
                    bert_config=BERTConfig(
                        model_name="yiyanghkust/finbert-tone",
                        max_length=512,
                        batch_size=16,
                        confidence_threshold=0.7
                    ),
                    lexicon_config=LexiconConfig(
                        enable_negation=True,
                        enable_intensifiers=True,
                        enable_diminishers=True,
                        confidence_threshold=0.5
                    ),
                    bert_weight=0.6,
                    lexicon_weight=0.4,
                    confidence_threshold=0.7,
                    enable_agreement_boost=True,
                    fallback_to_lexicon=True
                ),
                trend_config=TrendConfig(
                    short_window_minutes=15,
                    medium_window_hours=4,
                    long_window_hours=24,
                    min_data_points=5,
                    volatility_threshold=0.3,
                    trend_strength_threshold=0.2
                ),
                batch_size=50,
                max_processing_time_ms=5000.0,
                enable_trend_analysis=True,
                enable_caching=True,
                cache_ttl_seconds=3600
            )
            
            sentiment_processor = SentimentProcessor(config)
            await sentiment_processor.initialize()
            
            logger.info("Sentiment processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment processor: {e}")
            raise

# Get sentiment processor instance
def get_sentiment_processor() -> SentimentProcessor:
    """Get the sentiment processor instance"""
    global sentiment_processor
    if sentiment_processor is None:
        raise HTTPException(status_code=503, detail="Sentiment processor not initialized")
    return sentiment_processor

# Create router
router = APIRouter()

@router.on_event("startup")
async def startup_sentiment_processor():
    """Initialize sentiment processor on startup"""
    await initialize_sentiment_processor()

@router.on_event("shutdown")
async def shutdown_sentiment_processor():
    """Cleanup sentiment processor on shutdown"""
    global sentiment_processor
    if sentiment_processor:
        await sentiment_processor.close()
        sentiment_processor = None
    logger.info("Sentiment processor shutdown")

@router.get("/health", response_model=SentimentHealthResponse)
async def get_sentiment_health():
    """Get sentiment analysis health status"""
    try:
        processor = get_sentiment_processor()
        processor_info = await processor.get_processor_info()
        
        return SentimentHealthResponse(
            status="healthy" if processor_info.get("status") == "initialized" else "unhealthy",
            processor_initialized=processor_info.get("status") == "initialized",
            ensemble_info=processor_info.get("ensemble_analyzer", {}),
            trend_analyzer_info=processor_info.get("trend_analyzer", {})
        )
    except Exception as e:
        logger.error(f"Error getting sentiment health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment of a single text"""
    try:
        processor = get_sentiment_processor()
        
        result = await processor.analyze_text(
            text=request.text,
            language=request.language,
            currency_pairs=request.currency_pairs,
            use_cache=request.use_cache
        )
        
        return SentimentResponse(
            result=result,
            success=True,
            processing_time_ms=result.processing_time_ms or 0.0
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(request: BatchSentimentAnalysisRequest):
    """Analyze sentiment of multiple texts"""
    try:
        processor = get_sentiment_processor()
        
        results = await processor.analyze_batch(
            texts=request.texts,
            languages=request.languages,
            currency_pairs_list=request.currency_pairs_list,
            use_cache=request.use_cache
        )
        
        return BatchSentimentResponse(
            results=results,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trends", response_model=SentimentTrendResponse)
async def get_sentiment_trends(request: SentimentTrendRequest):
    """Get sentiment trends for currency pairs"""
    try:
        processor = get_sentiment_processor()
        
        trends = await processor.get_sentiment_trends(
            currency_pairs=request.currency_pairs,
            timeframe=request.timeframe,
            hours_back=request.hours_back
        )
        
        return SentimentTrendResponse(
            trends=trends,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error getting sentiment trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/simple")
async def analyze_sentiment_simple(
    text: str = Query(..., description="Text to analyze"),
    language: str = Query("en", description="Language of the text"),
    currency_pairs: Optional[str] = Query(None, description="Comma-separated currency pairs")
):
    """Simple sentiment analysis endpoint"""
    try:
        processor = get_sentiment_processor()
        
        # Parse currency pairs
        pairs = None
        if currency_pairs:
            pairs = [pair.strip().upper() for pair in currency_pairs.split(",")]
        
        result = await processor.analyze_text(
            text=text,
            language=language,
            currency_pairs=pairs
        )
        
        return {
            "text": result.text,
            "sentiment": result.label.value,
            "score": result.score,
            "confidence": result.confidence,
            "source": result.source.value,
            "processing_time_ms": result.processing_time_ms,
            "financial_entities": result.financial_entities,
            "market_impact": result.market_impact
        }
        
    except Exception as e:
        logger.error(f"Error in simple sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends/simple")
async def get_sentiment_trends_simple(
    currency_pairs: str = Query(..., description="Comma-separated currency pairs"),
    timeframe: str = Query("1h", description="Timeframe for analysis"),
    hours_back: int = Query(24, description="Hours to look back")
):
    """Simple sentiment trends endpoint"""
    try:
        processor = get_sentiment_processor()
        
        # Parse currency pairs
        pairs = [pair.strip().upper() for pair in currency_pairs.split(",")]
        
        trends = await processor.get_sentiment_trends(
            currency_pairs=pairs,
            timeframe=timeframe,
            hours_back=hours_back
        )
        
        return {
            "trends": [
                {
                    "currency_pair": trend.currency_pair,
                    "timeframe": trend.timeframe,
                    "average_sentiment": trend.average_sentiment,
                    "sentiment_volatility": trend.sentiment_volatility,
                    "trend_direction": trend.trend_direction,
                    "confidence": trend.confidence,
                    "start_time": trend.start_time.isoformat(),
                    "end_time": trend.end_time.isoformat()
                }
                for trend in trends
            ],
            "total_trends": len(trends)
        }
        
    except Exception as e:
        logger.error(f"Error in simple sentiment trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

