"""
Application 1: AI-Powered Forex Strategy Generator
Main entry point for the strategy generation application.
"""

import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app1.src.api.routes import strategy_router, data_router, sentiment_router
from app1.src.data_ingestion.market_data import MarketDataManager
from app1.src.sentiment_analysis.news_processor import NewsProcessor
from app1.src.strategy_generation.engine import StrategyEngine
from app1.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global managers
market_data_manager = None
news_processor = None
strategy_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global market_data_manager, news_processor, strategy_engine
    
    # Startup
    logger.info("Starting AI-Powered Forex Strategy Generator...")
    
    settings = Settings()
    
    # Initialize managers
    market_data_manager = MarketDataManager(settings)
    news_processor = NewsProcessor(settings)
    strategy_engine = StrategyEngine(settings)
    
    # Start background tasks
    await market_data_manager.start()
    await news_processor.start()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await market_data_manager.stop()
    await news_processor.stop()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AXF Bot - Strategy Generator",
    description="AI-Powered Forex Strategy Generator for axf-bot-0",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(strategy_router, prefix="/api/v1/strategies", tags=["strategies"])
app.include_router(data_router, prefix="/api/v1/data", tags=["data"])
app.include_router(sentiment_router, prefix="/api/v1/sentiment", tags=["sentiment"])


@app.get("/")
async def root():
    """Root endpoint with basic application information."""
    return {
        "application": "AXF Bot - Strategy Generator",
        "version": "1.0.0",
        "status": "running",
        "description": "AI-Powered Forex Strategy Generator"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "components": {
            "market_data": market_data_manager.is_healthy() if market_data_manager else False,
            "news_processor": news_processor.is_healthy() if news_processor else False,
            "strategy_engine": strategy_engine.is_healthy() if strategy_engine else False
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
