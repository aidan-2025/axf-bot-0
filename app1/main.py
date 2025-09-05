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
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv("../.env")

# Debug: Print environment variables
import os
print(f"Environment variables loaded:")
print(f"PERPLEXITY_API_KEY: {bool(os.getenv('PERPLEXITY_API_KEY'))}")
print(f"OPENAI_API_KEY: {bool(os.getenv('OPENAI_API_KEY'))}")

from src.api.routes import strategy_router, data_router, sentiment_router, performance_router, ai_router, orchestrator_router, tools_router, news_router, sentiment_analysis_router, economic_calendar_router, backtesting_router, strategy_validation_router, workflow_router, health
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create FastAPI application
app = FastAPI(
    title="AXF Bot - Strategy Generator",
    description="AI-Powered Forex Strategy Generator for axf-bot-0",
    version="1.0.0"
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
app.include_router(performance_router, prefix="/api/v1/performance", tags=["performance"])
app.include_router(ai_router, prefix="/api/v1/ai", tags=["ai"])
app.include_router(orchestrator_router, prefix="/api/v1/orchestrator", tags=["orchestrator"])
app.include_router(tools_router, prefix="/api/v1/tools", tags=["tools"])
app.include_router(news_router, tags=["news"])
app.include_router(sentiment_analysis_router, prefix="/api/v1/sentiment-analysis", tags=["sentiment-analysis"])
app.include_router(economic_calendar_router, prefix="/api/v1/economic-calendar", tags=["economic-calendar"])
app.include_router(backtesting_router, prefix="/api/v1", tags=["backtesting"])
app.include_router(strategy_validation_router, tags=["strategy-validation"])
app.include_router(workflow_router, tags=["workflow"])
app.include_router(health.router, tags=["health"])


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
        "message": "App1 is running"
    }

@app.get("/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables."""
    import os
    return {
        "perplexity_key": bool(os.getenv("PERPLEXITY_API_KEY")),
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "perplexity_value": os.getenv("PERPLEXITY_API_KEY", "")[:10] + "..." if os.getenv("PERPLEXITY_API_KEY") else "None",
        "openai_value": os.getenv("OPENAI_API_KEY", "")[:10] + "..." if os.getenv("OPENAI_API_KEY") else "None"
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
