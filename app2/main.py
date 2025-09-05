"""
Application 2: MetaTrader 4 Script Development Application
Main entry point for the MT4 EA generation application.
"""

import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create FastAPI application
app = FastAPI(
    title="AXF Bot - MT4 EA Generator",
    description="MetaTrader 4 Script Development Application for axf-bot-0",
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

# Include routers (will be added when modules are implemented)
# app.include_router(ea_router, prefix="/api/v1/ea", tags=["expert-advisors"])
# app.include_router(backtesting_router, prefix="/api/v1/backtesting", tags=["backtesting"])
# app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["evaluation"])


@app.get("/")
async def root():
    """Root endpoint with basic application information."""
    return {
        "application": "AXF Bot - MT4 EA Generator",
        "version": "1.0.0",
        "status": "running",
        "description": "MetaTrader 4 Script Development Application"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "message": "App2 is running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
