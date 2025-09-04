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

from app2.src.api.routes import ea_router, backtesting_router, evaluation_router
from app2.src.code_generation.engine import CodeGenerationEngine
from app2.src.self_evaluation.monitor import PerformanceMonitor
from app2.src.fault_detection.detector import FaultDetector
from app2.src.mt4_integration.manager import MT4Manager
from app2.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global managers
code_generation_engine = None
performance_monitor = None
fault_detector = None
mt4_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global code_generation_engine, performance_monitor, fault_detector, mt4_manager
    
    # Startup
    logger.info("Starting MetaTrader 4 Script Development Application...")
    
    settings = Settings()
    
    # Initialize managers
    code_generation_engine = CodeGenerationEngine(settings)
    performance_monitor = PerformanceMonitor(settings)
    fault_detector = FaultDetector(settings)
    mt4_manager = MT4Manager(settings)
    
    # Start background tasks
    await performance_monitor.start()
    await fault_detector.start()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await performance_monitor.stop()
    await fault_detector.stop()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AXF Bot - MT4 EA Generator",
    description="MetaTrader 4 Script Development Application for axf-bot-0",
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
app.include_router(ea_router, prefix="/api/v1/ea", tags=["expert-advisors"])
app.include_router(backtesting_router, prefix="/api/v1/backtesting", tags=["backtesting"])
app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["evaluation"])


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
        "components": {
            "code_generation": code_generation_engine.is_healthy() if code_generation_engine else False,
            "performance_monitor": performance_monitor.is_healthy() if performance_monitor else False,
            "fault_detector": fault_detector.is_healthy() if fault_detector else False,
            "mt4_manager": mt4_manager.is_healthy() if mt4_manager else False
        }
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
