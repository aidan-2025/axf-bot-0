"""
Performance monitoring API routes
Simple endpoints for checking strategy performance
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from app1.src.strategy_monitoring.performance_tracker import SimplePerformanceTracker, StrategyPerformance
from app1.config.settings import Settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency to get settings
def get_settings():
    return Settings()

# Dependency to get performance tracker
def get_performance_tracker(settings: Settings = Depends(get_settings)):
    return SimplePerformanceTracker(settings)

@router.get("/summary")
async def get_performance_summary(
    tracker: SimplePerformanceTracker = Depends(get_performance_tracker)
):
    """Get overall performance summary"""
    try:
        summary = await tracker.get_performance_summary()
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance summary")

@router.get("/well-performing")
async def get_well_performing_strategies(
    tracker: SimplePerformanceTracker = Depends(get_performance_tracker)
):
    """Get strategies that are currently performing well"""
    try:
        strategies = await tracker.get_well_performing_strategies()
        return {
            "status": "success",
            "data": [
                {
                    "strategy_id": s.strategy_id,
                    "name": s.name,
                    "current_performance": s.current_performance,
                    "win_rate": s.win_rate,
                    "profit_factor": s.profit_factor,
                    "max_drawdown": s.max_drawdown,
                    "total_trades": s.total_trades,
                    "performance_score": s.performance_score,
                    "last_updated": s.last_updated.isoformat()
                }
                for s in strategies
            ]
        }
    except Exception as e:
        logger.error(f"Error getting well-performing strategies: {e}")
        raise HTTPException(status_code=500, detail="Failed to get well-performing strategies")

@router.get("/poor-performing")
async def get_poor_performing_strategies(
    tracker: SimplePerformanceTracker = Depends(get_performance_tracker)
):
    """Get strategies that are performing poorly and need attention"""
    try:
        strategies = await tracker.get_poor_performing_strategies()
        return {
            "status": "success",
            "data": [
                {
                    "strategy_id": s.strategy_id,
                    "name": s.name,
                    "current_performance": s.current_performance,
                    "win_rate": s.win_rate,
                    "profit_factor": s.profit_factor,
                    "max_drawdown": s.max_drawdown,
                    "total_trades": s.total_trades,
                    "performance_score": s.performance_score,
                    "last_updated": s.last_updated.isoformat()
                }
                for s in strategies
            ]
        }
    except Exception as e:
        logger.error(f"Error getting poor-performing strategies: {e}")
        raise HTTPException(status_code=500, detail="Failed to get poor-performing strategies")

@router.get("/strategy/{strategy_id}")
async def get_strategy_performance(
    strategy_id: str,
    tracker: SimplePerformanceTracker = Depends(get_performance_tracker)
):
    """Get performance details for a specific strategy"""
    try:
        performance = await tracker.evaluate_strategy_performance(strategy_id)
        if not performance:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "status": "success",
            "data": {
                "strategy_id": performance.strategy_id,
                "name": performance.name,
                "current_performance": performance.current_performance,
                "win_rate": performance.win_rate,
                "profit_factor": performance.profit_factor,
                "max_drawdown": performance.max_drawdown,
                "total_trades": performance.total_trades,
                "is_performing_well": performance.is_performing_well,
                "performance_score": performance.performance_score,
                "last_updated": performance.last_updated.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get strategy performance")

@router.post("/strategy/{strategy_id}/update")
async def update_strategy_performance(
    strategy_id: str,
    performance_data: Dict[str, Any],
    tracker: SimplePerformanceTracker = Depends(get_performance_tracker)
):
    """Update performance data for a strategy"""
    try:
        await tracker.update_strategy_performance(strategy_id, performance_data)
        return {
            "status": "success",
            "message": f"Performance updated for strategy {strategy_id}"
        }
    except Exception as e:
        logger.error(f"Error updating strategy performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to update strategy performance")

@router.get("/health")
async def performance_health_check():
    """Health check for performance monitoring"""
    return {
        "status": "healthy",
        "service": "performance_monitoring",
        "message": "Performance monitoring service is running"
    }
