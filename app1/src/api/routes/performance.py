"""
Performance monitoring API routes
Simple endpoints for checking strategy performance
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta

from src.database.connection import get_db
from src.database.models import Strategy, StrategyPerformance as DBStrategyPerformance
from src.strategy_monitoring.performance_tracker import SimplePerformanceTracker, StrategyPerformance
from config.settings import Settings

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

@router.get("/db-summary")
async def get_database_performance_summary(db: Session = Depends(get_db)):
    """Get performance summary directly from database"""
    try:
        # Get all strategies
        strategies = db.query(Strategy).all()
        
        if not strategies:
            return {
                "status": "success",
                "data": {
                    "total_strategies": 0,
                    "well_performing_count": 0,
                    "poor_performing_count": 0,
                    "average_performance_score": 0,
                    "top_performer": None,
                    "needs_attention": []
                }
            }
        
        # Calculate performance metrics
        well_performing = []
        poor_performing = []
        total_performance_score = 0
        
        for strategy in strategies:
            # Simple performance scoring based on profit factor and win rate
            performance_score = (float(strategy.profit_factor) * 20 + float(strategy.win_rate) * 0.5) / 2
            
            if performance_score >= 70:
                well_performing.append(strategy)
            else:
                poor_performing.append(strategy)
            
            total_performance_score += performance_score
        
        average_performance_score = total_performance_score / len(strategies) if strategies else 0
        
        # Find top performer
        top_performer = max(strategies, key=lambda s: (float(s.profit_factor) * 20 + float(s.win_rate) * 0.5) / 2)
        
        performance_data = {
            "total_strategies": len(strategies),
            "well_performing_count": len(well_performing),
            "poor_performing_count": len(poor_performing),
            "average_performance_score": round(average_performance_score, 2),
            "top_performer": {
                "strategy_id": top_performer.strategy_id,
                "name": top_performer.name,
                "current_performance": float(top_performer.total_profit),
                "win_rate": float(top_performer.win_rate),
                "profit_factor": float(top_performer.profit_factor),
                "max_drawdown": float(top_performer.max_drawdown),
                "total_trades": top_performer.total_trades,
                "is_performing_well": True,
                "performance_score": round((float(top_performer.profit_factor) * 20 + float(top_performer.win_rate) * 0.5) / 2, 2),
                "last_updated": top_performer.updated_at.isoformat() if top_performer.updated_at else None
            },
            "needs_attention": [
                {
                    "strategy_id": s.strategy_id,
                    "name": s.name,
                    "performance_score": round((float(s.profit_factor) * 20 + float(s.win_rate) * 0.5) / 2, 2),
                    "reason": "Low performance score"
                }
                for s in poor_performing
            ]
        }
        
        return {
            "status": "success",
            "data": performance_data
        }
    except Exception as e:
        logger.error(f"Error getting database performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database performance summary")

@router.get("/health")
async def performance_health_check():
    """Health check for performance monitoring"""
    return {
        "status": "healthy",
        "service": "performance_monitoring",
        "message": "Performance monitoring service is running"
    }
