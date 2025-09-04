"""
Strategy API routes
Simple endpoints for strategy management
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_strategies():
    """Get all strategies"""
    try:
        # Mock data for now
        strategies = [
            {
                "id": "STRAT_001",
                "name": "EUR/USD Trend Following",
                "description": "Simple moving average crossover strategy",
                "strategy_type": "trend_following",
                "currency_pairs": ["EURUSD"],
                "timeframes": ["H1", "H4"],
                "status": "active",
                "performance": {
                    "profit_factor": 1.45,
                    "win_rate": 62.5,
                    "max_drawdown": 8.2,
                    "sharpe_ratio": 1.23,
                    "total_trades": 156,
                    "total_profit": 2340.50
                },
                "created_at": "2024-01-01T00:00:00Z",
                "last_updated": "2024-01-15T12:00:00Z"
            }
        ]
        
        return {
            "status": "success",
            "data": {
                "all": strategies,
                "active": len([s for s in strategies if s["status"] == "active"]),
                "recent": strategies[:5]
            }
        }
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail="Failed to get strategies")

@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get a specific strategy"""
    try:
        # Mock implementation
        if strategy_id == "STRAT_001":
            return {
                "status": "success",
                "data": {
                    "id": "STRAT_001",
                    "name": "EUR/USD Trend Following",
                    "description": "Simple moving average crossover strategy",
                    "strategy_type": "trend_following",
                    "currency_pairs": ["EURUSD"],
                    "timeframes": ["H1", "H4"],
                    "status": "active",
                    "performance": {
                        "profit_factor": 1.45,
                        "win_rate": 62.5,
                        "max_drawdown": 8.2,
                        "sharpe_ratio": 1.23,
                        "total_trades": 156,
                        "total_profit": 2340.50
                    }
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Strategy not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy: {e}")
        raise HTTPException(status_code=500, detail="Failed to get strategy")

@router.post("/")
async def create_strategy(strategy_data: Dict[str, Any]):
    """Create a new strategy"""
    try:
        # Mock implementation
        return {
            "status": "success",
            "data": {
                "id": "STRAT_NEW",
                "message": "Strategy created successfully"
            }
        }
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail="Failed to create strategy")
