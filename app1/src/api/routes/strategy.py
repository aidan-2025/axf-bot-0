"""
Strategy API routes
Comprehensive endpoints for strategy management using SQLAlchemy models
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from src.database.connection import get_db
from src.database.models import Strategy, StrategyCurrencyPair, StrategyTimeframe, CurrencyPair, Timeframe

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_strategies(db: Session = Depends(get_db)):
    """Get all strategies with their relationships"""
    try:
        # Query strategies with their currency pairs and timeframes
        strategies_query = db.query(Strategy).all()
        
        strategies = []
        for strategy in strategies_query:
            # Get currency pairs for this strategy
            currency_pairs = db.query(CurrencyPair.symbol).join(
                StrategyCurrencyPair, 
                CurrencyPair.id == StrategyCurrencyPair.currency_pair_id
            ).filter(StrategyCurrencyPair.strategy_id == strategy.id).all()
            
            # Get timeframes for this strategy
            timeframes = db.query(Timeframe.name).join(
                StrategyTimeframe,
                Timeframe.id == StrategyTimeframe.timeframe_id
            ).filter(StrategyTimeframe.strategy_id == strategy.id).all()
            
            strategy_data = {
                "id": strategy.strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "strategy_type": strategy.strategy_type,
                "currency_pairs": [cp.symbol for cp in currency_pairs],
                "timeframes": [tf.name for tf in timeframes],
                "status": strategy.status,
                "performance": {
                    "profit_factor": float(strategy.profit_factor),
                    "win_rate": float(strategy.win_rate),
                    "max_drawdown": float(strategy.max_drawdown),
                    "sharpe_ratio": float(strategy.sharpe_ratio),
                    "total_trades": strategy.total_trades,
                    "total_profit": float(strategy.total_profit)
                },
                "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
                "last_updated": strategy.updated_at.isoformat() if strategy.updated_at else None
            }
            strategies.append(strategy_data)
        
        active_count = len([s for s in strategies if s["status"] == "active"])
        
        return {
            "status": "success",
            "data": {
                "all": strategies,
                "active": active_count,
                "recent": strategies[:5]
            }
        }
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail="Failed to get strategies")

@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str, db: Session = Depends(get_db)):
    """Get a specific strategy by strategy_id"""
    try:
        # Query strategy by strategy_id
        strategy = db.query(Strategy).filter(Strategy.strategy_id == strategy_id).first()
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Get currency pairs for this strategy
        currency_pairs = db.query(CurrencyPair.symbol).join(
            StrategyCurrencyPair, 
            CurrencyPair.id == StrategyCurrencyPair.currency_pair_id
        ).filter(StrategyCurrencyPair.strategy_id == strategy.id).all()
        
        # Get timeframes for this strategy
        timeframes = db.query(Timeframe.name).join(
            StrategyTimeframe,
            Timeframe.id == StrategyTimeframe.timeframe_id
        ).filter(StrategyTimeframe.strategy_id == strategy.id).all()
        
        strategy_data = {
            "id": strategy.strategy_id,
            "name": strategy.name,
            "description": strategy.description,
            "strategy_type": strategy.strategy_type,
            "currency_pairs": [cp.symbol for cp in currency_pairs],
            "timeframes": [tf.name for tf in timeframes],
            "status": strategy.status,
            "performance": {
                "profit_factor": float(strategy.profit_factor),
                "win_rate": float(strategy.win_rate),
                "max_drawdown": float(strategy.max_drawdown),
                "sharpe_ratio": float(strategy.sharpe_ratio),
                "total_trades": strategy.total_trades,
                "total_profit": float(strategy.total_profit)
            },
            "parameters": strategy.parameters,
            "risk_management": strategy.risk_management,
            "entry_conditions": strategy.entry_conditions,
            "exit_conditions": strategy.exit_conditions,
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
            "last_updated": strategy.updated_at.isoformat() if strategy.updated_at else None
        }
        
        return {
            "status": "success",
            "data": strategy_data
        }
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
