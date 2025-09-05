"""
Simple Strategy Performance Tracker
Tracks and evaluates past strategies against current market conditions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class StrategyPerformance:
    """Simple data class for strategy performance metrics"""
    strategy_id: str
    name: str
    current_performance: float  # Current profit/loss
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    is_performing_well: bool
    performance_score: float  # 0-100 score
    last_updated: datetime

class SimplePerformanceTracker:
    """Simple performance tracker for monitoring past strategies"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.performance_threshold = 0.6  # 60% performance threshold
        self.min_trades = 10  # Minimum trades for evaluation
        
    async def evaluate_strategy_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """Evaluate if a past strategy is performing well in current market"""
        try:
            # Get strategy from database
            strategy = await self._get_strategy(strategy_id)
            if not strategy:
                return None
                
            # Get recent performance data
            recent_performance = await self._get_recent_performance(strategy_id)
            if not recent_performance:
                return None
                
            # Calculate performance metrics
            performance_score = self._calculate_performance_score(recent_performance)
            is_performing_well = performance_score >= (self.performance_threshold * 100)
            
            return StrategyPerformance(
                strategy_id=strategy_id,
                name=strategy['name'],
                current_performance=recent_performance['total_profit'],
                win_rate=recent_performance['win_rate'],
                profit_factor=recent_performance['profit_factor'],
                max_drawdown=recent_performance['max_drawdown'],
                total_trades=recent_performance['total_trades'],
                is_performing_well=is_performing_well,
                performance_score=performance_score,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy_id}: {e}")
            return None
    
    async def get_well_performing_strategies(self) -> List[StrategyPerformance]:
        """Get all strategies that are currently performing well"""
        try:
            # Get all active strategies
            strategies = await self._get_all_strategies()
            well_performing = []
            
            for strategy in strategies:
                performance = await self.evaluate_strategy_performance(strategy['strategy_id'])
                if performance and performance.is_performing_well:
                    well_performing.append(performance)
                    
            # Sort by performance score (best first)
            well_performing.sort(key=lambda x: x.performance_score, reverse=True)
            return well_performing
            
        except Exception as e:
            logger.error(f"Error getting well-performing strategies: {e}")
            return []
    
    async def get_poor_performing_strategies(self) -> List[StrategyPerformance]:
        """Get strategies that are performing poorly and might need attention"""
        try:
            strategies = await self._get_all_strategies()
            poor_performing = []
            
            for strategy in strategies:
                performance = await self.evaluate_strategy_performance(strategy['strategy_id'])
                if performance and not performance.is_performing_well:
                    poor_performing.append(performance)
                    
            # Sort by performance score (worst first)
            poor_performing.sort(key=lambda x: x.performance_score)
            return poor_performing
            
        except Exception as e:
            logger.error(f"Error getting poor-performing strategies: {e}")
            return []
    
    def _calculate_performance_score(self, performance: Dict[str, Any]) -> float:
        """Calculate a simple performance score (0-100)"""
        try:
            # Weight different metrics
            profit_factor_score = min(performance['profit_factor'] * 20, 40)  # Max 40 points
            win_rate_score = min(performance['win_rate'], 30)  # Max 30 points
            drawdown_score = max(0, 30 - (performance['max_drawdown'] * 2))  # Max 30 points, penalize high drawdown
            
            # Bonus for high trade count (indicates active strategy)
            trade_bonus = min(performance['total_trades'] / 10, 10)  # Max 10 points
            
            total_score = profit_factor_score + win_rate_score + drawdown_score + trade_bonus
            return min(total_score, 100)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    async def _get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy from database (mock implementation)"""
        # In a real implementation, this would query the database
        # For now, return mock data
        mock_strategies = {
            'STRAT_001': {
                'strategy_id': 'STRAT_001',
                'name': 'EUR/USD Trend Following',
                'strategy_type': 'trend_following',
                'currency_pairs': ['EURUSD'],
                'timeframes': ['H1', 'H4'],
                'parameters': {'ma_fast': 20, 'ma_slow': 50, 'stop_loss': 50, 'take_profit': 100}
            },
            'STRAT_002': {
                'strategy_id': 'STRAT_002',
                'name': 'GBP/USD Range Trading',
                'strategy_type': 'range_trading',
                'currency_pairs': ['GBPUSD'],
                'timeframes': ['M15', 'H1'],
                'parameters': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70}
            }
        }
        return mock_strategies.get(strategy_id)
    
    async def _get_recent_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get recent performance data for a strategy (mock implementation)"""
        # In a real implementation, this would query the performance table
        # For now, return mock data
        import random
        
        mock_performance = {
            'STRAT_001': {
                'total_profit': random.uniform(1000, 5000),
                'win_rate': random.uniform(55, 75),
                'profit_factor': random.uniform(1.2, 2.0),
                'max_drawdown': random.uniform(5, 15),
                'total_trades': random.randint(50, 200)
            },
            'STRAT_002': {
                'total_profit': random.uniform(500, 3000),
                'win_rate': random.uniform(45, 65),
                'profit_factor': random.uniform(1.1, 1.8),
                'max_drawdown': random.uniform(8, 20),
                'total_trades': random.randint(30, 150)
            }
        }
        return mock_performance.get(strategy_id)
    
    async def _get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get all strategies from database (mock implementation)"""
        return [
            {'strategy_id': 'STRAT_001', 'name': 'EUR/USD Trend Following'},
            {'strategy_id': 'STRAT_002', 'name': 'GBP/USD Range Trading'},
            {'strategy_id': 'STRAT_003', 'name': 'USD/JPY Breakout'}
        ]
    
    async def update_strategy_performance(self, strategy_id: str, performance_data: Dict[str, Any]):
        """Update strategy performance in database"""
        try:
            # In a real implementation, this would update the database
            logger.info(f"Updating performance for strategy {strategy_id}: {performance_data}")
            
            # Mock database update
            await asyncio.sleep(0.1)  # Simulate database operation
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        try:
            well_performing = await self.get_well_performing_strategies()
            poor_performing = await self.get_poor_performing_strategies()
            
            return {
                'total_strategies': len(well_performing) + len(poor_performing),
                'well_performing_count': len(well_performing),
                'poor_performing_count': len(poor_performing),
                'average_performance_score': sum(s.performance_score for s in well_performing + poor_performing) / max(1, len(well_performing) + len(poor_performing)),
                'top_performer': well_performing[0] if well_performing else None,
                'needs_attention': poor_performing[:3] if poor_performing else []
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'total_strategies': 0,
                'well_performing_count': 0,
                'poor_performing_count': 0,
                'average_performance_score': 0,
                'top_performer': None,
                'needs_attention': []
            }
