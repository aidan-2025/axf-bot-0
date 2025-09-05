#!/usr/bin/env python3
"""
Backtrader Validator

Integrates Backtrader v1.9.78.123 for comprehensive strategy backtesting and validation.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from ..criteria.performance_metrics import PerformanceMetrics
from ..scoring.strategy_scorer import StrategyScorer, ScoringMetrics
from .data_feeds import ForexDataFeed, DataFeedConfig
from .broker_simulation import ForexBroker, BrokerConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    
    # Data configuration
    start_date: datetime
    end_date: datetime
    timeframe: str = '1m'  # 1m, 5m, 15m, 1h, 4h, 1d
    symbols: List[str] = None
    
    # Strategy configuration
    initial_capital: float = 10000.0
    commission: float = 0.0001  # 0.01% commission
    slippage: float = 0.0001   # 0.01% slippage
    
    # Risk management
    max_position_size: float = 0.1  # 10% of capital per position
    stop_loss_pct: float = 0.02    # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    
    # Backtesting options
    exactbars: bool = True  # Memory optimization
    stdstats: bool = False  # Disable standard statistics for performance
    runonce: bool = True    # Optimize for speed
    
    # Data source configuration
    data_source: str = 'influxdb'  # 'influxdb', 'csv', 'mock'
    data_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['EURUSD']
        if self.data_config is None:
            self.data_config = {}


class BacktraderValidator:
    """Backtrader-based strategy validator"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize components
        self.data_feed = ForexDataFeed(DataFeedConfig(**self.config.data_config))
        self.broker = ForexBroker(BrokerConfig(
            commission=self.config.commission,
            slippage=self.config.slippage
        ))
        
        self.logger.info("BacktraderValidator initialized")
    
    async def validate_strategy(self, strategy_class, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a strategy using Backtrader backtesting"""
        self.logger.info(f"Starting strategy validation: {strategy_class.__name__}")
        
        try:
            # Create Cerebro engine
            cerebro = bt.Cerebro(
                exactbars=self.config.exactbars,
                stdstats=self.config.stdstats,
                runonce=self.config.runonce
            )
            
            # Add data feeds
            await self._add_data_feeds(cerebro)
            
            # Add strategy
            cerebro.addstrategy(strategy_class, **strategy_params)
            
            # Set broker
            cerebro.broker.setcash(self.config.initial_capital)
            cerebro.broker.addcommissioninfo(bt.CommInfoBase(
                comm=self.config.commission,
                mult=1.0,
                margin=None,
                commtype=bt.CommInfoBase.COMM_PERC,
                stocklike=False
            ))
            
            # Add analyzers
            self._add_analyzers(cerebro)
            
            # Run backtest
            results = cerebro.run()
            
            # Extract results
            validation_results = self._extract_results(results[0])
            
            self.logger.info("Strategy validation completed successfully")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Strategy validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'performance_metrics': PerformanceMetrics(),
                'scoring_metrics': ScoringMetrics(),
                'trades': []
            }
    
    async def _add_data_feeds(self, cerebro: bt.Cerebro) -> None:
        """Add data feeds to Cerebro"""
        for symbol in self.config.symbols:
            try:
                # Get data from data feed
                data = await self.data_feed.get_data(
                    symbol=symbol,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    timeframe=self.config.timeframe
                )
                
                if data is not None and not data.empty:
                    # Convert to Backtrader format
                    bt_data = self._convert_to_backtrader_data(data, symbol)
                    cerebro.adddata(bt_data)
                    self.logger.info(f"Added data feed for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to add data feed for {symbol}: {e}")
    
    def _convert_to_backtrader_data(self, data: pd.DataFrame, symbol: str) -> bt.feeds.PandasData:
        """Convert pandas DataFrame to Backtrader data format"""
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'volume':
                    data[col] = 0  # Default volume
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Convert to Backtrader format
        bt_data = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # Use index as datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        
        return bt_data
    
    def _add_analyzers(self, cerebro: bt.Cerebro) -> None:
        """Add performance analyzers to Cerebro"""
        # Returns analyzer
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Drawdown analyzer
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        # Sharpe ratio analyzer
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        # Trade analyzer
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Time return analyzer
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        # Custom performance analyzer
        cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')
    
    def _extract_results(self, strategy_instance) -> Dict[str, Any]:
        """Extract results from strategy instance"""
        try:
            # Get analyzers
            returns_analyzer = strategy_instance.analyzers.returns.get_analysis()
            drawdown_analyzer = strategy_instance.analyzers.drawdown.get_analysis()
            sharpe_analyzer = strategy_instance.analyzers.sharpe.get_analysis()
            trades_analyzer = strategy_instance.analyzers.trades.get_analysis()
            performance_analyzer = strategy_instance.analyzers.performance.get_analysis()
            
            # Extract performance metrics
            performance_metrics = self._extract_performance_metrics(
                returns_analyzer, drawdown_analyzer, sharpe_analyzer, 
                trades_analyzer, performance_analyzer
            )
            
            # Extract trades
            trades = self._extract_trades(strategy_instance)
            
            # Calculate scoring metrics
            scorer = StrategyScorer()
            scoring_metrics = scorer.score_strategy(
                performance_metrics, 
                trades,
                (self.config.end_date - self.config.start_date).days
            )
            
            return {
                'success': True,
                'performance_metrics': performance_metrics,
                'scoring_metrics': scoring_metrics,
                'trades': trades,
                'raw_analyzers': {
                    'returns': returns_analyzer,
                    'drawdown': drawdown_analyzer,
                    'sharpe': sharpe_analyzer,
                    'trades': trades_analyzer,
                    'performance': performance_analyzer
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract results: {e}")
            return {
                'success': False,
                'error': str(e),
                'performance_metrics': PerformanceMetrics(),
                'scoring_metrics': ScoringMetrics(),
                'trades': []
            }
    
    def _extract_performance_metrics(self, returns_analyzer, drawdown_analyzer, 
                                   sharpe_analyzer, trades_analyzer, 
                                   performance_analyzer) -> PerformanceMetrics:
        """Extract performance metrics from analyzers"""
        metrics = PerformanceMetrics()
        
        try:
            # Basic performance
            metrics.total_return = returns_analyzer.get('rtot', 0.0)
            metrics.annualized_return = returns_analyzer.get('rnorm100', 0.0) / 100.0
            metrics.net_profit = returns_analyzer.get('rtot', 0.0) * self.config.initial_capital
            
            # Risk metrics
            metrics.max_drawdown = abs(drawdown_analyzer.get('max', {}).get('drawdown', 0.0))
            metrics.current_drawdown = abs(drawdown_analyzer.get('drawdown', 0.0))
            
            # Risk-adjusted returns
            metrics.sharpe_ratio = sharpe_analyzer.get('sharperatio', 0.0)
            
            # Trading statistics
            trades_stats = trades_analyzer.get('total', {})
            metrics.total_trades = trades_stats.get('total', 0)
            metrics.winning_trades = trades_stats.get('won', 0)
            metrics.losing_trades = trades_stats.get('lost', 0)
            
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades
            
            # P&L statistics
            won_stats = trades_analyzer.get('won', {})
            lost_stats = trades_analyzer.get('lost', {})
            
            metrics.gross_profit = won_stats.get('pnl', {}).get('total', 0.0)
            metrics.gross_loss = abs(lost_stats.get('pnl', {}).get('total', 0.0))
            
            if metrics.gross_loss > 0:
                metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
            
            # Average win/loss
            if metrics.winning_trades > 0:
                metrics.avg_win = metrics.gross_profit / metrics.winning_trades
            if metrics.losing_trades > 0:
                metrics.avg_loss = metrics.gross_loss / metrics.losing_trades
            if metrics.avg_loss > 0:
                metrics.avg_win_loss_ratio = metrics.avg_win / metrics.avg_loss
            
            # Consecutive statistics
            metrics.max_consecutive_wins = won_stats.get('streak', {}).get('max', 0)
            metrics.max_consecutive_losses = lost_stats.get('streak', {}).get('max', 0)
            
            # Additional metrics from custom analyzer
            if performance_analyzer:
                metrics.volatility = performance_analyzer.get('volatility', 0.0)
                metrics.sortino_ratio = performance_analyzer.get('sortino_ratio', 0.0)
                metrics.calmar_ratio = performance_analyzer.get('calmar_ratio', 0.0)
                metrics.consistency_score = performance_analyzer.get('consistency_score', 0.0)
                metrics.stability_score = performance_analyzer.get('stability_score', 0.0)
            
        except Exception as e:
            self.logger.error(f"Failed to extract performance metrics: {e}")
        
        return metrics
    
    def _extract_trades(self, strategy_instance) -> List[Dict[str, Any]]:
        """Extract trade data from strategy instance"""
        trades = []
        
        try:
            # This would need to be implemented based on how trades are stored
            # in the strategy instance. For now, return empty list.
            # In a real implementation, you'd extract trade data from the strategy's
            # trade log or from the broker's trade history.
            pass
            
        except Exception as e:
            self.logger.error(f"Failed to extract trades: {e}")
        
        return trades
    
    async def validate_multiple_strategies(self, strategies: List[Tuple[type, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Validate multiple strategies in parallel"""
        self.logger.info(f"Starting validation of {len(strategies)} strategies")
        
        tasks = []
        for strategy_class, strategy_params in strategies:
            task = self.validate_strategy(strategy_class, strategy_params)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Strategy {i} validation failed: {result}")
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'performance_metrics': PerformanceMetrics(),
                    'scoring_metrics': ScoringMetrics(),
                    'trades': []
                })
            else:
                processed_results.append(result)
        
        self.logger.info("Multiple strategy validation completed")
        return processed_results
    
    def close(self):
        """Close the validator and cleanup resources"""
        self.executor.shutdown(wait=True)
        self.logger.info("BacktraderValidator closed")


class PerformanceAnalyzer(bt.Analyzer):
    """Custom performance analyzer for additional metrics"""
    
    def __init__(self):
        self.returns = []
        self.volatility = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.consistency_score = 0.0
        self.stability_score = 0.0
    
    def next(self):
        """Called for each bar"""
        if hasattr(self.strategy, 'broker') and self.strategy.broker.getvalue() > 0:
            # Calculate daily return
            current_value = self.strategy.broker.getvalue()
            if hasattr(self, 'prev_value') and self.prev_value > 0:
                daily_return = (current_value - self.prev_value) / self.prev_value
                self.returns.append(daily_return)
            self.prev_value = current_value
    
    def stop(self):
        """Called when strategy stops"""
        if len(self.returns) > 1:
            # Calculate volatility
            self.volatility = np.std(self.returns) * np.sqrt(252)
            
            # Calculate Sortino ratio
            downside_returns = [r for r in self.returns if r < 0]
            if downside_returns:
                downside_volatility = np.std(downside_returns) * np.sqrt(252)
                if downside_volatility > 0:
                    self.sortino_ratio = np.mean(self.returns) / downside_volatility
            
            # Calculate Calmar ratio
            if hasattr(self, 'max_drawdown') and self.max_drawdown > 0:
                self.calmar_ratio = np.mean(self.returns) / self.max_drawdown
            
            # Calculate consistency and stability scores
            self.consistency_score = self._calculate_consistency_score(self.returns)
            self.stability_score = self._calculate_stability_score(self.returns)
    
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0:
            return 0.0
        
        cv = std_return / abs(mean_return)
        consistency = max(0.0, 1.0 - cv)
        return min(1.0, consistency)
    
    def _calculate_stability_score(self, returns: List[float]) -> float:
        """Calculate stability score"""
        if not returns or len(returns) < 3:
            return 0.0
        
        window_size = min(10, len(returns) // 3)
        if window_size < 2:
            return 0.0
        
        rolling_vols = []
        for i in range(len(returns) - window_size + 1):
            window_returns = returns[i:i + window_size]
            rolling_vols.append(np.std(window_returns))
        
        if not rolling_vols:
            return 0.0
        
        vol_of_vol = np.std(rolling_vols)
        mean_vol = np.mean(rolling_vols)
        
        if mean_vol == 0:
            return 0.0
        
        stability = max(0.0, 1.0 - (vol_of_vol / mean_vol))
        return min(1.0, stability)
    
    def get_analysis(self):
        """Return analysis results"""
        return {
            'volatility': self.volatility,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'consistency_score': self.consistency_score,
            'stability_score': self.stability_score
        }

