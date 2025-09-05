"""
Walk-forward analysis for strategy validation
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta

from ..core.strategy_template import StrategyTemplate


class WalkForwardTester:
    """
    Walk-forward analysis for strategy validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Walk-forward parameters
        self.training_period = self.config.get('training_period', 252)  # 1 year
        self.testing_period = self.config.get('testing_period', 63)     # 3 months
        self.step_size = self.config.get('step_size', 21)               # 1 month
        
    def test(self, strategy: StrategyTemplate, market_data: Dict[str, Any], 
             config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run walk-forward analysis on strategy
        
        Args:
            strategy: Strategy to test
            market_data: Historical market data
            config: Additional configuration
            
        Returns:
            Dict[str, Any]: Walk-forward analysis results
        """
        try:
            # Update config
            if config:
                self.training_period = config.get('training_period', self.training_period)
                self.testing_period = config.get('testing_period', self.testing_period)
                self.step_size = config.get('step_size', self.step_size)
            
            self.logger.info(f"Starting walk-forward analysis")
            
            # Split data into periods
            periods = self._split_data_periods(market_data)
            
            # Run walk-forward tests
            results = []
            for i, period in enumerate(periods):
                result = self._run_walk_forward_period(strategy, period, i)
                results.append(result)
            
            # Analyze results
            analysis = self._analyze_walk_forward_results(results)
            
            self.logger.info(f"Walk-forward analysis completed. {len(results)} periods tested")
            
            return {
                "success": True,
                "periods": len(periods),
                "results": results,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "periods": 0,
                "results": [],
                "analysis": {}
            }
    
    def _split_data_periods(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split market data into training and testing periods"""
        periods = []
        
        # Extract OHLCV data
        ohlcv_data = market_data.get('ohlcv', {})
        if not ohlcv_data or not ohlcv_data.get('close'):
            return periods
        
        prices = ohlcv_data['close']
        total_length = len(prices)
        
        # Calculate number of periods
        min_period_length = self.training_period + self.testing_period
        if total_length < min_period_length:
            self.logger.warning(f"Data length {total_length} is less than minimum period length {min_period_length}")
            return periods
        
        # Create periods
        start_idx = 0
        period_num = 0
        
        while start_idx + min_period_length <= total_length:
            end_idx = min(start_idx + self.training_period + self.testing_period, total_length)
            
            # Split into training and testing
            training_end = start_idx + self.training_period
            testing_start = training_end
            testing_end = end_idx
            
            period_data = {
                "period": period_num,
                "training": self._extract_period_data(ohlcv_data, start_idx, training_end),
                "testing": self._extract_period_data(ohlcv_data, testing_start, testing_end),
                "start_idx": start_idx,
                "training_end": training_end,
                "testing_start": testing_start,
                "testing_end": testing_end
            }
            
            periods.append(period_data)
            
            # Move to next period
            start_idx += self.step_size
            period_num += 1
        
        return periods
    
    def _extract_period_data(self, ohlcv_data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Extract data for a specific period"""
        period_data = {}
        
        for key, values in ohlcv_data.items():
            if isinstance(values, list) and len(values) > end_idx:
                period_data[key] = values[start_idx:end_idx]
            else:
                period_data[key] = values
        
        return period_data
    
    def _run_walk_forward_period(self, strategy: StrategyTemplate, period: Dict[str, Any], 
                                period_num: int) -> Dict[str, Any]:
        """Run walk-forward test for a single period"""
        try:
            # Train strategy on training data
            training_data = {"ohlcv": period["training"]}
            training_signals = strategy.generate_signals(training_data)
            
            # Test strategy on testing data
            testing_data = {"ohlcv": period["testing"]}
            testing_signals = strategy.generate_signals(testing_data)
            
            # Calculate performance
            training_performance = self._calculate_performance(training_signals, training_data)
            testing_performance = self._calculate_performance(testing_signals, testing_data)
            
            return {
                "period": period_num,
                "training_signals": len(training_signals),
                "testing_signals": len(testing_signals),
                "training_performance": training_performance,
                "testing_performance": testing_performance,
                "performance_ratio": testing_performance.get("total_return", 0) / max(training_performance.get("total_return", 1), 1e-8)
            }
            
        except Exception as e:
            self.logger.warning(f"Error in walk-forward period {period_num}: {e}")
            return {
                "period": period_num,
                "training_signals": 0,
                "testing_signals": 0,
                "training_performance": {},
                "testing_performance": {},
                "performance_ratio": 0.0
            }
    
    def _calculate_performance(self, signals: List[Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for signals"""
        if not signals:
            return {
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0
            }
        
        # Simple performance calculation
        returns = []
        for signal in signals:
            if signal.signal_type in ['buy', 'sell']:
                # Simulate trade outcome
                outcome = np.random.normal(0, 0.01) * signal.strength
                returns.append(outcome)
        
        if returns:
            total_return = sum(returns)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # Calculate win rate
            winning_trades = sum(1 for r in returns if r > 0)
            win_rate = winning_trades / len(returns)
        else:
            total_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate
        }
    
    def _analyze_walk_forward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward test results"""
        if not results:
            return {}
        
        # Extract metrics
        training_returns = [r["training_performance"].get("total_return", 0) for r in results]
        testing_returns = [r["testing_performance"].get("total_return", 0) for r in results]
        performance_ratios = [r["performance_ratio"] for r in results]
        
        # Calculate statistics
        analysis = {
            "total_periods": len(results),
            "avg_training_return": np.mean(training_returns),
            "avg_testing_return": np.mean(testing_returns),
            "avg_performance_ratio": np.mean(performance_ratios),
            "std_performance_ratio": np.std(performance_ratios),
            "positive_periods": sum(1 for r in testing_returns if r > 0),
            "consistency_score": sum(1 for r in performance_ratios if 0.5 <= r <= 2.0) / len(performance_ratios),
            "degradation_trend": self._calculate_degradation_trend(testing_returns)
        }
        
        return analysis
    
    def _calculate_degradation_trend(self, returns: List[float]) -> float:
        """Calculate if strategy performance is degrading over time"""
        if len(returns) < 3:
            return 0.0
        
        # Calculate correlation between period number and returns
        periods = list(range(len(returns)))
        correlation = np.corrcoef(periods, returns)[0, 1]
        
        return correlation  # Negative means degrading performance

