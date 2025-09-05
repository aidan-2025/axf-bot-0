"""
Monte Carlo simulation for strategy robustness testing
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import random
import logging
from datetime import datetime

from ..core.strategy_template import StrategyTemplate


class MonteCarloSimulator:
    """
    Monte Carlo simulation for testing strategy robustness
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Simulation parameters
        self.iterations = self.config.get('iterations', 1000)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.noise_level = self.config.get('noise_level', 0.01)
        
    def simulate(self, strategy: StrategyTemplate, market_data: Dict[str, Any], 
                iterations: int = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on strategy
        
        Args:
            strategy: Strategy to simulate
            market_data: Historical market data
            iterations: Number of simulation iterations
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        try:
            iterations = iterations or self.iterations
            self.logger.info(f"Starting Monte Carlo simulation with {iterations} iterations")
            
            # Run simulations
            results = []
            for i in range(iterations):
                result = self._run_single_simulation(strategy, market_data, i)
                results.append(result)
                
                if i % 100 == 0:
                    self.logger.info(f"Completed {i}/{iterations} simulations")
            
            # Analyze results
            analysis = self._analyze_results(results)
            
            self.logger.info(f"Monte Carlo simulation completed. Mean return: {analysis['mean_return']:.4f}")
            
            return {
                "success": True,
                "iterations": iterations,
                "results": results,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            return {
                "success": False,
                "error": str(e),
                "iterations": 0,
                "results": [],
                "analysis": {}
            }
    
    def _run_single_simulation(self, strategy: StrategyTemplate, market_data: Dict[str, Any], 
                              iteration: int) -> Dict[str, Any]:
        """Run a single simulation iteration"""
        try:
            # Add noise to market data
            noisy_data = self._add_noise_to_data(market_data)
            
            # Generate signals with noisy data
            signals = strategy.generate_signals(noisy_data)
            
            # Simulate trading
            performance = self._simulate_trading(signals, noisy_data)
            
            return {
                "iteration": iteration,
                "signals_count": len(signals),
                "performance": performance,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Error in simulation iteration {iteration}: {e}")
            return {
                "iteration": iteration,
                "signals_count": 0,
                "performance": {"total_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0},
                "timestamp": datetime.now().isoformat()
            }
    
    def _add_noise_to_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add random noise to market data"""
        noisy_data = market_data.copy()
        
        # Add noise to OHLCV data
        if 'ohlcv' in noisy_data:
            ohlcv = noisy_data['ohlcv'].copy()
            
            for key in ['open', 'high', 'low', 'close']:
                if key in ohlcv and ohlcv[key]:
                    prices = np.array(ohlcv[key])
                    noise = np.random.normal(0, self.noise_level, len(prices))
                    ohlcv[key] = (prices * (1 + noise)).tolist()
            
            noisy_data['ohlcv'] = ohlcv
        
        return noisy_data
    
    def _simulate_trading(self, signals: List[Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate trading based on signals"""
        if not signals:
            return {
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0
            }
        
        # Simple simulation - can be enhanced
        total_return = 0.0
        returns = []
        
        for signal in signals:
            # Simulate trade outcome
            if signal.signal_type in ['buy', 'sell']:
                # Random outcome based on signal strength
                outcome = random.uniform(-0.02, 0.02) * signal.strength
                total_return += outcome
                returns.append(outcome)
        
        # Calculate metrics
        if returns:
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
            win_rate = winning_trades / len(returns) if returns else 0.0
            
            # Calculate profit factor
            profits = sum(r for r in returns if r > 0)
            losses = abs(sum(r for r in returns if r < 0))
            profit_factor = profits / losses if losses > 0 else float('inf')
        else:
            mean_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
            profit_factor = 0.0
        
        return {
            "total_return": total_return,
            "mean_return": mean_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze simulation results"""
        if not results:
            return {}
        
        # Extract performance metrics
        total_returns = [r['performance']['total_return'] for r in results]
        max_drawdowns = [r['performance']['max_drawdown'] for r in results]
        sharpe_ratios = [r['performance']['sharpe_ratio'] for r in results]
        win_rates = [r['performance']['win_rate'] for r in results]
        profit_factors = [r['performance']['profit_factor'] for r in results]
        
        # Calculate statistics
        analysis = {
            "mean_return": np.mean(total_returns),
            "std_return": np.std(total_returns),
            "min_return": np.min(total_returns),
            "max_return": np.max(total_returns),
            "mean_drawdown": np.mean(max_drawdowns),
            "max_drawdown": np.max(max_drawdowns),
            "mean_sharpe": np.mean(sharpe_ratios),
            "std_sharpe": np.std(sharpe_ratios),
            "mean_win_rate": np.mean(win_rates),
            "mean_profit_factor": np.mean(profit_factors),
            "positive_returns_pct": sum(1 for r in total_returns if r > 0) / len(total_returns) * 100,
            "confidence_intervals": self._calculate_confidence_intervals(total_returns)
        }
        
        return analysis
    
    def _calculate_confidence_intervals(self, returns: List[float]) -> Dict[str, float]:
        """Calculate confidence intervals for returns"""
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        alpha = 1 - self.confidence_level
        
        # Calculate percentiles
        lower_percentile = np.percentile(returns_array, (alpha / 2) * 100)
        upper_percentile = np.percentile(returns_array, (1 - alpha / 2) * 100)
        
        return {
            "lower_bound": lower_percentile,
            "upper_bound": upper_percentile,
            "confidence_level": self.confidence_level
        }

