"""
Advanced Monte Carlo simulation for strategy robustness testing
"""

import logging
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

from ..core.strategy_template import StrategyTemplate, Signal
from ..optimization.advanced_genetic_optimizer import FitnessMetrics


class SimulationType(Enum):
    """Types of Monte Carlo simulations"""
    BOOTSTRAP = "bootstrap"
    RANDOM_WALK = "random_walk"
    HISTORICAL_SIMULATION = "historical_simulation"
    STRESS_TEST = "stress_test"
    PARAMETRIC = "parametric"


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    # Simulation parameters
    iterations: int = 1000
    confidence_level: float = 0.95
    simulation_type: SimulationType = SimulationType.BOOTSTRAP
    
    # Data parameters
    noise_level: float = 0.01
    volatility_multiplier: float = 1.0
    correlation_preservation: bool = True
    
    # Risk parameters
    max_drawdown_threshold: float = 0.2
    var_confidence: float = 0.05
    cvar_confidence: float = 0.01
    
    # Performance parameters
    use_parallel: bool = True
    num_processes: int = 4
    chunk_size: int = 100


@dataclass
class SimulationResult:
    """Result of a single Monte Carlo simulation"""
    iteration: int
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float
    cvar_95: float
    total_trades: int
    equity_curve: List[float]
    drawdown_curve: List[float]
    trade_returns: List[float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "iteration": self.iteration,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "total_trades": self.total_trades,
            "equity_curve": self.equity_curve,
            "drawdown_curve": self.drawdown_curve,
            "trade_returns": self.trade_returns,
            "timestamp": self.timestamp.isoformat()
        }


class AdvancedMonteCarloSimulator:
    """
    Advanced Monte Carlo simulator with multiple simulation types
    and comprehensive risk analysis
    """
    
    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.simulation_results: List[SimulationResult] = []
        self.aggregate_metrics: Dict[str, Any] = {}
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    async def simulate_async(self, strategy: StrategyTemplate, market_data: Dict[str, Any],
                           config: MonteCarloConfig = None) -> Dict[str, Any]:
        """
        Run advanced Monte Carlo simulation asynchronously
        
        Args:
            strategy: Strategy to simulate
            market_data: Historical market data
            config: Simulation configuration
            
        Returns:
            Dict[str, Any]: Comprehensive simulation results
        """
        if config:
            self.config = config
        
        try:
            self.start_time = datetime.now()
            self.logger.info(f"Starting advanced Monte Carlo simulation: {self.config.simulation_type.value}")
            
            # Prepare market data
            prepared_data = await self._prepare_market_data(market_data)
            
            # Run simulations
            if self.config.use_parallel:
                results = await self._run_parallel_simulations(strategy, prepared_data)
            else:
                results = await self._run_sequential_simulations(strategy, prepared_data)
            
            # Analyze results
            analysis = self._analyze_simulation_results(results)
            
            self.end_time = datetime.now()
            
            return {
                "success": True,
                "simulation_type": self.config.simulation_type.value,
                "iterations": self.config.iterations,
                "results": [r.to_dict() for r in results],
                "analysis": analysis,
                "execution_time": (self.end_time - self.start_time).total_seconds(),
                "timestamp": self.end_time.isoformat()
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
    
    def simulate(self, strategy: StrategyTemplate, market_data: Dict[str, Any],
                config: MonteCarloConfig = None) -> Dict[str, Any]:
        """Synchronous wrapper for simulation"""
        return asyncio.run(self.simulate_async(strategy, market_data, config))
    
    async def _prepare_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare market data for simulation"""
        try:
            prepared_data = market_data.copy()
            
            # Ensure OHLCV data is available
            if 'ohlcv' not in prepared_data:
                self.logger.warning("No OHLCV data found, creating mock data")
                prepared_data['ohlcv'] = self._create_mock_ohlcv_data()
            
            # Add technical indicators if not present
            if 'indicators' not in prepared_data:
                prepared_data['indicators'] = self._calculate_basic_indicators(prepared_data['ohlcv'])
            
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"Error preparing market data: {e}")
            return market_data
    
    def _create_mock_ohlcv_data(self) -> Dict[str, Any]:
        """Create mock OHLCV data for testing"""
        np.random.seed(42)  # For reproducibility
        
        # Generate 1000 data points
        n_points = 1000
        base_price = 1.1000
        
        # Generate random walk
        returns = np.random.normal(0, 0.001, n_points)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        ohlcv = {
            'open': prices[:-1],
            'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices[:-1]],
            'close': prices[1:],
            'volume': [random.randint(1000, 5000) for _ in range(n_points)],
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_points, 0, -1)]
        }
        
        return ohlcv
    
    def _calculate_basic_indicators(self, ohlcv: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic technical indicators"""
        try:
            closes = np.array(ohlcv['close'])
            
            # Simple Moving Averages
            sma_20 = np.convolve(closes, np.ones(20)/20, mode='valid')
            sma_50 = np.convolve(closes, np.ones(50)/50, mode='valid')
            
            # RSI (simplified)
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.convolve(gains, np.ones(14)/14, mode='valid')
            avg_losses = np.convolve(losses, np.ones(14)/14, mode='valid')
            
            rs = avg_gains / (avg_losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            return {
                'sma_20': sma_20.tolist(),
                'sma_50': sma_50.tolist(),
                'rsi': rsi.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating indicators: {e}")
            return {}
    
    async def _run_parallel_simulations(self, strategy: StrategyTemplate, 
                                      market_data: Dict[str, Any]) -> List[SimulationResult]:
        """Run simulations in parallel"""
        # For now, use sequential processing
        # In production, you'd use multiprocessing or asyncio.gather
        return await self._run_sequential_simulations(strategy, market_data)
    
    async def _run_sequential_simulations(self, strategy: StrategyTemplate,
                                        market_data: Dict[str, Any]) -> List[SimulationResult]:
        """Run simulations sequentially"""
        results = []
        
        for i in range(self.config.iterations):
            try:
                # Generate simulation data based on type
                sim_data = self._generate_simulation_data(market_data, i)
                
                # Run single simulation
                result = await self._run_single_simulation(strategy, sim_data, i)
                results.append(result)
                
                # Log progress
                if i % 100 == 0:
                    self.logger.info(f"Completed {i}/{self.config.iterations} simulations")
                    
            except Exception as e:
                self.logger.warning(f"Error in simulation {i}: {e}")
                # Create failed simulation result
                result = SimulationResult(
                    iteration=i,
                    total_return=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    var_95=0.0,
                    cvar_95=0.0,
                    total_trades=0,
                    equity_curve=[],
                    drawdown_curve=[],
                    trade_returns=[],
                    timestamp=datetime.now()
                )
                results.append(result)
        
        return results
    
    def _generate_simulation_data(self, market_data: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Generate data for a single simulation based on simulation type"""
        if self.config.simulation_type == SimulationType.BOOTSTRAP:
            return self._bootstrap_data(market_data, iteration)
        elif self.config.simulation_type == SimulationType.RANDOM_WALK:
            return self._random_walk_data(market_data, iteration)
        elif self.config.simulation_type == SimulationType.HISTORICAL_SIMULATION:
            return self._historical_simulation_data(market_data, iteration)
        elif self.config.simulation_type == SimulationType.STRESS_TEST:
            return self._stress_test_data(market_data, iteration)
        else:
            return self._add_noise_to_data(market_data)
    
    def _bootstrap_data(self, market_data: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Generate bootstrap resampled data"""
        sim_data = market_data.copy()
        
        if 'ohlcv' in sim_data:
            ohlcv = sim_data['ohlcv'].copy()
            n_points = len(ohlcv['close'])
            
            # Bootstrap resample indices
            np.random.seed(iteration)
            indices = np.random.choice(n_points, size=n_points, replace=True)
            
            # Resample all OHLCV data
            for key in ['open', 'high', 'low', 'close', 'volume']:
                if key in ohlcv:
                    ohlcv[key] = [ohlcv[key][i] for i in indices]
            
            sim_data['ohlcv'] = ohlcv
        
        return sim_data
    
    def _random_walk_data(self, market_data: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Generate random walk data"""
        sim_data = market_data.copy()
        
        if 'ohlcv' in sim_data:
            ohlcv = sim_data['ohlcv'].copy()
            closes = np.array(ohlcv['close'])
            
            # Generate random walk
            np.random.seed(iteration)
            n_points = len(closes)
            returns = np.random.normal(0, np.std(np.diff(closes)), n_points)
            
            # Create new price series
            new_prices = [closes[0]]
            for ret in returns:
                new_prices.append(new_prices[-1] * (1 + ret))
            
            # Update OHLCV data
            ohlcv['close'] = new_prices[1:]
            ohlcv['open'] = [closes[0]] + new_prices[1:-1]
            ohlcv['high'] = [max(o, c) for o, c in zip(ohlcv['open'], ohlcv['close'])]
            ohlcv['low'] = [min(o, c) for o, c in zip(ohlcv['open'], ohlcv['close'])]
            
            sim_data['ohlcv'] = ohlcv
        
        return sim_data
    
    def _historical_simulation_data(self, market_data: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Generate historical simulation data"""
        # Similar to bootstrap but with different sampling strategy
        return self._bootstrap_data(market_data, iteration)
    
    def _stress_test_data(self, market_data: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Generate stress test data with extreme scenarios"""
        sim_data = market_data.copy()
        
        if 'ohlcv' in sim_data:
            ohlcv = sim_data['ohlcv'].copy()
            closes = np.array(ohlcv['close'])
            
            # Create extreme scenarios
            np.random.seed(iteration)
            
            # Randomly select stress scenario
            scenario = np.random.choice(['crash', 'boom', 'volatility_spike'])
            
            if scenario == 'crash':
                # Simulate market crash
                crash_factor = np.random.uniform(0.1, 0.3)  # 10-30% crash
                stress_returns = np.random.normal(-crash_factor, 0.05, len(closes))
            elif scenario == 'boom':
                # Simulate market boom
                boom_factor = np.random.uniform(0.1, 0.3)  # 10-30% boom
                stress_returns = np.random.normal(boom_factor, 0.05, len(closes))
            else:
                # Simulate volatility spike
                vol_multiplier = np.random.uniform(2, 5)  # 2-5x volatility
                stress_returns = np.random.normal(0, np.std(np.diff(closes)) * vol_multiplier, len(closes))
            
            # Apply stress to prices
            new_prices = [closes[0]]
            for ret in stress_returns:
                new_prices.append(new_prices[-1] * (1 + ret))
            
            ohlcv['close'] = new_prices[1:]
            sim_data['ohlcv'] = ohlcv
        
        return sim_data
    
    def _add_noise_to_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to market data (parametric simulation)"""
        noisy_data = market_data.copy()
        
        if 'ohlcv' in noisy_data:
            ohlcv = noisy_data['ohlcv'].copy()
            
            for key in ['open', 'high', 'low', 'close']:
                if key in ohlcv and ohlcv[key]:
                    prices = np.array(ohlcv[key])
                    noise = np.random.normal(0, self.config.noise_level, len(prices))
                    ohlcv[key] = (prices * (1 + noise)).tolist()
            
            noisy_data['ohlcv'] = ohlcv
        
        return noisy_data
    
    async def _run_single_simulation(self, strategy: StrategyTemplate, 
                                   sim_data: Dict[str, Any], iteration: int) -> SimulationResult:
        """Run a single simulation iteration"""
        try:
            # Generate signals
            signals = strategy.generate_signals(sim_data)
            
            # Simulate trading
            trading_result = self._simulate_trading_advanced(signals, sim_data)
            
            return SimulationResult(
                iteration=iteration,
                total_return=trading_result['total_return'],
                max_drawdown=trading_result['max_drawdown'],
                sharpe_ratio=trading_result['sharpe_ratio'],
                win_rate=trading_result['win_rate'],
                profit_factor=trading_result['profit_factor'],
                var_95=trading_result['var_95'],
                cvar_95=trading_result['cvar_95'],
                total_trades=trading_result['total_trades'],
                equity_curve=trading_result['equity_curve'],
                drawdown_curve=trading_result['drawdown_curve'],
                trade_returns=trading_result['trade_returns'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.warning(f"Error in simulation {iteration}: {e}")
            return SimulationResult(
                iteration=iteration,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                var_95=0.0,
                cvar_95=0.0,
                total_trades=0,
                equity_curve=[],
                drawdown_curve=[],
                trade_returns=[],
                timestamp=datetime.now()
            )
    
    def _simulate_trading_advanced(self, signals: List[Signal], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced trading simulation with comprehensive metrics"""
        if not signals:
            return self._empty_trading_result()
        
        # Extract price data
        ohlcv = market_data.get('ohlcv', {})
        closes = ohlcv.get('close', [])
        
        if not closes:
            return self._empty_trading_result()
        
        # Simulate trades
        trades = []
        equity_curve = [1.0]  # Starting equity
        current_equity = 1.0
        
        for signal in signals:
            if signal.signal_type in ['buy', 'sell']:
                # Simulate trade outcome
                trade_return = self._simulate_trade_outcome(signal, closes)
                trades.append(trade_return)
                
                # Update equity
                current_equity *= (1 + trade_return)
                equity_curve.append(current_equity)
        
        if not trades:
            return self._empty_trading_result()
        
        # Calculate comprehensive metrics
        trade_returns = np.array(trades)
        total_return = current_equity - 1.0
        
        # Calculate drawdown curve
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown_curve = (equity_array - running_max) / running_max
        
        # Risk metrics
        var_95 = np.percentile(trade_returns, 5)  # 5th percentile (95% VaR)
        cvar_95 = np.mean(trade_returns[trade_returns <= var_95])  # Conditional VaR
        
        # Performance metrics
        mean_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        
        max_drawdown = abs(np.min(drawdown_curve))
        
        # Win rate and profit factor
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0.0
        
        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "total_trades": len(trades),
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve.tolist(),
            "trade_returns": trade_returns.tolist()
        }
    
    def _simulate_trade_outcome(self, signal: Signal, closes: List[float]) -> float:
        """Simulate individual trade outcome"""
        try:
            # Base return based on signal strength and confidence
            base_return = (signal.strength * signal.confidence - 0.5) * 0.02  # Scale to ±1%
            
            # Add market noise
            market_noise = np.random.normal(0, 0.005)  # 0.5% standard deviation
            
            # Add signal-specific noise
            signal_noise = np.random.normal(0, 0.002)  # 0.2% standard deviation
            
            # Combine factors
            trade_return = base_return + market_noise + signal_noise
            
            # Cap extreme returns
            trade_return = max(-0.05, min(0.05, trade_return))  # Cap at ±5%
            
            return trade_return
            
        except Exception as e:
            self.logger.warning(f"Error simulating trade outcome: {e}")
            return 0.0
    
    def _empty_trading_result(self) -> Dict[str, Any]:
        """Return empty trading result"""
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "total_trades": 0,
            "equity_curve": [1.0],
            "drawdown_curve": [0.0],
            "trade_returns": []
        }
    
    def _analyze_simulation_results(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze comprehensive simulation results"""
        if not results:
            return {}
        
        # Extract metrics
        total_returns = [r.total_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        win_rates = [r.win_rate for r in results]
        profit_factors = [r.profit_factor for r in results]
        var_95s = [r.var_95 for r in results]
        cvar_95s = [r.cvar_95 for r in results]
        total_trades = [r.total_trades for r in results]
        
        # Calculate comprehensive statistics
        analysis = {
            # Return statistics
            "mean_return": np.mean(total_returns),
            "std_return": np.std(total_returns),
            "min_return": np.min(total_returns),
            "max_return": np.max(total_returns),
            "median_return": np.median(total_returns),
            
            # Drawdown statistics
            "mean_drawdown": np.mean(max_drawdowns),
            "max_drawdown": np.max(max_drawdowns),
            "std_drawdown": np.std(max_drawdowns),
            
            # Sharpe ratio statistics
            "mean_sharpe": np.mean(sharpe_ratios),
            "std_sharpe": np.std(sharpe_ratios),
            "min_sharpe": np.min(sharpe_ratios),
            "max_sharpe": np.max(sharpe_ratios),
            
            # Win rate statistics
            "mean_win_rate": np.mean(win_rates),
            "std_win_rate": np.std(win_rates),
            
            # Profit factor statistics
            "mean_profit_factor": np.mean(profit_factors),
            "std_profit_factor": np.std(profit_factors),
            
            # Risk statistics
            "mean_var_95": np.mean(var_95s),
            "mean_cvar_95": np.mean(cvar_95s),
            
            # Trade statistics
            "mean_trades": np.mean(total_trades),
            "std_trades": np.std(total_trades),
            
            # Probability statistics
            "positive_returns_pct": sum(1 for r in total_returns if r > 0) / len(total_returns) * 100,
            "profitable_simulations_pct": sum(1 for r in total_returns if r > 0.01) / len(total_returns) * 100,
            "high_drawdown_pct": sum(1 for d in max_drawdowns if d > 0.1) / len(max_drawdowns) * 100,
            
            # Confidence intervals
            "return_confidence_intervals": self._calculate_confidence_intervals(total_returns),
            "drawdown_confidence_intervals": self._calculate_confidence_intervals(max_drawdowns),
            "sharpe_confidence_intervals": self._calculate_confidence_intervals(sharpe_ratios),
            
            # Robustness metrics
            "robustness_score": self._calculate_robustness_score(results),
            "stability_score": self._calculate_stability_score(results),
            "consistency_score": self._calculate_consistency_score(results)
        }
        
        return analysis
    
    def _calculate_confidence_intervals(self, data: List[float]) -> Dict[str, float]:
        """Calculate confidence intervals for data"""
        if not data:
            return {}
        
        data_array = np.array(data)
        alpha = 1 - self.config.confidence_level
        
        lower_percentile = np.percentile(data_array, (alpha / 2) * 100)
        upper_percentile = np.percentile(data_array, (1 - alpha / 2) * 100)
        
        return {
            "lower_bound": lower_percentile,
            "upper_bound": upper_percentile,
            "confidence_level": self.config.confidence_level
        }
    
    def _calculate_robustness_score(self, results: List[SimulationResult]) -> float:
        """Calculate robustness score based on consistency across simulations"""
        if not results:
            return 0.0
        
        # Factors for robustness
        positive_returns_pct = sum(1 for r in results if r.total_return > 0) / len(results)
        low_drawdown_pct = sum(1 for r in results if r.max_drawdown < 0.1) / len(results)
        consistent_performance = 1 - np.std([r.total_return for r in results]) / max(0.01, abs(np.mean([r.total_return for r in results])))
        
        # Weighted robustness score
        robustness = (positive_returns_pct * 0.4 + low_drawdown_pct * 0.3 + consistent_performance * 0.3)
        
        return max(0.0, min(1.0, robustness))
    
    def _calculate_stability_score(self, results: List[SimulationResult]) -> float:
        """Calculate stability score based on low volatility of returns"""
        if not results:
            return 0.0
        
        returns = [r.total_return for r in results]
        if not returns:
            return 0.0
        
        # Stability based on low coefficient of variation
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if abs(mean_return) < 1e-8:
            return 0.0
        
        coefficient_of_variation = std_return / abs(mean_return)
        stability = max(0.0, 1 - coefficient_of_variation)
        
        return min(1.0, stability)
    
    def _calculate_consistency_score(self, results: List[SimulationResult]) -> float:
        """Calculate consistency score based on trade frequency and win rate stability"""
        if not results:
            return 0.0
        
        # Trade frequency consistency
        trade_counts = [r.total_trades for r in results]
        if not trade_counts:
            return 0.0
        
        trade_consistency = 1 - (np.std(trade_counts) / max(1, np.mean(trade_counts)))
        
        # Win rate consistency
        win_rates = [r.win_rate for r in results]
        if not win_rates:
            return 0.0
        
        win_rate_consistency = 1 - np.std(win_rates)
        
        # Combined consistency
        consistency = (trade_consistency * 0.6 + win_rate_consistency * 0.4)
        
        return max(0.0, min(1.0, consistency))
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of simulation results"""
        return {
            "total_simulations": len(self.simulation_results),
            "execution_time": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0,
            "simulation_type": self.config.simulation_type.value,
            "iterations": self.config.iterations
        }
    
    def save_results(self, filepath: str):
        """Save simulation results to file"""
        try:
            results = {
                "simulation_results": [r.to_dict() for r in self.simulation_results],
                "aggregate_metrics": self.aggregate_metrics,
                "config": self.config.__dict__,
                "summary": self.get_simulation_summary(),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Simulation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str) -> bool:
        """Load simulation results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Reconstruct simulation results
            self.simulation_results = []
            for r in results.get("simulation_results", []):
                result = SimulationResult(
                    iteration=r['iteration'],
                    total_return=r['total_return'],
                    max_drawdown=r['max_drawdown'],
                    sharpe_ratio=r['sharpe_ratio'],
                    win_rate=r['win_rate'],
                    profit_factor=r['profit_factor'],
                    var_95=r['var_95'],
                    cvar_95=r['cvar_95'],
                    total_trades=r['total_trades'],
                    equity_curve=r['equity_curve'],
                    drawdown_curve=r['drawdown_curve'],
                    trade_returns=r['trade_returns'],
                    timestamp=datetime.fromisoformat(r['timestamp'])
                )
                self.simulation_results.append(result)
            
            self.aggregate_metrics = results.get("aggregate_metrics", {})
            
            self.logger.info(f"Simulation results loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return False

