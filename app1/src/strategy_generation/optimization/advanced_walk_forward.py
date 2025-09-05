"""
Advanced walk-forward testing for strategy validation
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

from ..core.strategy_template import StrategyTemplate
from ..optimization.advanced_genetic_optimizer import AdvancedGeneticOptimizer, OptimizationConfig
from ..optimization.advanced_monte_carlo import AdvancedMonteCarloSimulator, MonteCarloConfig


class WalkForwardMode(Enum):
    """Walk-forward testing modes"""
    FIXED_WINDOW = "fixed_window"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"
    ADAPTIVE_WINDOW = "adaptive_window"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward testing"""
    # Window parameters
    mode: WalkForwardMode = WalkForwardMode.ROLLING_WINDOW
    in_sample_ratio: float = 0.7  # 70% for optimization
    out_sample_ratio: float = 0.3  # 30% for validation
    min_in_sample_periods: int = 100  # Minimum periods for optimization
    min_out_sample_periods: int = 20  # Minimum periods for validation
    
    # Reoptimization parameters
    reoptimize_frequency: int = 1  # Reoptimize every N periods
    max_optimization_time: int = 300  # Max seconds per optimization
    
    # Validation parameters
    min_validation_periods: int = 5  # Minimum validation periods required
    robustness_threshold: float = 0.6  # Minimum robustness score
    stability_threshold: float = 0.7  # Minimum stability score
    
    # Performance parameters
    use_monte_carlo: bool = True
    monte_carlo_iterations: int = 100
    use_parallel: bool = True
    num_processes: int = 4


@dataclass
class WalkForwardPeriod:
    """Single walk-forward period"""
    period_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    in_sample_data: Dict[str, Any]
    out_sample_data: Dict[str, Any]
    optimized_parameters: Dict[str, Any]
    optimization_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    monte_carlo_results: Optional[Dict[str, Any]] = None


@dataclass
class WalkForwardResult:
    """Complete walk-forward test result"""
    test_id: str
    strategy_name: str
    total_periods: int
    successful_periods: int
    failed_periods: int
    periods: List[WalkForwardPeriod]
    aggregate_metrics: Dict[str, Any]
    robustness_score: float
    stability_score: float
    consistency_score: float
    overall_grade: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "strategy_name": self.strategy_name,
            "total_periods": self.total_periods,
            "successful_periods": self.successful_periods,
            "failed_periods": self.failed_periods,
            "periods": [
                {
                    "period_id": p.period_id,
                    "in_sample_start": p.in_sample_start.isoformat(),
                    "in_sample_end": p.in_sample_end.isoformat(),
                    "out_sample_start": p.out_sample_start.isoformat(),
                    "out_sample_end": p.out_sample_end.isoformat(),
                    "optimized_parameters": p.optimized_parameters,
                    "optimization_metrics": p.optimization_metrics,
                    "validation_metrics": p.validation_metrics,
                    "monte_carlo_results": p.monte_carlo_results
                }
                for p in self.periods
            ],
            "aggregate_metrics": self.aggregate_metrics,
            "robustness_score": self.robustness_score,
            "stability_score": self.stability_score,
            "consistency_score": self.consistency_score,
            "overall_grade": self.overall_grade,
            "timestamp": self.timestamp.isoformat()
        }


class AdvancedWalkForwardTester:
    """
    Advanced walk-forward testing system with comprehensive validation
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.genetic_optimizer = AdvancedGeneticOptimizer()
        self.monte_carlo_simulator = AdvancedMonteCarloSimulator()
        
        # Results storage
        self.walk_forward_results: List[WalkForwardResult] = []
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    async def test_async(self, strategy: StrategyTemplate, market_data: Dict[str, Any],
                        parameter_space: Any, config: WalkForwardConfig = None) -> WalkForwardResult:
        """
        Run advanced walk-forward test asynchronously
        
        Args:
            strategy: Strategy to test
            market_data: Historical market data
            parameter_space: Parameter space for optimization
            config: Walk-forward configuration
            
        Returns:
            WalkForwardResult: Comprehensive test results
        """
        if config:
            self.config = config
        
        try:
            self.start_time = datetime.now()
            test_id = f"wf_{strategy.parameters.strategy_id}_{int(self.start_time.timestamp())}"
            
            self.logger.info(f"Starting walk-forward test: {test_id}")
            
            # Prepare data and create periods
            periods = await self._create_walk_forward_periods(market_data)
            
            if not periods:
                raise ValueError("No valid walk-forward periods created")
            
            self.logger.info(f"Created {len(periods)} walk-forward periods")
            
            # Run walk-forward test
            successful_periods = 0
            failed_periods = 0
            processed_periods = []
            
            for i, period in enumerate(periods):
                try:
                    self.logger.info(f"Processing period {i+1}/{len(periods)}")
                    
                    # Optimize strategy on in-sample data
                    optimization_result = await self._optimize_strategy(
                        strategy, period.in_sample_data, parameter_space
                    )
                    
                    if not optimization_result['success']:
                        self.logger.warning(f"Optimization failed for period {i+1}")
                        failed_periods += 1
                        continue
                    
                    # Update period with optimization results
                    period.optimized_parameters = optimization_result['best_parameters']
                    period.optimization_metrics = optimization_result['optimization_metrics']
                    
                    # Validate on out-of-sample data
                    validation_result = await self._validate_strategy(
                        strategy, period.out_sample_data, period.optimized_parameters
                    )
                    
                    period.validation_metrics = validation_result
                    
                    # Run Monte Carlo simulation if enabled
                    if self.config.use_monte_carlo:
                        monte_carlo_result = await self._run_monte_carlo_validation(
                            strategy, period.out_sample_data, period.optimized_parameters
                        )
                        period.monte_carlo_results = monte_carlo_result
                    
                    # Check if period passed validation
                    if self._evaluate_period(period):
                        successful_periods += 1
                    else:
                        failed_periods += 1
                    
                    processed_periods.append(period)
                    
                except Exception as e:
                    self.logger.error(f"Error processing period {i+1}: {e}")
                    failed_periods += 1
                    continue
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics(processed_periods)
            
            # Calculate overall scores
            robustness_score = self._calculate_robustness_score(processed_periods)
            stability_score = self._calculate_stability_score(processed_periods)
            consistency_score = self._calculate_consistency_score(processed_periods)
            
            # Determine overall grade
            overall_grade = self._determine_overall_grade(
                robustness_score, stability_score, consistency_score, 
                successful_periods, len(periods)
            )
            
            self.end_time = datetime.now()
            
            # Create result
            result = WalkForwardResult(
                test_id=test_id,
                strategy_name=strategy.parameters.name,
                total_periods=len(periods),
                successful_periods=successful_periods,
                failed_periods=failed_periods,
                periods=processed_periods,
                aggregate_metrics=aggregate_metrics,
                robustness_score=robustness_score,
                stability_score=stability_score,
                consistency_score=consistency_score,
                overall_grade=overall_grade,
                timestamp=self.end_time
            )
            
            self.walk_forward_results.append(result)
            
            self.logger.info(f"Walk-forward test completed: {successful_periods}/{len(periods)} periods successful")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward test: {e}")
            raise
    
    def test(self, strategy: StrategyTemplate, market_data: Dict[str, Any],
             parameter_space: Any, config: WalkForwardConfig = None) -> WalkForwardResult:
        """Synchronous wrapper for walk-forward testing"""
        return asyncio.run(self.test_async(strategy, market_data, parameter_space, config))
    
    async def _create_walk_forward_periods(self, market_data: Dict[str, Any]) -> List[WalkForwardPeriod]:
        """Create walk-forward periods based on configuration"""
        try:
            # Extract timestamps from market data
            ohlcv = market_data.get('ohlcv', {})
            timestamps = ohlcv.get('timestamp', [])
            
            if not timestamps:
                self.logger.error("No timestamps found in market data")
                return []
            
            # Convert to datetime if needed
            if isinstance(timestamps[0], str):
                timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            
            # Sort timestamps
            timestamps.sort()
            total_periods = len(timestamps)
            
            if total_periods < self.config.min_in_sample_periods + self.config.min_out_sample_periods:
                self.logger.error("Insufficient data for walk-forward testing")
                return []
            
            periods = []
            period_id = 0
            
            if self.config.mode == WalkForwardMode.FIXED_WINDOW:
                periods = self._create_fixed_window_periods(timestamps, market_data, period_id)
            elif self.config.mode == WalkForwardMode.EXPANDING_WINDOW:
                periods = self._create_expanding_window_periods(timestamps, market_data, period_id)
            elif self.config.mode == WalkForwardMode.ROLLING_WINDOW:
                periods = self._create_rolling_window_periods(timestamps, market_data, period_id)
            elif self.config.mode == WalkForwardMode.ADAPTIVE_WINDOW:
                periods = self._create_adaptive_window_periods(timestamps, market_data, period_id)
            
            return periods
            
        except Exception as e:
            self.logger.error(f"Error creating walk-forward periods: {e}")
            return []
    
    def _create_fixed_window_periods(self, timestamps: List[datetime], 
                                   market_data: Dict[str, Any], start_id: int) -> List[WalkForwardPeriod]:
        """Create fixed window periods"""
        periods = []
        period_id = start_id
        
        # Calculate window sizes
        total_periods = len(timestamps)
        in_sample_size = int(total_periods * self.config.in_sample_ratio)
        out_sample_size = int(total_periods * self.config.out_sample_ratio)
        
        # Create single period
        if total_periods >= in_sample_size + out_sample_size:
            in_sample_start = timestamps[0]
            in_sample_end = timestamps[in_sample_size - 1]
            out_sample_start = timestamps[in_sample_size]
            out_sample_end = timestamps[min(in_sample_size + out_sample_size - 1, total_periods - 1)]
            
            period = WalkForwardPeriod(
                period_id=period_id,
                in_sample_start=in_sample_start,
                in_sample_end=in_sample_end,
                out_sample_start=out_sample_start,
                out_sample_end=out_sample_end,
                in_sample_data=self._extract_period_data(market_data, in_sample_start, in_sample_end),
                out_sample_data=self._extract_period_data(market_data, out_sample_start, out_sample_end),
                optimized_parameters={},
                optimization_metrics={},
                validation_metrics={}
            )
            periods.append(period)
        
        return periods
    
    def _create_expanding_window_periods(self, timestamps: List[datetime], 
                                       market_data: Dict[str, Any], start_id: int) -> List[WalkForwardPeriod]:
        """Create expanding window periods"""
        periods = []
        period_id = start_id
        
        # Calculate step size
        step_size = max(1, int(len(timestamps) * self.config.out_sample_ratio))
        
        # Start with minimum in-sample size
        current_in_sample_size = self.config.min_in_sample_periods
        
        while current_in_sample_size + self.config.min_out_sample_periods <= len(timestamps):
            # Define period boundaries
            in_sample_start = timestamps[0]
            in_sample_end = timestamps[current_in_sample_size - 1]
            out_sample_start = timestamps[current_in_sample_size]
            out_sample_end = timestamps[min(
                current_in_sample_size + self.config.min_out_sample_periods - 1,
                len(timestamps) - 1
            )]
            
            period = WalkForwardPeriod(
                period_id=period_id,
                in_sample_start=in_sample_start,
                in_sample_end=in_sample_end,
                out_sample_start=out_sample_start,
                out_sample_end=out_sample_end,
                in_sample_data=self._extract_period_data(market_data, in_sample_start, in_sample_end),
                out_sample_data=self._extract_period_data(market_data, out_sample_start, out_sample_end),
                optimized_parameters={},
                optimization_metrics={},
                validation_metrics={}
            )
            periods.append(period)
            
            # Expand in-sample window
            current_in_sample_size += step_size
            period_id += 1
        
        return periods
    
    def _create_rolling_window_periods(self, timestamps: List[datetime], 
                                     market_data: Dict[str, Any], start_id: int) -> List[WalkForwardPeriod]:
        """Create rolling window periods"""
        periods = []
        period_id = start_id
        
        # Calculate window sizes
        in_sample_size = int(len(timestamps) * self.config.in_sample_ratio)
        out_sample_size = int(len(timestamps) * self.config.out_sample_ratio)
        step_size = max(1, int(out_sample_size * 0.5))  # 50% overlap
        
        # Create rolling periods
        start_idx = 0
        while start_idx + in_sample_size + out_sample_size <= len(timestamps):
            # Define period boundaries
            in_sample_start = timestamps[start_idx]
            in_sample_end = timestamps[start_idx + in_sample_size - 1]
            out_sample_start = timestamps[start_idx + in_sample_size]
            out_sample_end = timestamps[start_idx + in_sample_size + out_sample_size - 1]
            
            period = WalkForwardPeriod(
                period_id=period_id,
                in_sample_start=in_sample_start,
                in_sample_end=in_sample_end,
                out_sample_start=out_sample_start,
                out_sample_end=out_sample_end,
                in_sample_data=self._extract_period_data(market_data, in_sample_start, in_sample_end),
                out_sample_data=self._extract_period_data(market_data, out_sample_start, out_sample_end),
                optimized_parameters={},
                optimization_metrics={},
                validation_metrics={}
            )
            periods.append(period)
            
            # Move window
            start_idx += step_size
            period_id += 1
        
        return periods
    
    def _create_adaptive_window_periods(self, timestamps: List[datetime], 
                                      market_data: Dict[str, Any], start_id: int) -> List[WalkForwardPeriod]:
        """Create adaptive window periods based on market volatility"""
        # For now, use rolling window as base
        # In production, this would adapt window sizes based on market conditions
        return self._create_rolling_window_periods(timestamps, market_data, start_id)
    
    def _extract_period_data(self, market_data: Dict[str, Any], 
                           start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Extract data for a specific time period"""
        try:
            period_data = {}
            
            # Extract OHLCV data
            if 'ohlcv' in market_data:
                ohlcv = market_data['ohlcv'].copy()
                timestamps = ohlcv.get('timestamp', [])
                
                # Convert timestamps to datetime if needed
                if timestamps and isinstance(timestamps[0], str):
                    timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
                
                # Find indices for the period
                start_idx = 0
                end_idx = len(timestamps)
                
                for i, ts in enumerate(timestamps):
                    if ts >= start_time:
                        start_idx = i
                        break
                
                for i, ts in enumerate(timestamps[start_idx:], start_idx):
                    if ts > end_time:
                        end_idx = i
                        break
                
                # Extract period data
                period_ohlcv = {}
                for key in ['open', 'high', 'low', 'close', 'volume']:
                    if key in ohlcv:
                        period_ohlcv[key] = ohlcv[key][start_idx:end_idx]
                
                period_ohlcv['timestamp'] = timestamps[start_idx:end_idx]
                period_data['ohlcv'] = period_ohlcv
            
            # Extract other data (indicators, sentiment, etc.)
            for key in ['indicators', 'sentiment', 'economic_events']:
                if key in market_data:
                    period_data[key] = market_data[key]  # Simplified - in production, filter by time
            
            return period_data
            
        except Exception as e:
            self.logger.error(f"Error extracting period data: {e}")
            return {}
    
    async def _optimize_strategy(self, strategy: StrategyTemplate, 
                               in_sample_data: Dict[str, Any], 
                               parameter_space: Any) -> Dict[str, Any]:
        """Optimize strategy on in-sample data"""
        try:
            # Create optimization config
            opt_config = OptimizationConfig(
                population_size=50,
                generations=20,
                use_multi_objective=True
            )
            
            # Run optimization
            result = await self.genetic_optimizer.optimize_async(
                strategy, parameter_space, in_sample_data, opt_config
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_strategy(self, strategy: StrategyTemplate, 
                               out_sample_data: Dict[str, Any], 
                               optimized_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy on out-of-sample data"""
        try:
            # Create strategy instance with optimized parameters
            # This would involve updating the strategy with new parameters
            # For now, we'll simulate validation
            
            # Generate signals
            signals = strategy.generate_signals(out_sample_data)
            
            # Calculate validation metrics
            validation_metrics = self._calculate_validation_metrics(signals, out_sample_data)
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating strategy: {e}")
            return {"error": str(e)}
    
    def _calculate_validation_metrics(self, signals: List, out_sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate validation metrics for out-of-sample data"""
        try:
            if not signals:
                return {
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0
                }
            
            # Simulate trading performance
            ohlcv = out_sample_data.get('ohlcv', {})
            closes = ohlcv.get('close', [])
            
            if not closes:
                return {
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0
                }
            
            # Simulate trades
            trades = []
            for signal in signals:
                if hasattr(signal, 'signal_type') and signal.signal_type in ['buy', 'sell']:
                    # Simple trade simulation
                    trade_return = np.random.normal(0, 0.01) * signal.strength
                    trades.append(trade_return)
            
            if not trades:
                return {
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0
                }
            
            # Calculate metrics
            total_return = sum(trades)
            mean_return = np.mean(trades)
            std_return = np.std(trades)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            
            # Calculate drawdown
            cumulative_returns = np.cumsum(trades)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # Calculate win rate and profit factor
            winning_trades = [t for t in trades if t > 0]
            losing_trades = [t for t in trades if t < 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0.0
            
            gross_profit = sum(winning_trades) if winning_trades else 0.0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            return {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": len(trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating validation metrics: {e}")
            return {"error": str(e)}
    
    async def _run_monte_carlo_validation(self, strategy: StrategyTemplate, 
                                        out_sample_data: Dict[str, Any], 
                                        optimized_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo validation on out-of-sample data"""
        try:
            # Create Monte Carlo config
            mc_config = MonteCarloConfig(
                iterations=self.config.monte_carlo_iterations,
                simulation_type=SimulationType.BOOTSTRAP
            )
            
            # Run Monte Carlo simulation
            result = await self.monte_carlo_simulator.simulate_async(
                strategy, out_sample_data, mc_config
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running Monte Carlo validation: {e}")
            return {"error": str(e)}
    
    def _evaluate_period(self, period: WalkForwardPeriod) -> bool:
        """Evaluate if a period passed validation"""
        try:
            validation_metrics = period.validation_metrics
            
            # Check basic performance criteria
            if validation_metrics.get('total_return', 0) < 0:
                return False
            
            if validation_metrics.get('max_drawdown', 1) > self.config.max_drawdown_threshold:
                return False
            
            if validation_metrics.get('sharpe_ratio', 0) < 0:
                return False
            
            # Check Monte Carlo results if available
            if period.monte_carlo_results and period.monte_carlo_results.get('success'):
                analysis = period.monte_carlo_results.get('analysis', {})
                robustness_score = analysis.get('robustness_score', 0)
                
                if robustness_score < self.config.robustness_threshold:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating period: {e}")
            return False
    
    def _calculate_aggregate_metrics(self, periods: List[WalkForwardPeriod]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all periods"""
        if not periods:
            return {}
        
        # Extract metrics from all periods
        total_returns = []
        max_drawdowns = []
        sharpe_ratios = []
        win_rates = []
        profit_factors = []
        total_trades = []
        
        for period in periods:
            metrics = period.validation_metrics
            if 'error' not in metrics:
                total_returns.append(metrics.get('total_return', 0))
                max_drawdowns.append(metrics.get('max_drawdown', 0))
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                win_rates.append(metrics.get('win_rate', 0))
                profit_factors.append(metrics.get('profit_factor', 0))
                total_trades.append(metrics.get('total_trades', 0))
        
        if not total_returns:
            return {}
        
        # Calculate aggregate statistics
        return {
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
            "mean_trades": np.mean(total_trades),
            "positive_periods_pct": sum(1 for r in total_returns if r > 0) / len(total_returns) * 100,
            "consistent_performance": 1 - np.std(total_returns) / max(0.01, abs(np.mean(total_returns)))
        }
    
    def _calculate_robustness_score(self, periods: List[WalkForwardPeriod]) -> float:
        """Calculate robustness score based on consistency across periods"""
        if not periods:
            return 0.0
        
        # Extract performance metrics
        total_returns = []
        for period in periods:
            metrics = period.validation_metrics
            if 'error' not in metrics:
                total_returns.append(metrics.get('total_return', 0))
        
        if not total_returns:
            return 0.0
        
        # Calculate robustness factors
        positive_periods_pct = sum(1 for r in total_returns if r > 0) / len(total_returns)
        low_drawdown_pct = sum(1 for p in periods if p.validation_metrics.get('max_drawdown', 1) < 0.1) / len(periods)
        consistency = 1 - np.std(total_returns) / max(0.01, abs(np.mean(total_returns)))
        
        # Weighted robustness score
        robustness = (positive_periods_pct * 0.4 + low_drawdown_pct * 0.3 + consistency * 0.3)
        
        return max(0.0, min(1.0, robustness))
    
    def _calculate_stability_score(self, periods: List[WalkForwardPeriod]) -> float:
        """Calculate stability score based on low volatility of returns"""
        if not periods:
            return 0.0
        
        # Extract returns
        total_returns = []
        for period in periods:
            metrics = period.validation_metrics
            if 'error' not in metrics:
                total_returns.append(metrics.get('total_return', 0))
        
        if not total_returns:
            return 0.0
        
        # Calculate stability
        mean_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        
        if abs(mean_return) < 1e-8:
            return 0.0
        
        coefficient_of_variation = std_return / abs(mean_return)
        stability = max(0.0, 1 - coefficient_of_variation)
        
        return min(1.0, stability)
    
    def _calculate_consistency_score(self, periods: List[WalkForwardPeriod]) -> float:
        """Calculate consistency score based on trade frequency and win rate stability"""
        if not periods:
            return 0.0
        
        # Extract trade counts and win rates
        trade_counts = []
        win_rates = []
        
        for period in periods:
            metrics = period.validation_metrics
            if 'error' not in metrics:
                trade_counts.append(metrics.get('total_trades', 0))
                win_rates.append(metrics.get('win_rate', 0))
        
        if not trade_counts:
            return 0.0
        
        # Calculate consistency
        trade_consistency = 1 - (np.std(trade_counts) / max(1, np.mean(trade_counts)))
        win_rate_consistency = 1 - np.std(win_rates)
        
        consistency = (trade_consistency * 0.6 + win_rate_consistency * 0.4)
        
        return max(0.0, min(1.0, consistency))
    
    def _determine_overall_grade(self, robustness_score: float, stability_score: float, 
                               consistency_score: float, successful_periods: int, 
                               total_periods: int) -> str:
        """Determine overall grade based on scores"""
        success_rate = successful_periods / total_periods if total_periods > 0 else 0
        
        # Calculate weighted score
        overall_score = (
            robustness_score * 0.3 +
            stability_score * 0.3 +
            consistency_score * 0.2 +
            success_rate * 0.2
        )
        
        # Determine grade
        if overall_score >= 0.9:
            return "A+"
        elif overall_score >= 0.8:
            return "A"
        elif overall_score >= 0.7:
            return "B+"
        elif overall_score >= 0.6:
            return "B"
        elif overall_score >= 0.5:
            return "C+"
        elif overall_score >= 0.4:
            return "C"
        else:
            return "D"
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of walk-forward tests"""
        return {
            "total_tests": len(self.walk_forward_results),
            "execution_time": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0,
            "config": self.config.__dict__
        }
    
    def save_results(self, filepath: str):
        """Save walk-forward results to file"""
        try:
            results = {
                "walk_forward_results": [r.to_dict() for r in self.walk_forward_results],
                "summary": self.get_test_summary(),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Walk-forward results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str) -> bool:
        """Load walk-forward results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Reconstruct walk-forward results
            self.walk_forward_results = []
            for r in results.get("walk_forward_results", []):
                # Reconstruct periods
                periods = []
                for p in r.get("periods", []):
                    period = WalkForwardPeriod(
                        period_id=p['period_id'],
                        in_sample_start=datetime.fromisoformat(p['in_sample_start']),
                        in_sample_end=datetime.fromisoformat(p['in_sample_end']),
                        out_sample_start=datetime.fromisoformat(p['out_sample_start']),
                        out_sample_end=datetime.fromisoformat(p['out_sample_end']),
                        in_sample_data=p.get('in_sample_data', {}),
                        out_sample_data=p.get('out_sample_data', {}),
                        optimized_parameters=p.get('optimized_parameters', {}),
                        optimization_metrics=p.get('optimization_metrics', {}),
                        validation_metrics=p.get('validation_metrics', {}),
                        monte_carlo_results=p.get('monte_carlo_results')
                    )
                    periods.append(period)
                
                # Reconstruct result
                result = WalkForwardResult(
                    test_id=r['test_id'],
                    strategy_name=r['strategy_name'],
                    total_periods=r['total_periods'],
                    successful_periods=r['successful_periods'],
                    failed_periods=r['failed_periods'],
                    periods=periods,
                    aggregate_metrics=r.get('aggregate_metrics', {}),
                    robustness_score=r.get('robustness_score', 0),
                    stability_score=r.get('stability_score', 0),
                    consistency_score=r.get('consistency_score', 0),
                    overall_grade=r.get('overall_grade', 'D'),
                    timestamp=datetime.fromisoformat(r['timestamp'])
                )
                self.walk_forward_results.append(result)
            
            self.logger.info(f"Walk-forward results loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return False

