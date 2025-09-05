#!/usr/bin/env python3
"""
Parallel Backtesting Framework

Implements parallel processing capabilities to run multiple backtests simultaneously,
reducing overall computation time for large datasets and parameter optimization.
"""

import multiprocessing as mp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Iterable
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time
import pickle
import os
from pathlib import Path
import json

try:
    import backtrader as bt
    from backtrader import Cerebro, Strategy
except ImportError:
    bt = None
    # Create mock classes for testing when backtrader is not available
    class MockCerebro:
        def __init__(self):
            pass
        def setbroker(self, broker):
            pass
        def addsizer(self, sizer):
            pass
        def adddata(self, data, name=None):
            pass
        def addstrategy(self, strategy, **kwargs):
            pass
        def run(self):
            return []
    
    class MockStrategy:
        def __init__(self):
            pass
        def next(self):
            pass
        def notify_order(self, order):
            pass
        def notify_trade(self, trade):
            pass
        def buy(self):
            pass
        def sell(self):
            pass
        def log(self, txt, dt=None):
            pass
    
    Cerebro = MockCerebro
    Strategy = MockStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run"""
    strategy_class: str  # Strategy class name
    strategy_params: Dict[str, Any]
    data_config: Dict[str, Any]
    execution_config: Dict[str, Any]
    sizing_config: Dict[str, Any]
    start_date: str
    end_date: str
    initial_cash: float = 100000.0
    commission: float = 0.001
    run_id: str = None
    
    def __post_init__(self):
        if self.run_id is None:
            self.run_id = f"backtest_{int(time.time() * 1000)}"


@dataclass
class BacktestResult:
    """Result of a single backtest run"""
    run_id: str
    success: bool
    execution_time: float
    results: Dict[str, Any] = None
    error: str = None
    performance_metrics: Dict[str, Any] = None
    execution_statistics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class ParallelConfig:
    """Configuration for parallel backtesting"""
    max_workers: int = None  # None for auto-detection
    chunk_size: int = 1  # Number of backtests per worker
    timeout: float = 300.0  # Timeout per backtest in seconds
    memory_limit: int = 1024  # Memory limit per worker in MB
    result_aggregation: bool = True
    save_intermediate_results: bool = True
    intermediate_dir: str = "temp_backtest_results"
    cleanup_temp_files: bool = True
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        
        # Create intermediate directory if needed
        if self.save_intermediate_results:
            os.makedirs(self.intermediate_dir, exist_ok=True)


class ParallelBacktester:
    """
    Parallel backtesting engine that distributes backtest runs across multiple CPU cores.
    
    Features:
    - Multi-process execution for true parallelism
    - Automatic load balancing
    - Progress tracking and monitoring
    - Result aggregation and analysis
    - Error handling and recovery
    - Memory management
    - Configurable timeouts and resource limits
    """
    
    def __init__(self, config: ParallelConfig = None):
        self.config = config or ParallelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.total_runs = 0
        self.completed_runs = 0
        self.failed_runs = 0
        self.start_time = None
        self.end_time = None
        
        # Results storage
        self.results: List[BacktestResult] = []
        self.aggregated_results: Dict[str, Any] = {}
        
        self.logger.info(f"ParallelBacktester initialized with {self.config.max_workers} workers")
    
    def run_parallel_backtests(self, 
                             backtest_configs: List[BacktestConfig],
                             progress_callback: Callable[[int, int], None] = None) -> List[BacktestResult]:
        """
        Run multiple backtests in parallel.
        
        Args:
            backtest_configs: List of backtest configurations
            progress_callback: Optional callback for progress updates (completed, total)
            
        Returns:
            List of backtest results
        """
        self.logger.info(f"Starting parallel execution of {len(backtest_configs)} backtests")
        
        self.total_runs = len(backtest_configs)
        self.completed_runs = 0
        self.failed_runs = 0
        self.start_time = time.time()
        self.results = []
        
        # Prepare data for parallel execution
        prepared_configs = self._prepare_configs_for_parallel(backtest_configs)
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all backtests
            future_to_config = {
                executor.submit(self._execute_single_backtest, config): config 
                for config in prepared_configs
            }
            
            # Process completed backtests
            for future in as_completed(future_to_config, timeout=self.config.timeout * len(backtest_configs)):
                config = future_to_config[future]
                
                try:
                    result = future.result(timeout=self.config.timeout)
                    self.results.append(result)
                    
                    if result.success:
                        self.completed_runs += 1
                    else:
                        self.failed_runs += 1
                        self.logger.warning(f"Backtest {config.run_id} failed: {result.error}")
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(self.completed_runs + self.failed_runs, self.total_runs)
                    
                except Exception as e:
                    self.failed_runs += 1
                    error_result = BacktestResult(
                        run_id=config.run_id,
                        success=False,
                        execution_time=0.0,
                        error=f"Execution failed: {str(e)}"
                    )
                    self.results.append(error_result)
                    self.logger.error(f"Backtest {config.run_id} execution failed: {e}")
        
        self.end_time = time.time()
        
        # Aggregate results
        if self.config.result_aggregation:
            self.aggregated_results = self._aggregate_results()
        
        # Cleanup temporary files
        if self.config.cleanup_temp_files:
            self._cleanup_temp_files()
        
        self.logger.info(f"Parallel execution completed: {self.completed_runs} successful, {self.failed_runs} failed")
        
        return self.results
    
    def _prepare_configs_for_parallel(self, configs: List[BacktestConfig]) -> List[BacktestConfig]:
        """Prepare configurations for parallel execution"""
        prepared_configs = []
        
        for config in configs:
            # Serialize strategy class if it's a class object
            if hasattr(config.strategy_class, '__name__'):
                config.strategy_class = config.strategy_class.__name__
            
            prepared_configs.append(config)
        
        return prepared_configs
    
    @staticmethod
    def _execute_single_backtest(config: BacktestConfig) -> BacktestResult:
        """
        Execute a single backtest in a separate process.
        This method must be static for multiprocessing to work.
        """
        start_time = time.time()
        
        try:
            # Import required modules in the worker process
            from ..execution.execution_integration import HighFidelityExecutionIntegration, ExecutionIntegrationConfig
            from ..tick_data.enhanced_tick_feed import EnhancedTickDataFeed, EnhancedTickDataFeedFactory
            from ..tick_data.variable_spread_simulator import SpreadConfig, SpreadModel
            
            # Create execution integration
            exec_config = ExecutionIntegrationConfig(**config.execution_config)
            integration = HighFidelityExecutionIntegration(exec_config)
            
            # Setup Cerebro
            cerebro = integration.setup_cerebro()
            
            # Create data feed
            data_feed = EnhancedTickDataFeedFactory.create_feed(
                symbol=config.data_config.get('symbol', 'EURUSD'),
                start_date=config.start_date,
                end_date=config.end_date,
                **config.data_config
            )
            cerebro.adddata(data_feed)
            
            # Get strategy class
            strategy_class = globals().get(config.strategy_class)
            if strategy_class is None:
                # Try to import from common strategy modules
                try:
                    from ..strategies import get_strategy_class
                    strategy_class = get_strategy_class(config.strategy_class)
                except ImportError:
                    raise ValueError(f"Strategy class {config.strategy_class} not found")
            
            # Run backtest
            results = integration.run_backtest(strategy_class, **config.strategy_params)
            
            execution_time = time.time() - start_time
            
            return BacktestResult(
                run_id=config.run_id,
                success=True,
                execution_time=execution_time,
                results=results.get('results', []),
                performance_metrics=results.get('performance_metrics', {}),
                execution_statistics=results.get('execution_statistics', {})
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BacktestResult(
                run_id=config.run_id,
                success=False,
                execution_time=execution_time,
                error=str(e)
            )
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all backtest runs"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        # Calculate aggregate metrics
        total_execution_time = sum(r.execution_time for r in self.results)
        avg_execution_time = total_execution_time / len(self.results) if self.results else 0
        
        # Performance metrics aggregation
        performance_metrics = {}
        if successful_results:
            # Aggregate performance metrics from successful runs
            all_metrics = [r.performance_metrics for r in successful_results if r.performance_metrics]
            if all_metrics:
                performance_metrics = self._aggregate_performance_metrics(all_metrics)
        
        # Execution statistics aggregation
        execution_statistics = {}
        if successful_results:
            all_stats = [r.execution_statistics for r in successful_results if r.execution_statistics]
            if all_stats:
                execution_statistics = self._aggregate_execution_statistics(all_stats)
        
        return {
            'total_runs': len(self.results),
            'successful_runs': len(successful_results),
            'failed_runs': len(failed_results),
            'success_rate': len(successful_results) / len(self.results) if self.results else 0,
            'total_execution_time': total_execution_time,
            'average_execution_time': avg_execution_time,
            'parallel_efficiency': self._calculate_parallel_efficiency(),
            'performance_metrics': performance_metrics,
            'execution_statistics': execution_statistics,
            'failed_run_details': [{'run_id': r.run_id, 'error': r.error} for r in failed_results]
        }
    
    def _aggregate_performance_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate performance metrics from multiple runs"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Calculate averages for numeric metrics
        numeric_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        for key in numeric_keys:
            values = [m.get(key) for m in metrics_list if m.get(key) is not None]
            if values:
                aggregated[f'avg_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
                aggregated[f'min_{key}'] = np.min(values)
                aggregated[f'max_{key}'] = np.max(values)
        
        return aggregated
    
    def _aggregate_execution_statistics(self, stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate execution statistics from multiple runs"""
        if not stats_list:
            return {}
        
        aggregated = {}
        
        # Calculate totals and averages for execution stats
        total_keys = ['total_orders', 'total_trades', 'total_volume']
        for key in total_keys:
            values = [s.get(key, 0) for s in stats_list]
            aggregated[f'total_{key}'] = sum(values)
            aggregated[f'avg_{key}'] = np.mean(values) if values else 0
        
        # Calculate averages for execution quality metrics
        quality_keys = ['avg_slippage', 'avg_latency', 'execution_quality_score']
        for key in quality_keys:
            values = [s.get(key) for s in stats_list if s.get(key) is not None]
            if values:
                aggregated[f'avg_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
        
        return aggregated
    
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel execution efficiency"""
        if not self.results or self.start_time is None or self.end_time is None:
            return 0.0
        
        total_wall_time = self.end_time - self.start_time
        total_cpu_time = sum(r.execution_time for r in self.results)
        
        if total_wall_time == 0:
            return 0.0
        
        # Efficiency = (total CPU time) / (wall time * number of workers)
        efficiency = total_cpu_time / (total_wall_time * self.config.max_workers)
        return min(efficiency, 1.0)  # Cap at 100%
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during parallel execution"""
        if os.path.exists(self.config.intermediate_dir):
            try:
                import shutil
                shutil.rmtree(self.config.intermediate_dir)
                self.logger.info("Cleaned up temporary files")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp files: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of parallel execution performance"""
        if not self.results:
            return {}
        
        return {
            'total_runs': self.total_runs,
            'completed_runs': self.completed_runs,
            'failed_runs': self.failed_runs,
            'success_rate': self.completed_runs / self.total_runs if self.total_runs > 0 else 0,
            'total_execution_time': sum(r.execution_time for r in self.results),
            'wall_time': self.end_time - self.start_time if self.start_time and self.end_time else 0,
            'parallel_efficiency': self._calculate_parallel_efficiency(),
            'speedup_factor': self._calculate_speedup_factor(),
            'aggregated_results': self.aggregated_results
        }
    
    def _calculate_speedup_factor(self) -> float:
        """Calculate speedup factor compared to sequential execution"""
        if not self.results or self.start_time is None or self.end_time is None:
            return 1.0
        
        total_cpu_time = sum(r.execution_time for r in self.results)
        wall_time = self.end_time - self.start_time
        
        if wall_time == 0:
            return 1.0
        
        return total_cpu_time / wall_time
    
    def save_results(self, filepath: str):
        """Save results to file"""
        results_data = {
            'config': asdict(self.config),
            'results': [r.to_dict() for r in self.results],
            'aggregated_results': self.aggregated_results,
            'performance_summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from file"""
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.config = ParallelConfig(**results_data['config'])
        self.results = [BacktestResult(**r) for r in results_data['results']]
        self.aggregated_results = results_data['aggregated_results']
        
        self.logger.info(f"Results loaded from {filepath}")


class ParallelBacktestManager:
    """
    High-level manager for parallel backtesting operations.
    Provides convenient methods for common parallel backtesting scenarios.
    """
    
    def __init__(self, parallel_config: ParallelConfig = None):
        self.parallel_config = parallel_config or ParallelConfig()
        self.backtester = ParallelBacktester(self.parallel_config)
        self.logger = logging.getLogger(__name__)
    
    def run_parameter_optimization(self, 
                                 base_config: BacktestConfig,
                                 parameter_ranges: Dict[str, List[Any]],
                                 progress_callback: Callable[[int, int], None] = None) -> List[BacktestResult]:
        """
        Run parameter optimization using parallel backtesting.
        
        Args:
            base_config: Base backtest configuration
            parameter_ranges: Dictionary mapping parameter names to lists of values
            progress_callback: Optional progress callback
            
        Returns:
            List of backtest results for all parameter combinations
        """
        # Generate all parameter combinations
        configs = self._generate_parameter_combinations(base_config, parameter_ranges)
        
        self.logger.info(f"Running parameter optimization with {len(configs)} combinations")
        
        # Run parallel backtests
        results = self.backtester.run_parallel_backtests(configs, progress_callback)
        
        # Find best parameters
        best_result = self._find_best_result(results)
        if best_result:
            self.logger.info(f"Best result: {best_result.run_id} with parameters {best_result.results}")
        
        return results
    
    def run_multiple_strategies(self, 
                              strategy_configs: List[Tuple[str, Dict[str, Any]]],
                              data_config: Dict[str, Any],
                              execution_config: Dict[str, Any],
                              sizing_config: Dict[str, Any],
                              start_date: str,
                              end_date: str,
                              progress_callback: Callable[[int, int], None] = None) -> List[BacktestResult]:
        """
        Run multiple strategies in parallel.
        
        Args:
            strategy_configs: List of (strategy_class, strategy_params) tuples
            data_config: Data configuration
            execution_config: Execution configuration
            sizing_config: Sizing configuration
            start_date: Start date
            end_date: End date
            progress_callback: Optional progress callback
            
        Returns:
            List of backtest results
        """
        configs = []
        for i, (strategy_class, strategy_params) in enumerate(strategy_configs):
            config = BacktestConfig(
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                data_config=data_config,
                execution_config=execution_config,
                sizing_config=sizing_config,
                start_date=start_date,
                end_date=end_date,
                run_id=f"strategy_{i}_{strategy_class}"
            )
            configs.append(config)
        
        self.logger.info(f"Running {len(configs)} strategies in parallel")
        
        return self.backtester.run_parallel_backtests(configs, progress_callback)
    
    def _generate_parameter_combinations(self, 
                                       base_config: BacktestConfig, 
                                       parameter_ranges: Dict[str, List[Any]]) -> List[BacktestConfig]:
        """Generate all combinations of parameters"""
        import itertools
        
        # Get parameter names and values
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        configs = []
        
        # Generate all combinations
        for i, combination in enumerate(itertools.product(*param_values)):
            # Create new config based on base config
            config = BacktestConfig(
                strategy_class=base_config.strategy_class,
                strategy_params=base_config.strategy_params.copy(),
                data_config=base_config.data_config.copy(),
                execution_config=base_config.execution_config.copy(),
                sizing_config=base_config.sizing_config.copy(),
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                initial_cash=base_config.initial_cash,
                commission=base_config.commission,
                run_id=f"param_opt_{i}"
            )
            
            # Update strategy parameters with current combination
            for param_name, param_value in zip(param_names, combination):
                config.strategy_params[param_name] = param_value
            
            configs.append(config)
        
        return configs
    
    def _find_best_result(self, results: List[BacktestResult]) -> Optional[BacktestResult]:
        """Find the best result based on performance metrics"""
        successful_results = [r for r in results if r.success and r.performance_metrics]
        
        if not successful_results:
            return None
        
        # Sort by total return (assuming it exists in performance metrics)
        best_result = max(successful_results, 
                         key=lambda r: r.performance_metrics.get('total_return', 0))
        
        return best_result

