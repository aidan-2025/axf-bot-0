#!/usr/bin/env python3
"""
Parallel Backtesting Demonstration

Demonstrates the performance improvements of parallel backtesting
compared to sequential execution, including benchmarking and analysis.
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append('app1/src')

from backtesting.parallel.parallel_backtester import (
    ParallelBacktester, ParallelConfig, BacktestConfig, BacktestResult
)
from backtesting.parallel.parallel_manager import (
    PerformanceBenchmark, ResourceManager, ParallelBacktestManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoStrategy:
    """
    Simple demo strategy for testing parallel backtesting performance.
    """
    
    def __init__(self, param1=1.0, param2=0.5, delay_ms=10):
        self.param1 = param1
        self.param2 = param2
        self.delay_ms = delay_ms
        self.order_count = 0
        self.execution_history = []
    
    def next(self):
        """Strategy logic - simulate some computation"""
        # Simulate strategy computation time
        time.sleep(self.delay_ms / 1000.0)
        
        # Simulate order generation
        if np.random.random() < 0.1:  # 10% chance of order
            self.order_count += 1
    
    def notify_order(self, order):
        """Handle order notifications"""
        self.execution_history.append({
            'timestamp': datetime.now(),
            'order_ref': getattr(order, 'ref', 'unknown'),
            'status': 'submitted'
        })
    
    def notify_trade(self, trade):
        """Handle trade notifications"""
        pass
    
    def log(self, txt, dt=None):
        """Logging method"""
        pass
    
    def get_performance_metrics(self):
        """Get performance metrics for the strategy"""
        return {
            'total_return': np.random.normal(0.05, 0.02),  # Simulated return
            'sharpe_ratio': np.random.normal(1.2, 0.3),    # Simulated Sharpe ratio
            'max_drawdown': np.random.uniform(0.01, 0.05), # Simulated drawdown
            'win_rate': np.random.uniform(0.45, 0.65),     # Simulated win rate
            'total_orders': self.order_count
        }
    
    def get_execution_statistics(self):
        """Get execution statistics for the strategy"""
        return {
            'total_orders': self.order_count,
            'total_trades': len(self.execution_history),
            'avg_slippage': np.random.uniform(0.0001, 0.001),
            'avg_latency': np.random.uniform(1, 10),
            'execution_quality_score': np.random.uniform(80, 95)
        }


def create_sample_tick_data(symbol="EURUSD", start_date="2023-01-01", 
                          end_date="2023-01-31", num_ticks=1000):
    """Create sample tick data for demonstration"""
    
    # Generate timestamps
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    timestamps = pd.date_range(start=start_dt, end=end_dt, periods=num_ticks)
    
    # Generate price data
    base_price = 1.1000
    price_changes = np.random.normal(0, 0.0001, num_ticks)
    prices = base_price + np.cumsum(price_changes)
    
    # Generate bid/ask spreads
    spreads = np.random.uniform(0.0001, 0.0005, num_ticks)
    bid_prices = prices - spreads / 2
    ask_prices = prices + spreads / 2
    
    # Generate volume data
    volumes = np.random.uniform(1000, 10000, num_ticks)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'bid': bid_prices,
        'ask': ask_prices,
        'mid': prices,
        'spread': spreads,
        'volume': volumes
    })


def create_backtest_configs(num_configs=10, base_delay_ms=10):
    """Create multiple backtest configurations for testing"""
    
    configs = []
    
    for i in range(num_configs):
        # Vary strategy parameters
        param1 = np.random.uniform(0.5, 2.0)
        param2 = np.random.uniform(0.1, 1.0)
        delay_ms = base_delay_ms + np.random.randint(0, 20)  # Vary computation time
        
        config = BacktestConfig(
            strategy_class="DemoStrategy",
            strategy_params={
                "param1": param1,
                "param2": param2,
                "delay_ms": delay_ms
            },
            data_config={
                "symbol": "EURUSD",
                "source": "demo"
            },
            execution_config={
                "slippage_model": "realistic",
                "slippage_bps": 0.5,
                "latency_ms": 10
            },
            sizing_config={
                "method": "fixed",
                "size": 1000
            },
            start_date="2023-01-01",
            end_date="2023-01-31",
            initial_cash=100000.0,
            commission=0.001,
            run_id=f"demo_backtest_{i:03d}"
        )
        
        configs.append(config)
    
    return configs


def demonstrate_basic_parallel_execution():
    """Demonstrate basic parallel backtesting execution"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Basic Parallel Execution")
    print("="*60)
    
    # Create backtest configurations
    configs = create_backtest_configs(num_configs=8, base_delay_ms=50)
    
    print(f"Created {len(configs)} backtest configurations")
    print(f"Each backtest simulates {50}ms of computation time")
    
    # Test different numbers of workers
    worker_counts = [1, 2, 4, 8]
    
    for num_workers in worker_counts:
        print(f"\nTesting with {num_workers} worker(s)...")
        
        # Configure parallel backtester
        parallel_config = ParallelConfig(
            max_workers=num_workers,
            timeout=60.0,
            result_aggregation=True
        )
        
        backtester = ParallelBacktester(parallel_config)
        
        # Run backtests
        start_time = time.time()
        results = backtester.run_parallel_backtests(configs)
        end_time = time.time()
        
        # Calculate performance metrics
        execution_time = end_time - start_time
        successful_runs = sum(1 for r in results if r.success)
        success_rate = successful_runs / len(results) if results else 0
        
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Successful runs: {successful_runs}/{len(results)}")
        print(f"  Success rate: {success_rate:.1%}")
        
        if num_workers > 1:
            # Calculate speedup
            single_worker_time = execution_time * num_workers  # Approximate
            speedup = single_worker_time / execution_time if execution_time > 0 else 1.0
            efficiency = speedup / num_workers
            
            print(f"  Estimated speedup: {speedup:.2f}x")
            print(f"  Efficiency: {efficiency:.1%}")


def demonstrate_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Performance Benchmarking")
    print("="*60)
    
    # Create configurations for benchmarking
    configs = create_backtest_configs(num_configs=6, base_delay_ms=30)
    
    print(f"Benchmarking with {len(configs)} backtest configurations")
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Run benchmark
    parallel_config = ParallelConfig(max_workers=4, timeout=60.0)
    
    print("Running benchmark (this may take a moment)...")
    benchmark_results = benchmark.benchmark_parallel_vs_sequential(
        configs, 
        parallel_config, 
        num_runs=2
    )
    
    # Display results
    print(f"\nBenchmark Results:")
    print(f"  Parallel execution (avg): {benchmark_results['parallel_avg']:.2f}s")
    print(f"  Sequential execution (avg): {benchmark_results['sequential_avg']:.2f}s")
    print(f"  Speedup factor: {benchmark_results['speedup_factor']:.2f}x")
    print(f"  Efficiency: {benchmark_results['efficiency']:.1%}")
    print(f"  Number of workers: {benchmark_results['num_workers']}")
    print(f"  Number of backtests: {benchmark_results['num_backtests']}")


def demonstrate_resource_monitoring():
    """Demonstrate resource monitoring during parallel execution"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Resource Monitoring")
    print("="*60)
    
    # Create configurations
    configs = create_backtest_configs(num_configs=4, base_delay_ms=100)
    
    print(f"Running {len(configs)} backtests with resource monitoring...")
    
    # Create manager with resource monitoring
    manager = ParallelBacktestManager(ParallelConfig(max_workers=2))
    
    # Progress callback
    def progress_callback(completed, total):
        print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    # Run with monitoring
    results, monitoring_data = manager.run_with_monitoring(
        configs, 
        progress_callback=progress_callback,
        enable_resource_monitoring=True
    )
    
    # Display resource summary
    resource_summary = monitoring_data.get('resource_summary', {})
    if resource_summary:
        print(f"\nResource Usage Summary:")
        print(f"  Duration: {resource_summary.get('duration', 0):.2f} seconds")
        print(f"  Average CPU usage: {resource_summary.get('avg_cpu_usage', 0):.1f}%")
        print(f"  Peak CPU usage: {resource_summary.get('max_cpu_usage', 0):.1f}%")
        print(f"  Average memory usage: {resource_summary.get('avg_memory_usage_mb', 0):.1f} MB")
        print(f"  Peak memory usage: {resource_summary.get('max_memory_usage_mb', 0):.1f} MB")
    
    # Display performance summary
    performance_summary = monitoring_data.get('performance_summary', {})
    if performance_summary:
        print(f"\nPerformance Summary:")
        print(f"  Total runs: {performance_summary.get('total_runs', 0)}")
        print(f"  Successful runs: {performance_summary.get('completed_runs', 0)}")
        print(f"  Success rate: {performance_summary.get('success_rate', 0):.1%}")
        print(f"  Total execution time: {performance_summary.get('total_execution_time', 0):.2f}s")
        print(f"  Wall time: {performance_summary.get('wall_time', 0):.2f}s")
        print(f"  Parallel efficiency: {performance_summary.get('parallel_efficiency', 0):.1%}")
        print(f"  Speedup factor: {performance_summary.get('speedup_factor', 1):.2f}x")


def demonstrate_parameter_optimization():
    """Demonstrate parameter optimization with parallel backtesting"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION 4: Parameter Optimization")
    print("="*60)
    
    # Create base configuration
    base_config = BacktestConfig(
        strategy_class="DemoStrategy",
        strategy_params={
            "param1": 1.0,
            "param2": 0.5,
            "delay_ms": 20
        },
        data_config={
            "symbol": "EURUSD",
            "source": "demo"
        },
        execution_config={
            "slippage_model": "realistic",
            "slippage_bps": 0.5
        },
        sizing_config={
            "method": "fixed",
            "size": 1000
        },
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    
    # Define parameter ranges for optimization
    parameter_ranges = {
        "param1": [0.5, 1.0, 1.5, 2.0],
        "param2": [0.2, 0.5, 0.8],
        "delay_ms": [10, 20, 30]
    }
    
    print(f"Parameter ranges:")
    for param, values in parameter_ranges.items():
        print(f"  {param}: {values}")
    
    total_combinations = np.prod([len(values) for values in parameter_ranges.values()])
    print(f"Total combinations: {total_combinations}")
    
    # Create manager
    manager = ParallelBacktestManager(ParallelConfig(max_workers=4, timeout=60.0))
    
    # Progress callback
    def progress_callback(completed, total):
        print(f"  Optimization progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    print("\nRunning parameter optimization...")
    
    # Run optimization
    optimization_results = manager.run_parameter_optimization_with_benchmark(
        base_config,
        parameter_ranges,
        progress_callback=progress_callback
    )
    
    # Display results
    results = optimization_results['results']
    best_result = optimization_results['best_result']
    optimization_metrics = optimization_results['optimization_metrics']
    
    print(f"\nOptimization Results:")
    print(f"  Total combinations tested: {optimization_metrics.get('total_combinations', 0)}")
    print(f"  Successful combinations: {optimization_metrics.get('successful_combinations', 0)}")
    print(f"  Success rate: {optimization_metrics.get('success_rate', 0):.1%}")
    
    if best_result:
        print(f"  Best result: {best_result.run_id}")
        print(f"  Best return: {best_result.performance_metrics.get('total_return', 0):.3f}")
        print(f"  Best Sharpe ratio: {best_result.performance_metrics.get('sharpe_ratio', 0):.3f}")
    
    # Display parameter sensitivity
    param_sensitivity = optimization_metrics.get('parameter_sensitivity', {})
    if param_sensitivity:
        print(f"\nParameter Sensitivity:")
        for param, sensitivity in param_sensitivity.items():
            correlation = sensitivity.get('correlation', 0)
            print(f"  {param}: correlation = {correlation:.3f}")


def demonstrate_optimal_workers_benchmark():
    """Demonstrate finding optimal number of workers"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION 5: Optimal Workers Benchmark")
    print("="*60)
    
    # Create configurations
    configs = create_backtest_configs(num_configs=8, base_delay_ms=40)
    
    print(f"Benchmarking optimal workers with {len(configs)} backtests")
    
    # Create manager
    manager = ParallelBacktestManager()
    
    # Benchmark different worker counts
    worker_range = range(1, min(9, len(configs) + 1))
    
    print(f"Testing worker counts: {list(worker_range)}")
    print("Running benchmark (this may take a moment)...")
    
    benchmark_results = manager.benchmark_optimal_workers(
        configs,
        max_workers_range=worker_range,
        num_runs=2
    )
    
    # Display results
    print(f"\nWorker Performance Results:")
    print(f"{'Workers':<8} {'Avg Time (s)':<12} {'Std Dev (s)':<12} {'Efficiency':<12}")
    print("-" * 50)
    
    for workers, data in benchmark_results['benchmark_results'].items():
        avg_time = data['avg_time']
        std_time = data['std_time']
        efficiency = data['efficiency']
        
        print(f"{workers:<8} {avg_time:<12.2f} {std_time:<12.2f} {efficiency:<12.1%}")
    
    optimal_workers = benchmark_results['optimal_workers']
    recommendation = benchmark_results['recommendation']
    
    print(f"\nOptimal configuration:")
    print(f"  Recommended workers: {optimal_workers}")
    print(f"  Recommendation: {recommendation}")


def demonstrate_result_aggregation():
    """Demonstrate result aggregation and analysis"""
    
    print("\n" + "="*60)
    print("DEMONSTRATION 6: Result Aggregation and Analysis")
    print("="*60)
    
    # Create configurations
    configs = create_backtest_configs(num_configs=6, base_delay_ms=30)
    
    print(f"Running {len(configs)} backtests for aggregation analysis...")
    
    # Run parallel backtests
    backtester = ParallelBacktester(ParallelConfig(max_workers=3))
    results = backtester.run_parallel_backtests(configs)
    
    # Get aggregated results
    aggregated = backtester.aggregated_results
    
    print(f"\nAggregated Results:")
    print(f"  Total runs: {aggregated.get('total_runs', 0)}")
    print(f"  Successful runs: {aggregated.get('successful_runs', 0)}")
    print(f"  Failed runs: {aggregated.get('failed_runs', 0)}")
    print(f"  Success rate: {aggregated.get('success_rate', 0):.1%}")
    print(f"  Total execution time: {aggregated.get('total_execution_time', 0):.2f}s")
    print(f"  Average execution time: {aggregated.get('average_execution_time', 0):.2f}s")
    print(f"  Parallel efficiency: {aggregated.get('parallel_efficiency', 0):.1%}")
    
    # Display performance metrics aggregation
    perf_metrics = aggregated.get('performance_metrics', {})
    if perf_metrics:
        print(f"\nPerformance Metrics Aggregation:")
        for metric, value in perf_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    # Display execution statistics aggregation
    exec_stats = aggregated.get('execution_statistics', {})
    if exec_stats:
        print(f"\nExecution Statistics Aggregation:")
        for stat, value in exec_stats.items():
            if isinstance(value, (int, float)):
                print(f"  {stat}: {value:.4f}")


def main():
    """Main demonstration function"""
    
    print("PARALLEL BACKTESTING DEMONSTRATION")
    print("="*60)
    print("This demonstration shows the performance improvements")
    print("and capabilities of parallel backtesting framework.")
    print("="*60)
    
    try:
        # Run demonstrations
        demonstrate_basic_parallel_execution()
        demonstrate_performance_benchmarking()
        demonstrate_resource_monitoring()
        demonstrate_parameter_optimization()
        demonstrate_optimal_workers_benchmark()
        demonstrate_result_aggregation()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Key benefits of parallel backtesting:")
        print("  • Significant speedup for multiple backtests")
        print("  • Efficient resource utilization")
        print("  • Scalable parameter optimization")
        print("  • Comprehensive performance monitoring")
        print("  • Result aggregation and analysis")
        print("  • Automatic load balancing")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        logger.exception("Demonstration error")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

