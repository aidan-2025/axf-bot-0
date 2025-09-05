#!/usr/bin/env python3
"""
Parallel Backtesting Manager

High-level management utilities for parallel backtesting operations,
including performance monitoring, resource management, and result analysis.
"""

import multiprocessing as mp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
import time
import psutil
import threading
from queue import Queue, Empty
import json
from pathlib import Path

from .parallel_backtester import ParallelBacktester, ParallelConfig, BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ResourceMonitor:
    """Monitor system resources during parallel execution"""
    cpu_usage: List[float] = None
    memory_usage: List[float] = None
    disk_io: List[Dict[str, float]] = None
    network_io: List[Dict[str, float]] = None
    start_time: float = None
    end_time: float = None
    
    def __post_init__(self):
        if self.cpu_usage is None:
            self.cpu_usage = []
        if self.memory_usage is None:
            self.memory_usage = []
        if self.disk_io is None:
            self.disk_io = []
        if self.network_io is None:
            self.network_io = []


class PerformanceBenchmark:
    """
    Benchmark parallel backtesting performance against sequential execution.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def benchmark_parallel_vs_sequential(self, 
                                       backtest_configs: List[BacktestConfig],
                                       parallel_config: ParallelConfig = None,
                                       num_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark parallel execution against sequential execution.
        
        Args:
            backtest_configs: List of backtest configurations
            parallel_config: Parallel configuration
            num_runs: Number of benchmark runs for averaging
            
        Returns:
            Benchmark results comparing parallel vs sequential performance
        """
        self.logger.info(f"Starting benchmark: {len(backtest_configs)} backtests, {num_runs} runs")
        
        parallel_times = []
        sequential_times = []
        
        for run in range(num_runs):
            self.logger.info(f"Benchmark run {run + 1}/{num_runs}")
            
            # Benchmark parallel execution
            parallel_time = self._benchmark_parallel(backtest_configs, parallel_config)
            parallel_times.append(parallel_time)
            
            # Benchmark sequential execution
            sequential_time = self._benchmark_sequential(backtest_configs)
            sequential_times.append(sequential_time)
        
        # Calculate statistics
        parallel_avg = np.mean(parallel_times)
        parallel_std = np.std(parallel_times)
        sequential_avg = np.mean(sequential_times)
        sequential_std = np.std(sequential_times)
        
        speedup = sequential_avg / parallel_avg if parallel_avg > 0 else 1.0
        efficiency = speedup / (parallel_config.max_workers if parallel_config else mp.cpu_count())
        
        return {
            'parallel_times': parallel_times,
            'sequential_times': sequential_times,
            'parallel_avg': parallel_avg,
            'parallel_std': parallel_std,
            'sequential_avg': sequential_avg,
            'sequential_std': sequential_std,
            'speedup_factor': speedup,
            'efficiency': efficiency,
            'num_backtests': len(backtest_configs),
            'num_workers': parallel_config.max_workers if parallel_config else mp.cpu_count(),
            'num_runs': num_runs
        }
    
    def _benchmark_parallel(self, configs: List[BacktestConfig], 
                          parallel_config: ParallelConfig = None) -> float:
        """Benchmark parallel execution time"""
        parallel_config = parallel_config or ParallelConfig()
        backtester = ParallelBacktester(parallel_config)
        
        start_time = time.time()
        backtester.run_parallel_backtests(configs)
        end_time = time.time()
        
        return end_time - start_time
    
    def _benchmark_sequential(self, configs: List[BacktestConfig]) -> float:
        """Benchmark sequential execution time"""
        from ..execution.execution_integration import HighFidelityExecutionIntegration, ExecutionIntegrationConfig
        
        start_time = time.time()
        
        for config in configs:
            try:
                # Create execution integration
                exec_config = ExecutionIntegrationConfig(**config.execution_config)
                integration = HighFidelityExecutionIntegration(exec_config)
                
                # Setup Cerebro
                cerebro = integration.setup_cerebro()
                
                # Create data feed
                from ..tick_data.enhanced_tick_feed import EnhancedTickDataFeedFactory
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
                    try:
                        from ..strategies import get_strategy_class
                        strategy_class = get_strategy_class(config.strategy_class)
                    except ImportError:
                        continue
                
                # Run backtest
                integration.run_backtest(strategy_class, **config.strategy_params)
                
            except Exception as e:
                self.logger.warning(f"Sequential backtest failed: {e}")
                continue
        
        end_time = time.time()
        return end_time - start_time


class ResourceManager:
    """
    Manage system resources during parallel backtesting.
    """
    
    def __init__(self, max_memory_mb: int = 8192, max_cpu_percent: float = 90.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.logger = logging.getLogger(__name__)
        self.monitor_thread = None
        self.stop_monitoring_flag = False
        self.resource_data = ResourceMonitor()
    
    def start_monitoring(self):
        """Start resource monitoring in a separate thread"""
        self.stop_monitoring_flag = False
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.stop_monitoring_flag = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources"""
        self.resource_data.start_time = time.time()
        
        while not self.stop_monitoring_flag:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.resource_data.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.resource_data.memory_usage.append(memory_mb)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.resource_data.disk_io.append({
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes,
                        'read_count': disk_io.read_count,
                        'write_count': disk_io.write_count
                    })
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    self.resource_data.network_io.append({
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv,
                        'packets_sent': net_io.packets_sent,
                        'packets_recv': net_io.packets_recv
                    })
                
                # Check resource limits
                if memory_mb > self.max_memory_mb:
                    self.logger.warning(f"Memory usage exceeded limit: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                
                if cpu_percent > self.max_cpu_percent:
                    self.logger.warning(f"CPU usage exceeded limit: {cpu_percent:.1f}% > {self.max_cpu_percent}%")
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                break
        
        self.resource_data.end_time = time.time()
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.resource_data.cpu_usage:
            return {}
        
        return {
            'duration': self.resource_data.end_time - self.resource_data.start_time if self.resource_data.end_time else 0,
            'avg_cpu_usage': np.mean(self.resource_data.cpu_usage),
            'max_cpu_usage': np.max(self.resource_data.cpu_usage),
            'avg_memory_usage_mb': np.mean(self.resource_data.memory_usage),
            'max_memory_usage_mb': np.max(self.resource_data.memory_usage),
            'peak_memory_usage_mb': psutil.virtual_memory().total / (1024 * 1024),
            'total_disk_read_mb': self._calculate_total_disk_io('read_bytes') / (1024 * 1024),
            'total_disk_write_mb': self._calculate_total_disk_io('write_bytes') / (1024 * 1024),
            'total_network_sent_mb': self._calculate_total_network_io('bytes_sent') / (1024 * 1024),
            'total_network_recv_mb': self._calculate_total_network_io('bytes_recv') / (1024 * 1024)
        }
    
    def _calculate_total_disk_io(self, metric: str) -> float:
        """Calculate total disk I/O for a metric"""
        if not self.resource_data.disk_io:
            return 0.0
        
        values = [entry.get(metric, 0) for entry in self.resource_data.disk_io]
        if len(values) < 2:
            return 0.0
        
        return values[-1] - values[0]
    
    def _calculate_total_network_io(self, metric: str) -> float:
        """Calculate total network I/O for a metric"""
        if not self.resource_data.network_io:
            return 0.0
        
        values = [entry.get(metric, 0) for entry in self.resource_data.network_io]
        if len(values) < 2:
            return 0.0
        
        return values[-1] - values[0]


class ParallelBacktestManager:
    """
    High-level manager for parallel backtesting operations with advanced features.
    """
    
    def __init__(self, parallel_config: ParallelConfig = None):
        self.parallel_config = parallel_config or ParallelConfig()
        self.backtester = ParallelBacktester(self.parallel_config)
        self.benchmark = PerformanceBenchmark()
        self.resource_manager = ResourceManager()
        self.logger = logging.getLogger(__name__)
    
    def run_with_monitoring(self, 
                          backtest_configs: List[BacktestConfig],
                          progress_callback: Callable[[int, int], None] = None,
                          enable_resource_monitoring: bool = True) -> Tuple[List[BacktestResult], Dict[str, Any]]:
        """
        Run parallel backtests with comprehensive monitoring.
        
        Args:
            backtest_configs: List of backtest configurations
            progress_callback: Optional progress callback
            enable_resource_monitoring: Whether to monitor system resources
            
        Returns:
            Tuple of (results, monitoring_data)
        """
        self.logger.info(f"Starting monitored parallel execution of {len(backtest_configs)} backtests")
        
        # Start resource monitoring
        if enable_resource_monitoring:
            self.resource_manager.start_monitoring()
        
        try:
            # Run parallel backtests
            results = self.backtester.run_parallel_backtests(backtest_configs, progress_callback)
            
            # Get resource summary
            resource_summary = self.resource_manager.get_resource_summary() if enable_resource_monitoring else {}
            
            # Combine results with monitoring data
            monitoring_data = {
                'resource_summary': resource_summary,
                'performance_summary': self.backtester.get_performance_summary(),
                'system_info': self._get_system_info()
            }
            
            return results, monitoring_data
            
        finally:
            # Stop resource monitoring
            if enable_resource_monitoring:
                self.resource_manager.stop_monitoring()
    
    def run_parameter_optimization_with_benchmark(self, 
                                                base_config: BacktestConfig,
                                                parameter_ranges: Dict[str, List[Any]],
                                                progress_callback: Callable[[int, int], None] = None) -> Dict[str, Any]:
        """
        Run parameter optimization with performance benchmarking.
        
        Args:
            base_config: Base backtest configuration
            parameter_ranges: Parameter ranges for optimization
            progress_callback: Optional progress callback
            
        Returns:
            Optimization results with performance metrics
        """
        # Generate parameter combinations
        configs = self._generate_parameter_combinations(base_config, parameter_ranges)
        
        # Limit number of combinations for benchmarking if too many
        max_combinations = 100
        if len(configs) > max_combinations:
            self.logger.warning(f"Too many parameter combinations ({len(configs)}), limiting to {max_combinations}")
            configs = configs[:max_combinations]
        
        # Run with monitoring
        results, monitoring_data = self.run_with_monitoring(configs, progress_callback)
        
        # Find best parameters
        best_result = self._find_best_result(results)
        
        # Calculate optimization metrics
        optimization_metrics = self._calculate_optimization_metrics(results, parameter_ranges)
        
        return {
            'results': results,
            'best_result': best_result,
            'optimization_metrics': optimization_metrics,
            'monitoring_data': monitoring_data,
            'parameter_ranges': parameter_ranges,
            'total_combinations_tested': len(configs)
        }
    
    def benchmark_optimal_workers(self, 
                                backtest_configs: List[BacktestConfig],
                                max_workers_range: range = range(1, 9),
                                num_runs: int = 2) -> Dict[str, Any]:
        """
        Benchmark to find optimal number of workers.
        
        Args:
            backtest_configs: List of backtest configurations
            max_workers_range: Range of worker counts to test
            num_runs: Number of runs per worker count
            
        Returns:
            Benchmark results for different worker counts
        """
        self.logger.info(f"Benchmarking optimal workers: {list(max_workers_range)}")
        
        benchmark_results = {}
        
        for num_workers in max_workers_range:
            self.logger.info(f"Testing {num_workers} workers")
            
            config = ParallelConfig(max_workers=num_workers)
            times = []
            
            for run in range(num_runs):
                start_time = time.time()
                backtester = ParallelBacktester(config)
                backtester.run_parallel_backtests(backtest_configs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            benchmark_results[num_workers] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'times': times,
                'efficiency': self._calculate_efficiency_for_workers(avg_time, num_workers, backtest_configs)
            }
        
        # Find optimal number of workers
        optimal_workers = min(benchmark_results.keys(), 
                            key=lambda w: benchmark_results[w]['avg_time'])
        
        return {
            'benchmark_results': benchmark_results,
            'optimal_workers': optimal_workers,
            'recommendation': self._generate_worker_recommendation(benchmark_results, optimal_workers)
        }
    
    def _generate_parameter_combinations(self, 
                                       base_config: BacktestConfig, 
                                       parameter_ranges: Dict[str, List[Any]]) -> List[BacktestConfig]:
        """Generate all combinations of parameters"""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        configs = []
        
        for i, combination in enumerate(itertools.product(*param_values)):
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
            
            for param_name, param_value in zip(param_names, combination):
                config.strategy_params[param_name] = param_value
            
            configs.append(config)
        
        return configs
    
    def _find_best_result(self, results: List[BacktestResult]) -> Optional[BacktestResult]:
        """Find the best result based on performance metrics"""
        successful_results = [r for r in results if r.success and r.performance_metrics]
        
        if not successful_results:
            return None
        
        best_result = max(successful_results, 
                         key=lambda r: r.performance_metrics.get('total_return', 0))
        
        return best_result
    
    def _calculate_optimization_metrics(self, results: List[BacktestResult], 
                                      parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Calculate optimization-specific metrics"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {}
        
        # Calculate parameter sensitivity
        param_sensitivity = {}
        for param_name in parameter_ranges.keys():
            param_values = []
            returns = []
            
            for result in successful_results:
                if result.performance_metrics and 'total_return' in result.performance_metrics:
                    # Extract parameter value from run_id or results
                    param_value = self._extract_parameter_value(result, param_name)
                    if param_value is not None:
                        param_values.append(param_value)
                        returns.append(result.performance_metrics['total_return'])
            
            if param_values and returns:
                correlation = np.corrcoef(param_values, returns)[0, 1] if len(param_values) > 1 else 0
                param_sensitivity[param_name] = {
                    'correlation': correlation,
                    'value_range': [min(param_values), max(param_values)],
                    'return_range': [min(returns), max(returns)]
                }
        
        return {
            'total_combinations': len(results),
            'successful_combinations': len(successful_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'parameter_sensitivity': param_sensitivity,
            'best_return': max(r.performance_metrics.get('total_return', 0) for r in successful_results) if successful_results else 0,
            'worst_return': min(r.performance_metrics.get('total_return', 0) for r in successful_results) if successful_results else 0
        }
    
    def _extract_parameter_value(self, result: BacktestResult, param_name: str) -> Optional[Any]:
        """Extract parameter value from result (simplified implementation)"""
        # This is a simplified implementation
        # In practice, you'd need to store parameter values in the result
        return None
    
    def _calculate_efficiency_for_workers(self, avg_time: float, num_workers: int, 
                                        configs: List[BacktestConfig]) -> float:
        """Calculate efficiency for a specific number of workers"""
        # Simplified efficiency calculation
        # In practice, you'd compare against theoretical maximum
        return min(1.0, num_workers / len(configs)) if configs else 0.0
    
    def _generate_worker_recommendation(self, benchmark_results: Dict[int, Dict], 
                                      optimal_workers: int) -> str:
        """Generate recommendation for optimal number of workers"""
        optimal_time = benchmark_results[optimal_workers]['avg_time']
        single_worker_time = benchmark_results.get(1, {}).get('avg_time', optimal_time)
        
        speedup = single_worker_time / optimal_time if optimal_time > 0 else 1.0
        
        return f"Optimal workers: {optimal_workers}, Speedup: {speedup:.2f}x"
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'cpu_count': mp.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
        }
