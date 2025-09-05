#!/usr/bin/env python3
"""
Benchmark Runner
Comprehensive performance benchmarking and testing framework
"""

import asyncio
import logging
import time
import statistics
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np

from ..brokers.broker_manager import PriceData, CandleData

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    LOAD = "load"
    STRESS = "stress"
    FAILOVER = "failover"
    DATA_QUALITY = "data_quality"

class BenchmarkStatus(Enum):
    """Benchmark status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    benchmark_id: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Performance metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    throughput_ops_per_second: float = 0.0
    peak_throughput: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    
    # System metrics
    avg_cpu_usage: float = 0.0
    max_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    max_memory_usage: float = 0.0
    
    # Data quality metrics
    data_completeness: float = 1.0
    validation_success_rate: float = 1.0
    quality_score: float = 1.0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metrics
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    # General settings
    benchmark_duration_seconds: float = 60.0
    warmup_duration_seconds: float = 10.0
    cooldown_duration_seconds: float = 5.0
    
    # Latency benchmark
    latency_test_operations: int = 1000
    latency_test_concurrency: int = 10
    
    # Throughput benchmark
    throughput_test_duration: float = 30.0
    throughput_test_concurrency: int = 50
    
    # Load test settings
    load_test_ramp_up_seconds: float = 30.0
    load_test_sustained_seconds: float = 60.0
    load_test_ramp_down_seconds: float = 30.0
    max_load_operations_per_second: int = 1000
    
    # Stress test settings
    stress_test_duration: float = 300.0  # 5 minutes
    stress_test_max_concurrency: int = 100
    stress_test_ramp_up_seconds: float = 60.0
    
    # Data generation
    generate_realistic_data: bool = True
    data_variation_pct: float = 5.0
    error_injection_rate: float = 0.01  # 1%
    
    # Monitoring
    collect_system_metrics: bool = True
    metrics_collection_interval: float = 1.0
    
    # Reporting
    generate_detailed_report: bool = True
    save_results_to_file: bool = True
    results_file_path: str = "benchmark_results.json"

class BenchmarkRunner:
    """Comprehensive benchmark testing framework"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark runner"""
        self.config = config or BenchmarkConfig()
        
        # Benchmark state
        self.current_benchmark: Optional[BenchmarkResult] = None
        self.benchmark_history: List[BenchmarkResult] = []
        
        # Performance tracking
        self.latency_measurements: List[float] = []
        self.throughput_measurements: List[float] = []
        self.system_metrics: List[Dict[str, float]] = []
        
        # Data generation
        self.base_price_data: List[PriceData] = []
        self.base_candle_data: List[CandleData] = []
        
        # Callbacks
        self.benchmark_callbacks: List[Callable] = []
        self.progress_callbacks: List[Callable] = []
        
        # Task management
        self.benchmark_task: Optional[asyncio.Task] = None
        
        logger.info("BenchmarkRunner initialized with config: %s", self.config)
    
    async def run_latency_benchmark(self, 
                                  operation_func: Callable,
                                  operation_args: List[Any] = None,
                                  operation_kwargs: Dict[str, Any] = None) -> BenchmarkResult:
        """Run latency benchmark test"""
        benchmark_id = f"latency_{int(time.time())}"
        
        try:
            # Initialize benchmark result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                benchmark_type=BenchmarkType.LATENCY,
                status=BenchmarkStatus.RUNNING,
                start_time=datetime.utcnow(),
                config={
                    'operations': self.config.latency_test_operations,
                    'concurrency': self.config.latency_test_concurrency
                }
            )
            
            self.current_benchmark = result
            
            # Prepare operation arguments
            operation_args = operation_args or []
            operation_kwargs = operation_kwargs or {}
            
            # Run latency test
            await self._run_latency_test(operation_func, operation_args, operation_kwargs, result)
            
            # Complete benchmark
            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Calculate final metrics
            self._calculate_benchmark_metrics(result)
            
            # Save results
            self.benchmark_history.append(result)
            if self.config.save_results_to_file:
                await self._save_benchmark_result(result)
            
            # Call callbacks
            await self._notify_benchmark_completed(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running latency benchmark: {e}")
            if self.current_benchmark:
                self.current_benchmark.status = BenchmarkStatus.FAILED
                self.current_benchmark.end_time = datetime.utcnow()
            raise
    
    async def run_throughput_benchmark(self,
                                     operation_func: Callable,
                                     operation_args: List[Any] = None,
                                     operation_kwargs: Dict[str, Any] = None) -> BenchmarkResult:
        """Run throughput benchmark test"""
        benchmark_id = f"throughput_{int(time.time())}"
        
        try:
            # Initialize benchmark result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                benchmark_type=BenchmarkType.THROUGHPUT,
                status=BenchmarkStatus.RUNNING,
                start_time=datetime.utcnow(),
                config={
                    'duration': self.config.throughput_test_duration,
                    'concurrency': self.config.throughput_test_concurrency
                }
            )
            
            self.current_benchmark = result
            
            # Prepare operation arguments
            operation_args = operation_args or []
            operation_kwargs = operation_kwargs or {}
            
            # Run throughput test
            await self._run_throughput_test(operation_func, operation_args, operation_kwargs, result)
            
            # Complete benchmark
            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Calculate final metrics
            self._calculate_benchmark_metrics(result)
            
            # Save results
            self.benchmark_history.append(result)
            if self.config.save_results_to_file:
                await self._save_benchmark_result(result)
            
            # Call callbacks
            await self._notify_benchmark_completed(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running throughput benchmark: {e}")
            if self.current_benchmark:
                self.current_benchmark.status = BenchmarkStatus.FAILED
                self.current_benchmark.end_time = datetime.utcnow()
            raise
    
    async def run_load_benchmark(self,
                               operation_func: Callable,
                               operation_args: List[Any] = None,
                               operation_kwargs: Dict[str, Any] = None) -> BenchmarkResult:
        """Run load benchmark test with gradual ramp-up"""
        benchmark_id = f"load_{int(time.time())}"
        
        try:
            # Initialize benchmark result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                benchmark_type=BenchmarkType.LOAD,
                status=BenchmarkStatus.RUNNING,
                start_time=datetime.utcnow(),
                config={
                    'ramp_up_seconds': self.config.load_test_ramp_up_seconds,
                    'sustained_seconds': self.config.load_test_sustained_seconds,
                    'ramp_down_seconds': self.config.load_test_ramp_down_seconds,
                    'max_ops_per_second': self.config.max_load_operations_per_second
                }
            )
            
            self.current_benchmark = result
            
            # Prepare operation arguments
            operation_args = operation_args or []
            operation_kwargs = operation_kwargs or {}
            
            # Run load test
            await self._run_load_test(operation_func, operation_args, operation_kwargs, result)
            
            # Complete benchmark
            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Calculate final metrics
            self._calculate_benchmark_metrics(result)
            
            # Save results
            self.benchmark_history.append(result)
            if self.config.save_results_to_file:
                await self._save_benchmark_result(result)
            
            # Call callbacks
            await self._notify_benchmark_completed(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running load benchmark: {e}")
            if self.current_benchmark:
                self.current_benchmark.status = BenchmarkStatus.FAILED
                self.current_benchmark.end_time = datetime.utcnow()
            raise
    
    async def run_stress_benchmark(self,
                                 operation_func: Callable,
                                 operation_args: List[Any] = None,
                                 operation_kwargs: Dict[str, Any] = None) -> BenchmarkResult:
        """Run stress benchmark test to find system limits"""
        benchmark_id = f"stress_{int(time.time())}"
        
        try:
            # Initialize benchmark result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                benchmark_type=BenchmarkType.STRESS,
                status=BenchmarkStatus.RUNNING,
                start_time=datetime.utcnow(),
                config={
                    'duration': self.config.stress_test_duration,
                    'max_concurrency': self.config.stress_test_max_concurrency,
                    'ramp_up_seconds': self.config.stress_test_ramp_up_seconds
                }
            )
            
            self.current_benchmark = result
            
            # Prepare operation arguments
            operation_args = operation_args or []
            operation_kwargs = operation_kwargs or {}
            
            # Run stress test
            await self._run_stress_test(operation_func, operation_args, operation_kwargs, result)
            
            # Complete benchmark
            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Calculate final metrics
            self._calculate_benchmark_metrics(result)
            
            # Save results
            self.benchmark_history.append(result)
            if self.config.save_results_to_file:
                await self._save_benchmark_result(result)
            
            # Call callbacks
            await self._notify_benchmark_completed(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running stress benchmark: {e}")
            if self.current_benchmark:
                self.current_benchmark.status = BenchmarkStatus.FAILED
                self.current_benchmark.end_time = datetime.utcnow()
            raise
    
    async def _run_latency_test(self, operation_func: Callable, operation_args: List[Any], 
                              operation_kwargs: Dict[str, Any], result: BenchmarkResult) -> None:
        """Run latency test"""
        try:
            # Warmup phase
            if self.config.warmup_duration_seconds > 0:
                await self._run_warmup(operation_func, operation_args, operation_kwargs)
            
            # Main latency test
            operations = self.config.latency_test_operations
            concurrency = self.config.latency_test_concurrency
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(concurrency)
            
            async def run_operation():
                async with semaphore:
                    start_time = time.time()
                    try:
                        if asyncio.iscoroutinefunction(operation_func):
                            await operation_func(*operation_args, **operation_kwargs)
                        else:
                            operation_func(*operation_args, **operation_kwargs)
                        
                        latency_ms = (time.time() - start_time) * 1000
                        self.latency_measurements.append(latency_ms)
                        result.successful_operations += 1
                        
                    except Exception as e:
                        result.failed_operations += 1
                        error_type = type(e).__name__
                        if error_type not in result.errors_by_type:
                            result.errors_by_type[error_type] = 0
                        result.errors_by_type[error_type] += 1
                        logger.error(f"Error in latency test operation: {e}")
            
            # Run operations concurrently
            tasks = [run_operation() for _ in range(operations)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            result.total_operations = operations
            
        except Exception as e:
            logger.error(f"Error in latency test: {e}")
            raise
    
    async def _run_throughput_test(self, operation_func: Callable, operation_args: List[Any],
                                 operation_kwargs: Dict[str, Any], result: BenchmarkResult) -> None:
        """Run throughput test"""
        try:
            # Warmup phase
            if self.config.warmup_duration_seconds > 0:
                await self._run_warmup(operation_func, operation_args, operation_kwargs)
            
            # Main throughput test
            duration = self.config.throughput_test_duration
            concurrency = self.config.throughput_test_concurrency
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(concurrency)
            
            async def run_operation():
                async with semaphore:
                    start_time = time.time()
                    try:
                        if asyncio.iscoroutinefunction(operation_func):
                            await operation_func(*operation_args, **operation_kwargs)
                        else:
                            operation_func(*operation_args, **operation_kwargs)
                        
                        result.successful_operations += 1
                        
                    except Exception as e:
                        result.failed_operations += 1
                        error_type = type(e).__name__
                        if error_type not in result.errors_by_type:
                            result.errors_by_type[error_type] = 0
                        result.errors_by_type[error_type] += 1
                        logger.error(f"Error in throughput test operation: {e}")
            
            # Run throughput test for specified duration
            start_time = time.time()
            operations_count = 0
            
            while (time.time() - start_time) < duration:
                # Create tasks for current batch
                tasks = []
                for _ in range(concurrency):
                    tasks.append(run_operation())
                    operations_count += 1
                
                # Run batch
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)
            
            result.total_operations = operations_count
            
        except Exception as e:
            logger.error(f"Error in throughput test: {e}")
            raise
    
    async def _run_load_test(self, operation_func: Callable, operation_args: List[Any],
                           operation_kwargs: Dict[str, Any], result: BenchmarkResult) -> None:
        """Run load test with gradual ramp-up"""
        try:
            # Warmup phase
            if self.config.warmup_duration_seconds > 0:
                await self._run_warmup(operation_func, operation_args, operation_kwargs)
            
            # Ramp-up phase
            await self._run_ramp_up_phase(operation_func, operation_args, operation_kwargs, result)
            
            # Sustained load phase
            await self._run_sustained_phase(operation_func, operation_args, operation_kwargs, result)
            
            # Ramp-down phase
            await self._run_ramp_down_phase(operation_func, operation_args, operation_kwargs, result)
            
        except Exception as e:
            logger.error(f"Error in load test: {e}")
            raise
    
    async def _run_stress_test(self, operation_func: Callable, operation_args: List[Any],
                             operation_kwargs: Dict[str, Any], result: BenchmarkResult) -> None:
        """Run stress test to find system limits"""
        try:
            # Warmup phase
            if self.config.warmup_duration_seconds > 0:
                await self._run_warmup(operation_func, operation_args, operation_kwargs)
            
            # Gradually increase load until system breaks
            max_concurrency = self.config.stress_test_max_concurrency
            ramp_up_duration = self.config.stress_test_ramp_up_seconds
            total_duration = self.config.stress_test_duration
            
            current_concurrency = 1
            concurrency_increment = max(1, max_concurrency // 10)  # 10 steps
            step_duration = ramp_up_duration / 10
            
            start_time = time.time()
            
            while (time.time() - start_time) < total_duration and current_concurrency <= max_concurrency:
                # Run at current concurrency level
                await self._run_at_concurrency(operation_func, operation_args, operation_kwargs, 
                                             current_concurrency, step_duration, result)
                
                # Increase concurrency
                current_concurrency = min(current_concurrency + concurrency_increment, max_concurrency)
                
                # Check if system is still responsive
                if result.error_rate > 0.5:  # 50% error rate threshold
                    logger.warning(f"System stress limit reached at concurrency {current_concurrency}")
                    break
            
        except Exception as e:
            logger.error(f"Error in stress test: {e}")
            raise
    
    async def _run_warmup(self, operation_func: Callable, operation_args: List[Any],
                        operation_kwargs: Dict[str, Any]) -> None:
        """Run warmup phase"""
        logger.info(f"Running warmup for {self.config.warmup_duration_seconds} seconds")
        
        start_time = time.time()
        while (time.time() - start_time) < self.config.warmup_duration_seconds:
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(*operation_args, **operation_kwargs)
                else:
                    operation_func(*operation_args, **operation_kwargs)
            except Exception as e:
                logger.debug(f"Warmup error (expected): {e}")
            
            await asyncio.sleep(0.1)
    
    async def _run_ramp_up_phase(self, operation_func: Callable, operation_args: List[Any],
                               operation_kwargs: Dict[str, Any], result: BenchmarkResult) -> None:
        """Run ramp-up phase of load test"""
        logger.info("Starting ramp-up phase")
        
        ramp_up_duration = self.config.load_test_ramp_up_seconds
        max_ops_per_second = self.config.max_load_operations_per_second
        
        start_time = time.time()
        while (time.time() - start_time) < ramp_up_duration:
            # Calculate current target operations per second
            elapsed = time.time() - start_time
            progress = elapsed / ramp_up_duration
            current_ops_per_second = int(max_ops_per_second * progress)
            
            if current_ops_per_second > 0:
                await self._run_at_ops_per_second(operation_func, operation_args, operation_kwargs,
                                                current_ops_per_second, 1.0, result)
            
            await asyncio.sleep(0.1)
    
    async def _run_sustained_phase(self, operation_func: Callable, operation_args: List[Any],
                                 operation_kwargs: Dict[str, Any], result: BenchmarkResult) -> None:
        """Run sustained load phase"""
        logger.info("Starting sustained load phase")
        
        sustained_duration = self.config.load_test_sustained_seconds
        max_ops_per_second = self.config.max_load_operations_per_second
        
        await self._run_at_ops_per_second(operation_func, operation_args, operation_kwargs,
                                        max_ops_per_second, sustained_duration, result)
    
    async def _run_ramp_down_phase(self, operation_func: Callable, operation_args: List[Any],
                                 operation_kwargs: Dict[str, Any], result: BenchmarkResult) -> None:
        """Run ramp-down phase of load test"""
        logger.info("Starting ramp-down phase")
        
        ramp_down_duration = self.config.load_test_ramp_down_seconds
        max_ops_per_second = self.config.max_load_operations_per_second
        
        start_time = time.time()
        while (time.time() - start_time) < ramp_down_duration:
            # Calculate current target operations per second
            elapsed = time.time() - start_time
            progress = elapsed / ramp_down_duration
            current_ops_per_second = int(max_ops_per_second * (1 - progress))
            
            if current_ops_per_second > 0:
                await self._run_at_ops_per_second(operation_func, operation_args, operation_kwargs,
                                                current_ops_per_second, 1.0, result)
            
            await asyncio.sleep(0.1)
    
    async def _run_at_concurrency(self, operation_func: Callable, operation_args: List[Any],
                                operation_kwargs: Dict[str, Any], concurrency: int,
                                duration: float, result: BenchmarkResult) -> None:
        """Run operations at specific concurrency level"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def run_operation():
            async with semaphore:
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(operation_func):
                        await operation_func(*operation_args, **operation_kwargs)
                    else:
                        operation_func(*operation_args, **operation_kwargs)
                    
                    latency_ms = (time.time() - start_time) * 1000
                    self.latency_measurements.append(latency_ms)
                    result.successful_operations += 1
                    
                except Exception as e:
                    result.failed_operations += 1
                    error_type = type(e).__name__
                    if error_type not in result.errors_by_type:
                        result.errors_by_type[error_type] = 0
                    result.errors_by_type[error_type] += 1
        
        # Run for specified duration
        start_time = time.time()
        while (time.time() - start_time) < duration:
            tasks = [run_operation() for _ in range(concurrency)]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.01)  # Small delay
        
        result.total_operations += int(concurrency * duration)
    
    async def _run_at_ops_per_second(self, operation_func: Callable, operation_args: List[Any],
                                   operation_kwargs: Dict[str, Any], ops_per_second: int,
                                   duration: float, result: BenchmarkResult) -> None:
        """Run operations at specific rate"""
        interval = 1.0 / ops_per_second if ops_per_second > 0 else 1.0
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            operation_start = time.time()
            
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(*operation_args, **operation_kwargs)
                else:
                    operation_func(*operation_args, **operation_kwargs)
                
                latency_ms = (time.time() - operation_start) * 1000
                self.latency_measurements.append(latency_ms)
                result.successful_operations += 1
                
            except Exception as e:
                result.failed_operations += 1
                error_type = type(e).__name__
                if error_type not in result.errors_by_type:
                    result.errors_by_type[error_type] = 0
                result.errors_by_type[error_type] += 1
            
            # Wait for next operation
            elapsed = time.time() - operation_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            
            result.total_operations += 1
    
    def _calculate_benchmark_metrics(self, result: BenchmarkResult) -> None:
        """Calculate final benchmark metrics"""
        try:
            # Latency metrics
            if self.latency_measurements:
                result.avg_latency_ms = statistics.mean(self.latency_measurements)
                result.p50_latency_ms = self._calculate_percentile(self.latency_measurements, 50)
                result.p95_latency_ms = self._calculate_percentile(self.latency_measurements, 95)
                result.p99_latency_ms = self._calculate_percentile(self.latency_measurements, 99)
                result.max_latency_ms = max(self.latency_measurements)
                result.min_latency_ms = min(self.latency_measurements)
            
            # Throughput metrics
            if result.duration_seconds > 0:
                result.throughput_ops_per_second = result.total_operations / result.duration_seconds
                result.peak_throughput = result.throughput_ops_per_second
            
            # Error metrics
            if result.total_operations > 0:
                result.error_rate = result.failed_operations / result.total_operations
            
            # System metrics (if collected)
            if self.system_metrics:
                cpu_usage = [m.get('cpu_usage', 0) for m in self.system_metrics]
                memory_usage = [m.get('memory_usage', 0) for m in self.system_metrics]
                
                if cpu_usage:
                    result.avg_cpu_usage = statistics.mean(cpu_usage)
                    result.max_cpu_usage = max(cpu_usage)
                
                if memory_usage:
                    result.avg_memory_usage = statistics.mean(memory_usage)
                    result.max_memory_usage = max(memory_usage)
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {e}")
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        
        return sorted_data[index]
    
    async def _save_benchmark_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to file"""
        try:
            # Convert result to dictionary
            result_dict = {
                'benchmark_id': result.benchmark_id,
                'benchmark_type': result.benchmark_type.value,
                'status': result.status.value,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'duration_seconds': result.duration_seconds,
                'total_operations': result.total_operations,
                'successful_operations': result.successful_operations,
                'failed_operations': result.failed_operations,
                'avg_latency_ms': result.avg_latency_ms,
                'p95_latency_ms': result.p95_latency_ms,
                'p99_latency_ms': result.p99_latency_ms,
                'throughput_ops_per_second': result.throughput_ops_per_second,
                'error_rate': result.error_rate,
                'config': result.config
            }
            
            # Save to file
            with open(self.config.results_file_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"Benchmark result saved to {self.config.results_file_path}")
            
        except Exception as e:
            logger.error(f"Error saving benchmark result: {e}")
    
    async def _notify_benchmark_completed(self, result: BenchmarkResult) -> None:
        """Notify callbacks of benchmark completion"""
        for callback in self.benchmark_callbacks:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Error in benchmark callback: {e}")
    
    # Callback management
    def add_benchmark_callback(self, callback: Callable) -> None:
        """Add benchmark completion callback"""
        self.benchmark_callbacks.append(callback)
    
    def add_progress_callback(self, callback: Callable) -> None:
        """Add progress update callback"""
        self.progress_callbacks.append(callback)
    
    # Status and results access
    def get_benchmark_history(self) -> List[BenchmarkResult]:
        """Get benchmark history"""
        return self.benchmark_history
    
    def get_current_benchmark(self) -> Optional[BenchmarkResult]:
        """Get current running benchmark"""
        return self.current_benchmark
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get benchmark summary statistics"""
        if not self.benchmark_history:
            return {'total_benchmarks': 0}
        
        completed_benchmarks = [b for b in self.benchmark_history if b.status == BenchmarkStatus.COMPLETED]
        
        if not completed_benchmarks:
            return {'total_benchmarks': len(self.benchmark_history), 'completed_benchmarks': 0}
        
        # Calculate summary statistics
        avg_latencies = [b.avg_latency_ms for b in completed_benchmarks if b.avg_latency_ms > 0]
        avg_throughputs = [b.throughput_ops_per_second for b in completed_benchmarks if b.throughput_ops_per_second > 0]
        error_rates = [b.error_rate for b in completed_benchmarks]
        
        return {
            'total_benchmarks': len(self.benchmark_history),
            'completed_benchmarks': len(completed_benchmarks),
            'avg_latency_ms': statistics.mean(avg_latencies) if avg_latencies else 0,
            'avg_throughput_ops_per_second': statistics.mean(avg_throughputs) if avg_throughputs else 0,
            'avg_error_rate': statistics.mean(error_rates) if error_rates else 0,
            'benchmark_types': list(set(b.benchmark_type.value for b in completed_benchmarks))
        }

# Example usage and testing
async def test_benchmark_runner():
    """Test the benchmark runner"""
    
    # Mock operation function
    async def mock_operation(data=None):
        await asyncio.sleep(0.01)  # Simulate work
        if random.random() < 0.01:  # 1% error rate
            raise Exception("Simulated error")
    
    # Create benchmark runner
    config = BenchmarkConfig(
        latency_test_operations=100,
        throughput_test_duration=10.0
    )
    
    runner = BenchmarkRunner(config)
    
    # Add callbacks
    async def benchmark_callback(result):
        print(f"Benchmark completed: {result.benchmark_type.value} - "
              f"Latency: {result.avg_latency_ms:.2f}ms, "
              f"Throughput: {result.throughput_ops_per_second:.2f} ops/s")
    
    runner.add_benchmark_callback(benchmark_callback)
    
    # Run latency benchmark
    print("Running latency benchmark...")
    latency_result = await runner.run_latency_benchmark(mock_operation)
    print(f"Latency benchmark result: {latency_result.avg_latency_ms:.2f}ms avg latency")
    
    # Run throughput benchmark
    print("Running throughput benchmark...")
    throughput_result = await runner.run_throughput_benchmark(mock_operation)
    print(f"Throughput benchmark result: {throughput_result.throughput_ops_per_second:.2f} ops/s")
    
    # Get summary
    summary = runner.get_benchmark_summary()
    print(f"Benchmark summary: {summary}")

if __name__ == "__main__":
    asyncio.run(test_benchmark_runner())

