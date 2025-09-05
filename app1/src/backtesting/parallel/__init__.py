#!/usr/bin/env python3
"""
Parallel Backtesting Module

Provides parallel processing capabilities for backtesting operations,
including multi-process execution, performance benchmarking, and resource management.
"""

from .parallel_backtester import (
    ParallelBacktester,
    ParallelConfig,
    BacktestConfig,
    BacktestResult,
    ParallelBacktestManager
)

from .parallel_manager import (
    PerformanceBenchmark,
    ResourceManager,
    ParallelBacktestManager as AdvancedParallelBacktestManager
)

__all__ = [
    # Core parallel backtesting
    'ParallelBacktester',
    'ParallelConfig',
    'BacktestConfig',
    'BacktestResult',
    'ParallelBacktestManager',
    
    # Advanced management and benchmarking
    'PerformanceBenchmark',
    'ResourceManager',
    'AdvancedParallelBacktestManager'
]

__version__ = "1.0.0"
__author__ = "AXF Bot Development Team"
__description__ = "Parallel backtesting framework for high-performance quantitative analysis"

