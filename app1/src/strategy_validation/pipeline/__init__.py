"""
Backtesting Pipeline Module

Provides automated backtesting pipeline for strategy validation.
"""

from .backtesting_pipeline import BacktestingPipeline, PipelineConfig
from .strategy_loader import StrategyLoader, StrategyDefinition
from .batch_processor import BatchProcessor, BatchConfig
from .result_aggregator import ResultAggregator, AggregationConfig

__all__ = [
    'BacktestingPipeline',
    'PipelineConfig',
    'StrategyLoader',
    'StrategyDefinition',
    'BatchProcessor',
    'BatchConfig',
    'ResultAggregator',
    'AggregationConfig'
]
