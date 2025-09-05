"""
Technical Analysis Module for Multi-Timeframe Market Analysis

This module provides comprehensive technical analysis capabilities including:
- Multi-timeframe data aggregation (M1, M5, M15, M30, H1, H4, D1, W1)
- Technical indicators using TA-Lib
- Parallel processing for real-time analysis
- InfluxDB storage for analysis results
"""

from .models import (
    Timeframe,
    TechnicalIndicator,
    MarketAnalysis,
    AnalysisResult,
    IndicatorConfig,
    TimeframeConfig,
    INDICATOR_PRESETS,
    TIMEFRAME_CONFIGS
)

from .indicators.technical_indicators import TechnicalIndicatorCalculator
from .timeframes.timeframe_aggregator import TimeframeAggregator
from .processors.analysis_processor import AnalysisProcessor
from .storage.analysis_storage import AnalysisStorage

__all__ = [
    'Timeframe',
    'TechnicalIndicator', 
    'MarketAnalysis',
    'AnalysisResult',
    'IndicatorConfig',
    'TimeframeConfig',
    'INDICATOR_PRESETS',
    'TIMEFRAME_CONFIGS',
    'TechnicalIndicatorCalculator',
    'TimeframeAggregator',
    'AnalysisProcessor',
    'AnalysisStorage'
]
