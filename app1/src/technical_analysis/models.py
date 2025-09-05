"""
Data models for technical analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import pandas as pd


class Timeframe(Enum):
    """Supported timeframes for analysis"""
    M1 = "1m"      # 1 minute
    M5 = "5m"      # 5 minutes
    M15 = "15m"    # 15 minutes
    M30 = "30m"    # 30 minutes
    H1 = "1h"      # 1 hour
    H4 = "4h"      # 4 hours
    D1 = "1d"      # 1 day
    W1 = "1w"      # 1 week


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    timeframe: Timeframe
    parameters: Dict[str, Any]
    enabled: bool = True


@dataclass
class TimeframeConfig:
    """Configuration for timeframe analysis"""
    timeframe: Timeframe
    indicators: List[IndicatorConfig]
    enabled: bool = True


@dataclass
class TechnicalIndicator:
    """Technical indicator result"""
    name: str
    timeframe: Timeframe
    symbol: str
    values: List[float]
    timestamps: List[datetime]
    parameters: Dict[str, Any]
    calculated_at: datetime


@dataclass
class MarketAnalysis:
    """Complete market analysis for a symbol"""
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    indicators: Dict[str, TechnicalIndicator]
    price_data: pd.DataFrame
    analysis_metadata: Dict[str, Any]


@dataclass
class AnalysisResult:
    """Analysis result with metadata"""
    symbol: str
    timeframes: List[Timeframe]
    analyses: Dict[Timeframe, MarketAnalysis]
    calculated_at: datetime
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class OHLCVData:
    """OHLCV data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: Timeframe


@dataclass
class AggregatedData:
    """Aggregated data for a specific timeframe"""
    symbol: str
    timeframe: Timeframe
    data: pd.DataFrame
    start_time: datetime
    end_time: datetime
    record_count: int


# Indicator parameter presets
INDICATOR_PRESETS = {
    'sma': {
        'periods': [10, 20, 50, 100, 200],
        'default_period': 20
    },
    'ema': {
        'periods': [10, 20, 50, 100, 200],
        'default_period': 20
    },
    'rsi': {
        'periods': [14, 21, 28],
        'default_period': 14
    },
    'macd': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    'bollinger_bands': {
        'period': 20,
        'std_dev': 2
    },
    'stochastic': {
        'k_period': 14,
        'd_period': 3
    },
    'atr': {
        'periods': [14, 21, 28],
        'default_period': 14
    },
    'adx': {
        'periods': [14, 21, 28],
        'default_period': 14
    }
}

# Timeframe configurations
TIMEFRAME_CONFIGS = {
    Timeframe.M1: {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
        'min_data_points': 200
    },
    Timeframe.M5: {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic'],
        'min_data_points': 100
    },
    Timeframe.M15: {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic', 'atr'],
        'min_data_points': 50
    },
    Timeframe.M30: {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic', 'atr', 'adx'],
        'min_data_points': 30
    },
    Timeframe.H1: {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic', 'atr', 'adx'],
        'min_data_points': 20
    },
    Timeframe.H4: {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic', 'atr', 'adx'],
        'min_data_points': 10
    },
    Timeframe.D1: {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic', 'atr', 'adx'],
        'min_data_points': 5
    },
    Timeframe.W1: {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
        'min_data_points': 2
    }
}

