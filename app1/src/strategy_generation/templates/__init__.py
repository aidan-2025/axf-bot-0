"""
Strategy templates
"""

from .trend_strategy import TrendStrategy
from .range_strategy import RangeStrategy
from .breakout_strategy import BreakoutStrategy
from .sentiment_strategy import SentimentStrategy
from .news_strategy import NewsStrategy
from .multi_timeframe_strategy import MultiTimeframeStrategy
from .pairs_strategy import PairsStrategy

__all__ = [
    'TrendStrategy',
    'RangeStrategy',
    'BreakoutStrategy',
    'SentimentStrategy',
    'NewsStrategy',
    'MultiTimeframeStrategy',
    'PairsStrategy'
]

