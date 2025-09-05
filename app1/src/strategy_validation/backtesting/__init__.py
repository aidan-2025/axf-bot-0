"""
Backtesting Module

Provides Backtrader integration for strategy validation and backtesting.
"""

from .backtrader_validator import BacktraderValidator, BacktestConfig
from .data_feeds import ForexDataFeed, DataFeedConfig
from .broker_simulation import ForexBroker, BrokerConfig

__all__ = [
    'BacktraderValidator',
    'BacktestConfig',
    'ForexDataFeed',
    'DataFeedConfig',
    'ForexBroker',
    'BrokerConfig'
]

