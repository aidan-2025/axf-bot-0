"""
Broker API Clients
Provides integration with multiple forex broker APIs
"""

from .broker_manager import BrokerManager
from .oanda_client import OANDAClient, CandleData, PriceData, Granularity
from .fxcm_client import FXCMClient, FXCMCandleData, FXCMPriceData, TimeFrame

__all__ = [
    'BrokerManager',
    'OANDAClient', 'CandleData', 'PriceData', 'Granularity',
    'FXCMClient', 'FXCMCandleData', 'FXCMPriceData', 'TimeFrame'
]
