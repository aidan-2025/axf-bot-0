"""
Data Ingestion Module
Handles real-time market data ingestion from multiple sources
"""

from .brokers.broker_manager import BrokerManager
from .brokers.oanda_client import OANDAClient, CandleData, PriceData, Granularity
from .brokers.fxcm_client import FXCMClient, FXCMCandleData, FXCMPriceData, TimeFrame

__all__ = [
    'BrokerManager',
    'OANDAClient', 'CandleData', 'PriceData', 'Granularity',
    'FXCMClient', 'FXCMCandleData', 'FXCMPriceData', 'TimeFrame'
]
