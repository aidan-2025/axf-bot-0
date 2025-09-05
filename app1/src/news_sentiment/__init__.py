#!/usr/bin/env python3
"""
News and Sentiment Data Integration Package
Comprehensive news and sentiment data sources integration for forex trading
"""

from .models import NewsArticle, SentimentData, NewsSource, EconomicEvent, NewsIngestionConfig, NewsIngestionStats
from .clients import (
    ReutersClient,
    BloombergClient, 
    ForexFactoryClient,
    CentralBankClient,
    TwitterClient,
    FinageClient,
    AlphaVantageClient
)
from .news_ingestion_service import NewsIngestionService
from .sentiment_analyzer import SentimentAnalyzer
from .news_processor import NewsProcessor

__all__ = [
    'NewsArticle',
    'SentimentData', 
    'NewsSource',
    'EconomicEvent',
    'NewsIngestionConfig',
    'NewsIngestionStats',
    'ReutersClient',
    'BloombergClient',
    'ForexFactoryClient', 
    'CentralBankClient',
    'TwitterClient',
    'FinageClient',
    'AlphaVantageClient',
    'NewsIngestionService',
    'SentimentAnalyzer',
    'NewsProcessor'
]
