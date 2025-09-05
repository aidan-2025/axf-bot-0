#!/usr/bin/env python3
"""
Sentiment Analysis Package
Advanced BERT-based sentiment analysis for financial text
"""

from .bert_sentiment_analyzer import BERTSentimentAnalyzer, BERTConfig
from .ensemble_sentiment_analyzer import EnsembleSentimentAnalyzer, EnsembleConfig
from .lexicon_sentiment_analyzer import LexiconSentimentAnalyzer, LexiconConfig
from .sentiment_processor import SentimentProcessor, SentimentProcessorConfig
from .models import SentimentResult, SentimentLabel, SentimentConfidence, SentimentSource, SentimentTrend, SentimentBatchResult
from .finbert_client import FinBERTClient, FinBERTConfig
from .sentiment_trend_analyzer import SentimentTrendAnalyzer, TrendConfig

__all__ = [
    # Core analyzers
    'BERTSentimentAnalyzer',
    'BERTConfig',
    'EnsembleSentimentAnalyzer', 
    'EnsembleConfig',
    'LexiconSentimentAnalyzer',
    'LexiconConfig',
    'SentimentProcessor',
    'SentimentProcessorConfig',
    
    # Models
    'SentimentResult',
    'SentimentLabel', 
    'SentimentConfidence',
    'SentimentSource',
    'SentimentTrend',
    'SentimentBatchResult',
    
    # Specialized clients
    'FinBERTClient',
    'FinBERTConfig',
    'SentimentTrendAnalyzer',
    'TrendConfig'
]
