#!/usr/bin/env python3
"""
News and Sentiment Data Models
Data structures for news articles, sentiment analysis, and economic events
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

class NewsSource(Enum):
    """News source enumeration"""
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    FOREX_FACTORY = "forex_factory"
    CENTRAL_BANK = "central_bank"
    TWITTER = "twitter"
    FINAGE = "finage"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    INTRINIO = "intrinio"

class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class ImpactLevel(Enum):
    """Economic event impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class Currency(Enum):
    """Currency enumeration"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    NZD = "NZD"
    CNY = "CNY"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    PLN = "PLN"
    CZK = "CZK"
    HUF = "HUF"
    RON = "RON"
    BGN = "BGN"
    HRK = "HRK"
    RUB = "RUB"
    TRY = "TRY"
    ZAR = "ZAR"
    MXN = "MXN"
    BRL = "BRL"
    ARS = "ARS"
    CLP = "CLP"
    COP = "COP"
    PEN = "PEN"
    UYU = "UYU"

@dataclass
class NewsArticle:
    """News article data model"""
    # Core fields
    article_id: str
    title: str
    content: str
    source: NewsSource
    published_at: datetime
    summary: Optional[str] = None
    author: Optional[str] = None
    language: str = "en"
    url: Optional[str] = None
    
    # Forex relevance
    currency_pairs: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    
    # Sentiment data
    sentiment_label: Optional[SentimentLabel] = None
    sentiment_score: Optional[float] = None
    confidence: Optional[float] = None
    
    # Technical fields
    raw_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Deduplication
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate content hash for deduplication"""
        if not self.content_hash:
            content = f"{self.title}|{self.content}|{self.source.value}|{self.published_at.isoformat()}"
            self.content_hash = hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'article_id': self.article_id,
            'title': self.title,
            'content': self.content,
            'summary': self.summary,
            'source': self.source.value,
            'author': self.author,
            'published_at': self.published_at.isoformat(),
            'language': self.language,
            'url': self.url,
            'currency_pairs': self.currency_pairs,
            'keywords': self.keywords,
            'relevance_score': self.relevance_score,
            'sentiment_label': self.sentiment_label.value if self.sentiment_label else None,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'raw_data': self.raw_data,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'content_hash': self.content_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create from dictionary"""
        return cls(
            article_id=data['article_id'],
            title=data['title'],
            content=data['content'],
            summary=data.get('summary'),
            source=NewsSource(data['source']),
            author=data.get('author'),
            published_at=datetime.fromisoformat(data['published_at']),
            language=data.get('language', 'en'),
            url=data.get('url'),
            currency_pairs=data.get('currency_pairs', []),
            keywords=data.get('keywords', []),
            relevance_score=data.get('relevance_score', 0.0),
            sentiment_label=SentimentLabel(data['sentiment_label']) if data.get('sentiment_label') else None,
            sentiment_score=data.get('sentiment_score'),
            confidence=data.get('confidence'),
            raw_data=data.get('raw_data', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            content_hash=data.get('content_hash')
        )

@dataclass
class SentimentData:
    """Sentiment analysis data model"""
    # Core fields
    sentiment_id: str
    text: str
    label: SentimentLabel
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    model_name: str
    model_version: str
    source: NewsSource
    language: str = "en"
    article_id: Optional[str] = None
    currency_pairs: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'sentiment_id': self.sentiment_id,
            'text': self.text,
            'language': self.language,
            'label': self.label.value,
            'score': self.score,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'source': self.source.value,
            'article_id': self.article_id,
            'currency_pairs': self.currency_pairs,
            'processed_at': self.processed_at.isoformat(),
            'processing_time_ms': self.processing_time_ms
        }

@dataclass
class EconomicEvent:
    """Economic calendar event data model"""
    # Core fields
    event_id: str
    title: str
    event_time: datetime
    impact: ImpactLevel
    currency: Currency
    source: NewsSource
    description: Optional[str] = None
    timezone: str = "UTC"
    currency_pairs: List[str] = field(default_factory=list)
    
    # Data values
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    unit: Optional[str] = None
    
    # Source information
    country: Optional[str] = None
    category: Optional[str] = None
    
    # Relevance scoring
    relevance_score: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'title': self.title,
            'description': self.description,
            'event_time': self.event_time.isoformat(),
            'timezone': self.timezone,
            'impact': self.impact.value,
            'currency': self.currency.value,
            'currency_pairs': self.currency_pairs,
            'actual': self.actual,
            'forecast': self.forecast,
            'previous': self.previous,
            'unit': self.unit,
            'source': self.source.value,
            'country': self.country,
            'category': self.category,
            'relevance_score': self.relevance_score,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

@dataclass
class NewsIngestionConfig:
    """Configuration for news ingestion"""
    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    
    # Data processing
    enable_sentiment_analysis: bool = True
    enable_deduplication: bool = True
    enable_language_detection: bool = True
    
    # Storage
    batch_size: int = 100
    flush_interval_seconds: float = 30.0
    
    # Filtering
    min_relevance_score: float = 0.1
    supported_languages: List[str] = field(default_factory=lambda: ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja'])
    forex_keywords: List[str] = field(default_factory=lambda: [
        'forex', 'fx', 'currency', 'exchange rate', 'central bank', 'interest rate',
        'inflation', 'gdp', 'unemployment', 'trade balance', 'current account',
        'EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'
    ])
    
    # Source-specific settings
    reuters_api_key: Optional[str] = None
    bloomberg_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    finage_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    intrinio_api_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'max_requests_per_minute': self.max_requests_per_minute,
            'max_requests_per_hour': self.max_requests_per_hour,
            'max_retries': self.max_retries,
            'retry_delay_seconds': self.retry_delay_seconds,
            'exponential_backoff': self.exponential_backoff,
            'enable_sentiment_analysis': self.enable_sentiment_analysis,
            'enable_deduplication': self.enable_deduplication,
            'enable_language_detection': self.enable_language_detection,
            'batch_size': self.batch_size,
            'flush_interval_seconds': self.flush_interval_seconds,
            'min_relevance_score': self.min_relevance_score,
            'supported_languages': self.supported_languages,
            'forex_keywords': self.forex_keywords,
            'reuters_api_key': self.reuters_api_key,
            'bloomberg_api_key': self.bloomberg_api_key,
            'twitter_bearer_token': self.twitter_bearer_token,
            'finage_api_key': self.finage_api_key,
            'alpha_vantage_api_key': self.alpha_vantage_api_key,
            'finnhub_api_key': self.finnhub_api_key,
            'intrinio_api_key': self.intrinio_api_key
        }

@dataclass
class NewsIngestionStats:
    """Statistics for news ingestion monitoring"""
    # Source statistics
    total_articles: int = 0
    articles_by_source: Dict[str, int] = field(default_factory=dict)
    
    # Processing statistics
    processed_articles: int = 0
    failed_articles: int = 0
    duplicate_articles: int = 0
    
    # Sentiment analysis statistics
    sentiment_analyzed: int = 0
    sentiment_failed: int = 0
    
    # Language statistics
    articles_by_language: Dict[str, int] = field(default_factory=dict)
    
    # Time statistics
    last_ingestion: Optional[datetime] = None
    average_processing_time_ms: float = 0.0
    
    # Error statistics
    errors_by_source: Dict[str, int] = field(default_factory=dict)
    rate_limit_hits: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_articles': self.total_articles,
            'articles_by_source': self.articles_by_source,
            'processed_articles': self.processed_articles,
            'failed_articles': self.failed_articles,
            'duplicate_articles': self.duplicate_articles,
            'sentiment_analyzed': self.sentiment_analyzed,
            'sentiment_failed': self.sentiment_failed,
            'articles_by_language': self.articles_by_language,
            'last_ingestion': self.last_ingestion.isoformat() if self.last_ingestion else None,
            'average_processing_time_ms': self.average_processing_time_ms,
            'errors_by_source': self.errors_by_source,
            'rate_limit_hits': self.rate_limit_hits
        }
