#!/usr/bin/env python3
"""
Sentiment Analysis Models
Data models for sentiment analysis results and configuration
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json

class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class SentimentConfidence(Enum):
    """Confidence levels for sentiment predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class SentimentSource(Enum):
    """Source of sentiment analysis"""
    BERT = "bert"
    FINBERT = "finbert"
    LEXICON = "lexicon"
    ENSEMBLE = "ensemble"
    LLM = "llm"

@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    # Core fields
    text: str
    label: SentimentLabel
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: SentimentSource
    
    # Metadata
    language: str = "en"
    processing_time_ms: Optional[float] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    
    # Financial context
    currency_pairs: List[str] = field(default_factory=list)
    financial_entities: List[str] = field(default_factory=list)
    market_impact: Optional[float] = None  # -1.0 to 1.0
    
    # Additional analysis
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    topic_scores: Dict[str, float] = field(default_factory=dict)
    risk_indicators: List[str] = field(default_factory=list)
    
    # Timestamps
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'text': self.text,
            'label': self.label.value,
            'score': self.score,
            'confidence': self.confidence,
            'source': self.source.value,
            'language': self.language,
            'processing_time_ms': self.processing_time_ms,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'currency_pairs': self.currency_pairs,
            'financial_entities': self.financial_entities,
            'market_impact': self.market_impact,
            'emotion_scores': self.emotion_scores,
            'topic_scores': self.topic_scores,
            'risk_indicators': self.risk_indicators,
            'analyzed_at': self.analyzed_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentResult':
        """Create from dictionary"""
        return cls(
            text=data['text'],
            label=SentimentLabel(data['label']),
            score=data['score'],
            confidence=data['confidence'],
            source=SentimentSource(data['source']),
            language=data.get('language', 'en'),
            processing_time_ms=data.get('processing_time_ms'),
            model_name=data.get('model_name'),
            model_version=data.get('model_version'),
            currency_pairs=data.get('currency_pairs', []),
            financial_entities=data.get('financial_entities', []),
            market_impact=data.get('market_impact'),
            emotion_scores=data.get('emotion_scores', {}),
            topic_scores=data.get('topic_scores', {}),
            risk_indicators=data.get('risk_indicators', []),
            analyzed_at=datetime.fromisoformat(data['analyzed_at'])
        )

@dataclass
class SentimentTrend:
    """Sentiment trend over time"""
    currency_pair: str
    timeframe: str  # e.g., "1h", "4h", "1d"
    start_time: datetime
    end_time: datetime
    
    # Trend metrics
    average_sentiment: float
    sentiment_volatility: float
    trend_direction: str  # "bullish", "bearish", "sideways"
    confidence: float
    
    # Data points
    sentiment_history: List[float] = field(default_factory=list)
    volume_weighted_sentiment: Optional[float] = None
    
    # Market context
    price_change: Optional[float] = None
    correlation: Optional[float] = None

@dataclass
class SentimentBatchResult:
    """Result of batch sentiment analysis"""
    results: List[SentimentResult]
    total_processed: int
    successful: int
    failed: int
    processing_time_ms: float
    average_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'results': [result.to_dict() for result in self.results],
            'total_processed': self.total_processed,
            'successful': self.successful,
            'failed': self.failed,
            'processing_time_ms': self.processing_time_ms,
            'average_confidence': self.average_confidence
        }

