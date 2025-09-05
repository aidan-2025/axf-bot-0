#!/usr/bin/env python3
"""
Economic Calendar Models
Data models for economic events and indicators
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import json

class EventImpact(Enum):
    """Impact level of economic events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class EventStatus(Enum):
    """Status of economic events"""
    UPCOMING = "upcoming"
    LIVE = "live"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"

class EventCategory(Enum):
    """Category of economic events"""
    # Monetary Policy
    INTEREST_RATE = "interest_rate"
    MONETARY_POLICY = "monetary_policy"
    QUANTITATIVE_EASING = "quantitative_easing"
    
    # Economic Indicators
    GDP = "gdp"
    INFLATION = "inflation"
    UNEMPLOYMENT = "unemployment"
    RETAIL_SALES = "retail_sales"
    MANUFACTURING = "manufacturing"
    TRADE_BALANCE = "trade_balance"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    
    # Central Bank
    CENTRAL_BANK_MEETING = "central_bank_meeting"
    CENTRAL_BANK_SPEECH = "central_bank_speech"
    CENTRAL_BANK_REPORT = "central_bank_report"
    
    # Government
    BUDGET = "budget"
    FISCAL_POLICY = "fiscal_policy"
    ELECTION = "election"
    
    # Other
    OTHER = "other"

class Country(Enum):
    """Country codes for economic events"""
    # Major Economies
    US = "US"
    EU = "EU"
    UK = "UK"
    JP = "JP"
    CA = "CA"
    AU = "AU"
    NZ = "NZ"
    CH = "CH"
    
    # Emerging Markets
    CN = "CN"
    IN = "IN"
    BR = "BR"
    RU = "RU"
    ZA = "ZA"
    MX = "MX"
    KR = "KR"
    SG = "SG"
    
    # Other
    OTHER = "OTHER"

class Currency(Enum):
    """Currency codes"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    NZD = "NZD"
    CHF = "CHF"
    CNY = "CNY"
    INR = "INR"
    BRL = "BRL"
    RUB = "RUB"
    ZAR = "ZAR"
    MXN = "MXN"
    KRW = "KRW"
    SGD = "SGD"

@dataclass
class EconomicIndicator:
    """Economic indicator data model"""
    # Core fields
    indicator_id: str
    name: str
    country: Country
    currency: Currency
    category: EventCategory
    impact: EventImpact
    
    # Timing
    event_time: datetime
    timezone: str = "UTC"
    
    # Values
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    unit: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    
    # Market impact
    market_impact_score: Optional[float] = None
    volatility_expected: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'indicator_id': self.indicator_id,
            'name': self.name,
            'country': self.country.value,
            'currency': self.currency.value,
            'category': self.category.value,
            'impact': self.impact.value,
            'event_time': self.event_time.isoformat(),
            'timezone': self.timezone,
            'actual': self.actual,
            'forecast': self.forecast,
            'previous': self.previous,
            'unit': self.unit,
            'description': self.description,
            'source': self.source,
            'url': self.url,
            'market_impact_score': self.market_impact_score,
            'volatility_expected': self.volatility_expected,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EconomicIndicator':
        """Create from dictionary"""
        return cls(
            indicator_id=data['indicator_id'],
            name=data['name'],
            country=Country(data['country']),
            currency=Currency(data['currency']),
            category=EventCategory(data['category']),
            impact=EventImpact(data['impact']),
            event_time=datetime.fromisoformat(data['event_time']),
            timezone=data.get('timezone', 'UTC'),
            actual=data.get('actual'),
            forecast=data.get('forecast'),
            previous=data.get('previous'),
            unit=data.get('unit'),
            description=data.get('description'),
            source=data.get('source'),
            url=data.get('url'),
            market_impact_score=data.get('market_impact_score'),
            volatility_expected=data.get('volatility_expected'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )

@dataclass
class EconomicEvent:
    """Economic event data model"""
    # Core fields
    event_id: str
    title: str
    country: Country
    currency: Currency
    category: EventCategory
    impact: EventImpact
    status: EventStatus
    
    # Timing
    event_time: datetime
    timezone: str = "UTC"
    
    # Values
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    unit: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    
    # Market impact
    market_impact_score: Optional[float] = None
    volatility_expected: Optional[float] = None
    affected_pairs: List[str] = field(default_factory=list)
    
    # Analysis
    sentiment_impact: Optional[float] = None
    price_impact: Optional[float] = None
    volume_impact: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'title': self.title,
            'country': self.country.value,
            'currency': self.currency.value,
            'category': self.category.value,
            'impact': self.impact.value,
            'status': self.status.value,
            'event_time': self.event_time.isoformat(),
            'timezone': self.timezone,
            'actual': self.actual,
            'forecast': self.forecast,
            'previous': self.previous,
            'unit': self.unit,
            'description': self.description,
            'source': self.source,
            'url': self.url,
            'market_impact_score': self.market_impact_score,
            'volatility_expected': self.volatility_expected,
            'affected_pairs': self.affected_pairs,
            'sentiment_impact': self.sentiment_impact,
            'price_impact': self.price_impact,
            'volume_impact': self.volume_impact,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EconomicEvent':
        """Create from dictionary"""
        return cls(
            event_id=data['event_id'],
            title=data['title'],
            country=Country(data['country']),
            currency=Currency(data['currency']),
            category=EventCategory(data['category']),
            impact=EventImpact(data['impact']),
            status=EventStatus(data['status']),
            event_time=datetime.fromisoformat(data['event_time']),
            timezone=data.get('timezone', 'UTC'),
            actual=data.get('actual'),
            forecast=data.get('forecast'),
            previous=data.get('previous'),
            unit=data.get('unit'),
            description=data.get('description'),
            source=data.get('source'),
            url=data.get('url'),
            market_impact_score=data.get('market_impact_score'),
            volatility_expected=data.get('volatility_expected'),
            affected_pairs=data.get('affected_pairs', []),
            sentiment_impact=data.get('sentiment_impact'),
            price_impact=data.get('price_impact'),
            volume_impact=data.get('volume_impact'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )

@dataclass
class EventAnalysisResult:
    """Result of economic event analysis"""
    event_id: str
    analysis_type: str
    confidence_score: float
    impact_prediction: EventImpact
    market_impact: float
    volatility_prediction: float
    correlation_analysis: Dict[str, float]
    recommendations: List[str]
    risk_factors: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'event_id': self.event_id,
            'analysis_type': self.analysis_type,
            'confidence_score': self.confidence_score,
            'impact_prediction': self.impact_prediction.value,
            'market_impact': self.market_impact,
            'volatility_prediction': self.volatility_prediction,
            'correlation_analysis': self.correlation_analysis,
            'recommendations': self.recommendations,
            'risk_factors': self.risk_factors,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class CalendarFilter:
    """Filter for economic calendar queries"""
    countries: Optional[List[Country]] = None
    currencies: Optional[List[Currency]] = None
    categories: Optional[List[EventCategory]] = None
    impacts: Optional[List[EventImpact]] = None
    statuses: Optional[List[EventStatus]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            'countries': [c.value for c in self.countries] if self.countries else None,
            'currencies': [c.value for c in self.currencies] if self.currencies else None,
            'categories': [c.value for c in self.categories] if self.categories else None,
            'impacts': [i.value for i in self.impacts] if self.impacts else None,
            'statuses': [s.value for s in self.statuses] if self.statuses else None,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'limit': self.limit
        }

@dataclass
class CalendarStats:
    """Statistics for economic calendar data"""
    total_events: int
    events_by_country: Dict[str, int]
    events_by_category: Dict[str, int]
    events_by_impact: Dict[str, int]
    upcoming_events: int
    completed_events: int
    high_impact_events: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_events': self.total_events,
            'events_by_country': self.events_by_country,
            'events_by_category': self.events_by_category,
            'events_by_impact': self.events_by_impact,
            'upcoming_events': self.upcoming_events,
            'completed_events': self.completed_events,
            'high_impact_events': self.high_impact_events
        }

