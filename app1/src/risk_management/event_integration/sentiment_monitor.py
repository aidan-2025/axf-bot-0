#!/usr/bin/env python3
"""
Sentiment Monitor

Monitors market sentiment and integrates it with the risk management system.
Provides real-time sentiment analysis, trend detection, and risk recommendations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import statistics

from ..models import (
    SentimentData, SentimentLevel, RiskLevel, RiskEvent, RiskAction
)

logger = logging.getLogger(__name__)


@dataclass
class SentimentMonitorConfig:
    """Configuration for sentiment monitoring"""
    # Sentiment thresholds
    very_bearish_threshold: float = -0.8
    bearish_threshold: float = -0.5
    neutral_threshold: float = -0.2
    bullish_threshold: float = 0.2
    very_bullish_threshold: float = 0.8
    
    # Risk assessment
    sentiment_weight: float = 0.3
    news_weight: float = 0.4
    social_weight: float = 0.2
    technical_weight: float = 0.1
    
    # Monitoring settings
    lookback_hours: int = 24
    min_confidence: float = 0.6
    trend_detection_periods: int = 4
    
    # Currency filtering
    monitored_currencies: List[str] = None
    currency_relevance_threshold: float = 0.7
    
    # Update intervals
    check_interval_seconds: int = 300  # 5 minutes
    cache_duration_minutes: int = 60
    
    def __post_init__(self):
        if self.monitored_currencies is None:
            self.monitored_currencies = [
                'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'
            ]


class SentimentMonitor:
    """
    Monitors market sentiment and assesses its impact on trading risk.
    
    Integrates with existing sentiment analysis modules to:
    - Track sentiment trends across multiple sources
    - Detect sentiment shifts and anomalies
    - Provide risk recommendations based on sentiment
    - Correlate sentiment with market events
    """
    
    def __init__(self, config: SentimentMonitorConfig = None):
        """Initialize sentiment monitor"""
        self.config = config or SentimentMonitorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Sentiment storage
        self.sentiment_cache: Dict[str, List[SentimentData]] = {}
        self.current_sentiment: Dict[str, SentimentData] = {}
        self.sentiment_history: List[SentimentData] = []
        
        # Risk tracking
        self.sentiment_risk_events: List[RiskEvent] = []
        self.last_check_time = datetime.utcnow()
        
        # Performance tracking
        self.sentiment_updates_processed = 0
        self.risk_events_generated = 0
        
        self.logger.info("SentimentMonitor initialized")
    
    async def start_monitoring(self, sentiment_service=None):
        """Start monitoring market sentiment"""
        self.logger.info("Starting sentiment monitoring")
        
        try:
            while True:
                await self._check_sentiment(sentiment_service)
                await asyncio.sleep(self.config.check_interval_seconds)
        except asyncio.CancelledError:
            self.logger.info("Sentiment monitoring stopped")
        except Exception as e:
            self.logger.error(f"Error in sentiment monitoring: {e}")
    
    async def _check_sentiment(self, sentiment_service=None):
        """Check for sentiment updates"""
        try:
            current_time = datetime.utcnow()
            
            # Get sentiment data from service if available
            if sentiment_service:
                sentiment_data = await self._fetch_sentiment_from_service(sentiment_service)
            else:
                sentiment_data = await self._get_mock_sentiment()
            
            # Process sentiment data
            await self._process_sentiment_data(sentiment_data)
            
            # Update current sentiment
            self._update_current_sentiment()
            
            # Generate risk events if needed
            await self._generate_sentiment_risk_events()
            
            self.last_check_time = current_time
            self.sentiment_updates_processed += len(sentiment_data)
            
        except Exception as e:
            self.logger.error(f"Error checking sentiment: {e}")
    
    async def _fetch_sentiment_from_service(self, service) -> List[SentimentData]:
        """Fetch sentiment data from sentiment service"""
        try:
            # This would integrate with the existing sentiment analysis service
            # For now, return empty list
            return []
        except Exception as e:
            self.logger.error(f"Error fetching sentiment from service: {e}")
            return []
    
    async def _get_mock_sentiment(self) -> List[SentimentData]:
        """Get mock sentiment data for testing"""
        current_time = datetime.utcnow()
        
        # Generate mock sentiment data for different currency pairs
        mock_sentiment = [
            SentimentData(
                currency_pair="EUR/USD",
                sentiment_level=SentimentLevel.BEARISH,
                sentiment_score=-0.6,
                confidence=0.8,
                timestamp=current_time,
                source="news_analysis",
                factors={
                    "news_sentiment": -0.7,
                    "social_sentiment": -0.5,
                    "technical_sentiment": -0.6
                }
            ),
            SentimentData(
                currency_pair="GBP/USD",
                sentiment_level=SentimentLevel.VERY_BEARISH,
                sentiment_score=-0.9,
                confidence=0.9,
                timestamp=current_time,
                source="news_analysis",
                factors={
                    "news_sentiment": -0.9,
                    "social_sentiment": -0.8,
                    "technical_sentiment": -0.9
                }
            ),
            SentimentData(
                currency_pair="USD/JPY",
                sentiment_level=SentimentLevel.NEUTRAL,
                sentiment_score=0.1,
                confidence=0.6,
                timestamp=current_time,
                source="news_analysis",
                factors={
                    "news_sentiment": 0.2,
                    "social_sentiment": 0.0,
                    "technical_sentiment": 0.1
                }
            ),
            SentimentData(
                currency_pair="AUD/USD",
                sentiment_level=SentimentLevel.BULLISH,
                sentiment_score=0.4,
                confidence=0.7,
                timestamp=current_time,
                source="news_analysis",
                factors={
                    "news_sentiment": 0.5,
                    "social_sentiment": 0.3,
                    "technical_sentiment": 0.4
                }
            )
        ]
        
        return mock_sentiment
    
    async def _process_sentiment_data(self, sentiment_data: List[SentimentData]):
        """Process and categorize sentiment data"""
        for sentiment in sentiment_data:
            # Check if sentiment meets minimum confidence
            if sentiment.confidence < self.config.min_confidence:
                continue
            
            # Check currency relevance
            base_currency = sentiment.currency_pair.split('/')[0]
            quote_currency = sentiment.currency_pair.split('/')[1]
            
            if (base_currency not in self.config.monitored_currencies and 
                quote_currency not in self.config.monitored_currencies):
                continue
            
            # Add to cache
            if sentiment.currency_pair not in self.sentiment_cache:
                self.sentiment_cache[sentiment.currency_pair] = []
            
            self.sentiment_cache[sentiment.currency_pair].append(sentiment)
            
            # Add to history
            self.sentiment_history.append(sentiment)
            
            # Update current sentiment
            self.current_sentiment[sentiment.currency_pair] = sentiment
            
            self.logger.debug(f"Processed sentiment: {sentiment.currency_pair} = {sentiment.sentiment_score:.2f}")
    
    def _update_current_sentiment(self):
        """Update current sentiment based on recent data"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=self.config.lookback_hours)
        
        # Update current sentiment for each currency pair
        for currency_pair, sentiment_list in self.sentiment_cache.items():
            # Get recent sentiment data
            recent_sentiment = [
                s for s in sentiment_list
                if s.timestamp > cutoff_time
            ]
            
            if recent_sentiment:
                # Calculate weighted average sentiment
                weighted_sentiment = self._calculate_weighted_sentiment(recent_sentiment)
                self.current_sentiment[currency_pair] = weighted_sentiment
    
    def _calculate_weighted_sentiment(self, sentiment_list: List[SentimentData]) -> SentimentData:
        """Calculate weighted average sentiment from multiple sources"""
        if not sentiment_list:
            return None
        
        # Calculate weighted average score
        total_weight = 0
        weighted_score = 0
        
        for sentiment in sentiment_list:
            weight = sentiment.confidence
            weighted_score += sentiment.sentiment_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return sentiment_list[-1]  # Return most recent if no weights
        
        avg_score = weighted_score / total_weight
        avg_confidence = statistics.mean([s.confidence for s in sentiment_list])
        
        # Determine sentiment level
        sentiment_level = self._score_to_sentiment_level(avg_score)
        
        # Create aggregated sentiment data
        return SentimentData(
            currency_pair=sentiment_list[0].currency_pair,
            sentiment_level=sentiment_level,
            sentiment_score=avg_score,
            confidence=avg_confidence,
            timestamp=datetime.utcnow(),
            source="aggregated",
            factors={
                "news_sentiment": statistics.mean([s.factors.get("news_sentiment", 0) for s in sentiment_list]),
                "social_sentiment": statistics.mean([s.factors.get("social_sentiment", 0) for s in sentiment_list]),
                "technical_sentiment": statistics.mean([s.factors.get("technical_sentiment", 0) for s in sentiment_list])
            }
        )
    
    def _score_to_sentiment_level(self, score: float) -> SentimentLevel:
        """Convert sentiment score to sentiment level"""
        if score <= self.config.very_bearish_threshold:
            return SentimentLevel.VERY_BEARISH
        elif score <= self.config.bearish_threshold:
            return SentimentLevel.BEARISH
        elif score <= self.config.neutral_threshold:
            return SentimentLevel.NEUTRAL
        elif score <= self.config.bullish_threshold:
            return SentimentLevel.BULLISH
        else:
            return SentimentLevel.VERY_BULLISH
    
    async def _generate_sentiment_risk_events(self):
        """Generate risk events based on sentiment analysis"""
        if not self.current_sentiment:
            return
        
        # Analyze sentiment trends
        for currency_pair, sentiment in self.current_sentiment.items():
            # Check for very bearish sentiment
            if sentiment.sentiment_level == SentimentLevel.VERY_BEARISH:
                await self._create_sentiment_risk_event(
                    "very_bearish_sentiment",
                    RiskLevel.CRITICAL,
                    f"Very bearish sentiment detected for {currency_pair}: {sentiment.sentiment_score:.2f}",
                    {
                        "currency_pair": currency_pair,
                        "sentiment_score": sentiment.sentiment_score,
                        "confidence": sentiment.confidence
                    }
                )
            elif sentiment.sentiment_level == SentimentLevel.BEARISH:
                await self._create_sentiment_risk_event(
                    "bearish_sentiment",
                    RiskLevel.HIGH,
                    f"Bearish sentiment detected for {currency_pair}: {sentiment.sentiment_score:.2f}",
                    {
                        "currency_pair": currency_pair,
                        "sentiment_score": sentiment.sentiment_score,
                        "confidence": sentiment.confidence
                    }
                )
            
            # Check for sentiment trend changes
            trend_change = self._detect_sentiment_trend_change(currency_pair)
            if trend_change:
                await self._create_sentiment_risk_event(
                    "sentiment_trend_change",
                    RiskLevel.MEDIUM,
                    f"Sentiment trend change detected for {currency_pair}: {trend_change}",
                    {
                        "currency_pair": currency_pair,
                        "trend_change": trend_change,
                        "current_sentiment": sentiment.sentiment_score
                    }
                )
    
    def _detect_sentiment_trend_change(self, currency_pair: str) -> Optional[str]:
        """Detect significant sentiment trend changes"""
        if currency_pair not in self.sentiment_cache:
            return None
        
        sentiment_list = self.sentiment_cache[currency_pair]
        if len(sentiment_list) < self.config.trend_detection_periods:
            return None
        
        # Get recent sentiment scores
        recent_scores = [s.sentiment_score for s in sentiment_list[-self.config.trend_detection_periods:]]
        
        # Check for trend
        if len(recent_scores) >= 3:
            # Calculate trend direction
            first_half = recent_scores[:len(recent_scores)//2]
            second_half = recent_scores[len(recent_scores)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            # Check for significant change
            change_threshold = 0.3
            if second_avg - first_avg > change_threshold:
                return "improving"
            elif first_avg - second_avg > change_threshold:
                return "deteriorating"
        
        return None
    
    async def _create_sentiment_risk_event(self, event_type: str, risk_level: RiskLevel,
                                        description: str, data: Dict[str, Any]):
        """Create a sentiment-based risk event"""
        risk_event = RiskEvent(
            event_id=f"sentiment_{event_type}_{datetime.utcnow().timestamp()}",
            event_type=event_type,
            risk_level=risk_level,
            description=description,
            timestamp=datetime.utcnow(),
            source="sentiment_monitor",
            data=data
        )
        
        self.sentiment_risk_events.append(risk_event)
        self.risk_events_generated += 1
        
        self.logger.info(f"Generated sentiment risk event: {description}")
    
    def get_current_sentiment(self, currency_pair: str = None) -> Dict[str, SentimentData]:
        """Get current sentiment data"""
        if currency_pair:
            return {currency_pair: self.current_sentiment.get(currency_pair)}
        return self.current_sentiment.copy()
    
    def get_sentiment_history(self, currency_pair: str = None, 
                            hours_back: int = 24) -> List[SentimentData]:
        """Get sentiment history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        if currency_pair:
            return [
                s for s in self.sentiment_cache.get(currency_pair, [])
                if s.timestamp > cutoff_time
            ]
        else:
            return [
                s for s in self.sentiment_history
                if s.timestamp > cutoff_time
            ]
    
    def get_sentiment_risk_events(self) -> List[RiskEvent]:
        """Get generated sentiment risk events"""
        return self.sentiment_risk_events.copy()
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get sentiment summary"""
        if not self.current_sentiment:
            return {
                "currency_pairs": 0,
                "average_sentiment": 0.0,
                "bearish_pairs": 0,
                "bullish_pairs": 0,
                "risk_events_generated": self.risk_events_generated,
                "updates_processed": self.sentiment_updates_processed
            }
        
        sentiment_scores = [s.sentiment_score for s in self.current_sentiment.values()]
        bearish_count = sum(1 for s in self.current_sentiment.values() 
                          if s.sentiment_level in [SentimentLevel.BEARISH, SentimentLevel.VERY_BEARISH])
        bullish_count = sum(1 for s in self.current_sentiment.values() 
                          if s.sentiment_level in [SentimentLevel.BULLISH, SentimentLevel.VERY_BULLISH])
        
        return {
            "currency_pairs": len(self.current_sentiment),
            "average_sentiment": statistics.mean(sentiment_scores) if sentiment_scores else 0.0,
            "bearish_pairs": bearish_count,
            "bullish_pairs": bullish_count,
            "risk_events_generated": self.risk_events_generated,
            "updates_processed": self.sentiment_updates_processed,
            "last_check": self.last_check_time.isoformat()
        }
    
    def get_sentiment_trend(self, currency_pair: str, periods: int = 5) -> Dict[str, Any]:
        """Get sentiment trend for a currency pair"""
        if currency_pair not in self.sentiment_cache:
            return {"trend": "unknown", "direction": "stable", "change": 0.0}
        
        sentiment_list = self.sentiment_cache[currency_pair]
        if len(sentiment_list) < periods:
            return {"trend": "insufficient_data", "direction": "stable", "change": 0.0}
        
        # Get recent sentiment scores
        recent_scores = [s.sentiment_score for s in sentiment_list[-periods:]]
        
        # Calculate trend
        if len(recent_scores) >= 2:
            first_half = recent_scores[:len(recent_scores)//2]
            second_half = recent_scores[len(recent_scores)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            change = second_avg - first_avg
            
            if change > 0.2:
                direction = "improving"
            elif change < -0.2:
                direction = "deteriorating"
            else:
                direction = "stable"
            
            return {
                "trend": "detected",
                "direction": direction,
                "change": change,
                "current_score": recent_scores[-1],
                "periods_analyzed": len(recent_scores)
            }
        
        return {"trend": "insufficient_data", "direction": "stable", "change": 0.0}
    
    def clear_old_sentiment(self, max_age_hours: int = 24):
        """Clear old sentiment data from cache"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        # Clear old sentiment from cache
        for currency_pair in list(self.sentiment_cache.keys()):
            self.sentiment_cache[currency_pair] = [
                s for s in self.sentiment_cache[currency_pair]
                if s.timestamp > cutoff_time
            ]
            
            # Remove empty entries
            if not self.sentiment_cache[currency_pair]:
                del self.sentiment_cache[currency_pair]
        
        # Clear old risk events
        self.sentiment_risk_events = [
            event for event in self.sentiment_risk_events
            if event.timestamp > cutoff_time
        ]
        
        self.logger.info("Cleared old sentiment data")

