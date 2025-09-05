#!/usr/bin/env python3
"""
Sentiment Trend Analyzer
Analyzes sentiment trends over time for market impact assessment
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque

from .models import SentimentResult, SentimentTrend, SentimentLabel

logger = logging.getLogger(__name__)

@dataclass
class TrendConfig:
    """Configuration for sentiment trend analyzer"""
    # Time windows for trend analysis
    short_window_minutes: int = 15
    medium_window_hours: int = 4
    long_window_hours: int = 24
    
    # Trend calculation parameters
    min_data_points: int = 5
    volatility_threshold: float = 0.3
    trend_strength_threshold: float = 0.2
    
    # Market impact calculation
    enable_market_correlation: bool = True
    correlation_threshold: float = 0.5

class SentimentTrendAnalyzer:
    """Analyzes sentiment trends and market impact"""
    
    def __init__(self, config: TrendConfig):
        """Initialize sentiment trend analyzer"""
        self.config = config
        
        # Store sentiment history by currency pair
        self.sentiment_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.trend_cache: Dict[str, Dict[str, Any]] = {}
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the trend analyzer"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing sentiment trend analyzer")
            self._initialized = True
            logger.info("Sentiment trend analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trend analyzer: {e}")
            raise
    
    async def analyze_trends(self, 
                           result: SentimentResult,
                           currency_pairs: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze trends for a sentiment result"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Store sentiment data
            for pair in currency_pairs:
                self.sentiment_history[pair].append({
                    'timestamp': result.analyzed_at,
                    'score': result.score,
                    'confidence': result.confidence,
                    'label': result.label,
                    'text': result.text[:100]  # Store first 100 chars for context
                })
            
            # Calculate trend metrics
            trend_data = {}
            
            for pair in currency_pairs:
                pair_trend = await self._calculate_pair_trend(pair)
                if pair_trend:
                    trend_data[pair] = pair_trend
            
            # Calculate overall market impact
            if trend_data:
                market_impact = self._calculate_market_impact(trend_data)
                risk_indicators = self._identify_risk_indicators(trend_data)
                
                return {
                    'market_impact': market_impact,
                    'risk_indicators': risk_indicators,
                    'pair_trends': trend_data
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return None
    
    async def get_trends(self, 
                        currency_pairs: List[str],
                        timeframe: str = "1h",
                        hours_back: int = 24) -> List[SentimentTrend]:
        """Get sentiment trends for currency pairs"""
        if not self._initialized:
            await self.initialize()
        
        trends = []
        
        try:
            for pair in currency_pairs:
                trend = await self._calculate_trend_for_pair(pair, timeframe, hours_back)
                if trend:
                    trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            return []
    
    async def _calculate_pair_trend(self, currency_pair: str) -> Optional[Dict[str, Any]]:
        """Calculate trend metrics for a currency pair"""
        history = self.sentiment_history[currency_pair]
        
        if len(history) < self.config.min_data_points:
            return None
        
        # Get recent data points
        recent_data = list(history)[-self.config.min_data_points:]
        scores = [point['score'] for point in recent_data]
        confidences = [point['confidence'] for point in recent_data]
        timestamps = [point['timestamp'] for point in recent_data]
        
        # Calculate basic metrics
        avg_sentiment = np.mean(scores)
        sentiment_volatility = np.std(scores)
        avg_confidence = np.mean(confidences)
        
        # Calculate trend direction
        trend_direction = self._calculate_trend_direction(scores)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(scores)
        
        # Calculate momentum
        momentum = self._calculate_momentum(scores)
        
        return {
            'currency_pair': currency_pair,
            'avg_sentiment': avg_sentiment,
            'sentiment_volatility': sentiment_volatility,
            'avg_confidence': avg_confidence,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'momentum': momentum,
            'data_points': len(recent_data),
            'time_range': {
                'start': min(timestamps),
                'end': max(timestamps)
            }
        }
    
    async def _calculate_trend_for_pair(self, 
                                      currency_pair: str,
                                      timeframe: str,
                                      hours_back: int) -> Optional[SentimentTrend]:
        """Calculate detailed trend for a currency pair"""
        history = self.sentiment_history[currency_pair]
        
        if len(history) < self.config.min_data_points:
            return None
        
        # Filter data by time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        filtered_data = [
            point for point in history 
            if point['timestamp'] >= cutoff_time
        ]
        
        if len(filtered_data) < self.config.min_data_points:
            return None
        
        # Extract data
        scores = [point['score'] for point in filtered_data]
        timestamps = [point['timestamp'] for point in filtered_data]
        
        # Calculate trend metrics
        avg_sentiment = np.mean(scores)
        sentiment_volatility = np.std(scores)
        trend_direction = self._calculate_trend_direction(scores)
        trend_strength = self._calculate_trend_strength(scores)
        
        # Determine trend direction label
        if trend_strength < self.config.trend_strength_threshold:
            direction_label = "sideways"
        elif trend_direction > 0:
            direction_label = "bullish"
        else:
            direction_label = "bearish"
        
        # Calculate confidence
        confidence = min(1.0, trend_strength * 2)  # Scale to 0-1
        
        return SentimentTrend(
            currency_pair=currency_pair,
            timeframe=timeframe,
            start_time=min(timestamps),
            end_time=max(timestamps),
            average_sentiment=avg_sentiment,
            sentiment_volatility=sentiment_volatility,
            trend_direction=direction_label,
            confidence=confidence,
            sentiment_history=scores
        )
    
    def _calculate_trend_direction(self, scores: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(scores) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize to -1 to 1 range
        return np.tanh(slope * len(scores))
    
    def _calculate_trend_strength(self, scores: List[float]) -> float:
        """Calculate trend strength (0 to 1)"""
        if len(scores) < 2:
            return 0.0
        
        # Calculate R-squared for linear trend
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        
        # Calculate R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    
    def _calculate_momentum(self, scores: List[float]) -> float:
        """Calculate momentum (-1 to 1)"""
        if len(scores) < 3:
            return 0.0
        
        # Calculate rate of change
        recent_scores = scores[-3:]  # Last 3 points
        momentum = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        return np.tanh(momentum * 2)  # Scale and normalize
    
    def _calculate_market_impact(self, trend_data: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall market impact (-1 to 1)"""
        if not trend_data:
            return 0.0
        
        # Weight by confidence and trend strength
        weighted_impacts = []
        
        for pair_data in trend_data.values():
            sentiment = pair_data['avg_sentiment']
            confidence = pair_data['avg_confidence']
            strength = pair_data['trend_strength']
            
            # Weight by confidence and strength
            weight = confidence * strength
            impact = sentiment * weight
            
            weighted_impacts.append(impact)
        
        if not weighted_impacts:
            return 0.0
        
        # Return weighted average
        return np.mean(weighted_impacts)
    
    def _identify_risk_indicators(self, trend_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify risk indicators from trend data"""
        risk_indicators = []
        
        for pair_data in trend_data.values():
            # High volatility
            if pair_data['sentiment_volatility'] > self.config.volatility_threshold:
                risk_indicators.append(f"High volatility in {pair_data['currency_pair']}")
            
            # Low confidence
            if pair_data['avg_confidence'] < 0.3:
                risk_indicators.append(f"Low confidence in {pair_data['currency_pair']}")
            
            # Strong negative trend
            if (pair_data['trend_direction'] < -0.5 and 
                pair_data['trend_strength'] > self.config.trend_strength_threshold):
                risk_indicators.append(f"Strong negative trend in {pair_data['currency_pair']}")
        
        return list(set(risk_indicators))  # Remove duplicates
    
    async def get_trend_info(self) -> Dict[str, Any]:
        """Get information about the trend analyzer"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        # Calculate cache statistics
        total_data_points = sum(len(history) for history in self.sentiment_history.values())
        active_pairs = len(self.sentiment_history)
        
        return {
            "status": "initialized",
            "short_window_minutes": self.config.short_window_minutes,
            "medium_window_hours": self.config.medium_window_hours,
            "long_window_hours": self.config.long_window_hours,
            "min_data_points": self.config.min_data_points,
            "volatility_threshold": self.config.volatility_threshold,
            "trend_strength_threshold": self.config.trend_strength_threshold,
            "active_pairs": active_pairs,
            "total_data_points": total_data_points,
            "enable_market_correlation": self.config.enable_market_correlation
        }
    
    async def close(self):
        """Close the trend analyzer"""
        self.sentiment_history.clear()
        self.trend_cache.clear()
        self._initialized = False
        logger.info("Sentiment trend analyzer closed")

