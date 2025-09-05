"""
Sentiment-based trading strategy template
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import logging

from ..core.strategy_template import StrategyTemplate, StrategyType, Signal, StrategyParameters
from ..core.parameter_space import ParameterSpace, ParameterType, ParameterDefinition


class SentimentStrategy(StrategyTemplate):
    """
    Sentiment-based trading strategy using news and social media sentiment
    """
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self.parameters.name
        
        # Strategy-specific attributes
        self.sentiment_threshold = 0.6
        self.news_weight = 0.4
        self.social_weight = 0.3
        self.technical_weight = 0.3
        self.lookback_hours = 24
        self.min_confidence = 0.5
        
    def initialize(self) -> bool:
        """Initialize sentiment strategy with parameters"""
        try:
            params = self.parameters.parameters
            
            # Extract parameters
            self.sentiment_threshold = params.get('sentiment_threshold', 0.6)
            self.news_weight = params.get('news_weight', 0.4)
            self.social_weight = params.get('social_weight', 0.3)
            self.technical_weight = params.get('technical_weight', 0.3)
            self.lookback_hours = params.get('lookback_hours', 24)
            self.min_confidence = params.get('min_confidence', 0.5)
            
            # Validate parameters
            is_valid, errors = self.validate_parameters()
            if not is_valid:
                self.logger.error(f"Invalid parameters: {errors}")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Sentiment strategy initialized: {self}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing sentiment strategy: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate sentiment-based signals"""
        if not self.is_initialized:
            self.logger.warning("Strategy not initialized")
            return []
        
        try:
            signals = []
            
            # Extract sentiment data
            sentiment_data = market_data.get('sentiment', {})
            news_data = sentiment_data.get('news', [])
            social_data = sentiment_data.get('social', [])
            
            if not news_data and not social_data:
                return signals
            
            # Calculate composite sentiment score
            sentiment_score = self._calculate_composite_sentiment(news_data, social_data)
            
            # Generate signal based on sentiment
            if abs(sentiment_score) >= self.sentiment_threshold:
                signal = self._create_sentiment_signal(
                    sentiment_score=sentiment_score,
                    news_data=news_data,
                    social_data=social_data,
                    market_data=market_data
                )
                
                if signal:
                    signals.append(signal)
            
            self.logger.info(f"Generated {len(signals)} sentiment signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment signals: {e}")
            return []
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate sentiment strategy parameters"""
        errors = []
        
        # Validate sentiment threshold
        if not (0.1 <= self.sentiment_threshold <= 1.0):
            errors.append("Sentiment threshold must be between 0.1 and 1.0")
        
        # Validate weights
        total_weight = self.news_weight + self.social_weight + self.technical_weight
        if abs(total_weight - 1.0) > 0.01:
            errors.append("Weights must sum to 1.0")
        
        # Validate lookback hours
        if self.lookback_hours < 1 or self.lookback_hours > 168:  # Max 1 week
            errors.append("Lookback hours must be between 1 and 168")
        
        return len(errors) == 0, errors
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization"""
        param_space = ParameterSpace()
        
        # Sentiment threshold
        param_space.add_parameter(ParameterDefinition(
            name="sentiment_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.3,
            max_value=0.9,
            default_value=0.6,
            description="Minimum sentiment score threshold for signals"
        ))
        
        # News weight
        param_space.add_parameter(ParameterDefinition(
            name="news_weight",
            param_type=ParameterType.FLOAT,
            min_value=0.1,
            max_value=0.7,
            default_value=0.4,
            description="Weight for news sentiment"
        ))
        
        # Social weight
        param_space.add_parameter(ParameterDefinition(
            name="social_weight",
            param_type=ParameterType.FLOAT,
            min_value=0.1,
            max_value=0.7,
            default_value=0.3,
            description="Weight for social media sentiment"
        ))
        
        # Technical weight
        param_space.add_parameter(ParameterDefinition(
            name="technical_weight",
            param_type=ParameterType.FLOAT,
            min_value=0.1,
            max_value=0.7,
            default_value=0.3,
            description="Weight for technical analysis"
        ))
        
        # Lookback hours
        param_space.add_parameter(ParameterDefinition(
            name="lookback_hours",
            param_type=ParameterType.INTEGER,
            min_value=6,
            max_value=72,
            default_value=24,
            description="Hours to look back for sentiment data"
        ))
        
        return param_space
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """Update strategy parameters"""
        try:
            for key, value in new_parameters.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            self.parameters.parameters.update(new_parameters)
            self.parameters.updated_at = datetime.now()
            return self.initialize()
            
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")
            return False
    
    def _calculate_composite_sentiment(self, news_data: List[Dict], social_data: List[Dict]) -> float:
        """Calculate composite sentiment score"""
        news_sentiment = self._calculate_news_sentiment(news_data)
        social_sentiment = self._calculate_social_sentiment(social_data)
        
        # Combine with weights
        composite = (news_sentiment * self.news_weight + 
                    social_sentiment * self.social_weight)
        
        return composite
    
    def _calculate_news_sentiment(self, news_data: List[Dict]) -> float:
        """Calculate news sentiment score"""
        if not news_data:
            return 0.0
        
        scores = [item.get('sentiment_score', 0.0) for item in news_data]
        weights = [item.get('relevance', 1.0) for item in news_data]
        
        if not scores:
            return 0.0
        
        # Weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_social_sentiment(self, social_data: List[Dict]) -> float:
        """Calculate social media sentiment score"""
        if not social_data:
            return 0.0
        
        scores = [item.get('sentiment_score', 0.0) for item in social_data]
        weights = [item.get('engagement', 1.0) for item in social_data]
        
        if not scores:
            return 0.0
        
        # Weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _create_sentiment_signal(self, sentiment_score: float, news_data: List[Dict],
                               social_data: List[Dict], market_data: Dict[str, Any]) -> Signal:
        """Create sentiment-based signal"""
        
        # Determine signal direction
        signal_type = "buy" if sentiment_score > 0 else "sell"
        
        # Calculate signal strength
        strength = min(1.0, abs(sentiment_score) / self.sentiment_threshold)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_signal_confidence(news_data, social_data)
        
        if confidence < self.min_confidence:
            return None
        
        # Get current price
        ohlcv_data = market_data.get('ohlcv', {})
        current_price = ohlcv_data.get('close', [0])[-1] if ohlcv_data.get('close') else 0.0
        
        return Signal(
            timestamp=datetime.now(),
            symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price=current_price,
            metadata={
                "strategy_type": "sentiment",
                "sentiment_score": sentiment_score,
                "news_count": len(news_data),
                "social_count": len(social_data)
            }
        )
    
    def _calculate_signal_confidence(self, news_data: List[Dict], social_data: List[Dict]) -> float:
        """Calculate signal confidence based on data quality"""
        confidence = 0.0
        
        # News confidence
        if news_data:
            news_confidence = min(1.0, len(news_data) / 10.0)  # More news = higher confidence
            confidence += news_confidence * self.news_weight
        
        # Social confidence
        if social_data:
            social_confidence = min(1.0, len(social_data) / 50.0)  # More social data = higher confidence
            confidence += social_confidence * self.social_weight
        
        return min(1.0, confidence)
