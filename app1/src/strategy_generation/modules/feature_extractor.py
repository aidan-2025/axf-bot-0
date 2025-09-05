"""
Feature extraction module for market data analysis
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta

from ..core.strategy_template import Signal


class FeatureExtractor:
    """
    Extracts features from market data for strategy generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def extract_technical_features(self, ohlcv_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract technical features from OHLCV data
        
        Args:
            ohlcv_data: OHLCV market data
            
        Returns:
            Dict[str, Any]: Extracted technical features
        """
        try:
            features = {}
            
            # Extract price data
            prices = ohlcv_data.get('close', [])
            highs = ohlcv_data.get('high', [])
            lows = ohlcv_data.get('low', [])
            volumes = ohlcv_data.get('volume', [])
            
            if not prices or len(prices) < 2:
                return features
            
            # Price-based features
            features.update(self._extract_price_features(prices, highs, lows))
            
            # Volume-based features
            if volumes:
                features.update(self._extract_volume_features(volumes))
            
            # Time-based features
            features.update(self._extract_time_features(ohlcv_data.get('timestamp', [])))
            
            # Volatility features
            features.update(self._extract_volatility_features(prices, highs, lows))
            
            # Trend features
            features.update(self._extract_trend_features(prices))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting technical features: {e}")
            return {}
    
    def extract_sentiment_features(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from sentiment data
        
        Args:
            sentiment_data: Sentiment data
            
        Returns:
            Dict[str, Any]: Extracted sentiment features
        """
        try:
            features = {}
            
            # News sentiment features
            if 'news' in sentiment_data:
                features.update(self._extract_news_features(sentiment_data['news']))
            
            # Social sentiment features
            if 'social' in sentiment_data:
                features.update(self._extract_social_features(sentiment_data['social']))
            
            # Combined sentiment features
            features.update(self._extract_combined_sentiment_features(sentiment_data))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting sentiment features: {e}")
            return {}
    
    def extract_economic_features(self, events_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract features from economic events data
        
        Args:
            events_data: Economic events data
            
        Returns:
            Dict[str, Any]: Extracted economic features
        """
        try:
            features = {}
            
            if not events_data:
                return features
            
            # Event count features
            features['total_events'] = len(events_data)
            features['high_impact_events'] = sum(1 for e in events_data if e.get('impact') == 'high')
            features['medium_impact_events'] = sum(1 for e in events_data if e.get('impact') == 'medium')
            features['low_impact_events'] = sum(1 for e in events_data if e.get('impact') == 'low')
            
            # Impact score features
            impact_scores = [e.get('market_impact_score', 0) for e in events_data]
            if impact_scores:
                features['avg_impact_score'] = np.mean(impact_scores)
                features['max_impact_score'] = np.max(impact_scores)
                features['impact_score_std'] = np.std(impact_scores)
            
            # Volatility features
            volatility_scores = [e.get('volatility_expected', 0) for e in events_data]
            if volatility_scores:
                features['avg_volatility_expected'] = np.mean(volatility_scores)
                features['max_volatility_expected'] = np.max(volatility_scores)
            
            # Time-based features
            now = datetime.now()
            upcoming_events = [e for e in events_data if e.get('event_time') and 
                             datetime.fromisoformat(e['event_time'].replace('Z', '+00:00')) > now]
            features['upcoming_events'] = len(upcoming_events)
            
            # Currency features
            currencies = [e.get('currency', '') for e in events_data if e.get('currency')]
            if currencies:
                features['unique_currencies'] = len(set(currencies))
                features['most_common_currency'] = max(set(currencies), key=currencies.count)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting economic features: {e}")
            return {}
    
    def _extract_price_features(self, prices: List[float], highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """Extract price-based features"""
        features = {}
        
        if len(prices) < 2:
            return features
        
        # Basic price statistics
        features['price_mean'] = np.mean(prices)
        features['price_std'] = np.std(prices)
        features['price_min'] = np.min(prices)
        features['price_max'] = np.max(prices)
        features['price_range'] = features['price_max'] - features['price_min']
        
        # Price changes
        price_changes = np.diff(prices)
        features['avg_price_change'] = np.mean(price_changes)
        features['price_change_std'] = np.std(price_changes)
        features['positive_changes'] = np.sum(price_changes > 0)
        features['negative_changes'] = np.sum(price_changes < 0)
        
        # High-low features
        if highs and lows and len(highs) == len(lows) == len(prices):
            daily_ranges = [h - l for h, l in zip(highs, lows)]
            features['avg_daily_range'] = np.mean(daily_ranges)
            features['max_daily_range'] = np.max(daily_ranges)
            features['daily_range_std'] = np.std(daily_ranges)
        
        return features
    
    def _extract_volume_features(self, volumes: List[float]) -> Dict[str, Any]:
        """Extract volume-based features"""
        features = {}
        
        if not volumes:
            return features
        
        # Basic volume statistics
        features['volume_mean'] = np.mean(volumes)
        features['volume_std'] = np.std(volumes)
        features['volume_min'] = np.min(volumes)
        features['volume_max'] = np.max(volumes)
        
        # Volume changes
        volume_changes = np.diff(volumes)
        features['avg_volume_change'] = np.mean(volume_changes)
        features['volume_change_std'] = np.std(volume_changes)
        
        # Volume trend
        if len(volumes) > 5:
            recent_volumes = volumes[-5:]
            older_volumes = volumes[-10:-5] if len(volumes) >= 10 else volumes[:-5]
            features['volume_trend'] = np.mean(recent_volumes) - np.mean(older_volumes)
        
        return features
    
    def _extract_time_features(self, timestamps: List[datetime]) -> Dict[str, Any]:
        """Extract time-based features"""
        features = {}
        
        if not timestamps:
            return features
        
        # Time range
        features['time_span_hours'] = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        
        # Hour distribution
        hours = [t.hour for t in timestamps]
        features['most_active_hour'] = max(set(hours), key=hours.count)
        
        # Day of week distribution
        weekdays = [t.weekday() for t in timestamps]
        features['most_active_weekday'] = max(set(weekdays), key=weekdays.count)
        
        return features
    
    def _extract_volatility_features(self, prices: List[float], highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """Extract volatility features"""
        features = {}
        
        if len(prices) < 2:
            return features
        
        # Price volatility
        returns = np.diff(np.log(prices))
        features['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
        
        # High-low volatility
        if highs and lows and len(highs) == len(lows) == len(prices):
            hl_volatility = np.std([np.log(h/l) for h, l in zip(highs, lows)])
            features['hl_volatility'] = hl_volatility * np.sqrt(252)
        
        # Rolling volatility
        if len(returns) >= 20:
            rolling_vol = [np.std(returns[i-20:i]) for i in range(20, len(returns))]
            features['avg_rolling_volatility'] = np.mean(rolling_vol)
            features['volatility_trend'] = np.mean(rolling_vol[-5:]) - np.mean(rolling_vol[:5])
        
        return features
    
    def _extract_trend_features(self, prices: List[float]) -> Dict[str, Any]:
        """Extract trend features"""
        features = {}
        
        if len(prices) < 10:
            return features
        
        # Linear trend
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        features['trend_slope'] = slope
        features['trend_strength'] = abs(slope) / np.std(prices)
        
        # Moving average trends
        if len(prices) >= 20:
            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])
            features['ma_trend'] = ma_short - ma_long
            features['ma_trend_ratio'] = ma_short / ma_long if ma_long != 0 else 1.0
        
        # Price momentum
        if len(prices) >= 5:
            recent_momentum = (prices[-1] - prices[-5]) / prices[-5]
            features['price_momentum'] = recent_momentum
        
        return features
    
    def _extract_news_features(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from news data"""
        features = {}
        
        if not news_data:
            return features
        
        # Sentiment scores
        sentiment_scores = [item.get('sentiment_score', 0) for item in news_data]
        features['news_sentiment_mean'] = np.mean(sentiment_scores)
        features['news_sentiment_std'] = np.std(sentiment_scores)
        features['news_sentiment_max'] = np.max(sentiment_scores)
        features['news_sentiment_min'] = np.min(sentiment_scores)
        
        # Relevance scores
        relevance_scores = [item.get('relevance', 1) for item in news_data]
        features['news_relevance_mean'] = np.mean(relevance_scores)
        
        # News count
        features['news_count'] = len(news_data)
        
        return features
    
    def _extract_social_features(self, social_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from social media data"""
        features = {}
        
        if not social_data:
            return features
        
        # Sentiment scores
        sentiment_scores = [item.get('sentiment_score', 0) for item in social_data]
        features['social_sentiment_mean'] = np.mean(sentiment_scores)
        features['social_sentiment_std'] = np.std(sentiment_scores)
        
        # Engagement scores
        engagement_scores = [item.get('engagement', 1) for item in social_data]
        features['social_engagement_mean'] = np.mean(engagement_scores)
        
        # Social count
        features['social_count'] = len(social_data)
        
        return features
    
    def _extract_combined_sentiment_features(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract combined sentiment features"""
        features = {}
        
        # Overall sentiment
        all_sentiment_scores = []
        
        if 'news' in sentiment_data:
            news_scores = [item.get('sentiment_score', 0) for item in sentiment_data['news']]
            all_sentiment_scores.extend(news_scores)
        
        if 'social' in sentiment_data:
            social_scores = [item.get('sentiment_score', 0) for item in sentiment_data['social']]
            all_sentiment_scores.extend(social_scores)
        
        if all_sentiment_scores:
            features['overall_sentiment_mean'] = np.mean(all_sentiment_scores)
            features['overall_sentiment_std'] = np.std(all_sentiment_scores)
            features['sentiment_bias'] = 1 if features['overall_sentiment_mean'] > 0 else -1
        
        return features

