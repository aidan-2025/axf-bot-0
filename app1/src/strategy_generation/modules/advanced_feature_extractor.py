"""
Advanced feature extraction module with real-time capabilities
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

from ..core.strategy_template import Signal


class FeatureType(Enum):
    """Feature types for categorization"""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    ECONOMIC = "economic"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    TIME = "time"
    CROSS_ASSET = "cross_asset"


@dataclass
class FeatureDefinition:
    """Definition of a feature"""
    name: str
    feature_type: FeatureType
    description: str
    calculation_method: str
    parameters: Dict[str, Any] = None
    dependencies: List[str] = None
    update_frequency: str = "real_time"  # real_time, minute, hour, daily
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []


class AdvancedFeatureExtractor:
    """
    Advanced feature extraction engine with real-time capabilities,
    multi-timeframe analysis, and sophisticated technical indicators
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Feature extraction configuration
        self.enable_real_time = self.config.get('enable_real_time', True)
        self.max_features = self.config.get('max_features', 1000)
        self.feature_cache_size = self.config.get('feature_cache_size', 10000)
        
        # Feature definitions registry
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_cache: Dict[str, Any] = {}
        self.feature_history: Dict[str, List[Any]] = {}
        
        # Processing statistics
        self.extraction_stats = {
            "features_extracted": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "processing_time_ms": 0,
            "errors": 0
        }
        
        # Initialize default features
        self._initialize_default_features()
    
    def _initialize_default_features(self):
        """Initialize default feature definitions"""
        # Technical indicators
        self._register_feature(FeatureDefinition(
            name="sma_20",
            feature_type=FeatureType.TECHNICAL,
            description="20-period Simple Moving Average",
            calculation_method="sma",
            parameters={"period": 20}
        ))
        
        self._register_feature(FeatureDefinition(
            name="ema_12",
            feature_type=FeatureType.TECHNICAL,
            description="12-period Exponential Moving Average",
            calculation_method="ema",
            parameters={"period": 12}
        ))
        
        self._register_feature(FeatureDefinition(
            name="rsi_14",
            feature_type=FeatureType.TECHNICAL,
            description="14-period Relative Strength Index",
            calculation_method="rsi",
            parameters={"period": 14}
        ))
        
        self._register_feature(FeatureDefinition(
            name="macd",
            feature_type=FeatureType.TECHNICAL,
            description="MACD (12,26,9)",
            calculation_method="macd",
            parameters={"fast": 12, "slow": 26, "signal": 9}
        ))
        
        self._register_feature(FeatureDefinition(
            name="bollinger_bands",
            feature_type=FeatureType.TECHNICAL,
            description="Bollinger Bands (20,2)",
            calculation_method="bollinger_bands",
            parameters={"period": 20, "std": 2}
        ))
        
        # Volatility features
        self._register_feature(FeatureDefinition(
            name="atr_14",
            feature_type=FeatureType.VOLATILITY,
            description="14-period Average True Range",
            calculation_method="atr",
            parameters={"period": 14}
        ))
        
        self._register_feature(FeatureDefinition(
            name="volatility_20",
            feature_type=FeatureType.VOLATILITY,
            description="20-period rolling volatility",
            calculation_method="rolling_volatility",
            parameters={"period": 20}
        ))
        
        # Momentum features
        self._register_feature(FeatureDefinition(
            name="momentum_10",
            feature_type=FeatureType.MOMENTUM,
            description="10-period price momentum",
            calculation_method="momentum",
            parameters={"period": 10}
        ))
        
        self._register_feature(FeatureDefinition(
            name="rate_of_change",
            feature_type=FeatureType.MOMENTUM,
            description="Rate of change",
            calculation_method="rate_of_change",
            parameters={"period": 10}
        ))
        
        # Volume features
        self._register_feature(FeatureDefinition(
            name="volume_sma",
            feature_type=FeatureType.VOLUME,
            description="Volume moving average",
            calculation_method="volume_sma",
            parameters={"period": 20}
        ))
        
        self._register_feature(FeatureDefinition(
            name="volume_ratio",
            feature_type=FeatureType.VOLUME,
            description="Current volume to average volume ratio",
            calculation_method="volume_ratio",
            parameters={"period": 20}
        ))
        
        # Time features
        self._register_feature(FeatureDefinition(
            name="hour_of_day",
            feature_type=FeatureType.TIME,
            description="Hour of day (0-23)",
            calculation_method="hour_of_day",
            parameters={}
        ))
        
        self._register_feature(FeatureDefinition(
            name="day_of_week",
            feature_type=FeatureType.TIME,
            description="Day of week (0-6)",
            calculation_method="day_of_week",
            parameters={}
        ))
        
        # Sentiment features
        self._register_feature(FeatureDefinition(
            name="news_sentiment",
            feature_type=FeatureType.SENTIMENT,
            description="News sentiment score",
            calculation_method="news_sentiment",
            parameters={}
        ))
        
        self._register_feature(FeatureDefinition(
            name="social_sentiment",
            feature_type=FeatureType.SENTIMENT,
            description="Social media sentiment score",
            calculation_method="social_sentiment",
            parameters={}
        ))
        
        # Economic features
        self._register_feature(FeatureDefinition(
            name="event_impact",
            feature_type=FeatureType.ECONOMIC,
            description="Economic event impact score",
            calculation_method="event_impact",
            parameters={}
        ))
        
        self._register_feature(FeatureDefinition(
            name="event_proximity",
            feature_type=FeatureType.ECONOMIC,
            description="Time to next economic event",
            calculation_method="event_proximity",
            parameters={}
        ))
    
    def _register_feature(self, feature_def: FeatureDefinition):
        """Register a feature definition"""
        self.feature_definitions[feature_def.name] = feature_def
    
    async def extract_features_async(self, market_data: Dict[str, Any], 
                                   feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Extract features asynchronously from market data
        
        Args:
            market_data: Processed market data
            feature_names: Specific features to extract (None for all)
            
        Returns:
            Dict[str, Any]: Extracted features
        """
        start_time = datetime.now()
        
        try:
            if feature_names is None:
                feature_names = list(self.feature_definitions.keys())
            
            features = {}
            
            # Extract features in parallel where possible
            feature_tasks = []
            for feature_name in feature_names:
                if feature_name in self.feature_definitions:
                    task = self._extract_single_feature_async(feature_name, market_data)
                    feature_tasks.append((feature_name, task))
            
            # Execute feature extraction tasks
            for feature_name, task in feature_tasks:
                try:
                    feature_value = await task
                    if feature_value is not None:
                        features[feature_name] = feature_value
                except Exception as e:
                    self.logger.warning(f"Error extracting feature {feature_name}: {e}")
                    self.extraction_stats["errors"] += 1
            
            # Add metadata
            features["_metadata"] = {
                "extraction_timestamp": datetime.now().isoformat(),
                "features_extracted": len(features) - 1,  # Exclude metadata
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
            self.extraction_stats["features_extracted"] += len(features) - 1
            self.extraction_stats["processing_time_ms"] += features["_metadata"]["processing_time_ms"]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {e}")
            self.extraction_stats["errors"] += 1
            return {}
    
    def extract_features(self, market_data: Dict[str, Any], 
                        feature_names: List[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for feature extraction"""
        return asyncio.run(self.extract_features_async(market_data, feature_names))
    
    async def _extract_single_feature_async(self, feature_name: str, 
                                          market_data: Dict[str, Any]) -> Any:
        """Extract a single feature asynchronously"""
        try:
            feature_def = self.feature_definitions[feature_name]
            
            # Check cache first
            cache_key = f"{feature_name}_{hash(str(market_data))}"
            if cache_key in self.feature_cache:
                self.extraction_stats["cache_hits"] += 1
                return self.feature_cache[cache_key]
            
            self.extraction_stats["cache_misses"] += 1
            
            # Extract feature based on method
            method = feature_def.calculation_method
            parameters = feature_def.parameters
            
            if method == "sma":
                result = await self._calculate_sma_async(market_data, parameters)
            elif method == "ema":
                result = await self._calculate_ema_async(market_data, parameters)
            elif method == "rsi":
                result = await self._calculate_rsi_async(market_data, parameters)
            elif method == "macd":
                result = await self._calculate_macd_async(market_data, parameters)
            elif method == "bollinger_bands":
                result = await self._calculate_bollinger_bands_async(market_data, parameters)
            elif method == "atr":
                result = await self._calculate_atr_async(market_data, parameters)
            elif method == "rolling_volatility":
                result = await self._calculate_rolling_volatility_async(market_data, parameters)
            elif method == "momentum":
                result = await self._calculate_momentum_async(market_data, parameters)
            elif method == "rate_of_change":
                result = await self._calculate_rate_of_change_async(market_data, parameters)
            elif method == "volume_sma":
                result = await self._calculate_volume_sma_async(market_data, parameters)
            elif method == "volume_ratio":
                result = await self._calculate_volume_ratio_async(market_data, parameters)
            elif method == "hour_of_day":
                result = await self._calculate_hour_of_day_async(market_data, parameters)
            elif method == "day_of_week":
                result = await self._calculate_day_of_week_async(market_data, parameters)
            elif method == "news_sentiment":
                result = await self._calculate_news_sentiment_async(market_data, parameters)
            elif method == "social_sentiment":
                result = await self._calculate_social_sentiment_async(market_data, parameters)
            elif method == "event_impact":
                result = await self._calculate_event_impact_async(market_data, parameters)
            elif method == "event_proximity":
                result = await self._calculate_event_proximity_async(market_data, parameters)
            else:
                self.logger.warning(f"Unknown feature calculation method: {method}")
                return None
            
            # Cache result
            if len(self.feature_cache) < self.feature_cache_size:
                self.feature_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting feature {feature_name}: {e}")
            return None
    
    async def _calculate_sma_async(self, market_data: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> List[float]:
        """Calculate Simple Moving Average"""
        period = parameters.get("period", 20)
        prices = market_data.get("ohlcv", {}).get("close", [])
        
        if len(prices) < period:
            return []
        
        return pd.Series(prices).rolling(window=period).mean().tolist()
    
    async def _calculate_ema_async(self, market_data: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> List[float]:
        """Calculate Exponential Moving Average"""
        period = parameters.get("period", 12)
        prices = market_data.get("ohlcv", {}).get("close", [])
        
        if len(prices) < period:
            return []
        
        return pd.Series(prices).ewm(span=period).mean().tolist()
    
    async def _calculate_rsi_async(self, market_data: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> List[float]:
        """Calculate Relative Strength Index"""
        period = parameters.get("period", 14)
        prices = market_data.get("ohlcv", {}).get("close", [])
        
        if len(prices) < period + 1:
            return []
        
        prices_series = pd.Series(prices)
        delta = prices_series.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).tolist()
    
    async def _calculate_macd_async(self, market_data: Dict[str, Any], 
                                  parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate MACD"""
        fast = parameters.get("fast", 12)
        slow = parameters.get("slow", 26)
        signal = parameters.get("signal", 9)
        
        prices = market_data.get("ohlcv", {}).get("close", [])
        
        if len(prices) < slow:
            return {"macd": [], "signal": [], "histogram": []}
        
        prices_series = pd.Series(prices)
        
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line.tolist(),
            "signal": signal_line.tolist(),
            "histogram": histogram.tolist()
        }
    
    async def _calculate_bollinger_bands_async(self, market_data: Dict[str, Any], 
                                             parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands"""
        period = parameters.get("period", 20)
        std_mult = parameters.get("std", 2)
        
        prices = market_data.get("ohlcv", {}).get("close", [])
        
        if len(prices) < period:
            return {"upper": [], "middle": [], "lower": []}
        
        prices_series = pd.Series(prices)
        sma = prices_series.rolling(window=period).mean()
        std = prices_series.rolling(window=period).std()
        
        upper_band = sma + (std * std_mult)
        lower_band = sma - (std * std_mult)
        
        return {
            "upper": upper_band.tolist(),
            "middle": sma.tolist(),
            "lower": lower_band.tolist()
        }
    
    async def _calculate_atr_async(self, market_data: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> List[float]:
        """Calculate Average True Range"""
        period = parameters.get("period", 14)
        
        ohlcv = market_data.get("ohlcv", {})
        highs = ohlcv.get("high", [])
        lows = ohlcv.get("low", [])
        closes = ohlcv.get("close", [])
        
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return []
        
        highs_series = pd.Series(highs)
        lows_series = pd.Series(lows)
        closes_series = pd.Series(closes)
        
        tr1 = highs_series - lows_series
        tr2 = (highs_series - closes_series.shift(1)).abs()
        tr3 = (lows_series - closes_series.shift(1)).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.tolist()
    
    async def _calculate_rolling_volatility_async(self, market_data: Dict[str, Any], 
                                                parameters: Dict[str, Any]) -> List[float]:
        """Calculate rolling volatility"""
        period = parameters.get("period", 20)
        prices = market_data.get("ohlcv", {}).get("close", [])
        
        if len(prices) < period + 1:
            return []
        
        prices_series = pd.Series(prices)
        returns = prices_series.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        
        return volatility.tolist()
    
    async def _calculate_momentum_async(self, market_data: Dict[str, Any], 
                                      parameters: Dict[str, Any]) -> List[float]:
        """Calculate price momentum"""
        period = parameters.get("period", 10)
        prices = market_data.get("ohlcv", {}).get("close", [])
        
        if len(prices) < period + 1:
            return []
        
        prices_series = pd.Series(prices)
        momentum = prices_series - prices_series.shift(period)
        
        return momentum.tolist()
    
    async def _calculate_rate_of_change_async(self, market_data: Dict[str, Any], 
                                            parameters: Dict[str, Any]) -> List[float]:
        """Calculate rate of change"""
        period = parameters.get("period", 10)
        prices = market_data.get("ohlcv", {}).get("close", [])
        
        if len(prices) < period + 1:
            return []
        
        prices_series = pd.Series(prices)
        roc = ((prices_series - prices_series.shift(period)) / prices_series.shift(period)) * 100
        
        return roc.tolist()
    
    async def _calculate_volume_sma_async(self, market_data: Dict[str, Any], 
                                        parameters: Dict[str, Any]) -> List[float]:
        """Calculate volume moving average"""
        period = parameters.get("period", 20)
        volumes = market_data.get("ohlcv", {}).get("volume", [])
        
        if len(volumes) < period:
            return []
        
        return pd.Series(volumes).rolling(window=period).mean().tolist()
    
    async def _calculate_volume_ratio_async(self, market_data: Dict[str, Any], 
                                          parameters: Dict[str, Any]) -> List[float]:
        """Calculate volume ratio"""
        period = parameters.get("period", 20)
        volumes = market_data.get("ohlcv", {}).get("volume", [])
        
        if len(volumes) < period + 1:
            return []
        
        volumes_series = pd.Series(volumes)
        avg_volume = volumes_series.rolling(window=period).mean()
        volume_ratio = volumes_series / avg_volume
        
        return volume_ratio.tolist()
    
    async def _calculate_hour_of_day_async(self, market_data: Dict[str, Any], 
                                         parameters: Dict[str, Any]) -> List[int]:
        """Calculate hour of day"""
        timestamps = market_data.get("ohlcv", {}).get("timestamp", [])
        
        if not timestamps:
            return []
        
        hours = []
        for timestamp in timestamps:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hours.append(timestamp.hour)
        
        return hours
    
    async def _calculate_day_of_week_async(self, market_data: Dict[str, Any], 
                                         parameters: Dict[str, Any]) -> List[int]:
        """Calculate day of week"""
        timestamps = market_data.get("ohlcv", {}).get("timestamp", [])
        
        if not timestamps:
            return []
        
        days = []
        for timestamp in timestamps:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            days.append(timestamp.weekday())
        
        return days
    
    async def _calculate_news_sentiment_async(self, market_data: Dict[str, Any], 
                                            parameters: Dict[str, Any]) -> float:
        """Calculate news sentiment score"""
        sentiment_data = market_data.get("sentiment", {})
        news_data = sentiment_data.get("news", [])
        
        if not news_data:
            return 0.0
        
        scores = [item.get("sentiment_score", 0) for item in news_data]
        weights = [item.get("relevance", 1) for item in news_data]
        
        if not scores:
            return 0.0
        
        return np.average(scores, weights=weights)
    
    async def _calculate_social_sentiment_async(self, market_data: Dict[str, Any], 
                                              parameters: Dict[str, Any]) -> float:
        """Calculate social media sentiment score"""
        sentiment_data = market_data.get("sentiment", {})
        social_data = sentiment_data.get("social", [])
        
        if not social_data:
            return 0.0
        
        scores = [item.get("sentiment_score", 0) for item in social_data]
        weights = [item.get("engagement", 1) for item in social_data]
        
        if not scores:
            return 0.0
        
        return np.average(scores, weights=weights)
    
    async def _calculate_event_impact_async(self, market_data: Dict[str, Any], 
                                          parameters: Dict[str, Any]) -> float:
        """Calculate economic event impact score"""
        events_data = market_data.get("economic_events", [])
        
        if not events_data:
            return 0.0
        
        # Calculate weighted impact score
        impact_scores = [event.get("market_impact_score", 0) for event in events_data]
        return np.mean(impact_scores) if impact_scores else 0.0
    
    async def _calculate_event_proximity_async(self, market_data: Dict[str, Any], 
                                             parameters: Dict[str, Any]) -> float:
        """Calculate time to next economic event"""
        events_data = market_data.get("economic_events", [])
        
        if not events_data:
            return float('inf')
        
        now = datetime.now()
        min_time_diff = float('inf')
        
        for event in events_data:
            event_time = event.get("event_time")
            if event_time:
                if isinstance(event_time, str):
                    event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                
                time_diff = (event_time - now).total_seconds() / 3600  # Hours
                if time_diff > 0:  # Future events only
                    min_time_diff = min(min_time_diff, time_diff)
        
        return min_time_diff if min_time_diff != float('inf') else 0.0
    
    def get_feature_definitions(self) -> Dict[str, FeatureDefinition]:
        """Get all feature definitions"""
        return self.feature_definitions.copy()
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get feature extraction statistics"""
        return self.extraction_stats.copy()
    
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        self.logger.info("Feature cache cleared")
    
    def reset_stats(self):
        """Reset extraction statistics"""
        self.extraction_stats = {
            "features_extracted": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "processing_time_ms": 0,
            "errors": 0
        }

