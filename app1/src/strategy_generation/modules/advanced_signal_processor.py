"""
Advanced signal processing module with real-time capabilities
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum

from ..core.strategy_template import Signal


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class ProcessedDataPoint:
    """Processed data point with quality metrics"""
    timestamp: datetime
    symbol: str
    data_type: str  # 'price', 'sentiment', 'event', 'indicator'
    value: Any
    quality_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DataStreamConfig:
    """Configuration for data stream processing"""
    symbol: str
    data_types: List[str]
    timeframes: List[str]
    buffer_size: int = 1000
    quality_threshold: float = 0.7
    latency_threshold_ms: int = 1000
    enable_interpolation: bool = True
    enable_outlier_detection: bool = True


class AdvancedSignalProcessor:
    """
    Advanced signal processor with real-time capabilities, data quality assessment,
    and multi-source integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Processing configuration
        self.max_latency_ms = self.config.get('max_latency_ms', 1000)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        self.enable_interpolation = self.config.get('enable_interpolation', True)
        self.enable_outlier_detection = self.config.get('enable_outlier_detection', True)
        
        # Data buffers for real-time processing
        self.data_buffers: Dict[str, List[ProcessedDataPoint]] = {}
        self.stream_configs: Dict[str, DataStreamConfig] = {}
        
        # Quality assessment metrics
        self.quality_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "quality_failures": 0,
            "latency_failures": 0,
            "outliers_detected": 0,
            "interpolations_performed": 0
        }
    
    def configure_stream(self, symbol: str, config: DataStreamConfig):
        """Configure a data stream for processing"""
        self.stream_configs[symbol] = config
        self.data_buffers[symbol] = []
        self.quality_metrics[symbol] = {
            "avg_quality": 0.0,
            "avg_latency": 0.0,
            "total_points": 0,
            "quality_trend": []
        }
        
        self.logger.info(f"Configured stream for {symbol}: {config.data_types}")
    
    async def process_market_data_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data asynchronously with real-time capabilities
        
        Args:
            market_data: Raw market data from multiple sources
            
        Returns:
            Dict[str, Any]: Processed market data with quality metrics
        """
        try:
            processed_data = {}
            
            # Process OHLCV data
            if 'ohlcv' in market_data:
                processed_data['ohlcv'] = await self._process_ohlcv_async(market_data['ohlcv'])
            
            # Process sentiment data
            if 'sentiment' in market_data:
                processed_data['sentiment'] = await self._process_sentiment_async(market_data['sentiment'])
            
            # Process economic events
            if 'economic_events' in market_data:
                processed_data['economic_events'] = await self._process_events_async(market_data['economic_events'])
            
            # Process technical indicators
            if 'indicators' in market_data:
                processed_data['indicators'] = await self._process_indicators_async(market_data['indicators'])
            
            # Add processing metadata
            processed_data['processing_metadata'] = {
                "timestamp": datetime.now().isoformat(),
                "quality_score": self._calculate_overall_quality(),
                "processing_stats": self.processing_stats.copy()
            }
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in async market data processing: {e}")
            return market_data
    
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for market data processing
        
        Args:
            market_data: Raw market data
            
        Returns:
            Dict[str, Any]: Processed market data
        """
        return asyncio.run(self.process_market_data_async(market_data))
    
    async def _process_ohlcv_async(self, ohlcv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OHLCV data asynchronously"""
        try:
            processed = ohlcv_data.copy()
            
            # Extract price arrays
            price_keys = ['open', 'high', 'low', 'close', 'volume']
            price_arrays = {key: processed.get(key, []) for key in price_keys}
            
            # Validate data consistency
            lengths = [len(arr) for arr in price_arrays.values() if arr]
            if not lengths:
                return processed
            
            min_length = min(lengths)
            max_length = max(lengths)
            
            # Handle length inconsistencies
            if max_length - min_length > 1:
                self.logger.warning(f"OHLCV data length inconsistency: {min_length}-{max_length}")
                # Truncate to minimum length
                for key in price_keys:
                    if key in processed and len(processed[key]) > min_length:
                        processed[key] = processed[key][:min_length]
            
            # Add timestamps if missing
            if 'timestamp' not in processed or not processed['timestamp']:
                processed['timestamp'] = [datetime.now() - timedelta(minutes=i) 
                                       for i in range(len(processed['close']))]
            
            # Detect and handle outliers
            if self.enable_outlier_detection:
                processed = await self._detect_and_handle_outliers_async(processed)
            
            # Interpolate missing values
            if self.enable_interpolation:
                processed = await self._interpolate_missing_values_async(processed)
            
            # Calculate data quality metrics
            quality_score = self._assess_ohlcv_quality(processed)
            
            # Add quality metadata
            processed['quality_metrics'] = {
                "quality_score": quality_score,
                "data_points": len(processed['close']),
                "missing_values": self._count_missing_values(processed),
                "outliers_detected": processed.get('outliers_detected', 0)
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing OHLCV data: {e}")
            return ohlcv_data
    
    async def _process_sentiment_async(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment data asynchronously"""
        try:
            processed = sentiment_data.copy()
            
            # Process news sentiment
            if 'news' in processed:
                processed['news'] = await self._process_news_sentiment_async(processed['news'])
            
            # Process social sentiment
            if 'social' in processed:
                processed['social'] = await self._process_social_sentiment_async(processed['social'])
            
            # Calculate combined sentiment metrics
            combined_metrics = self._calculate_combined_sentiment_metrics(processed)
            processed['combined_metrics'] = combined_metrics
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing sentiment data: {e}")
            return sentiment_data
    
    async def _process_events_async(self, events_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process economic events data asynchronously"""
        try:
            processed_events = []
            
            for event in events_data:
                processed_event = await self._process_single_event_async(event)
                if processed_event:
                    processed_events.append(processed_event)
            
            # Sort by event time
            processed_events.sort(key=lambda x: x.get('event_time', ''))
            
            return processed_events
            
        except Exception as e:
            self.logger.error(f"Error processing events data: {e}")
            return events_data
    
    async def _process_indicators_async(self, indicators_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process technical indicators data asynchronously"""
        try:
            processed = indicators_data.copy()
            
            # Ensure all indicator arrays have consistent length
            indicator_keys = [k for k, v in processed.items() if isinstance(v, list)]
            if not indicator_keys:
                return processed
            
            lengths = [len(processed[key]) for key in indicator_keys]
            min_length = min(lengths)
            
            # Truncate to consistent length
            for key in indicator_keys:
                if len(processed[key]) > min_length:
                    processed[key] = processed[key][:min_length]
            
            # Calculate indicator quality metrics
            quality_metrics = {}
            for key in indicator_keys:
                values = processed[key]
                if values:
                    quality_metrics[key] = {
                        "valid_values": sum(1 for v in values if v is not None and not np.isnan(v)),
                        "missing_values": sum(1 for v in values if v is None or np.isnan(v)),
                        "quality_score": sum(1 for v in values if v is not None and not np.isnan(v)) / len(values)
                    }
            
            processed['quality_metrics'] = quality_metrics
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing indicators data: {e}")
            return indicators_data
    
    async def _detect_and_handle_outliers_async(self, ohlcv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and handle outliers in OHLCV data"""
        try:
            processed = ohlcv_data.copy()
            outliers_detected = 0
            
            # Detect outliers in price data
            for key in ['open', 'high', 'low', 'close']:
                if key in processed and processed[key]:
                    values = np.array(processed[key])
                    
                    # Use IQR method for outlier detection
                    Q1 = np.percentile(values, 25)
                    Q3 = np.percentile(values, 75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = (values < lower_bound) | (values > upper_bound)
                    outlier_count = np.sum(outliers)
                    
                    if outlier_count > 0:
                        # Replace outliers with interpolated values
                        if self.enable_interpolation:
                            values[outliers] = np.nan
                            values = pd.Series(values).interpolate().values
                            processed[key] = values.tolist()
                            outliers_detected += outlier_count
                        else:
                            # Cap outliers instead of interpolating
                            values[values < lower_bound] = lower_bound
                            values[values > upper_bound] = upper_bound
                            processed[key] = values.tolist()
            
            processed['outliers_detected'] = outliers_detected
            self.processing_stats['outliers_detected'] += outliers_detected
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            return ohlcv_data
    
    async def _interpolate_missing_values_async(self, ohlcv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Interpolate missing values in OHLCV data"""
        try:
            processed = ohlcv_data.copy()
            interpolations_performed = 0
            
            for key in ['open', 'high', 'low', 'close', 'volume']:
                if key in processed and processed[key]:
                    values = np.array(processed[key])
                    nan_mask = np.isnan(values)
                    nan_count = np.sum(nan_mask)
                    
                    if nan_count > 0:
                        # Interpolate missing values
                        values[nan_mask] = np.nan
                        values = pd.Series(values).interpolate(method='linear').values
                        processed[key] = values.tolist()
                        interpolations_performed += nan_count
            
            processed['interpolations_performed'] = interpolations_performed
            self.processing_stats['interpolations_performed'] += interpolations_performed
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error interpolating missing values: {e}")
            return ohlcv_data
    
    def _assess_ohlcv_quality(self, ohlcv_data: Dict[str, Any]) -> float:
        """Assess quality of OHLCV data"""
        try:
            quality_factors = []
            
            # Check for missing values
            price_keys = ['open', 'high', 'low', 'close']
            for key in price_keys:
                if key in ohlcv_data and ohlcv_data[key]:
                    values = ohlcv_data[key]
                    valid_ratio = sum(1 for v in values if v is not None and not np.isnan(v)) / len(values)
                    quality_factors.append(valid_ratio)
            
            # Check for logical consistency (high >= low, etc.)
            if all(key in ohlcv_data for key in price_keys):
                highs = ohlcv_data['high']
                lows = ohlcv_data['low']
                opens = ohlcv_data['open']
                closes = ohlcv_data['close']
                
                if len(highs) == len(lows) == len(opens) == len(closes):
                    logical_consistency = 0
                    total_checks = 0
                    
                    for i in range(len(highs)):
                        if (highs[i] is not None and lows[i] is not None and 
                            opens[i] is not None and closes[i] is not None):
                            if (highs[i] >= lows[i] and 
                                highs[i] >= opens[i] and highs[i] >= closes[i] and
                                lows[i] <= opens[i] and lows[i] <= closes[i]):
                                logical_consistency += 1
                            total_checks += 1
                    
                    if total_checks > 0:
                        quality_factors.append(logical_consistency / total_checks)
            
            # Check for reasonable price ranges
            if 'close' in ohlcv_data and ohlcv_data['close']:
                closes = [c for c in ohlcv_data['close'] if c is not None and not np.isnan(c)]
                if closes:
                    price_range = max(closes) - min(closes)
                    avg_price = np.mean(closes)
                    if avg_price > 0:
                        volatility_ratio = price_range / avg_price
                        # Reasonable volatility should be between 0.01 and 0.5
                        if 0.01 <= volatility_ratio <= 0.5:
                            quality_factors.append(1.0)
                        else:
                            quality_factors.append(0.5)
            
            return np.mean(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Error assessing OHLCV quality: {e}")
            return 0.0
    
    def _count_missing_values(self, ohlcv_data: Dict[str, Any]) -> int:
        """Count missing values in OHLCV data"""
        missing_count = 0
        for key in ['open', 'high', 'low', 'close', 'volume']:
            if key in ohlcv_data and ohlcv_data[key]:
                missing_count += sum(1 for v in ohlcv_data[key] 
                                   if v is None or np.isnan(v))
        return missing_count
    
    async def _process_news_sentiment_async(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process news sentiment data"""
        processed_news = []
        
        for item in news_data:
            processed_item = item.copy()
            
            # Normalize sentiment score
            if 'sentiment_score' in processed_item:
                score = processed_item['sentiment_score']
                processed_item['sentiment_score'] = max(-1.0, min(1.0, float(score)))
            
            # Normalize relevance score
            if 'relevance' in processed_item:
                relevance = processed_item['relevance']
                processed_item['relevance'] = max(0.0, min(1.0, float(relevance)))
            
            # Add quality metrics
            processed_item['quality_score'] = self._assess_news_quality(processed_item)
            
            processed_news.append(processed_item)
        
        return processed_news
    
    async def _process_social_sentiment_async(self, social_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process social media sentiment data"""
        processed_social = []
        
        for item in social_data:
            processed_item = item.copy()
            
            # Normalize sentiment score
            if 'sentiment_score' in processed_item:
                score = processed_item['sentiment_score']
                processed_item['sentiment_score'] = max(-1.0, min(1.0, float(score)))
            
            # Normalize engagement score
            if 'engagement' in processed_item:
                engagement = processed_item['engagement']
                processed_item['engagement'] = max(0.0, min(1.0, float(engagement)))
            
            # Add quality metrics
            processed_item['quality_score'] = self._assess_social_quality(processed_item)
            
            processed_social.append(processed_item)
        
        return processed_social
    
    async def _process_single_event_async(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single economic event"""
        try:
            processed_event = event.copy()
            
            # Normalize impact scores
            if 'market_impact_score' in processed_event:
                score = processed_event['market_impact_score']
                processed_event['market_impact_score'] = max(0.0, min(1.0, float(score)))
            
            if 'volatility_expected' in processed_event:
                volatility = processed_event['volatility_expected']
                processed_event['volatility_expected'] = max(0.0, min(1.0, float(volatility)))
            
            # Add quality metrics
            processed_event['quality_score'] = self._assess_event_quality(processed_event)
            
            return processed_event
            
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
            return None
    
    def _assess_news_quality(self, news_item: Dict[str, Any]) -> float:
        """Assess quality of news item"""
        quality_factors = []
        
        # Check for required fields
        required_fields = ['sentiment_score', 'relevance']
        for field in required_fields:
            if field in news_item and news_item[field] is not None:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
        
        # Check sentiment score validity
        if 'sentiment_score' in news_item:
            score = news_item['sentiment_score']
            if -1.0 <= score <= 1.0:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.5)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _assess_social_quality(self, social_item: Dict[str, Any]) -> float:
        """Assess quality of social media item"""
        quality_factors = []
        
        # Check for required fields
        required_fields = ['sentiment_score', 'engagement']
        for field in required_fields:
            if field in social_item and social_item[field] is not None:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _assess_event_quality(self, event: Dict[str, Any]) -> float:
        """Assess quality of economic event"""
        quality_factors = []
        
        # Check for required fields
        required_fields = ['title', 'event_time', 'impact']
        for field in required_fields:
            if field in event and event[field]:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)
        
        # Check impact score validity
        if 'market_impact_score' in event:
            score = event['market_impact_score']
            if 0.0 <= score <= 1.0:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.5)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _calculate_combined_sentiment_metrics(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate combined sentiment metrics"""
        metrics = {}
        
        # News sentiment metrics
        if 'news' in sentiment_data and sentiment_data['news']:
            news_scores = [item.get('sentiment_score', 0) for item in sentiment_data['news']]
            news_weights = [item.get('relevance', 1) for item in sentiment_data['news']]
            
            if news_scores:
                metrics['news_sentiment_mean'] = np.mean(news_scores)
                metrics['news_sentiment_weighted'] = np.average(news_scores, weights=news_weights)
                metrics['news_count'] = len(news_scores)
        
        # Social sentiment metrics
        if 'social' in sentiment_data and sentiment_data['social']:
            social_scores = [item.get('sentiment_score', 0) for item in sentiment_data['social']]
            social_weights = [item.get('engagement', 1) for item in sentiment_data['social']]
            
            if social_scores:
                metrics['social_sentiment_mean'] = np.mean(social_scores)
                metrics['social_sentiment_weighted'] = np.average(social_scores, weights=social_weights)
                metrics['social_count'] = len(social_scores)
        
        # Combined metrics
        all_scores = []
        all_weights = []
        
        if 'news' in sentiment_data and sentiment_data['news']:
            all_scores.extend([item.get('sentiment_score', 0) for item in sentiment_data['news']])
            all_weights.extend([item.get('relevance', 1) for item in sentiment_data['news']])
        
        if 'social' in sentiment_data and sentiment_data['social']:
            all_scores.extend([item.get('sentiment_score', 0) for item in sentiment_data['social']])
            all_weights.extend([item.get('engagement', 1) for item in sentiment_data['social']])
        
        if all_scores:
            metrics['combined_sentiment_mean'] = np.mean(all_scores)
            metrics['combined_sentiment_weighted'] = np.average(all_scores, weights=all_weights)
            metrics['total_sentiment_count'] = len(all_scores)
            metrics['sentiment_bias'] = 1 if metrics['combined_sentiment_weighted'] > 0 else -1
        
        return metrics
    
    def _calculate_overall_quality(self) -> float:
        """Calculate overall processing quality"""
        if not self.quality_metrics:
            return 0.0
        
        quality_scores = [metrics['avg_quality'] for metrics in self.quality_metrics.values()]
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            "overall_quality": self._calculate_overall_quality(),
            "active_streams": len(self.stream_configs),
            "total_buffers": sum(len(buffer) for buffer in self.data_buffers.values())
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            "total_processed": 0,
            "quality_failures": 0,
            "latency_failures": 0,
            "outliers_detected": 0,
            "interpolations_performed": 0
        }
        
        for symbol in self.quality_metrics:
            self.quality_metrics[symbol] = {
                "avg_quality": 0.0,
                "avg_latency": 0.0,
                "total_points": 0,
                "quality_trend": []
            }
    
    async def process_signals_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process signals asynchronously"""
        try:
            self.logger.info("Processing signals asynchronously")
            
            # For now, return the market data with some basic processing
            # In production, this would do actual signal processing
            processed_data = market_data.copy()
            processed_data['signal_processing'] = {
                'processed_at': datetime.now().isoformat(),
                'data_quality': 'good',
                'signals_count': 0
            }
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
            return market_data
