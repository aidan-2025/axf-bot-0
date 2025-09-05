"""
Signal processing module for market data
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import logging
from datetime import datetime

from ..core.strategy_template import Signal


class SignalProcessor:
    """
    Processes market data and signals for strategy generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw market data for strategy consumption
        
        Args:
            market_data: Raw market data
            
        Returns:
            Dict[str, Any]: Processed market data
        """
        try:
            processed_data = market_data.copy()
            
            # Process OHLCV data
            if 'ohlcv' in processed_data:
                processed_data['ohlcv'] = self._process_ohlcv_data(processed_data['ohlcv'])
            
            # Process sentiment data
            if 'sentiment' in processed_data:
                processed_data['sentiment'] = self._process_sentiment_data(processed_data['sentiment'])
            
            # Process economic events
            if 'economic_events' in processed_data:
                processed_data['economic_events'] = self._process_economic_events(processed_data['economic_events'])
            
            # Process technical indicators
            if 'indicators' in processed_data:
                processed_data['indicators'] = self._process_indicators(processed_data['indicators'])
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return market_data
    
    def _process_ohlcv_data(self, ohlcv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OHLCV data"""
        processed = ohlcv_data.copy()
        
        # Ensure all price arrays have the same length
        price_keys = ['open', 'high', 'low', 'close', 'volume']
        lengths = [len(processed.get(key, [])) for key in price_keys if processed.get(key)]
        
        if lengths:
            min_length = min(lengths)
            for key in price_keys:
                if key in processed and len(processed[key]) > min_length:
                    processed[key] = processed[key][:min_length]
        
        # Add timestamp if missing
        if 'timestamp' not in processed or not processed['timestamp']:
            processed['timestamp'] = [datetime.now()] * min_length
        
        return processed
    
    def _process_sentiment_data(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment data"""
        processed = sentiment_data.copy()
        
        # Normalize sentiment scores
        if 'news' in processed:
            for item in processed['news']:
                if 'sentiment_score' in item:
                    item['sentiment_score'] = max(-1.0, min(1.0, item['sentiment_score']))
        
        if 'social' in processed:
            for item in processed['social']:
                if 'sentiment_score' in item:
                    item['sentiment_score'] = max(-1.0, min(1.0, item['sentiment_score']))
        
        return processed
    
    def _process_economic_events(self, events_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process economic events data"""
        processed = []
        
        for event in events_data:
            processed_event = event.copy()
            
            # Normalize impact scores
            if 'market_impact_score' in processed_event:
                processed_event['market_impact_score'] = max(0.0, min(1.0, processed_event['market_impact_score']))
            
            if 'volatility_expected' in processed_event:
                processed_event['volatility_expected'] = max(0.0, min(1.0, processed_event['volatility_expected']))
            
            processed.append(processed_event)
        
        return processed
    
    def _process_indicators(self, indicators_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process technical indicators data"""
        processed = indicators_data.copy()
        
        # Ensure all indicator arrays have the same length
        indicator_keys = list(processed.keys())
        lengths = [len(processed[key]) for key in indicator_keys if isinstance(processed[key], list)]
        
        if lengths:
            min_length = min(lengths)
            for key in indicator_keys:
                if isinstance(processed[key], list) and len(processed[key]) > min_length:
                    processed[key] = processed[key][:min_length]
        
        return processed
    
    def filter_signals(self, signals: List[Signal], filters: Dict[str, Any]) -> List[Signal]:
        """
        Filter signals based on criteria
        
        Args:
            signals: List of signals to filter
            filters: Filter criteria
            
        Returns:
            List[Signal]: Filtered signals
        """
        try:
            filtered_signals = signals.copy()
            
            # Filter by signal type
            if 'signal_types' in filters:
                allowed_types = filters['signal_types']
                filtered_signals = [s for s in filtered_signals if s.signal_type in allowed_types]
            
            # Filter by strength
            if 'min_strength' in filters:
                min_strength = filters['min_strength']
                filtered_signals = [s for s in filtered_signals if s.strength >= min_strength]
            
            # Filter by confidence
            if 'min_confidence' in filters:
                min_confidence = filters['min_confidence']
                filtered_signals = [s for s in filtered_signals if s.confidence >= min_confidence]
            
            # Filter by symbol
            if 'symbols' in filters:
                allowed_symbols = filters['symbols']
                filtered_signals = [s for s in filtered_signals if s.symbol in allowed_symbols]
            
            # Filter by time range
            if 'start_time' in filters or 'end_time' in filters:
                start_time = filters.get('start_time')
                end_time = filters.get('end_time')
                filtered_signals = [
                    s for s in filtered_signals
                    if (not start_time or s.timestamp >= start_time) and
                       (not end_time or s.timestamp <= end_time)
                ]
            
            self.logger.info(f"Filtered {len(signals)} signals to {len(filtered_signals)} signals")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error filtering signals: {e}")
            return signals
    
    def aggregate_signals(self, signals: List[Signal], aggregation_method: str = "weighted_average") -> Dict[str, Any]:
        """
        Aggregate signals into summary metrics
        
        Args:
            signals: List of signals to aggregate
            aggregation_method: Method for aggregation
            
        Returns:
            Dict[str, Any]: Aggregated signal metrics
        """
        try:
            if not signals:
                return {
                    "total_signals": 0,
                    "avg_strength": 0.0,
                    "avg_confidence": 0.0,
                    "signal_distribution": {},
                    "timestamp_range": None
                }
            
            # Basic metrics
            total_signals = len(signals)
            avg_strength = np.mean([s.strength for s in signals])
            avg_confidence = np.mean([s.confidence for s in signals])
            
            # Signal type distribution
            signal_types = [s.signal_type for s in signals]
            signal_distribution = {signal_type: signal_types.count(signal_type) for signal_type in set(signal_types)}
            
            # Timestamp range
            timestamps = [s.timestamp for s in signals]
            timestamp_range = {
                "start": min(timestamps),
                "end": max(timestamps)
            } if timestamps else None
            
            # Weighted aggregation if requested
            if aggregation_method == "weighted_average":
                weights = [s.strength * s.confidence for s in signals]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    weighted_strength = sum(s.strength * w for s, w in zip(signals, weights)) / total_weight
                    weighted_confidence = sum(s.confidence * w for s, w in zip(signals, weights)) / total_weight
                else:
                    weighted_strength = avg_strength
                    weighted_confidence = avg_confidence
            else:
                weighted_strength = avg_strength
                weighted_confidence = avg_confidence
            
            return {
                "total_signals": total_signals,
                "avg_strength": avg_strength,
                "avg_confidence": avg_confidence,
                "weighted_strength": weighted_strength,
                "weighted_confidence": weighted_confidence,
                "signal_distribution": signal_distribution,
                "timestamp_range": timestamp_range,
                "aggregation_method": aggregation_method
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating signals: {e}")
            return {
                "total_signals": 0,
                "avg_strength": 0.0,
                "avg_confidence": 0.0,
                "signal_distribution": {},
                "timestamp_range": None,
                "error": str(e)
            }

