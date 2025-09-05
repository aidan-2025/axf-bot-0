"""
Timeframe aggregation for multi-timeframe analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..models import Timeframe, OHLCVData, AggregatedData

logger = logging.getLogger(__name__)


class TimeframeAggregator:
    """Aggregates tick data to different timeframes"""
    
    def __init__(self):
        self.timeframe_minutes = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.M30: 30,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.D1: 1440,
            Timeframe.W1: 10080
        }
    
    def aggregate_to_timeframe(
        self,
        data: pd.DataFrame,
        symbol: str,
        target_timeframe: Timeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[AggregatedData]:
        """Aggregate data to a specific timeframe"""
        try:
            if data.empty:
                logger.warning(f"No data to aggregate for {symbol}")
                return None
            
            # Ensure data has timestamp index
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            
            # Filter by time range if specified
            if start_time or end_time:
                if start_time:
                    data = data[data.index >= start_time]
                if end_time:
                    data = data[data.index <= end_time]
            
            if data.empty:
                logger.warning(f"No data in specified time range for {symbol}")
                return None
            
            # Resample data to target timeframe
            aggregated = self._resample_ohlcv(data, target_timeframe)
            
            if aggregated.empty:
                logger.warning(f"No data after aggregation for {symbol} at {target_timeframe.value}")
                return None
            
            return AggregatedData(
                symbol=symbol,
                timeframe=target_timeframe,
                data=aggregated,
                start_time=aggregated.index.min(),
                end_time=aggregated.index.max(),
                record_count=len(aggregated)
            )
            
        except Exception as e:
            logger.error(f"Error aggregating data for {symbol} to {target_timeframe.value}: {str(e)}")
            return None
    
    def aggregate_multiple_timeframes(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframes: List[Timeframe],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[Timeframe, AggregatedData]:
        """Aggregate data to multiple timeframes"""
        results = {}
        
        for timeframe in timeframes:
            aggregated = self.aggregate_to_timeframe(
                data, symbol, timeframe, start_time, end_time
            )
            
            if aggregated:
                results[timeframe] = aggregated
        
        return results
    
    def _resample_ohlcv(self, data: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """Resample OHLCV data to target timeframe"""
        try:
            # Get the resample frequency
            freq = self._get_resample_frequency(timeframe)
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns for OHLCV resampling: {data.columns.tolist()}")
                return pd.DataFrame()
            
            # Resample with OHLCV aggregation rules
            resampled = data.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Remove rows with NaN values
            resampled = resampled.dropna()
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    def _get_resample_frequency(self, timeframe: Timeframe) -> str:
        """Get pandas resample frequency string for timeframe"""
        frequency_map = {
            Timeframe.M1: '1T',      # 1 minute
            Timeframe.M5: '5T',      # 5 minutes
            Timeframe.M15: '15T',    # 15 minutes
            Timeframe.M30: '30T',    # 30 minutes
            Timeframe.H1: '1H',      # 1 hour
            Timeframe.H4: '4H',      # 4 hours
            Timeframe.D1: '1D',      # 1 day
            Timeframe.W1: '1W'       # 1 week
        }
        
        return frequency_map.get(timeframe, '1T')
    
    def validate_timeframe_data(
        self,
        data: pd.DataFrame,
        timeframe: Timeframe,
        min_data_points: int = 10
    ) -> bool:
        """Validate if data has enough points for the timeframe"""
        try:
            if data.empty:
                return False
            
            # Check minimum data points
            if len(data) < min_data_points:
                logger.warning(f"Insufficient data points: {len(data)} < {min_data_points}")
                return False
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Missing required columns: {required_columns}")
                return False
            
            # Check for valid OHLCV data
            if data[required_columns].isnull().any().any():
                logger.warning("Found null values in OHLCV data")
                return False
            
            # Check OHLC consistency (High >= Low, High >= Open, High >= Close, etc.)
            if not self._validate_ohlc_consistency(data):
                logger.warning("OHLC data consistency check failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating timeframe data: {str(e)}")
            return False
    
    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> bool:
        """Validate OHLC data consistency"""
        try:
            # High should be >= Low
            if not (data['high'] >= data['low']).all():
                return False
            
            # High should be >= Open
            if not (data['high'] >= data['open']).all():
                return False
            
            # High should be >= Close
            if not (data['high'] >= data['close']).all():
                return False
            
            # Low should be <= Open
            if not (data['low'] <= data['open']).all():
                return False
            
            # Low should be <= Close
            if not (data['low'] <= data['close']).all():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating OHLC consistency: {str(e)}")
            return False
    
    def get_timeframe_info(self, timeframe: Timeframe) -> Dict[str, any]:
        """Get information about a timeframe"""
        return {
            'timeframe': timeframe.value,
            'minutes': self.timeframe_minutes.get(timeframe, 0),
            'description': self._get_timeframe_description(timeframe)
        }
    
    def _get_timeframe_description(self, timeframe: Timeframe) -> str:
        """Get human-readable description of timeframe"""
        descriptions = {
            Timeframe.M1: "1 Minute",
            Timeframe.M5: "5 Minutes",
            Timeframe.M15: "15 Minutes",
            Timeframe.M30: "30 Minutes",
            Timeframe.H1: "1 Hour",
            Timeframe.H4: "4 Hours",
            Timeframe.D1: "1 Day",
            Timeframe.W1: "1 Week"
        }
        
        return descriptions.get(timeframe, "Unknown")
    
    def calculate_timeframe_metrics(self, data: pd.DataFrame) -> Dict[str, any]:
        """Calculate metrics for timeframe data"""
        try:
            if data.empty:
                return {}
            
            metrics = {
                'total_records': len(data),
                'time_span_minutes': 0,
                'average_volume': 0,
                'price_range': 0,
                'volatility': 0
            }
            
            if len(data) > 0:
                # Time span
                if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                    time_span = data.index.max() - data.index.min()
                    metrics['time_span_minutes'] = time_span.total_seconds() / 60
                
                # Average volume
                if 'volume' in data.columns:
                    metrics['average_volume'] = data['volume'].mean()
                
                # Price range
                if all(col in data.columns for col in ['high', 'low']):
                    metrics['price_range'] = (data['high'] - data['low']).mean()
                
                # Volatility (standard deviation of returns)
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    metrics['volatility'] = returns.std()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating timeframe metrics: {str(e)}")
            return {}

