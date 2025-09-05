#!/usr/bin/env python3
"""
Simple Tick Data Feed for Backtrader

A simplified tick data feed that works with standard Backtrader interfaces.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import logging

try:
    import backtrader as bt
    from backtrader.feed import DataBase
    from backtrader.utils import date2num
except ImportError:
    bt = None
    DataBase = None

logger = logging.getLogger(__name__)


class SimpleTickDataFeed(DataBase):
    """
    Simple tick data feed for Backtrader.
    
    This feed provides tick-level data using the standard OHLCV format:
    - Open: Bid price
    - High: Ask price  
    - Low: Bid price
    - Close: Ask price
    - Volume: Tick volume
    """
    
    def __init__(self, tick_data: pd.DataFrame, symbol: str = "EURUSD"):
        """
        Initialize the simple tick data feed.
        
        Args:
            tick_data: DataFrame with columns ['timestamp', 'bid', 'ask', 'volume']
            symbol: Symbol name for the data
        """
        super().__init__()
        
        self.tick_data = tick_data.copy()
        self.symbol = symbol
        self.current_index = 0
        self.total_ticks = len(tick_data)
        
        # Validate data
        self._validate_tick_data()
        
        # Prepare data for Backtrader
        self._prepare_data()
        
        logger.info(f"Initialized SimpleTickDataFeed for {symbol} with {self.total_ticks} ticks")
    
    def _validate_tick_data(self):
        """Validate the tick data format"""
        required_columns = ['timestamp', 'bid', 'ask']
        missing_columns = [col for col in required_columns if col not in self.tick_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if self.tick_data.empty:
            raise ValueError("Tick data is empty")
        
        # Check for valid prices
        if (self.tick_data['bid'] <= 0).any() or (self.tick_data['ask'] <= 0).any():
            raise ValueError("Found non-positive bid or ask prices")
    
    def _prepare_data(self):
        """Prepare data for Backtrader consumption"""
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.tick_data['timestamp']):
            self.tick_data['timestamp'] = pd.to_datetime(self.tick_data['timestamp'])
        
        # Sort by timestamp
        self.tick_data = self.tick_data.sort_values('timestamp').reset_index(drop=True)
        
        # Add volume if not present
        if 'volume' not in self.tick_data.columns:
            self.tick_data['volume'] = 1
        
        # Convert timestamps to Backtrader format
        self.tick_data['datetime_num'] = self.tick_data['timestamp'].apply(
            lambda x: date2num(x.to_pydatetime()) if hasattr(x, 'to_pydatetime') else date2num(x)
        )
        
        logger.info(f"Prepared {len(self.tick_data)} ticks for Backtrader")
    
    def start(self):
        """Start the data feed"""
        self.current_index = 0
        logger.debug(f"Started simple tick data feed for {self.symbol}")
    
    def stop(self):
        """Stop the data feed"""
        logger.debug(f"Stopped simple tick data feed for {self.symbol}")
    
    def _load(self):
        """Load the next tick"""
        
        if self.current_index >= self.total_ticks:
            return False
        
        # Get current tick data
        tick = self.tick_data.iloc[self.current_index]
        
        # Set the datetime
        self.lines.datetime[0] = tick['datetime_num']
        
        # Set the OHLCV data using bid/ask prices
        self.lines.open[0] = tick['bid']    # Use bid as open
        self.lines.high[0] = tick['ask']    # Use ask as high
        self.lines.low[0] = tick['bid']     # Use bid as low
        self.lines.close[0] = tick['ask']   # Use ask as close
        self.lines.volume[0] = tick['volume']
        
        # Move to next tick
        self.current_index += 1
        
        return True
    
    def get_current_tick(self) -> Optional[Dict[str, Any]]:
        """Get current tick data"""
        if self.current_index == 0 or self.current_index > self.total_ticks:
            return None
        
        current_tick = self.tick_data.iloc[self.current_index - 1]
        return {
            'timestamp': current_tick['timestamp'],
            'bid': current_tick['bid'],
            'ask': current_tick['ask'],
            'spread': current_tick['ask'] - current_tick['bid'],
            'mid': (current_tick['ask'] + current_tick['bid']) / 2,
            'volume': current_tick['volume']
        }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the data feed"""
        if self.tick_data.empty:
            return {}
        
        spreads = self.tick_data['ask'] - self.tick_data['bid']
        
        return {
            'symbol': self.symbol,
            'total_ticks': self.total_ticks,
            'start_time': self.tick_data['timestamp'].min(),
            'end_time': self.tick_data['timestamp'].max(),
            'avg_spread': spreads.mean(),
            'min_spread': spreads.min(),
            'max_spread': spreads.max(),
            'current_index': self.current_index,
            'progress_percentage': (self.current_index / self.total_ticks) * 100
        }


class TickDataFeedFactory:
    """Factory for creating simple tick data feeds"""
    
    @staticmethod
    def from_dataframe(tick_data: pd.DataFrame, symbol: str = "EURUSD") -> SimpleTickDataFeed:
        """Create feed from DataFrame"""
        return SimpleTickDataFeed(tick_data, symbol)
    
    @staticmethod
    def create_sample_feed(symbol: str = "EURUSD", days: int = 7) -> SimpleTickDataFeed:
        """Create sample tick data feed for testing"""
        
        from .tick_data_loader import TickDataLoader, TickDataConfig, TickDataFormat
        
        config = TickDataConfig(data_format=TickDataFormat.INFLUXDB, symbol=symbol)
        loader = TickDataLoader(config)
        tick_data = loader.create_sample_data(symbol, days)
        
        return SimpleTickDataFeed(tick_data, symbol)

