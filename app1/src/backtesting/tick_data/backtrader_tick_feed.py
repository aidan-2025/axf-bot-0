#!/usr/bin/env python3
"""
Backtrader Tick Data Feed

Custom Backtrader data feed for tick-level data with bid/ask prices.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

try:
    import backtrader as bt
    from backtrader import DataBase
    from backtrader.feed import DataBase
    from backtrader.utils import date2num
except ImportError:
    bt = None
    DataBase = None

from .tick_data_loader import TickDataLoader, TickDataConfig, TickDataInfo

logger = logging.getLogger(__name__)


class TickDataFeed(DataBase):
    """
    Custom Backtrader data feed for tick-level data with bid/ask prices.
    
    This feed provides:
    - Tick-level precision (not just OHLCV)
    - Separate bid and ask prices
    - Volume information
    - Real-time spread calculation
    - Custom data fields for advanced strategies
    """
    
    params = (
        ('datetime', None),
        ('timeframe', bt.TimeFrame.Ticks),
        ('compression', 1),
        ('bid', None),
        ('ask', None),
        ('volume', None),
        ('openinterest', -1),
        ('spread', None),
        ('mid', None),
        ('tick_volume', None),
    )
    
    def __init__(self, tick_data: pd.DataFrame, symbol: str = "EURUSD"):
        """
        Initialize the tick data feed.
        
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
        
        logger.info(f"Initialized TickDataFeed for {symbol} with {self.total_ticks} ticks")
    
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
        
        # Check for valid spreads
        invalid_spreads = self.tick_data['ask'] < self.tick_data['bid']
        if invalid_spreads.any():
            invalid_count = invalid_spreads.sum()
            logger.warning(f"Found {invalid_count} ticks with ask < bid")
    
    def _prepare_data(self):
        """Prepare data for Backtrader consumption"""
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.tick_data['timestamp']):
            self.tick_data['timestamp'] = pd.to_datetime(self.tick_data['timestamp'])
        
        # Sort by timestamp
        self.tick_data = self.tick_data.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate additional fields
        self.tick_data['spread'] = self.tick_data['ask'] - self.tick_data['bid']
        self.tick_data['mid'] = (self.tick_data['ask'] + self.tick_data['bid']) / 2
        
        # Add tick volume if not present
        if 'volume' not in self.tick_data.columns:
            self.tick_data['volume'] = 1
        self.tick_data['tick_volume'] = self.tick_data['volume']
        
        # Convert timestamps to Backtrader format
        self.tick_data['datetime_num'] = self.tick_data['timestamp'].apply(
            lambda x: date2num(x.to_pydatetime()) if hasattr(x, 'to_pydatetime') else date2num(x)
        )
        
        logger.info(f"Prepared {len(self.tick_data)} ticks for Backtrader")
    
    def start(self):
        """Start the data feed"""
        self.current_index = 0
        logger.debug(f"Started tick data feed for {self.symbol}")
    
    def stop(self):
        """Stop the data feed"""
        logger.debug(f"Stopped tick data feed for {self.symbol}")
    
    def _load(self):
        """Load the next tick"""
        
        if self.current_index >= self.total_ticks:
            return False
        
        # Get current tick data
        tick = self.tick_data.iloc[self.current_index]
        
        # Set the datetime
        self.lines.datetime[0] = tick['datetime_num']
        
        # Set the OHLCV data (using bid/ask for realistic simulation)
        self.lines.open[0] = tick['bid']  # Use bid as open
        self.lines.high[0] = tick['ask']  # Use ask as high
        self.lines.low[0] = tick['bid']   # Use bid as low
        self.lines.close[0] = tick['ask'] # Use ask as close
        self.lines.volume[0] = tick['volume']
        
        # Set custom tick data (only if lines exist)
        if hasattr(self.lines, 'bid'):
            self.lines.bid[0] = tick['bid']
        if hasattr(self.lines, 'ask'):
            self.lines.ask[0] = tick['ask']
        if hasattr(self.lines, 'spread'):
            self.lines.spread[0] = tick['spread']
        if hasattr(self.lines, 'mid'):
            self.lines.mid[0] = tick['mid']
        if hasattr(self.lines, 'tick_volume'):
            self.lines.tick_volume[0] = tick['tick_volume']
        
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
            'spread': current_tick['spread'],
            'mid': current_tick['mid'],
            'volume': current_tick['volume']
        }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the data feed"""
        if self.tick_data.empty:
            return {}
        
        return {
            'symbol': self.symbol,
            'total_ticks': self.total_ticks,
            'start_time': self.tick_data['timestamp'].min(),
            'end_time': self.tick_data['timestamp'].max(),
            'avg_spread': self.tick_data['spread'].mean(),
            'min_spread': self.tick_data['spread'].min(),
            'max_spread': self.tick_data['spread'].max(),
            'current_index': self.current_index,
            'progress_percentage': (self.current_index / self.total_ticks) * 100
        }


class TickDataFeedFactory:
    """Factory for creating tick data feeds from various sources"""
    
    @staticmethod
    def from_dataframe(tick_data: pd.DataFrame, symbol: str = "EURUSD") -> TickDataFeed:
        """Create feed from DataFrame"""
        return TickDataFeed(tick_data, symbol)
    
    @staticmethod
    def from_file(file_path: str, symbol: str = "EURUSD", 
                  config: Optional[TickDataConfig] = None) -> TickDataFeed:
        """Create feed from file"""
        
        if config is None:
            config = TickDataConfig(
                data_path=file_path,
                symbol=symbol
            )
        
        loader = TickDataLoader(config)
        tick_data = loader.load_data()
        
        return TickDataFeed(tick_data, symbol)
    
    @staticmethod
    def from_influxdb(symbol: str = "EURUSD", start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> TickDataFeed:
        """Create feed from InfluxDB (placeholder)"""
        
        # This would integrate with the existing InfluxDB setup
        logger.warning("InfluxDB integration not yet implemented")
        
        # Create sample data for now
        config = TickDataConfig(
            data_format=TickDataFormat.INFLUXDB,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        loader = TickDataLoader(config)
        tick_data = loader.create_sample_data(symbol)
        
        return TickDataFeed(tick_data, symbol)
    
    @staticmethod
    def create_sample_feed(symbol: str = "EURUSD", days: int = 7) -> TickDataFeed:
        """Create sample tick data feed for testing"""
        
        config = TickDataConfig(symbol=symbol)
        loader = TickDataLoader(config)
        tick_data = loader.create_sample_data(symbol, days)
        
        return TickDataFeed(tick_data, symbol)


class TickDataAnalyzer:
    """Analyzer for tick data feeds"""
    
    def __init__(self, feed: TickDataFeed):
        self.feed = feed
        self.tick_data = feed.tick_data
    
    def analyze_spread_distribution(self) -> Dict[str, Any]:
        """Analyze spread distribution"""
        
        spreads = self.tick_data['spread']
        
        return {
            'mean': spreads.mean(),
            'median': spreads.median(),
            'std': spreads.std(),
            'min': spreads.min(),
            'max': spreads.max(),
            'q25': spreads.quantile(0.25),
            'q75': spreads.quantile(0.75),
            'q95': spreads.quantile(0.95),
            'q99': spreads.quantile(0.99)
        }
    
    def analyze_tick_frequency(self) -> Dict[str, Any]:
        """Analyze tick frequency patterns"""
        
        # Calculate time differences between ticks
        time_diffs = self.tick_data['timestamp'].diff().dt.total_seconds()
        time_diffs = time_diffs.dropna()
        
        return {
            'avg_interval_seconds': time_diffs.mean(),
            'median_interval_seconds': time_diffs.median(),
            'min_interval_seconds': time_diffs.min(),
            'max_interval_seconds': time_diffs.max(),
            'ticks_per_second': 1 / time_diffs.mean() if time_diffs.mean() > 0 else 0,
            'ticks_per_minute': 60 / time_diffs.mean() if time_diffs.mean() > 0 else 0
        }
    
    def analyze_price_movements(self) -> Dict[str, Any]:
        """Analyze price movement patterns"""
        
        # Calculate price changes
        mid_prices = self.tick_data['mid']
        price_changes = mid_prices.diff().dropna()
        
        return {
            'avg_price_change': price_changes.mean(),
            'std_price_change': price_changes.std(),
            'max_upward_move': price_changes.max(),
            'max_downward_move': price_changes.min(),
            'volatility': price_changes.std() * np.sqrt(252 * 24 * 60 * 60)  # Annualized
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        return {
            'symbol': self.feed.symbol,
            'data_info': self.feed.get_data_info(),
            'spread_analysis': self.analyze_spread_distribution(),
            'frequency_analysis': self.analyze_tick_frequency(),
            'price_analysis': self.analyze_price_movements()
        }
