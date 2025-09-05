#!/usr/bin/env python3
"""
InfluxDB Data Feed for Backtrader

Provides Backtrader-compatible data feeds using InfluxDB as the data source.
Supports both OHLCV candle data and tick data for high-fidelity backtesting.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import asyncio
from dataclasses import dataclass

from ...data_ingestion.storage.influxdb_writer import InfluxDBWriter, InfluxDBConfig
from ...data_ingestion.storage.storage_manager import StorageManager, StorageConfig

logger = logging.getLogger(__name__)


@dataclass
class FeedConfig:
    """Configuration for InfluxDB data feed"""
    symbol: str
    timeframe: str = '1m'
    start_date: datetime = None
    end_date: datetime = None
    compression: int = 1
    fromdate: datetime = None
    todate: datetime = None
    
    # InfluxDB configuration
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_org: str = "axf-bot"
    influxdb_bucket: str = "forex_data"
    
    # Data quality settings
    fill_missing: bool = True
    max_gap_minutes: int = 60
    validate_data: bool = True


class InfluxDBDataFeed(bt.feeds.GenericCSVData):
    """
    Backtrader data feed that loads data from InfluxDB
    
    This feed extends GenericCSVData but loads data from InfluxDB instead of CSV files.
    It supports both OHLCV candle data and tick data for high-fidelity backtesting.
    """
    
    params = (
        ('datetime', 0),
        ('time', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('tmformat', '%H:%M:%S'),
    )
    
    def __init__(self, config: FeedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize InfluxDB connection
        self.influxdb_config = InfluxDBConfig(
            url=config.influxdb_url,
            token=config.influxdb_token,
            org=config.influxdb_org,
            bucket=config.influxdb_bucket
        )
        self.influxdb_writer = InfluxDBWriter(self.influxdb_config)
        
        # Load data from InfluxDB
        self.data_df = self._load_data()
        
        if self.data_df is None or self.data_df.empty:
            raise ValueError(f"No data found for {config.symbol} in the specified time range")
        
        # Initialize parent class with the loaded data
        super().__init__(
            dataname=self.data_df,
            datetime=self.params.datetime,
            time=self.params.time,
            open=self.params.open,
            high=self.params.high,
            low=self.params.low,
            close=self.params.close,
            volume=self.params.volume,
            openinterest=self.params.openinterest,
            dtformat=self.params.dtformat,
            tmformat=self.params.tmformat,
            fromdate=config.fromdate,
            todate=config.todate,
            compression=config.compression
        )
        
        self.logger.info(f"Loaded {len(self.data_df)} data points for {config.symbol}")
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load data from InfluxDB and convert to pandas DataFrame"""
        try:
            # Set date range
            start_date = self.config.start_date or self.config.fromdate
            end_date = self.config.end_date or self.config.todate or datetime.now()
            
            if start_date is None:
                # Default to last 30 days
                start_date = end_date - timedelta(days=30)
            
            # Query data from InfluxDB (synchronous version)
            data = self._query_influxdb_data_sync(start_date, end_date)
            
            if not data:
                self.logger.warning(f"No data found for {self.config.symbol} in time range {start_date} to {end_date}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            
            # Sort by time
            df = df.sort_index()
            
            # Validate data quality
            if self.config.validate_data:
                df = self._validate_and_clean_data(df)
            
            # Fill missing data if requested
            if self.config.fill_missing:
                df = self._fill_missing_data(df)
            
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return None
            
            # Convert to the format expected by Backtrader
            df = df[required_columns].copy()
            
            # Reset index to make time a column
            df = df.reset_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from InfluxDB: {e}")
            return None
    
    def _query_influxdb_data_sync(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Query data from InfluxDB (synchronous version)"""
        try:
            # Connect to InfluxDB synchronously
            if not self.influxdb_writer.connect_sync():
                self.logger.error("Failed to connect to InfluxDB")
                return []
            
            # Use the synchronous query method
            data = self.influxdb_writer.query_candle_data_sync(
                instrument=self.config.symbol,
                granularity=self.config.timeframe,
                start_time=start_date,
                end_time=end_date
            )
            return data
        except Exception as e:
            self.logger.error(f"Failed to query InfluxDB: {e}")
            return []
    
    async def _query_influxdb_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Query data from InfluxDB (async version)"""
        try:
            data = await self.influxdb_writer.query_candle_data(
                instrument=self.config.symbol,
                granularity=self.config.timeframe,
                start_time=start_date,
                end_time=end_date
            )
            return data
        except Exception as e:
            self.logger.error(f"Failed to query InfluxDB: {e}")
            return []
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data"""
        original_length = len(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Remove rows with invalid OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        df = df[~invalid_ohlc]
        
        # Remove rows with negative volume
        df = df[df['volume'] >= 0]
        
        # Remove rows with zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df = df[df[col] > 0]
        
        cleaned_length = len(df)
        if cleaned_length < original_length:
            self.logger.warning(f"Removed {original_length - cleaned_length} invalid data points")
        
        return df
    
    def _fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing data points using forward fill"""
        # Resample to ensure consistent time intervals
        timeframe_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        freq = timeframe_map.get(self.config.timeframe, '1T')
        
        # Resample and forward fill
        df_resampled = df.resample(freq).ffill()
        
        # Check for gaps larger than max_gap_minutes
        if len(df_resampled) > 1:
            time_diffs = df_resampled.index.to_series().diff()
            max_gap = timedelta(minutes=self.config.max_gap_minutes)
            
            # Remove data after large gaps
            large_gaps = time_diffs > max_gap
            if large_gaps.any():
                first_large_gap = large_gaps.idxmax()
                df_resampled = df_resampled.loc[:first_large_gap]
                self.logger.warning(f"Removed data after large gap at {first_large_gap}")
        
        return df_resampled


class TickDataFeed(bt.feeds.GenericCSVData):
    """
    High-frequency tick data feed for Backtrader
    
    This feed loads tick-by-tick data from InfluxDB for maximum precision backtesting.
    """
    
    params = (
        ('datetime', 0),
        ('time', 0),
        ('bid', 1),
        ('ask', 2),
        ('volume', 3),
        ('openinterest', -1),
        ('dtformat', '%Y-%m-%d %H:%M:%S.%f'),
        ('tmformat', '%H:%M:%S.%f'),
    )
    
    def __init__(self, config: FeedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize InfluxDB connection
        self.influxdb_config = InfluxDBConfig(
            url=config.influxdb_url,
            token=config.influxdb_token,
            org=config.influxdb_org,
            bucket=config.influxdb_bucket
        )
        self.influxdb_writer = InfluxDBWriter(self.influxdb_config)
        
        # Load tick data from InfluxDB
        self.data_df = self._load_tick_data()
        
        if self.data_df is None or self.data_df.empty:
            raise ValueError(f"No tick data found for {config.symbol} in the specified time range")
        
        # Initialize parent class with the loaded data
        super().__init__(
            dataname=self.data_df,
            datetime=self.params.datetime,
            time=self.params.time,
            open=self.params.bid,  # Use bid as open
            high=self.params.ask,  # Use ask as high
            low=self.params.bid,   # Use bid as low
            close=self.params.ask, # Use ask as close
            volume=self.params.volume,
            openinterest=self.params.openinterest,
            dtformat=self.params.dtformat,
            tmformat=self.params.tmformat,
            fromdate=config.fromdate,
            todate=config.todate,
            compression=1  # No compression for tick data
        )
        
        self.logger.info(f"Loaded {len(self.data_df)} tick data points for {config.symbol}")
    
    def _load_tick_data(self) -> Optional[pd.DataFrame]:
        """Load tick data from InfluxDB"""
        try:
            # Connect to InfluxDB
            if not self.influxdb_writer.connect():
                self.logger.error("Failed to connect to InfluxDB")
                return None
            
            # Set date range
            start_date = self.config.start_date or self.config.fromdate
            end_date = self.config.end_date or self.config.todate or datetime.now()
            
            if start_date is None:
                # Default to last 7 days for tick data
                start_date = end_date - timedelta(days=7)
            
            # Query tick data from InfluxDB
            data = asyncio.run(self._query_tick_data(start_date, end_date))
            
            if not data:
                self.logger.warning(f"No tick data found for {self.config.symbol} in time range {start_date} to {end_date}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            
            # Sort by time
            df = df.sort_index()
            
            # Validate data quality
            df = self._validate_tick_data(df)
            
            # Reset index to make time a column
            df = df.reset_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load tick data from InfluxDB: {e}")
            return None
    
    async def _query_tick_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Query tick data from InfluxDB"""
        try:
            # Query tick data (assuming it's stored as 'tick_data' measurement)
            query = f'''
            from(bucket: "{self.influxdb_config.bucket}")
              |> range(start: {start_date.isoformat()}, stop: {end_date.isoformat()})
              |> filter(fn: (r) => r._measurement == "tick_data")
              |> filter(fn: (r) => r.instrument == "{self.config.symbol}")
              |> sort(columns: ["_time"])
            '''
            
            result = self.influxdb_writer.query_api.query(query)
            
            # Group by time to reconstruct tick data
            ticks = {}
            for table in result:
                for record in table.records:
                    time_key = record.get_time().isoformat()
                    if time_key not in ticks:
                        ticks[time_key] = {
                            'time': time_key,
                            'instrument': record.values.get('instrument')
                        }
                    
                    field = record.get_field()
                    value = record.values.get('_value')
                    if field in ['bid', 'ask', 'volume']:
                        ticks[time_key][field] = value
            
            data = list(ticks.values())
            data.sort(key=lambda x: x['time'])
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to query tick data: {e}")
            return []
    
    def _validate_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate tick data"""
        original_length = len(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Remove rows with invalid bid/ask relationships
        invalid_spread = df['ask'] <= df['bid']
        df = df[~invalid_spread]
        
        # Remove rows with negative prices
        df = df[(df['bid'] > 0) & (df['ask'] > 0)]
        
        # Remove rows with negative volume
        df = df[df['volume'] >= 0]
        
        cleaned_length = len(df)
        if cleaned_length < original_length:
            self.logger.warning(f"Removed {original_length - cleaned_length} invalid tick data points")
        
        return df


def create_data_feed(symbol: str, timeframe: str = '1m', 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    use_tick_data: bool = False) -> bt.feeds.GenericCSVData:
    """
    Factory function to create a data feed
    
    Args:
        symbol: Currency pair symbol (e.g., 'EURUSD')
        timeframe: Data timeframe ('1m', '5m', '1h', etc.)
        start_date: Start date for data
        end_date: End date for data
        use_tick_data: Whether to use tick data instead of OHLCV data
    
    Returns:
        Backtrader data feed
    """
    config = FeedConfig(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        fromdate=start_date,
        todate=end_date
    )
    
    if use_tick_data:
        return TickDataFeed(config)
    else:
        return InfluxDBDataFeed(config)
