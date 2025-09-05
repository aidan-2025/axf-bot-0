#!/usr/bin/env python3
"""
Tick Data Loader

Loads and preprocesses historical tick data for use with Backtrader.
Supports multiple data formats and sources.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TickDataFormat(Enum):
    """Supported tick data formats"""
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    INFLUXDB = "influxdb"
    MT4 = "mt4"
    METAQUOTES = "metaquotes"


@dataclass
class TickDataConfig:
    """Configuration for tick data loading"""
    
    # Data source configuration
    data_format: TickDataFormat = TickDataFormat.CSV
    data_path: Optional[str] = None
    symbol: str = "EURUSD"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Data processing configuration
    timezone: str = "UTC"
    resample_frequency: Optional[str] = None  # e.g., "1T" for 1-minute
    fill_missing: bool = True
    remove_duplicates: bool = True
    validate_data: bool = True
    
    # Performance configuration
    chunk_size: int = 100000  # For large datasets
    use_memory_mapping: bool = False
    compression: Optional[str] = None  # "gzip", "bz2", etc.
    
    # Column mapping (for different data formats)
    column_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.column_mapping is None:
            self.column_mapping = {
                'timestamp': 'timestamp',
                'bid': 'bid',
                'ask': 'ask',
                'volume': 'volume'
            }


@dataclass
class TickDataInfo:
    """Information about loaded tick data"""
    
    symbol: str
    start_time: datetime
    end_time: datetime
    total_ticks: int
    duration_days: float
    avg_spread: float
    min_spread: float
    max_spread: float
    data_quality_score: float
    missing_data_percentage: float
    duplicate_ticks: int
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_ticks': self.total_ticks,
            'duration_days': self.duration_days,
            'avg_spread': self.avg_spread,
            'min_spread': self.min_spread,
            'max_spread': self.max_spread,
            'data_quality_score': self.data_quality_score,
            'missing_data_percentage': self.missing_data_percentage,
            'duplicate_ticks': self.duplicate_ticks
        }


class TickDataLoader:
    """Loads and preprocesses historical tick data"""
    
    def __init__(self, config: TickDataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize data info
        self.data_info: Optional[TickDataInfo] = None
        
    def _validate_config(self):
        """Validate the configuration"""
        if not self.config.symbol:
            raise ValueError("Symbol must be specified")
        
        if self.config.data_format in [TickDataFormat.CSV, TickDataFormat.PARQUET, TickDataFormat.HDF5]:
            if not self.config.data_path:
                raise ValueError("Data path must be specified for file-based formats")
            if not os.path.exists(self.config.data_path):
                raise ValueError(f"Data path does not exist: {self.config.data_path}")
        
        if self.config.start_date and self.config.end_date:
            if self.config.start_date >= self.config.end_date:
                raise ValueError("Start date must be before end date")
    
    def load_data(self) -> pd.DataFrame:
        """Load tick data based on configuration"""
        
        self.logger.info(f"Loading tick data for {self.config.symbol} from {self.config.data_format.value}")
        
        try:
            # Load data based on format
            if self.config.data_format == TickDataFormat.CSV:
                data = self._load_csv()
            elif self.config.data_format == TickDataFormat.PARQUET:
                data = self._load_parquet()
            elif self.config.data_format == TickDataFormat.HDF5:
                data = self._load_hdf5()
            elif self.config.data_format == TickDataFormat.INFLUXDB:
                data = self._load_influxdb()
            elif self.config.data_format == TickDataFormat.MT4:
                data = self._load_mt4()
            elif self.config.data_format == TickDataFormat.METAQUOTES:
                data = self._load_metaquotes()
            else:
                raise ValueError(f"Unsupported data format: {self.config.data_format}")
            
            # Preprocess data
            data = self._preprocess_data(data)
            
            # Generate data info
            self.data_info = self._generate_data_info(data)
            
            self.logger.info(f"Successfully loaded {len(data)} ticks for {self.config.symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load tick data: {e}")
            raise
    
    def _load_csv(self) -> pd.DataFrame:
        """Load data from CSV file"""
        
        # Determine if we need to read in chunks
        if self.config.chunk_size > 0:
            chunks = []
            for chunk in pd.read_csv(
                self.config.data_path,
                chunksize=self.config.chunk_size,
                compression=self.config.compression
            ):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
        else:
            data = pd.read_csv(
                self.config.data_path,
                compression=self.config.compression
            )
        
        return data
    
    def _load_parquet(self) -> pd.DataFrame:
        """Load data from Parquet file"""
        data = pd.read_parquet(self.config.data_path)
        return data
    
    def _load_hdf5(self) -> pd.DataFrame:
        """Load data from HDF5 file"""
        data = pd.read_hdf(self.config.data_path, key='tick_data')
        return data
    
    def _load_influxdb(self) -> pd.DataFrame:
        """Load data from InfluxDB"""
        # This would integrate with the existing InfluxDB setup
        # For now, return empty DataFrame as placeholder
        self.logger.warning("InfluxDB loading not yet implemented")
        return pd.DataFrame()
    
    def _load_mt4(self) -> pd.DataFrame:
        """Load data from MT4 format"""
        # This would handle MT4's specific data format
        # For now, return empty DataFrame as placeholder
        self.logger.warning("MT4 loading not yet implemented")
        return pd.DataFrame()
    
    def _load_metaquotes(self) -> pd.DataFrame:
        """Load data from MetaQuotes format"""
        # This would handle MetaQuotes' specific data format
        # For now, return empty DataFrame as placeholder
        self.logger.warning("MetaQuotes loading not yet implemented")
        return pd.DataFrame()
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the loaded data"""
        
        if data.empty:
            return data
        
        # Apply column mapping
        if self.config.column_mapping:
            data = data.rename(columns=self.config.column_mapping)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'bid', 'ask']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Set timezone
        if data['timestamp'].dt.tz is None:
            data['timestamp'] = data['timestamp'].dt.tz_localize(self.config.timezone)
        else:
            data['timestamp'] = data['timestamp'].dt.tz_convert(self.config.timezone)
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        if self.config.remove_duplicates:
            initial_count = len(data)
            data = data.drop_duplicates(subset=['timestamp'], keep='last')
            removed_count = initial_count - len(data)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} duplicate timestamps")
        
        # Filter by date range
        if self.config.start_date:
            data = data[data['timestamp'] >= self.config.start_date]
        if self.config.end_date:
            data = data[data['timestamp'] <= self.config.end_date]
        
        # Fill missing values
        if self.config.fill_missing:
            data = self._fill_missing_data(data)
        
        # Validate data
        if self.config.validate_data:
            self._validate_data(data)
        
        # Resample if requested
        if self.config.resample_frequency:
            data = self._resample_data(data)
        
        return data
    
    def _fill_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing data using forward fill"""
        
        # Forward fill missing bid/ask prices
        data['bid'] = data['bid'].fillna(method='ffill')
        data['ask'] = data['ask'].fillna(method='ffill')
        
        # Fill volume with 0 if missing
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        
        return data
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate data quality"""
        
        # Check for negative prices
        if (data['bid'] <= 0).any() or (data['ask'] <= 0).any():
            raise ValueError("Found non-positive bid or ask prices")
        
        # Check for invalid spreads (ask < bid)
        invalid_spreads = data['ask'] < data['bid']
        if invalid_spreads.any():
            invalid_count = invalid_spreads.sum()
            self.logger.warning(f"Found {invalid_count} ticks with ask < bid")
        
        # Check for extreme spreads
        spreads = data['ask'] - data['bid']
        extreme_spreads = spreads > (spreads.quantile(0.99) * 10)
        if extreme_spreads.any():
            extreme_count = extreme_spreads.sum()
            self.logger.warning(f"Found {extreme_count} ticks with extreme spreads")
        
        # Check for missing timestamps
        if data['timestamp'].isna().any():
            missing_count = data['timestamp'].isna().sum()
            self.logger.warning(f"Found {missing_count} missing timestamps")
    
    def _resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample data to specified frequency"""
        
        # Set timestamp as index
        data_indexed = data.set_index('timestamp')
        
        # Resample using OHLCV aggregation
        resampled = data_indexed.resample(self.config.resample_frequency).agg({
            'bid': 'last',  # Use last bid price
            'ask': 'last',  # Use last ask price
            'volume': 'sum' if 'volume' in data.columns else 'count'
        })
        
        # Remove empty periods
        resampled = resampled.dropna()
        
        # Reset index
        resampled = resampled.reset_index()
        
        self.logger.info(f"Resampled data to {self.config.resample_frequency} frequency")
        
        return resampled
    
    def _generate_data_info(self, data: pd.DataFrame) -> TickDataInfo:
        """Generate information about the loaded data"""
        
        if data.empty:
            return TickDataInfo(
                symbol=self.config.symbol,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_ticks=0,
                duration_days=0.0,
                avg_spread=0.0,
                min_spread=0.0,
                max_spread=0.0,
                data_quality_score=0.0,
                missing_data_percentage=0.0,
                duplicate_ticks=0
            )
        
        # Calculate basic statistics
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        duration_days = (end_time - start_time).total_seconds() / 86400
        
        # Calculate spread statistics
        spreads = data['ask'] - data['bid']
        avg_spread = spreads.mean()
        min_spread = spreads.min()
        max_spread = spreads.max()
        
        # Calculate data quality metrics
        total_ticks = len(data)
        missing_data = data.isna().sum().sum()
        missing_data_percentage = (missing_data / (total_ticks * len(data.columns))) * 100
        
        # Calculate data quality score (0-100)
        quality_factors = []
        
        # No missing critical data
        if missing_data_percentage < 1.0:
            quality_factors.append(1.0)
        else:
            quality_factors.append(max(0, 1.0 - missing_data_percentage / 10.0))
        
        # Reasonable spread range
        if avg_spread > 0 and avg_spread < 0.01:  # Less than 1 pip for major pairs
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)
        
        # No extreme outliers
        if max_spread < avg_spread * 10:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.7)
        
        data_quality_score = np.mean(quality_factors) * 100
        
        return TickDataInfo(
            symbol=self.config.symbol,
            start_time=start_time,
            end_time=end_time,
            total_ticks=total_ticks,
            duration_days=duration_days,
            avg_spread=avg_spread,
            min_spread=min_spread,
            max_spread=max_spread,
            data_quality_score=data_quality_score,
            missing_data_percentage=missing_data_percentage,
            duplicate_ticks=0  # This would be calculated during preprocessing
        )
    
    def get_data_info(self) -> Optional[TickDataInfo]:
        """Get information about the loaded data"""
        return self.data_info
    
    def save_data(self, data: pd.DataFrame, output_path: str, format: TickDataFormat = TickDataFormat.CSV):
        """Save processed data to file"""
        
        self.logger.info(f"Saving data to {output_path} in {format.value} format")
        
        if format == TickDataFormat.CSV:
            data.to_csv(output_path, index=False, compression=self.config.compression)
        elif format == TickDataFormat.PARQUET:
            data.to_parquet(output_path, compression=self.config.compression)
        elif format == TickDataFormat.HDF5:
            data.to_hdf(output_path, key='tick_data', mode='w')
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        self.logger.info(f"Data saved successfully to {output_path}")
    
    def create_sample_data(self, symbol: str = "EURUSD", days: int = 7, 
                          tick_frequency: str = "1S") -> pd.DataFrame:
        """Create sample tick data for testing"""
        
        self.logger.info(f"Creating sample tick data for {symbol} ({days} days)")
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=days)
        timestamps = pd.date_range(
            start=start_time,
            periods=days * 24 * 60 * 60,  # 1 second intervals
            freq=tick_frequency,
            tz=self.config.timezone
        )
        
        # Generate realistic price data
        base_price = 1.1000 if "USD" in symbol else 1.0000
        price_changes = np.random.normal(0, 0.0001, len(timestamps)).cumsum()
        mid_prices = base_price + price_changes
        
        # Generate realistic spreads (1-3 pips)
        spreads = np.random.uniform(0.0001, 0.0003, len(timestamps))
        
        # Calculate bid and ask prices
        bid_prices = mid_prices - spreads / 2
        ask_prices = mid_prices + spreads / 2
        
        # Generate volume data
        volumes = np.random.randint(1, 100, len(timestamps))
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'bid': bid_prices,
            'ask': ask_prices,
            'volume': volumes
        })
        
        self.logger.info(f"Created {len(data)} sample ticks")
        return data
