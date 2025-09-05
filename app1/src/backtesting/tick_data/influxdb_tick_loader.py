#!/usr/bin/env python3
"""
InfluxDB Tick Data Loader

Loads historical tick data from InfluxDB for use with Backtrader.
Integrates with the existing InfluxDB setup.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import asyncio

try:
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    InfluxDBClient = None
    SYNCHRONOUS = None

from .tick_data_loader import TickDataLoader, TickDataConfig, TickDataFormat, TickDataInfo

logger = logging.getLogger(__name__)


class InfluxDBTickLoader:
    """Loads tick data from InfluxDB"""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        """
        Initialize InfluxDB tick loader.
        
        Args:
            url: InfluxDB server URL
            token: InfluxDB authentication token
            org: InfluxDB organization
            bucket: InfluxDB bucket name
        """
        if InfluxDBClient is None:
            raise ImportError("influxdb_client is required for InfluxDB integration")
        
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        
        # Initialize InfluxDB client
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        
        logger.info(f"Initialized InfluxDB tick loader for bucket: {bucket}")
    
    def load_tick_data(self, symbol: str, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load tick data from InfluxDB.
        
        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with tick data
        """
        
        logger.info(f"Loading tick data for {symbol} from InfluxDB")
        
        # Build query
        query = self._build_query(symbol, start_date, end_date, limit)
        
        try:
            # Execute query
            result = self.query_api.query(query, org=self.org)
            
            # Convert to DataFrame
            data = self._result_to_dataframe(result)
            
            if data.empty:
                logger.warning(f"No tick data found for {symbol}")
                return data
            
            # Process data
            data = self._process_tick_data(data, symbol)
            
            logger.info(f"Loaded {len(data)} ticks for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load tick data from InfluxDB: {e}")
            raise
    
    def _build_query(self, symbol: str, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None, limit: Optional[int] = None) -> str:
        """Build InfluxDB query for tick data"""
        
        # Base query
        query = f'from(bucket: "{self.bucket}")'
        
        # Add time range
        if start_date:
            start_rfc3339 = start_date.isoformat() + "Z"
            query += f' |> range(start: {start_rfc3339}'
        else:
            query += ' |> range(start: -30d'  # Default to last 30 days
        
        if end_date:
            end_rfc3339 = end_date.isoformat() + "Z"
            query += f', stop: {end_rfc3339}'
        
        query += ')'
        
        # Add filter for symbol
        query += f' |> filter(fn: (r) => r["symbol"] == "{symbol}")'
        
        # Add filter for measurement type
        query += ' |> filter(fn: (r) => r["_measurement"] == "tick_data")'
        
        # Add field filters
        query += ' |> filter(fn: (r) => r["_field"] == "bid" or r["_field"] == "ask" or r["_field"] == "volume")'
        
        # Add limit if specified
        if limit:
            query += f' |> limit(n: {limit})'
        
        # Sort by time
        query += ' |> sort(columns: ["_time"])'
        
        logger.debug(f"InfluxDB query: {query}")
        return query
    
    def _result_to_dataframe(self, result) -> pd.DataFrame:
        """Convert InfluxDB query result to DataFrame"""
        
        records = []
        
        for table in result:
            for record in table.records:
                records.append({
                    'timestamp': record.get_time(),
                    'field': record.get_field(),
                    'value': record.get_value(),
                    'symbol': record.values.get('symbol', ''),
                    'measurement': record.get_measurement()
                })
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Pivot to get bid, ask, volume as columns
        df_pivot = df.pivot_table(
            index=['timestamp', 'symbol', 'measurement'],
            columns='field',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        df_pivot.columns.name = None
        
        return df_pivot
    
    def _process_tick_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process tick data for Backtrader compatibility"""
        
        if data.empty:
            return data
        
        # Ensure required columns exist
        required_columns = ['bid', 'ask']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            # Fill missing columns with NaN
            for col in missing_columns:
                data[col] = np.nan
        
        # Add volume if missing
        if 'volume' not in data.columns:
            data['volume'] = 1
        
        # Remove rows with missing critical data
        data = data.dropna(subset=['bid', 'ask'])
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Validate data
        self._validate_tick_data(data)
        
        return data
    
    def _validate_tick_data(self, data: pd.DataFrame):
        """Validate tick data quality"""
        
        if data.empty:
            return
        
        # Check for valid prices
        if (data['bid'] <= 0).any() or (data['ask'] <= 0).any():
            invalid_count = ((data['bid'] <= 0) | (data['ask'] <= 0)).sum()
            logger.warning(f"Found {invalid_count} ticks with non-positive prices")
        
        # Check for invalid spreads
        invalid_spreads = data['ask'] < data['bid']
        if invalid_spreads.any():
            invalid_count = invalid_spreads.sum()
            logger.warning(f"Found {invalid_count} ticks with ask < bid")
    
    def save_tick_data(self, data: pd.DataFrame, symbol: str, 
                      measurement: str = "tick_data") -> bool:
        """
        Save tick data to InfluxDB.
        
        Args:
            data: DataFrame with tick data
            symbol: Currency pair symbol
            measurement: InfluxDB measurement name
            
        Returns:
            True if successful, False otherwise
        """
        
        logger.info(f"Saving {len(data)} ticks for {symbol} to InfluxDB")
        
        try:
            # Prepare data for InfluxDB
            points = self._prepare_data_points(data, symbol, measurement)
            
            # Write to InfluxDB
            self.write_api.write(bucket=self.bucket, record=points)
            
            logger.info(f"Successfully saved tick data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tick data to InfluxDB: {e}")
            return False
    
    def _prepare_data_points(self, data: pd.DataFrame, symbol: str, 
                           measurement: str) -> List[Dict]:
        """Prepare data points for InfluxDB"""
        
        points = []
        
        for _, row in data.iterrows():
            timestamp = row['timestamp']
            
            # Create point for bid
            if 'bid' in row and pd.notna(row['bid']):
                points.append({
                    "measurement": measurement,
                    "tags": {
                        "symbol": symbol,
                        "field": "bid"
                    },
                    "fields": {
                        "value": float(row['bid'])
                    },
                    "time": timestamp
                })
            
            # Create point for ask
            if 'ask' in row and pd.notna(row['ask']):
                points.append({
                    "measurement": measurement,
                    "tags": {
                        "symbol": symbol,
                        "field": "ask"
                    },
                    "fields": {
                        "value": float(row['ask'])
                    },
                    "time": timestamp
                })
            
            # Create point for volume
            if 'volume' in row and pd.notna(row['volume']):
                points.append({
                    "measurement": measurement,
                    "tags": {
                        "symbol": symbol,
                        "field": "volume"
                    },
                    "fields": {
                        "value": float(row['volume'])
                    },
                    "time": timestamp
                })
        
        return points
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in InfluxDB"""
        
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -30d)
        |> filter(fn: (r) => r["_measurement"] == "tick_data")
        |> filter(fn: (r) => r["_field"] == "bid")
        |> group(columns: ["symbol"])
        |> distinct(column: "symbol")
        |> yield(name: "symbols")
        '''
        
        try:
            result = self.query_api.query(query, org=self.org)
            symbols = []
            
            for table in result:
                for record in table.records:
                    symbol = record.values.get('symbol')
                    if symbol:
                        symbols.append(symbol)
            
            return list(set(symbols))
            
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []
    
    def get_data_info(self, symbol: str) -> Dict[str, any]:
        """Get information about available data for a symbol"""
        
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -30d)
        |> filter(fn: (r) => r["_measurement"] == "tick_data")
        |> filter(fn: (r) => r["symbol"] == "{symbol}")
        |> filter(fn: (r) => r["_field"] == "bid")
        |> group()
        |> count()
        |> yield(name: "count")
        '''
        
        try:
            result = self.query_api.query(query, org=self.org)
            
            count = 0
            for table in result:
                for record in table.records:
                    count = record.get_value()
                    break
            
            return {
                'symbol': symbol,
                'total_ticks': count,
                'available': count > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get data info for {symbol}: {e}")
            return {
                'symbol': symbol,
                'total_ticks': 0,
                'available': False
            }
    
    def close(self):
        """Close InfluxDB connection"""
        if self.client:
            self.client.close()
            logger.info("InfluxDB connection closed")


class InfluxDBTickDataConfig(TickDataConfig):
    """Configuration for InfluxDB tick data loading"""
    
    def __init__(self, symbol: str, influxdb_url: str, influxdb_token: str,
                 influxdb_org: str, influxdb_bucket: str, **kwargs):
        
        super().__init__(symbol=symbol, **kwargs)
        
        self.data_format = TickDataFormat.INFLUXDB
        self.influxdb_url = influxdb_url
        self.influxdb_token = influxdb_token
        self.influxdb_org = influxdb_org
        self.influxdb_bucket = influxdb_bucket


def create_influxdb_loader_from_env() -> InfluxDBTickLoader:
    """Create InfluxDB loader from environment variables"""
    
    import os
    
    url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    token = os.getenv('INFLUXDB_TOKEN', '')
    org = os.getenv('INFLUXDB_ORG', 'axf-bot')
    bucket = os.getenv('INFLUXDB_BUCKET', 'tick_data')
    
    if not token:
        raise ValueError("INFLUXDB_TOKEN environment variable is required")
    
    return InfluxDBTickLoader(url, token, org, bucket)

