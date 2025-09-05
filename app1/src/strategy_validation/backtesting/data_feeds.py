#!/usr/bin/env python3
"""
Data Feeds for Backtrader Integration

Provides data feed implementations for various data sources.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
import json

logger = logging.getLogger(__name__)


@dataclass
class DataFeedConfig:
    """Configuration for data feeds"""
    
    # Data source configuration
    source: str = 'influxdb'  # 'influxdb', 'csv', 'mock'
    host: str = 'localhost'
    port: int = 8086
    database: str = 'forex_data'
    username: str = ''
    password: str = ''
    token: str = ''
    
    # Data filtering
    symbols: List[str] = None
    timeframes: List[str] = None
    
    # Cache configuration
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        if self.timeframes is None:
            self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']


class ForexDataFeed:
    """Forex data feed for Backtrader integration"""
    
    def __init__(self, config: DataFeedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {} if config.cache_enabled else None
        
        self.logger.info(f"ForexDataFeed initialized with source: {config.source}")
    
    async def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                      timeframe: str = '1m') -> Optional[pd.DataFrame]:
        """Get forex data for a symbol and timeframe"""
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        # Check cache first
        if self.cache is not None and cache_key in self.cache:
            self.logger.debug(f"Retrieved data from cache: {cache_key}")
            return self.cache[cache_key]
        
        try:
            # Get data based on source
            if self.config.source == 'influxdb':
                data = await self._get_influxdb_data(symbol, start_date, end_date, timeframe)
            elif self.config.source == 'csv':
                data = await self._get_csv_data(symbol, start_date, end_date, timeframe)
            elif self.config.source == 'mock':
                data = await self._get_mock_data(symbol, start_date, end_date, timeframe)
            else:
                raise ValueError(f"Unsupported data source: {self.config.source}")
            
            # Cache the data
            if self.cache is not None and data is not None:
                self.cache[cache_key] = data
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get data for {symbol}: {e}")
            return None
    
    async def _get_influxdb_data(self, symbol: str, start_date: datetime, 
                               end_date: datetime, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data from InfluxDB"""
        try:
            # This would integrate with your existing InfluxDB setup
            # For now, return mock data
            self.logger.info(f"Getting InfluxDB data for {symbol} ({timeframe})")
            return await self._get_mock_data(symbol, start_date, end_date, timeframe)
            
        except Exception as e:
            self.logger.error(f"InfluxDB data retrieval failed: {e}")
            return None
    
    async def _get_csv_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data from CSV files"""
        try:
            # This would read from CSV files
            # For now, return mock data
            self.logger.info(f"Getting CSV data for {symbol} ({timeframe})")
            return await self._get_mock_data(symbol, start_date, end_date, timeframe)
            
        except Exception as e:
            self.logger.error(f"CSV data retrieval failed: {e}")
            return None
    
    async def _get_mock_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Generate mock forex data for testing"""
        self.logger.info(f"Generating mock data for {symbol} ({timeframe})")
        
        # Calculate time delta based on timeframe
        time_deltas = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        
        delta = time_deltas.get(timeframe, timedelta(minutes=1))
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq=delta)
        
        # Generate mock OHLCV data
        np.random.seed(42)  # For reproducible results
        
        # Base price for different symbols
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.2500,
            'USDJPY': 110.00,
            'AUDUSD': 0.7500,
            'USDCAD': 1.3000
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate price data with realistic forex characteristics
        n_periods = len(timestamps)
        returns = np.random.normal(0, 0.0001, n_periods)  # Small daily returns
        prices = [base_price]
        
        for i in range(1, n_periods):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC from price
            volatility = 0.0002  # 0.02% volatility
            
            high = price * (1 + abs(np.random.normal(0, volatility)))
            low = price * (1 - abs(np.random.normal(0, volatility)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume (higher during volatile periods)
            volume = int(np.random.exponential(1000) * (1 + abs(returns[i]) * 100))
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Generated {len(df)} bars of mock data for {symbol}")
        return df
    
    async def get_multiple_symbols(self, symbols: List[str], start_date: datetime, 
                                 end_date: datetime, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols"""
        results = {}
        
        tasks = []
        for symbol in symbols:
            task = self.get_data(symbol, start_date, end_date, timeframe)
            tasks.append((symbol, task))
        
        for symbol, task in tasks:
            try:
                data = await task
                if data is not None:
                    results[symbol] = data
            except Exception as e:
                self.logger.error(f"Failed to get data for {symbol}: {e}")
        
        return results
    
    def clear_cache(self):
        """Clear the data cache"""
        if self.cache is not None:
            self.cache.clear()
            self.logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache is None:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'size': len(self.cache),
            'keys': list(self.cache.keys())
        }

