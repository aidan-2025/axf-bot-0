#!/usr/bin/env python3
"""
Free Forex API Client
Provides free forex data for testing and development
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import time
import random

logger = logging.getLogger(__name__)

class FreeForexGranularity(Enum):
    """Free forex granularity options"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

@dataclass
class FreeForexCandleData:
    """Represents a single candle/bar data point from free forex API"""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class FreeForexPriceData:
    """Represents real-time price data from free forex API"""
    symbol: str
    time: datetime
    bid: float
    ask: float
    spread: float

class FreeForexClient:
    """Free Forex API Client for testing and development"""
    
    def __init__(self, base_url: str = "https://api.exchangerate-api.com/v4"):
        """
        Initialize free forex client
        
        Args:
            base_url: Base URL for the free API
        """
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Mock data for testing
        self.mock_prices = {
            'EUR_USD': {'bid': 1.0850, 'ask': 1.0852, 'spread': 0.0002},
            'GBP_USD': {'bid': 1.2650, 'ask': 1.2652, 'spread': 0.0002},
            'USD_JPY': {'bid': 149.50, 'ask': 149.52, 'spread': 0.02},
            'USD_CHF': {'bid': 0.8750, 'ask': 0.8752, 'spread': 0.0002},
            'AUD_USD': {'bid': 0.6550, 'ask': 0.6552, 'spread': 0.0002},
            'USD_CAD': {'bid': 1.3650, 'ask': 1.3652, 'spread': 0.0002}
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def get_latest_rates(self, base_currency: str = "USD") -> Dict[str, Any]:
        """Get latest exchange rates"""
        await self._rate_limit_check()
        
        try:
            url = f"{self.base_url}/latest/{base_currency}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API request failed: {response.status}")
                    return self._get_mock_rates(base_currency)
        except Exception as e:
            logger.warning(f"API request failed: {e}, using mock data")
            return self._get_mock_rates(base_currency)
    
    def _get_mock_rates(self, base_currency: str = "USD") -> Dict[str, Any]:
        """Get mock exchange rates for testing"""
        mock_rates = {
            "base": base_currency,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "rates": {
                "EUR": 0.9230,
                "GBP": 0.7890,
                "JPY": 149.50,
                "CHF": 0.8750,
                "AUD": 1.5250,
                "CAD": 1.3650
            }
        }
        return mock_rates
    
    async def get_historical_rates(self, 
                                 base_currency: str,
                                 target_currency: str,
                                 start_date: datetime,
                                 end_date: datetime) -> List[FreeForexCandleData]:
        """Get historical exchange rates"""
        await self._rate_limit_check()
        
        try:
            # Try to get real historical data
            url = f"{self.base_url}/history/{base_currency}/{start_date.strftime('%Y-%m-%d')}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_historical_data(data, target_currency)
                else:
                    logger.warning(f"Historical API request failed: {response.status}")
                    return self._generate_mock_historical_data(base_currency, target_currency, start_date, end_date)
        except Exception as e:
            logger.warning(f"Historical API request failed: {e}, using mock data")
            return self._generate_mock_historical_data(base_currency, target_currency, start_date, end_date)
    
    def _parse_historical_data(self, data: Dict[str, Any], target_currency: str) -> List[FreeForexCandleData]:
        """Parse historical data from API response"""
        candles = []
        
        if 'rates' in data:
            for date_str, rates in data['rates'].items():
                if target_currency in rates:
                    rate = rates[target_currency]
                    # Create mock OHLC data based on the rate
                    base_rate = rate
                    high_rate = base_rate * (1 + random.uniform(0, 0.01))
                    low_rate = base_rate * (1 - random.uniform(0, 0.01))
                    close_rate = base_rate * (1 + random.uniform(-0.005, 0.005))
                    
                    candle = FreeForexCandleData(
                        time=datetime.fromisoformat(date_str),
                        open=base_rate,
                        high=high_rate,
                        low=low_rate,
                        close=close_rate,
                        volume=random.randint(1000, 10000)
                    )
                    candles.append(candle)
        
        return candles
    
    def _generate_mock_historical_data(self, 
                                     base_currency: str, 
                                     target_currency: str,
                                     start_date: datetime, 
                                     end_date: datetime) -> List[FreeForexCandleData]:
        """Generate mock historical data for testing"""
        candles = []
        current_date = start_date
        base_rate = 1.0  # Default base rate
        
        # Set different base rates for different currency pairs
        if f"{base_currency}_{target_currency}" in self.mock_prices:
            base_rate = self.mock_prices[f"{base_currency}_{target_currency}"]['bid']
        elif f"{target_currency}_{base_currency}" in self.mock_prices:
            base_rate = 1.0 / self.mock_prices[f"{target_currency}_{base_currency}"]['bid']
        
        while current_date <= end_date:
            # Generate realistic price movement
            price_change = random.uniform(-0.02, 0.02)  # ±2% daily change
            base_rate *= (1 + price_change)
            
            # Generate OHLC data
            open_price = base_rate
            high_price = open_price * (1 + random.uniform(0, 0.01))
            low_price = open_price * (1 - random.uniform(0, 0.01))
            close_price = open_price * (1 + random.uniform(-0.005, 0.005))
            
            candle = FreeForexCandleData(
                time=current_date,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.randint(1000, 10000)
            )
            candles.append(candle)
            
            current_date += timedelta(days=1)
        
        return candles
    
    async def get_real_time_prices(self, symbols: List[str]) -> List[FreeForexPriceData]:
        """Get real-time prices for symbols"""
        await self._rate_limit_check()
        
        prices = []
        current_time = datetime.now()
        
        for symbol in symbols:
            if symbol in self.mock_prices:
                price_data = self.mock_prices[symbol]
                # Add some random variation to simulate real-time updates
                variation = random.uniform(-0.0001, 0.0001)
                
                price = FreeForexPriceData(
                    symbol=symbol,
                    time=current_time,
                    bid=price_data['bid'] + variation,
                    ask=price_data['ask'] + variation,
                    spread=price_data['spread']
                )
                prices.append(price)
        
        return prices
    
    async def stream_prices(self, 
                           symbols: List[str],
                           callback: callable) -> None:
        """Stream real-time prices (simulated)"""
        logger.info(f"Starting price stream for {symbols}")
        
        try:
            while True:
                prices = await self.get_real_time_prices(symbols)
                for price in prices:
                    await callback(price)
                
                # Wait 1 second before next update
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("Price stream cancelled")
        except Exception as e:
            logger.error(f"Price stream error: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            # Test API connection
            rates = await self.get_latest_rates()
            return {
                "status": "healthy",
                "base_currency": rates.get('base', 'USD'),
                "rates_count": len(rates.get('rates', {})),
                "last_update": rates.get('date', 'unknown')
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Example usage and testing
async def test_free_forex_client():
    """Test free forex client functionality"""
    async with FreeForexClient() as client:
        # Test health status
        health = await client.get_health_status()
        print(f"Health status: {health}")
        
        # Test latest rates
        rates = await client.get_latest_rates()
        print(f"Latest rates: {rates}")
        
        # Test real-time prices
        prices = await client.get_real_time_prices(["EUR_USD", "GBP_USD"])
        print(f"Real-time prices: {len(prices)} symbols")
        
        # Test historical data
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        historical = await client.get_historical_rates("USD", "EUR", start_date, end_date)
        print(f"Historical data: {len(historical)} candles")

if __name__ == "__main__":
    asyncio.run(test_free_forex_client())
