#!/usr/bin/env python3
"""
OANDA v20 API Client
Provides real-time and historical forex data from OANDA
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class Granularity(Enum):
    """OANDA granularity options"""
    S5 = "S5"      # 5 seconds
    S10 = "S10"    # 10 seconds
    S15 = "S15"    # 15 seconds
    S30 = "S30"    # 30 seconds
    M1 = "M1"      # 1 minute
    M2 = "M2"      # 2 minutes
    M4 = "M4"      # 4 minutes
    M5 = "M5"      # 5 minutes
    M10 = "M10"    # 10 minutes
    M15 = "M15"    # 15 minutes
    M30 = "M30"    # 30 minutes
    H1 = "H1"      # 1 hour
    H2 = "H2"      # 2 hours
    H3 = "H3"      # 3 hours
    H4 = "H4"      # 4 hours
    H6 = "H6"      # 6 hours
    H8 = "H8"      # 8 hours
    H12 = "H12"    # 12 hours
    D = "D"        # 1 day
    W = "W"        # 1 week
    M = "M"        # 1 month

@dataclass
class CandleData:
    """Represents a single candle/bar data point"""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool

@dataclass
class PriceData:
    """Represents real-time price data"""
    instrument: str
    time: datetime
    bid: float
    ask: float
    spread: float

class OANDAClient:
    """OANDA v20 API Client for forex data"""
    
    def __init__(self, 
                 api_key: str,
                 account_id: str,
                 environment: str = "practice",  # "practice" or "live"
                 base_url: Optional[str] = None):
        """
        Initialize OANDA client
        
        Args:
            api_key: OANDA API key
            account_id: OANDA account ID
            environment: "practice" or "live"
            base_url: Custom base URL (optional)
        """
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        
        # Set base URLs
        if base_url:
            self.base_url = base_url
        else:
            if environment == "live":
                self.base_url = "https://api-fxtrade.oanda.com"
            else:
                self.base_url = "https://api-fxpractice.oanda.com"
        
        # Rate limiting
        self.rate_limit_remaining = 100
        self.rate_limit_reset = time.time()
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        
        # Headers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
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
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Check if we need to wait
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, 
                           method: str, 
                           endpoint: str, 
                           params: Optional[Dict] = None,
                           data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling and rate limiting"""
        await self._rate_limit_check()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data
            ) as response:
                
                # Update rate limit info
                if 'X-RateLimit-Remaining' in response.headers:
                    self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                if 'X-RateLimit-Reset' in response.headers:
                    self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, endpoint, params, data)
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    raise Exception(f"OANDA API error: {response.status} - {error_text}")
        
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        endpoint = f"/v3/accounts/{self.account_id}"
        return await self._make_request("GET", endpoint)
    
    async def get_instruments(self) -> List[Dict[str, Any]]:
        """Get available trading instruments"""
        endpoint = f"/v3/accounts/{self.account_id}/instruments"
        response = await self._make_request("GET", endpoint)
        return response.get('instruments', [])
    
    async def get_pricing(self, instruments: List[str]) -> Dict[str, Any]:
        """Get real-time pricing for instruments"""
        endpoint = f"/v3/accounts/{self.account_id}/pricing"
        params = {
            "instruments": ",".join(instruments)
        }
        return await self._make_request("GET", endpoint, params=params)
    
    async def get_candles(self, 
                         instrument: str,
                         granularity: Granularity,
                         count: Optional[int] = None,
                         from_time: Optional[datetime] = None,
                         to_time: Optional[datetime] = None) -> List[CandleData]:
        """
        Get historical candle data
        
        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: Time granularity
            count: Number of candles to retrieve
            from_time: Start time
            to_time: End time
        """
        endpoint = f"/v3/instruments/{instrument}/candles"
        
        params = {
            "granularity": granularity.value,
            "price": "M"  # Mid prices
        }
        
        if count:
            params["count"] = count
        if from_time:
            params["from"] = from_time.isoformat() + "Z"
        if to_time:
            params["to"] = to_time.isoformat() + "Z"
        
        response = await self._make_request("GET", endpoint, params=params)
        
        candles = []
        for candle_data in response.get('candles', []):
            if candle_data.get('complete', False):
                candle = CandleData(
                    time=datetime.fromisoformat(candle_data['time'].replace('Z', '+00:00')),
                    open=float(candle_data['mid']['o']),
                    high=float(candle_data['mid']['h']),
                    low=float(candle_data['mid']['l']),
                    close=float(candle_data['mid']['c']),
                    volume=int(candle_data['volume']),
                    complete=candle_data['complete']
                )
                candles.append(candle)
        
        return candles
    
    async def stream_prices(self, 
                           instruments: List[str],
                           callback: callable) -> None:
        """
        Stream real-time price data
        
        Args:
            instruments: List of instruments to stream
            callback: Function to call with price data
        """
        endpoint = f"/v3/accounts/{self.account_id}/pricing/stream"
        params = {
            "instruments": ",".join(instruments)
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Stream connection failed: {response.status}")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'price' in data:
                                price_data = PriceData(
                                    instrument=data['price']['instrument'],
                                    time=datetime.fromisoformat(
                                        data['price']['time'].replace('Z', '+00:00')
                                    ),
                                    bid=float(data['price']['bids'][0]['price']),
                                    ask=float(data['price']['asks'][0]['price']),
                                    spread=float(data['price']['asks'][0]['price']) - 
                                          float(data['price']['bids'][0]['price'])
                                )
                                await callback(price_data)
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing price data: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Price stream error: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            account_info = await self.get_account_info()
            return {
                "status": "healthy",
                "account_id": account_info.get('account', {}).get('id'),
                "rate_limit_remaining": self.rate_limit_remaining,
                "rate_limit_reset": self.rate_limit_reset
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Example usage and testing
async def test_oanda_client():
    """Test OANDA client functionality"""
    import os
    
    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    
    if not api_key or not account_id:
        print("Please set OANDA_API_KEY and OANDA_ACCOUNT_ID environment variables")
        return
    
    async with OANDAClient(api_key, account_id) as client:
        # Test account info
        account_info = await client.get_account_info()
        print(f"Account: {account_info['account']['id']}")
        
        # Test instruments
        instruments = await client.get_instruments()
        print(f"Available instruments: {len(instruments)}")
        
        # Test pricing
        pricing = await client.get_pricing(["EUR_USD", "GBP_USD"])
        print(f"Pricing data: {len(pricing.get('prices', []))} instruments")
        
        # Test candles
        candles = await client.get_candles(
            "EUR_USD", 
            Granularity.M1, 
            count=10
        )
        print(f"Retrieved {len(candles)} candles")

if __name__ == "__main__":
    asyncio.run(test_oanda_client())
