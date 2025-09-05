#!/usr/bin/env python3
"""
FXCM REST API Client
Provides real-time and historical forex data from FXCM as backup
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

class TimeFrame(Enum):
    """FXCM timeframe options"""
    M1 = "m1"      # 1 minute
    M5 = "m5"      # 5 minutes
    M15 = "m15"    # 15 minutes
    M30 = "m30"    # 30 minutes
    H1 = "H1"      # 1 hour
    H2 = "H2"      # 2 hours
    H3 = "H3"      # 3 hours
    H4 = "H4"      # 4 hours
    H6 = "H6"      # 6 hours
    H8 = "H8"      # 8 hours
    H12 = "H12"    # 12 hours
    D1 = "D1"      # 1 day
    W1 = "W1"      # 1 week
    MN1 = "MN1"    # 1 month

@dataclass
class FXCMCandleData:
    """Represents a single candle/bar data point from FXCM"""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class FXCMPriceData:
    """Represents real-time price data from FXCM"""
    symbol: str
    time: datetime
    bid: float
    ask: float
    spread: float

class FXCMClient:
    """FXCM REST API Client for forex data"""
    
    def __init__(self, 
                 token: str,
                 environment: str = "demo",  # "demo" or "live"
                 base_url: Optional[str] = None):
        """
        Initialize FXCM client
        
        Args:
            token: FXCM API token
            environment: "demo" or "live"
            base_url: Custom base URL (optional)
        """
        self.token = token
        self.environment = environment
        
        # Set base URLs
        if base_url:
            self.base_url = base_url
        else:
            if environment == "live":
                self.base_url = "https://api.fxcm.com"
            else:
                self.base_url = "https://api-demo.fxcm.com"
        
        # Rate limiting
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = time.time()
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        
        # Headers
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
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
                    raise Exception(f"FXCM API error: {response.status} - {error_text}")
        
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        endpoint = "/v1/accounts"
        return await self._make_request("GET", endpoint)
    
    async def get_symbols(self) -> List[Dict[str, Any]]:
        """Get available trading symbols"""
        endpoint = "/v1/symbols"
        response = await self._make_request("GET", endpoint)
        return response.get('symbols', [])
    
    async def get_offers(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get real-time offers for symbols"""
        endpoint = "/v1/offers"
        params = {}
        if symbols:
            params["symbols"] = ",".join(symbols)
        return await self._make_request("GET", endpoint, params=params)
    
    async def get_historical_data(self, 
                                 symbol: str,
                                 timeframe: TimeFrame,
                                 count: Optional[int] = None,
                                 from_time: Optional[datetime] = None,
                                 to_time: Optional[datetime] = None) -> List[FXCMCandleData]:
        """
        Get historical candle data
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Time timeframe
            count: Number of candles to retrieve
            from_time: Start time
            to_time: End time
        """
        endpoint = f"/v1/candles/{symbol}"
        
        params = {
            "timeframe": timeframe.value
        }
        
        if count:
            params["count"] = count
        if from_time:
            params["from"] = int(from_time.timestamp() * 1000)
        if to_time:
            params["to"] = int(to_time.timestamp() * 1000)
        
        response = await self._make_request("GET", endpoint, params=params)
        
        candles = []
        for candle_data in response.get('candles', []):
            candle = FXCMCandleData(
                time=datetime.fromtimestamp(candle_data['time'] / 1000),
                open=float(candle_data['open']),
                high=float(candle_data['high']),
                low=float(candle_data['low']),
                close=float(candle_data['close']),
                volume=int(candle_data.get('volume', 0))
            )
            candles.append(candle)
        
        return candles
    
    async def stream_offers(self, 
                           symbols: List[str],
                           callback: callable) -> None:
        """
        Stream real-time offer data
        
        Args:
            symbols: List of symbols to stream
            callback: Function to call with offer data
        """
        endpoint = "/v1/offers/stream"
        params = {
            "symbols": ",".join(symbols)
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
                            if 'offer' in data:
                                offer_data = FXCMPriceData(
                                    symbol=data['offer']['symbol'],
                                    time=datetime.fromtimestamp(
                                        data['offer']['time'] / 1000
                                    ),
                                    bid=float(data['offer']['bid']),
                                    ask=float(data['offer']['ask']),
                                    spread=float(data['offer']['ask']) - 
                                          float(data['offer']['bid'])
                                )
                                await callback(offer_data)
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing offer data: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Offer stream error: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            account_info = await self.get_account_info()
            return {
                "status": "healthy",
                "accounts": len(account_info.get('accounts', [])),
                "rate_limit_remaining": self.rate_limit_remaining,
                "rate_limit_reset": self.rate_limit_reset
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Example usage and testing
async def test_fxcm_client():
    """Test FXCM client functionality"""
    import os
    
    token = os.getenv("FXCM_TOKEN")
    
    if not token:
        print("Please set FXCM_TOKEN environment variable")
        return
    
    async with FXCMClient(token) as client:
        # Test account info
        account_info = await client.get_account_info()
        print(f"Accounts: {len(account_info.get('accounts', []))}")
        
        # Test symbols
        symbols = await client.get_symbols()
        print(f"Available symbols: {len(symbols)}")
        
        # Test offers
        offers = await client.get_offers(["EURUSD", "GBPUSD"])
        print(f"Offers data: {len(offers.get('offers', []))} symbols")
        
        # Test historical data
        candles = await client.get_historical_data(
            "EURUSD", 
            TimeFrame.M1, 
            count=10
        )
        print(f"Retrieved {len(candles)} candles")

if __name__ == "__main__":
    asyncio.run(test_fxcm_client())
