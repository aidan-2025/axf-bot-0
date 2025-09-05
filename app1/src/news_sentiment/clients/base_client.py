#!/usr/bin/env python3
"""
Base News Client
Abstract base class for all news API clients
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import httpx
from dataclasses import dataclass

from ..models import NewsArticle, EconomicEvent, NewsSource

logger = logging.getLogger(__name__)

@dataclass
class RateLimiter:
    """Rate limiter for API requests"""
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    def __post_init__(self):
        self.minute_requests: List[float] = []
        self.hour_requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            
            # Clean old requests
            self.minute_requests = [req_time for req_time in self.minute_requests if now - req_time < 60]
            self.hour_requests = [req_time for req_time in self.hour_requests if now - req_time < 3600]
            
            # Check limits
            if len(self.minute_requests) >= self.max_requests_per_minute:
                sleep_time = 60 - (now - self.minute_requests[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached for minute, sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            
            if len(self.hour_requests) >= self.max_requests_per_hour:
                sleep_time = 3600 - (now - self.hour_requests[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached for hour, sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            
            # Record this request
            self.minute_requests.append(now)
            self.hour_requests.append(now)

class BaseNewsClient(ABC):
    """Abstract base class for news API clients"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 rate_limiter: Optional[RateLimiter] = None,
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """Initialize base news client"""
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limiter = rate_limiter or RateLimiter()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # HTTP client configuration
        self.client_config = {
            'timeout': httpx.Timeout(timeout),
            'limits': httpx.Limits(max_keepalive_connections=20, max_connections=100)
        }
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'last_request': None
        }
    
    @property
    @abstractmethod
    def source(self) -> NewsSource:
        """News source identifier"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Client name"""
        pass
    
    async def _make_request(self, 
                          method: str,
                          url: str,
                          headers: Optional[Dict[str, str]] = None,
                          params: Optional[Dict[str, Any]] = None,
                          json_data: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """Make HTTP request with rate limiting and retry logic"""
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Prepare headers
        request_headers = self._get_default_headers()
        if headers:
            request_headers.update(headers)
        
        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(**self.client_config) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=request_headers,
                        params=params,
                        json=json_data
                    )
                    
                    # Update statistics
                    self.stats['total_requests'] += 1
                    self.stats['last_request'] = datetime.utcnow()
                    
                    if response.is_success:
                        self.stats['successful_requests'] += 1
                        return response
                    else:
                        # Handle rate limiting
                        if response.status_code == 429:
                            self.stats['rate_limit_hits'] += 1
                            retry_after = int(response.headers.get('Retry-After', 60))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        
                        # Handle other HTTP errors
                        response.raise_for_status()
                    
            except httpx.RequestError as e:
                last_exception = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    break
        
        # All retries failed
        self.stats['failed_requests'] += 1
        raise last_exception or Exception("All retry attempts failed")
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests"""
        headers = {
            'User-Agent': 'ForexBot/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    @abstractmethod
    async def fetch_news(self, 
                        limit: int = 100,
                        since: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        keywords: Optional[List[str]] = None) -> List[NewsArticle]:
        """Fetch news articles"""
        pass
    
    @abstractmethod
    async def fetch_events(self,
                          limit: int = 100,
                          since: Optional[datetime] = None,
                          until: Optional[datetime] = None,
                          currency: Optional[str] = None) -> List[EconomicEvent]:
        """Fetch economic events"""
        pass
    
    async def stream_news(self,
                         keywords: Optional[List[str]] = None,
                         interval_seconds: int = 60) -> AsyncGenerator[NewsArticle, None]:
        """Stream news articles continuously"""
        while True:
            try:
                articles = await self.fetch_news(
                    limit=100,
                    since=datetime.utcnow() - timedelta(minutes=5),
                    keywords=keywords
                )
                
                for article in articles:
                    yield article
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in news streaming: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check client health and connectivity"""
        try:
            # Try to fetch a small number of articles
            articles = await self.fetch_news(limit=1)
            
            return {
                'status': 'healthy',
                'source': self.source.value,
                'name': self.name,
                'last_request': self.stats['last_request'].isoformat() if self.stats['last_request'] else None,
                'total_requests': self.stats['total_requests'],
                'success_rate': self.stats['successful_requests'] / max(self.stats['total_requests'], 1),
                'rate_limit_hits': self.stats['rate_limit_hits']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'source': self.source.value,
                'name': self.name,
                'error': str(e),
                'last_request': self.stats['last_request'].isoformat() if self.stats['last_request'] else None,
                'total_requests': self.stats['total_requests'],
                'success_rate': self.stats['successful_requests'] / max(self.stats['total_requests'], 1),
                'rate_limit_hits': self.stats['rate_limit_hits']
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'source': self.source.value,
            'name': self.name,
            'stats': self.stats.copy(),
            'rate_limiter': {
                'max_requests_per_minute': self.rate_limiter.max_requests_per_minute,
                'max_requests_per_hour': self.rate_limiter.max_requests_per_hour,
                'current_minute_requests': len(self.rate_limiter.minute_requests),
                'current_hour_requests': len(self.rate_limiter.hour_requests)
            }
        }
    
    async def close(self):
        """Close client and cleanup resources"""
        # Override in subclasses if needed
        pass

