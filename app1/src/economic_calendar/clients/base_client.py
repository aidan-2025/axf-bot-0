#!/usr/bin/env python3
"""
Base Economic Calendar Client
Base class for economic calendar API clients
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import httpx

from ..models import EconomicEvent, CalendarFilter

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_requests_per_minute: int = 60, max_requests_per_hour: int = 1000):
        """Initialize rate limiter"""
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.minute_requests = []
        self.hour_requests = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Clean old requests
        self.minute_requests = [req_time for req_time in self.minute_requests if now - req_time < 60]
        self.hour_requests = [req_time for req_time in self.hour_requests if now - req_time < 3600]
        
        # Check minute limit
        if len(self.minute_requests) >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self.minute_requests[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Check hour limit
        if len(self.hour_requests) >= self.max_requests_per_hour:
            sleep_time = 3600 - (now - self.hour_requests[0])
            if sleep_time > 0:
                logger.info(f"Hourly rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.minute_requests.append(now)
        self.hour_requests.append(now)

class BaseCalendarClient(ABC):
    """Base class for economic calendar clients"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: str = "",
                 rate_limiter: Optional[RateLimiter] = None,
                 timeout: int = 30):
        """Initialize base client"""
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limiter = rate_limiter or RateLimiter()
        self.timeout = timeout
        self.client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the HTTP client"""
        if self._initialized:
            return
        
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers=self._get_headers()
        )
        self._initialized = True
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        headers = {
            "User-Agent": "AXF-Bot-Economic-Calendar/1.0",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers.update(self._get_auth_headers())
        
        return headers
    
    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers (implemented by subclasses)"""
        pass
    
    @abstractmethod
    async def fetch_events(self, 
                          start_date: datetime,
                          end_date: datetime,
                          countries: Optional[List[str]] = None,
                          categories: Optional[List[str]] = None) -> List[EconomicEvent]:
        """Fetch economic events from the API"""
        pass
    
    @abstractmethod
    async def fetch_upcoming_events(self, 
                                   hours_ahead: int = 24,
                                   countries: Optional[List[str]] = None) -> List[EconomicEvent]:
        """Fetch upcoming economic events"""
        pass
    
    async def _make_request(self, 
                           endpoint: str,
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request with rate limiting"""
        if not self._initialized:
            await self.initialize()
        
        # Apply rate limiting
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def _parse_event_data(self, data: Dict[str, Any]) -> EconomicEvent:
        """Parse raw event data into EconomicEvent object"""
        # This is a base implementation - subclasses should override
        return EconomicEvent(
            event_id=data.get('id', ''),
            title=data.get('title', ''),
            country=self._parse_country(data.get('country', '')),
            currency=self._parse_currency(data.get('currency', '')),
            category=self._parse_category(data.get('category', '')),
            impact=self._parse_impact(data.get('impact', '')),
            status=self._parse_status(data.get('status', '')),
            event_time=self._parse_datetime(data.get('event_time', '')),
            actual=data.get('actual'),
            forecast=data.get('forecast'),
            previous=data.get('previous'),
            unit=data.get('unit'),
            description=data.get('description'),
            source=self.__class__.__name__,
            url=data.get('url')
        )
    
    def _parse_country(self, country_str: str) -> 'Country':
        """Parse country string to Country enum"""
        from ..models import Country
        country_mapping = {
            'US': Country.US,
            'United States': Country.US,
            'USA': Country.US,
            'EU': Country.EU,
            'Europe': Country.EU,
            'Eurozone': Country.EU,
            'UK': Country.UK,
            'United Kingdom': Country.UK,
            'GB': Country.UK,
            'JP': Country.JP,
            'Japan': Country.JP,
            'CA': Country.CA,
            'Canada': Country.CA,
            'AU': Country.AU,
            'Australia': Country.AU,
            'NZ': Country.NZ,
            'New Zealand': Country.NZ,
            'CH': Country.CH,
            'Switzerland': Country.CH,
            'CN': Country.CN,
            'China': Country.CN,
            'IN': Country.IN,
            'India': Country.IN,
            'BR': Country.BR,
            'Brazil': Country.BR,
            'RU': Country.RU,
            'Russia': Country.RU,
            'ZA': Country.ZA,
            'South Africa': Country.ZA,
            'MX': Country.MX,
            'Mexico': Country.MX,
            'KR': Country.KR,
            'South Korea': Country.KR,
            'SG': Country.SG,
            'Singapore': Country.SG
        }
        return country_mapping.get(country_str.upper(), Country.OTHER)
    
    def _parse_currency(self, currency_str: str) -> 'Currency':
        """Parse currency string to Currency enum"""
        from ..models import Currency
        currency_mapping = {
            'USD': Currency.USD,
            'EUR': Currency.EUR,
            'GBP': Currency.GBP,
            'JPY': Currency.JPY,
            'CAD': Currency.CAD,
            'AUD': Currency.AUD,
            'NZD': Currency.NZD,
            'CHF': Currency.CHF,
            'CNY': Currency.CNY,
            'INR': Currency.INR,
            'BRL': Currency.BRL,
            'RUB': Currency.RUB,
            'ZAR': Currency.ZAR,
            'MXN': Currency.MXN,
            'KRW': Currency.KRW,
            'SGD': Currency.SGD
        }
        return currency_mapping.get(currency_str.upper(), Currency.USD)
    
    def _parse_category(self, category_str: str) -> 'EventCategory':
        """Parse category string to EventCategory enum"""
        from ..models import EventCategory
        category_mapping = {
            'interest_rate': EventCategory.INTEREST_RATE,
            'monetary_policy': EventCategory.MONETARY_POLICY,
            'gdp': EventCategory.GDP,
            'inflation': EventCategory.INFLATION,
            'unemployment': EventCategory.UNEMPLOYMENT,
            'retail_sales': EventCategory.RETAIL_SALES,
            'manufacturing': EventCategory.MANUFACTURING,
            'trade_balance': EventCategory.TRADE_BALANCE,
            'consumer_confidence': EventCategory.CONSUMER_CONFIDENCE,
            'central_bank_meeting': EventCategory.CENTRAL_BANK_MEETING,
            'central_bank_speech': EventCategory.CENTRAL_BANK_SPEECH,
            'budget': EventCategory.BUDGET,
            'fiscal_policy': EventCategory.FISCAL_POLICY,
            'election': EventCategory.ELECTION
        }
        return category_mapping.get(category_str.lower(), EventCategory.OTHER)
    
    def _parse_impact(self, impact_str: str) -> 'EventImpact':
        """Parse impact string to EventImpact enum"""
        from ..models import EventImpact
        impact_mapping = {
            'low': EventImpact.LOW,
            'medium': EventImpact.MEDIUM,
            'high': EventImpact.HIGH,
            'very_high': EventImpact.VERY_HIGH,
            '1': EventImpact.LOW,
            '2': EventImpact.MEDIUM,
            '3': EventImpact.HIGH,
            '4': EventImpact.VERY_HIGH
        }
        return impact_mapping.get(impact_str.lower(), EventImpact.MEDIUM)
    
    def _parse_status(self, status_str: str) -> 'EventStatus':
        """Parse status string to EventStatus enum"""
        from ..models import EventStatus
        status_mapping = {
            'upcoming': EventStatus.UPCOMING,
            'live': EventStatus.LIVE,
            'completed': EventStatus.COMPLETED,
            'cancelled': EventStatus.CANCELLED,
            'postponed': EventStatus.POSTPONED
        }
        return status_mapping.get(status_str.lower(), EventStatus.UPCOMING)
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """Parse datetime string to datetime object"""
        try:
            # Try ISO format first
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try common formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                    return datetime.strptime(datetime_str, fmt)
            except ValueError:
                logger.warning(f"Could not parse datetime: {datetime_str}")
                return datetime.now()
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
        self._initialized = False
        logger.info(f"Closed {self.__class__.__name__}")

