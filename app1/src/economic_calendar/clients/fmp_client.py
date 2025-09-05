#!/usr/bin/env python3
"""
Financial Modeling Prep Economic Calendar Client
Client for FMP Economic Calendar API
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base_client import BaseCalendarClient, RateLimiter
from ..models import EconomicEvent, EventCategory, EventImpact, EventStatus, Country, Currency

logger = logging.getLogger(__name__)

class FMPCalendarClient(BaseCalendarClient):
    """Client for Financial Modeling Prep Economic Calendar API"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://financialmodelingprep.com/api/v3",
                 rate_limiter: Optional[RateLimiter] = None):
        """Initialize FMP client"""
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter or RateLimiter(max_requests_per_minute=30, max_requests_per_hour=500)
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for FMP API"""
        return {
            "apikey": self.api_key
        }
    
    async def fetch_events(self, 
                          start_date: datetime,
                          end_date: datetime,
                          countries: Optional[List[str]] = None,
                          categories: Optional[List[str]] = None) -> List[EconomicEvent]:
        """Fetch economic events from FMP API"""
        try:
            params = {
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "apikey": self.api_key
            }
            
            if countries:
                params["country"] = ",".join(countries)
            
            data = await self._make_request("/economic_calendar", params)
            
            events = []
            for event_data in data:
                try:
                    event = self._parse_fmp_event(event_data)
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse FMP event: {e}")
                    continue
            
            logger.info(f"Fetched {len(events)} events from FMP")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching events from FMP: {e}")
            return []
    
    async def fetch_upcoming_events(self, 
                                   hours_ahead: int = 24,
                                   countries: Optional[List[str]] = None) -> List[EconomicEvent]:
        """Fetch upcoming economic events"""
        start_date = datetime.now()
        end_date = start_date + timedelta(hours=hours_ahead)
        
        return await self.fetch_events(start_date, end_date, countries)
    
    def _parse_fmp_event(self, data: Dict[str, Any]) -> EconomicEvent:
        """Parse FMP event data into EconomicEvent object"""
        # Map FMP fields to our model
        event_id = data.get('id', f"fmp_{data.get('date', '')}_{data.get('event', '')}")
        title = data.get('event', '')
        country = self._parse_country(data.get('country', ''))
        currency = self._parse_currency(data.get('currency', ''))
        category = self._parse_fmp_category(data.get('event', ''))
        impact = self._parse_fmp_impact(data.get('impact', ''))
        status = self._parse_fmp_status(data.get('actual', ''), data.get('date', ''))
        
        # Parse datetime
        event_time = self._parse_fmp_datetime(data.get('date', ''), data.get('time', ''))
        
        # Parse values
        actual = self._parse_numeric_value(data.get('actual'))
        forecast = self._parse_numeric_value(data.get('estimate'))
        previous = self._parse_numeric_value(data.get('previous'))
        
        return EconomicEvent(
            event_id=event_id,
            title=title,
            country=country,
            currency=currency,
            category=category,
            impact=impact,
            status=status,
            event_time=event_time,
            actual=actual,
            forecast=forecast,
            previous=previous,
            unit=data.get('unit'),
            description=data.get('event'),
            source="Financial Modeling Prep",
            url=f"https://financialmodelingprep.com/economic-calendar/{event_id}"
        )
    
    def _parse_fmp_category(self, event_name: str) -> EventCategory:
        """Parse FMP event name to category"""
        event_lower = event_name.lower()
        
        if any(word in event_lower for word in ['interest rate', 'fed rate', 'ecb rate', 'boe rate']):
            return EventCategory.INTEREST_RATE
        elif any(word in event_lower for word in ['gdp', 'gross domestic product']):
            return EventCategory.GDP
        elif any(word in event_lower for word in ['cpi', 'inflation', 'consumer price']):
            return EventCategory.INFLATION
        elif any(word in event_lower for word in ['unemployment', 'jobless', 'employment']):
            return EventCategory.UNEMPLOYMENT
        elif any(word in event_lower for word in ['retail sales', 'consumer spending']):
            return EventCategory.RETAIL_SALES
        elif any(word in event_lower for word in ['manufacturing', 'pmi', 'industrial']):
            return EventCategory.MANUFACTURING
        elif any(word in event_lower for word in ['trade balance', 'trade deficit', 'trade surplus']):
            return EventCategory.TRADE_BALANCE
        elif any(word in event_lower for word in ['consumer confidence', 'sentiment']):
            return EventCategory.CONSUMER_CONFIDENCE
        elif any(word in event_lower for word in ['fed meeting', 'ecb meeting', 'boe meeting']):
            return EventCategory.CENTRAL_BANK_MEETING
        elif any(word in event_lower for word in ['fed speech', 'ecb speech', 'boe speech']):
            return EventCategory.CENTRAL_BANK_SPEECH
        else:
            return EventCategory.OTHER
    
    def _parse_fmp_impact(self, impact_str: str) -> EventImpact:
        """Parse FMP impact to EventImpact enum"""
        if not impact_str:
            return EventImpact.MEDIUM
        
        impact_lower = impact_str.lower()
        if 'high' in impact_lower:
            return EventImpact.HIGH
        elif 'medium' in impact_lower:
            return EventImpact.MEDIUM
        elif 'low' in impact_lower:
            return EventImpact.LOW
        else:
            return EventImpact.MEDIUM
    
    def _parse_fmp_status(self, actual: str, date: str) -> EventStatus:
        """Parse FMP status based on actual value and date"""
        if actual and actual.strip():
            return EventStatus.COMPLETED
        else:
            return EventStatus.UPCOMING
    
    def _parse_fmp_datetime(self, date_str: str, time_str: str) -> datetime:
        """Parse FMP datetime"""
        try:
            if time_str and time_str.strip():
                datetime_str = f"{date_str} {time_str}"
                return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            else:
                return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Could not parse FMP datetime: {date_str} {time_str}")
            return datetime.now()
    
    def _parse_numeric_value(self, value: str) -> Optional[float]:
        """Parse numeric value from string"""
        if not value or not value.strip():
            return None
        
        try:
            # Remove common suffixes and clean the string
            cleaned = value.strip().replace('%', '').replace(',', '')
            return float(cleaned)
        except ValueError:
            return None

