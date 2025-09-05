#!/usr/bin/env python3
"""
Trading Economics Economic Calendar Client
Client for Trading Economics Economic Calendar API
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base_client import BaseCalendarClient, RateLimiter
from ..models import EconomicEvent, EventCategory, EventImpact, EventStatus, Country, Currency

logger = logging.getLogger(__name__)

class TradingEconomicsClient(BaseCalendarClient):
    """Client for Trading Economics Economic Calendar API"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.tradingeconomics.com",
                 rate_limiter: Optional[RateLimiter] = None):
        """Initialize Trading Economics client"""
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter or RateLimiter(max_requests_per_minute=20, max_requests_per_hour=300)
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Trading Economics API"""
        return {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def fetch_events(self, 
                          start_date: datetime,
                          end_date: datetime,
                          countries: Optional[List[str]] = None,
                          categories: Optional[List[str]] = None) -> List[EconomicEvent]:
        """Fetch economic events from Trading Economics API"""
        try:
            params = {
                "c": self.api_key,
                "calendar": "1",
                "d1": start_date.strftime("%Y-%m-%d"),
                "d2": end_date.strftime("%Y-%m-%d")
            }
            
            if countries:
                params["country"] = ",".join(countries)
            
            data = await self._make_request("/calendar", params)
            
            events = []
            for event_data in data:
                try:
                    event = self._parse_te_event(event_data)
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse Trading Economics event: {e}")
                    continue
            
            logger.info(f"Fetched {len(events)} events from Trading Economics")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching events from Trading Economics: {e}")
            return []
    
    async def fetch_upcoming_events(self, 
                                   hours_ahead: int = 24,
                                   countries: Optional[List[str]] = None) -> List[EconomicEvent]:
        """Fetch upcoming economic events"""
        start_date = datetime.now()
        end_date = start_date + timedelta(hours=hours_ahead)
        
        return await self.fetch_events(start_date, end_date, countries)
    
    def _parse_te_event(self, data: Dict[str, Any]) -> EconomicEvent:
        """Parse Trading Economics event data into EconomicEvent object"""
        # Map TE fields to our model
        event_id = data.get('CalendarId', f"te_{data.get('Date', '')}_{data.get('Event', '')}")
        title = data.get('Event', '')
        country = self._parse_country(data.get('Country', ''))
        currency = self._parse_currency(data.get('Currency', ''))
        category = self._parse_te_category(data.get('Category', ''), data.get('Event', ''))
        impact = self._parse_te_impact(data.get('Importance', ''))
        status = self._parse_te_status(data.get('Actual', ''), data.get('Date', ''))
        
        # Parse datetime
        event_time = self._parse_te_datetime(data.get('Date', ''), data.get('Time', ''))
        
        # Parse values
        actual = self._parse_numeric_value(data.get('Actual'))
        forecast = self._parse_numeric_value(data.get('Forecast'))
        previous = self._parse_numeric_value(data.get('Previous'))
        
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
            unit=data.get('Unit'),
            description=data.get('Event'),
            source="Trading Economics",
            url=f"https://tradingeconomics.com/calendar/{event_id}"
        )
    
    def _parse_te_category(self, category: str, event_name: str) -> EventCategory:
        """Parse Trading Economics category to EventCategory"""
        category_lower = category.lower()
        event_lower = event_name.lower()
        
        if 'interest rate' in category_lower or 'monetary policy' in category_lower:
            return EventCategory.INTEREST_RATE
        elif 'gdp' in category_lower or 'gross domestic product' in event_lower:
            return EventCategory.GDP
        elif 'inflation' in category_lower or 'cpi' in event_lower:
            return EventCategory.INFLATION
        elif 'unemployment' in category_lower or 'employment' in category_lower:
            return EventCategory.UNEMPLOYMENT
        elif 'retail sales' in category_lower or 'consumer' in category_lower:
            return EventCategory.RETAIL_SALES
        elif 'manufacturing' in category_lower or 'industrial' in category_lower:
            return EventCategory.MANUFACTURING
        elif 'trade' in category_lower:
            return EventCategory.TRADE_BALANCE
        elif 'confidence' in category_lower or 'sentiment' in category_lower:
            return EventCategory.CONSUMER_CONFIDENCE
        elif 'central bank' in category_lower:
            return EventCategory.CENTRAL_BANK_MEETING
        else:
            return EventCategory.OTHER
    
    def _parse_te_impact(self, importance: str) -> EventImpact:
        """Parse Trading Economics importance to EventImpact"""
        if not importance:
            return EventImpact.MEDIUM
        
        try:
            importance_val = float(importance)
            if importance_val >= 3:
                return EventImpact.HIGH
            elif importance_val >= 2:
                return EventImpact.MEDIUM
            else:
                return EventImpact.LOW
        except ValueError:
            return EventImpact.MEDIUM
    
    def _parse_te_status(self, actual: str, date: str) -> EventStatus:
        """Parse Trading Economics status"""
        if actual and actual.strip() and actual.strip() != 'N/A':
            return EventStatus.COMPLETED
        else:
            return EventStatus.UPCOMING
    
    def _parse_te_datetime(self, date_str: str, time_str: str) -> datetime:
        """Parse Trading Economics datetime"""
        try:
            if time_str and time_str.strip():
                datetime_str = f"{date_str} {time_str}"
                return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            else:
                return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Could not parse Trading Economics datetime: {date_str} {time_str}")
            return datetime.now()
    
    def _parse_numeric_value(self, value: str) -> Optional[float]:
        """Parse numeric value from string"""
        if not value or not value.strip() or value.strip() == 'N/A':
            return None
        
        try:
            # Remove common suffixes and clean the string
            cleaned = value.strip().replace('%', '').replace(',', '').replace('K', '000').replace('M', '000000')
            return float(cleaned)
        except ValueError:
            return None

