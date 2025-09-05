#!/usr/bin/env python3
"""
Forex Factory Client
Client for Forex Factory economic calendar via JBlanked API
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import httpx

from .base_client import BaseNewsClient, RateLimiter
from ..models import NewsSource, EconomicEvent, NewsArticle, ImpactLevel, Currency

logger = logging.getLogger(__name__)

class ForexFactoryClient(BaseNewsClient):
    """Client for Forex Factory economic calendar"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: str = "https://www.forexfactory.com",
                 rate_limiter: Optional[RateLimiter] = None):
        """Initialize Forex Factory client"""
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter or RateLimiter(max_requests_per_minute=10, max_requests_per_hour=100)  # Conservative limits for scraping
        )
    
    @property
    def source(self) -> NewsSource:
        return NewsSource.FOREX_FACTORY
    
    @property
    def name(self) -> str:
        return "Forex Factory (JBlanked API)"
    
    async def fetch_news(self, 
                        limit: int = 100,
                        since: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        keywords: Optional[List[str]] = None) -> List[NewsArticle]:
        """Fetch news articles (Forex Factory doesn't have traditional news)"""
        # Forex Factory is primarily for economic events, not news
        # Return empty list since we don't have news articles
        logger.warning("Forex Factory doesn't provide news articles, only economic events")
        return []
    
    async def fetch_events(self,
                          limit: int = 100,
                          since: Optional[datetime] = None,
                          until: Optional[datetime] = None,
                          currency: Optional[str] = None) -> List[EconomicEvent]:
        """Fetch economic calendar events via web scraping"""
        try:
            # For now, return mock data since Forex Factory doesn't have a public API
            # In a production system, you would implement web scraping here
            logger.warning("Forex Factory API not available, returning mock data")
            
            # Generate mock economic events for testing
            events = self._generate_mock_events(limit, since, until, currency)
            
            logger.info(f"Generated {len(events)} mock economic events for Forex Factory")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching Forex Factory events: {e}")
            return []
    
    def _parse_event(self, event_data: Dict[str, Any]) -> Optional[EconomicEvent]:
        """Parse event data from API response"""
        try:
            # Extract basic information
            event_id = str(event_data.get('id', ''))
            title = event_data.get('title', '')
            description = event_data.get('description', '')
            
            # Parse event time
            event_time_str = event_data.get('datetime', '')
            if event_time_str:
                try:
                    event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                except ValueError:
                    event_time = datetime.utcnow()
            else:
                event_time = datetime.utcnow()
            
            # Parse impact level
            impact_str = event_data.get('impact', 'low').lower()
            impact_mapping = {
                'low': ImpactLevel.LOW,
                'medium': ImpactLevel.MEDIUM,
                'high': ImpactLevel.HIGH,
                'very_high': ImpactLevel.VERY_HIGH
            }
            impact = impact_mapping.get(impact_str, ImpactLevel.LOW)
            
            # Parse currency
            currency_str = event_data.get('currency', 'USD')
            try:
                currency = Currency(currency_str.upper())
            except ValueError:
                currency = Currency.USD
            
            # Extract data values
            actual = event_data.get('actual')
            forecast = event_data.get('forecast')
            previous = event_data.get('previous')
            unit = event_data.get('unit', '')
            
            # Convert to float if possible
            if actual is not None:
                try:
                    actual = float(actual)
                except (ValueError, TypeError):
                    actual = None
            
            if forecast is not None:
                try:
                    forecast = float(forecast)
                except (ValueError, TypeError):
                    forecast = None
            
            if previous is not None:
                try:
                    previous = float(previous)
                except (ValueError, TypeError):
                    previous = None
            
            # Determine currency pairs
            currency_pairs = self._get_currency_pairs(currency)
            
            # Extract additional metadata
            country = event_data.get('country', '')
            category = event_data.get('category', '')
            
            return EconomicEvent(
                event_id=event_id,
                title=title,
                description=description,
                event_time=event_time,
                timezone='UTC',
                impact=impact,
                currency=currency,
                currency_pairs=currency_pairs,
                actual=actual,
                forecast=forecast,
                previous=previous,
                unit=unit,
                source=self.source,
                country=country,
                category=category
            )
            
        except Exception as e:
            logger.error(f"Error parsing Forex Factory event: {e}")
            return None
    
    def _get_currency_pairs(self, currency: Currency) -> List[str]:
        """Get relevant currency pairs for a currency"""
        major_pairs = {
            Currency.USD: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD'],
            Currency.EUR: ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD'],
            Currency.GBP: ['GBPUSD', 'EURGBP', 'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD'],
            Currency.JPY: ['USDJPY', 'EURJPY', 'GBPJPY', 'CHFJPY', 'CADJPY', 'AUDJPY', 'NZDJPY'],
            Currency.CHF: ['USDCHF', 'EURCHF', 'GBPCHF', 'CHFJPY', 'CHFCAD', 'CHFAUD', 'CHFNZD'],
            Currency.CAD: ['USDCAD', 'EURCAD', 'GBPCAD', 'CADJPY', 'CHFCAD', 'AUDCAD', 'NZDCAD'],
            Currency.AUD: ['AUDUSD', 'EURAUD', 'GBPAUD', 'AUDJPY', 'CHFAUD', 'AUDCAD', 'AUDNZD'],
            Currency.NZD: ['NZDUSD', 'EURNZD', 'GBPNZD', 'NZDJPY', 'CHFNZD', 'NZDCAD', 'AUDNZD']
        }
        
        return major_pairs.get(currency, [f'{currency.value}USD'])
    
    async def get_high_impact_events(self, 
                                   hours_ahead: int = 24) -> List[EconomicEvent]:
        """Get high impact events in the next N hours"""
        since = datetime.utcnow()
        until = since + timedelta(hours=hours_ahead)
        
        events = await self.fetch_events(since=since, until=until)
        
        # Filter for high impact events
        high_impact_events = [
            event for event in events 
            if event.impact in [ImpactLevel.HIGH, ImpactLevel.VERY_HIGH]
        ]
        
        logger.info(f"Found {len(high_impact_events)} high impact events in next {hours_ahead} hours")
        return high_impact_events
    
    async def get_currency_events(self, 
                                currency: str,
                                hours_ahead: int = 24) -> List[EconomicEvent]:
        """Get events for a specific currency"""
        since = datetime.utcnow()
        until = since + timedelta(hours=hours_ahead)
        
        events = await self.fetch_events(
            since=since,
            until=until,
            currency=currency
        )
        
        logger.info(f"Found {len(events)} events for {currency} in next {hours_ahead} hours")
        return events
    
    def _generate_mock_events(self, 
                             limit: int,
                             since: Optional[datetime] = None,
                             until: Optional[datetime] = None,
                             currency: Optional[str] = None) -> List[EconomicEvent]:
        """Generate mock economic events for testing"""
        import random
        from datetime import timedelta
        
        # Mock economic events data
        mock_events_data = [
            {
                'title': 'Non-Farm Payrolls',
                'description': 'Change in the number of employed people during the previous month',
                'impact': ImpactLevel.HIGH,
                'currency': Currency.USD,
                'unit': 'K',
                'category': 'Employment'
            },
            {
                'title': 'Consumer Price Index (CPI)',
                'description': 'Change in the price of goods and services purchased by consumers',
                'impact': ImpactLevel.HIGH,
                'currency': Currency.USD,
                'unit': '%',
                'category': 'Inflation'
            },
            {
                'title': 'Federal Funds Rate',
                'description': 'Interest rate at which depository institutions lend funds maintained at the Federal Reserve',
                'impact': ImpactLevel.VERY_HIGH,
                'currency': Currency.USD,
                'unit': '%',
                'category': 'Interest Rate'
            },
            {
                'title': 'GDP Growth Rate',
                'description': 'Annualized change in the value of all goods and services produced',
                'impact': ImpactLevel.HIGH,
                'currency': Currency.USD,
                'unit': '%',
                'category': 'GDP'
            },
            {
                'title': 'Unemployment Rate',
                'description': 'Percentage of the total work force that is unemployed',
                'impact': ImpactLevel.MEDIUM,
                'currency': Currency.USD,
                'unit': '%',
                'category': 'Employment'
            },
            {
                'title': 'Retail Sales',
                'description': 'Change in the total value of sales at the retail level',
                'impact': ImpactLevel.MEDIUM,
                'currency': Currency.USD,
                'unit': '%',
                'category': 'Consumption'
            }
        ]
        
        events = []
        now = datetime.utcnow()
        
        # Set default time range if not provided
        if since is None:
            since = now - timedelta(days=1)
        if until is None:
            until = now + timedelta(days=7)
        
        # Generate events within the time range
        for i in range(min(limit, len(mock_events_data))):
            event_data = mock_events_data[i % len(mock_events_data)]
            
            # Skip if currency filter doesn't match
            if currency and event_data['currency'].value != currency.upper():
                continue
            
            # Generate random time within range
            time_diff = until - since
            random_seconds = random.randint(0, int(time_diff.total_seconds()))
            event_time = since + timedelta(seconds=random_seconds)
            
            # Generate random values
            actual = round(random.uniform(-2.0, 5.0), 2)
            forecast = round(actual + random.uniform(-0.5, 0.5), 2)
            previous = round(actual + random.uniform(-1.0, 1.0), 2)
            
            event = EconomicEvent(
                event_id=f"mock_event_{i+1}",
                title=event_data['title'],
                description=event_data['description'],
                event_time=event_time,
                timezone='UTC',
                impact=event_data['impact'],
                currency=event_data['currency'],
                currency_pairs=self._get_currency_pairs(event_data['currency']),
                actual=actual,
                forecast=forecast,
                previous=previous,
                unit=event_data['unit'],
                source=self.source,
                country='US',
                category=event_data['category'],
                relevance_score=0.8  # High relevance for economic events
            )
            
            events.append(event)
        
        return events
