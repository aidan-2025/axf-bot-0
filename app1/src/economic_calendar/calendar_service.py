#!/usr/bin/env python3
"""
Economic Calendar Service
Main service for economic calendar data management
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .models import EconomicEvent, CalendarFilter, CalendarStats, EventStatus, EventImpact
from .clients import (
    FMPCalendarClient, TradingEconomicsClient, 
    EODHDCalendarClient, ForexFactoryCalendarClient
)

logger = logging.getLogger(__name__)

@dataclass
class CalendarServiceConfig:
    """Configuration for economic calendar service"""
    # API Keys
    fmp_api_key: Optional[str] = None
    trading_economics_api_key: Optional[str] = None
    eodhd_api_key: Optional[str] = None
    
    # Client settings
    enable_fmp: bool = True
    enable_trading_economics: bool = True
    enable_eodhd: bool = True
    enable_forex_factory: bool = True
    
    # Data settings
    cache_ttl_hours: int = 1
    max_events_per_source: int = 1000
    deduplicate_events: bool = True
    
    # Update settings
    auto_update_interval_minutes: int = 30
    update_on_startup: bool = True

class EconomicCalendarService:
    """Main service for economic calendar data management"""
    
    def __init__(self, config: CalendarServiceConfig):
        """Initialize economic calendar service"""
        self.config = config
        self.clients = {}
        self.events_cache: Dict[str, List[EconomicEvent]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the calendar service and clients"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing economic calendar service")
            
            # Initialize clients based on configuration
            if self.config.enable_fmp and self.config.fmp_api_key:
                self.clients['fmp'] = FMPCalendarClient(self.config.fmp_api_key)
                await self.clients['fmp'].initialize()
                logger.info("FMP client initialized")
            
            if self.config.enable_trading_economics and self.config.trading_economics_api_key:
                self.clients['trading_economics'] = TradingEconomicsClient(self.config.trading_economics_api_key)
                await self.clients['trading_economics'].initialize()
                logger.info("Trading Economics client initialized")
            
            if self.config.enable_eodhd and self.config.eodhd_api_key:
                self.clients['eodhd'] = EODHDCalendarClient(self.config.eodhd_api_key)
                await self.clients['eodhd'].initialize()
                logger.info("EODHD client initialized")
            
            if self.config.enable_forex_factory:
                self.clients['forex_factory'] = ForexFactoryCalendarClient()
                await self.clients['forex_factory'].initialize()
                logger.info("Forex Factory client initialized")
            
            # Update data on startup if configured
            if self.config.update_on_startup:
                await self.update_events()
            
            self._initialized = True
            logger.info(f"Economic calendar service initialized with {len(self.clients)} clients")
            
        except Exception as e:
            logger.error(f"Failed to initialize economic calendar service: {e}")
            raise
    
    async def get_events(self, 
                        filter_criteria: Optional[CalendarFilter] = None,
                        use_cache: bool = True) -> List[EconomicEvent]:
        """Get economic events based on filter criteria"""
        if not self._initialized:
            await self.initialize()
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(filter_criteria)
            cached_events = self._get_cached_events(cache_key)
            if cached_events is not None:
                return cached_events
        
        # Fetch events from all sources
        all_events = []
        
        for client_name, client in self.clients.items():
            try:
                events = await self._fetch_events_from_client(
                    client, filter_criteria
                )
                all_events.extend(events)
                logger.info(f"Fetched {len(events)} events from {client_name}")
                
            except Exception as e:
                logger.error(f"Error fetching events from {client_name}: {e}")
                continue
        
        # Deduplicate events if configured
        if self.config.deduplicate_events:
            all_events = self._deduplicate_events(all_events)
        
        # Apply filters
        filtered_events = self._apply_filters(all_events, filter_criteria)
        
        # Cache results
        if use_cache:
            cache_key = self._get_cache_key(filter_criteria)
            self._cache_events(cache_key, filtered_events)
        
        logger.info(f"Returning {len(filtered_events)} events")
        return filtered_events
    
    async def get_upcoming_events(self, 
                                 hours_ahead: int = 24,
                                 countries: Optional[List[str]] = None,
                                 impacts: Optional[List[EventImpact]] = None) -> List[EconomicEvent]:
        """Get upcoming economic events"""
        filter_criteria = CalendarFilter(
            countries=[self._parse_country(c) for c in countries] if countries else None,
            impacts=impacts,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(hours=hours_ahead),
            statuses=[EventStatus.UPCOMING, EventStatus.LIVE]
        )
        
        return await self.get_events(filter_criteria)
    
    async def get_high_impact_events(self, 
                                   hours_ahead: int = 24,
                                   countries: Optional[List[str]] = None) -> List[EconomicEvent]:
        """Get high impact economic events"""
        return await self.get_upcoming_events(
            hours_ahead=hours_ahead,
            countries=countries,
            impacts=[EventImpact.HIGH, EventImpact.VERY_HIGH]
        )
    
    async def update_events(self):
        """Update events from all sources"""
        if not self._initialized:
            logger.warning("Service not initialized, skipping event update")
            return
        
        logger.info("Updating economic events from all sources")
        
        # Update events for next 7 days
        end_date = datetime.now() + timedelta(days=7)
        filter_criteria = CalendarFilter(
            start_date=datetime.now(),
            end_date=end_date
        )
        
        await self.get_events(filter_criteria, use_cache=False)
        logger.info("Economic events updated successfully")
    
    async def get_calendar_stats(self) -> CalendarStats:
        """Get statistics about the calendar data"""
        if not self._initialized:
            await self.initialize()
        
        # Get all events from cache
        all_events = []
        for events in self.events_cache.values():
            all_events.extend(events)
        
        # Calculate statistics
        events_by_country = {}
        events_by_category = {}
        events_by_impact = {}
        
        upcoming_count = 0
        completed_count = 0
        high_impact_count = 0
        
        for event in all_events:
            # Count by country
            country = event.country.value
            events_by_country[country] = events_by_country.get(country, 0) + 1
            
            # Count by category
            category = event.category.value
            events_by_category[category] = events_by_category.get(category, 0) + 1
            
            # Count by impact
            impact = event.impact.value
            events_by_impact[impact] = events_by_impact.get(impact, 0) + 1
            
            # Count by status
            if event.status in [EventStatus.UPCOMING, EventStatus.LIVE]:
                upcoming_count += 1
            elif event.status == EventStatus.COMPLETED:
                completed_count += 1
            
            # Count high impact
            if event.impact in [EventImpact.HIGH, EventImpact.VERY_HIGH]:
                high_impact_count += 1
        
        return CalendarStats(
            total_events=len(all_events),
            events_by_country=events_by_country,
            events_by_category=events_by_category,
            events_by_impact=events_by_impact,
            upcoming_events=upcoming_count,
            completed_events=completed_count,
            high_impact_events=high_impact_count
        )
    
    async def _fetch_events_from_client(self, 
                                       client, 
                                       filter_criteria: Optional[CalendarFilter]) -> List[EconomicEvent]:
        """Fetch events from a specific client"""
        start_date = filter_criteria.start_date if filter_criteria else datetime.now()
        end_date = filter_criteria.end_date if filter_criteria else datetime.now() + timedelta(days=7)
        
        countries = None
        if filter_criteria and filter_criteria.countries:
            countries = [c.value for c in filter_criteria.countries]
        
        return await client.fetch_events(start_date, end_date, countries)
    
    def _get_cache_key(self, filter_criteria: Optional[CalendarFilter]) -> str:
        """Generate cache key for filter criteria"""
        if not filter_criteria:
            return "all_events"
        
        key_parts = []
        if filter_criteria.countries:
            key_parts.append(f"countries:{','.join([c.value for c in filter_criteria.countries])}")
        if filter_criteria.currencies:
            key_parts.append(f"currencies:{','.join([c.value for c in filter_criteria.currencies])}")
        if filter_criteria.categories:
            key_parts.append(f"categories:{','.join([c.value for c in filter_criteria.categories])}")
        if filter_criteria.impacts:
            key_parts.append(f"impacts:{','.join([i.value for i in filter_criteria.impacts])}")
        if filter_criteria.start_date:
            key_parts.append(f"start:{filter_criteria.start_date.strftime('%Y-%m-%d')}")
        if filter_criteria.end_date:
            key_parts.append(f"end:{filter_criteria.end_date.strftime('%Y-%m-%d')}")
        
        return "|".join(key_parts) if key_parts else "all_events"
    
    def _get_cached_events(self, cache_key: str) -> Optional[List[EconomicEvent]]:
        """Get cached events if not expired"""
        if cache_key not in self.events_cache:
            return None
        
        if cache_key in self.cache_timestamps:
            age = datetime.now() - self.cache_timestamps[cache_key]
            if age.total_seconds() > self.config.cache_ttl_hours * 3600:
                # Cache expired
                del self.events_cache[cache_key]
                del self.cache_timestamps[cache_key]
                return None
        
        return self.events_cache[cache_key]
    
    def _cache_events(self, cache_key: str, events: List[EconomicEvent]):
        """Cache events with timestamp"""
        self.events_cache[cache_key] = events
        self.cache_timestamps[cache_key] = datetime.now()
    
    def _deduplicate_events(self, events: List[EconomicEvent]) -> List[EconomicEvent]:
        """Remove duplicate events based on title, country, and time"""
        seen = set()
        unique_events = []
        
        for event in events:
            # Create a key for deduplication
            key = (event.title.lower(), event.country.value, event.event_time.strftime('%Y-%m-%d'))
            
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        return unique_events
    
    def _apply_filters(self, 
                      events: List[EconomicEvent], 
                      filter_criteria: Optional[CalendarFilter]) -> List[EconomicEvent]:
        """Apply filter criteria to events"""
        if not filter_criteria:
            return events
        
        filtered_events = events
        
        # Filter by countries
        if filter_criteria.countries:
            filtered_events = [e for e in filtered_events if e.country in filter_criteria.countries]
        
        # Filter by currencies
        if filter_criteria.currencies:
            filtered_events = [e for e in filtered_events if e.currency in filter_criteria.currencies]
        
        # Filter by categories
        if filter_criteria.categories:
            filtered_events = [e for e in filtered_events if e.category in filter_criteria.categories]
        
        # Filter by impacts
        if filter_criteria.impacts:
            filtered_events = [e for e in filtered_events if e.impact in filter_criteria.impacts]
        
        # Filter by statuses
        if filter_criteria.statuses:
            filtered_events = [e for e in filtered_events if e.status in filter_criteria.statuses]
        
        # Filter by date range
        if filter_criteria.start_date:
            filtered_events = [e for e in filtered_events if e.event_time >= filter_criteria.start_date]
        
        if filter_criteria.end_date:
            filtered_events = [e for e in filtered_events if e.event_time <= filter_criteria.end_date]
        
        # Apply limit
        if filter_criteria.limit:
            filtered_events = filtered_events[:filter_criteria.limit]
        
        return filtered_events
    
    def _parse_country(self, country_str: str):
        """Parse country string to Country enum"""
        from .models import Country
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
            'Switzerland': Country.CH
        }
        return country_mapping.get(country_str.upper(), Country.OTHER)
    
    async def close(self):
        """Close all clients and cleanup"""
        for client in self.clients.values():
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
        
        self.clients.clear()
        self.events_cache.clear()
        self.cache_timestamps.clear()
        self._initialized = False
        logger.info("Economic calendar service closed")

