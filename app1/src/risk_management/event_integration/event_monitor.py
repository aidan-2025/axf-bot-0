#!/usr/bin/env python3
"""
Event Monitor

Monitors economic calendar events and integrates them with the risk management system.
Provides real-time event detection, impact assessment, and risk recommendations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..models import (
    EconomicEventData, EventImpact, RiskLevel, RiskEvent, RiskAction
)

logger = logging.getLogger(__name__)


@dataclass
class EventMonitorConfig:
    """Configuration for event monitoring"""
    # Event filtering
    min_impact_level: EventImpact = EventImpact.MEDIUM
    lookahead_hours: int = 24
    lookback_hours: int = 1
    
    # Risk assessment
    high_impact_threshold: int = 1
    critical_impact_threshold: int = 1
    multiple_events_threshold: int = 3
    
    # Currency filtering
    monitored_currencies: List[str] = None
    currency_relevance_threshold: float = 0.7
    
    # Update intervals
    check_interval_seconds: int = 60
    cache_duration_minutes: int = 30
    
    def __post_init__(self):
        if self.monitored_currencies is None:
            self.monitored_currencies = [
                'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'
            ]


class EventMonitor:
    """
    Monitors economic calendar events and assesses their impact on trading risk.
    
    Integrates with existing economic calendar modules to:
    - Detect high-impact events
    - Assess event relevance to trading positions
    - Provide risk recommendations based on event timing and impact
    - Track event outcomes and market reactions
    """
    
    def __init__(self, config: EventMonitorConfig = None):
        """Initialize event monitor"""
        self.config = config or EventMonitorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Event storage
        self.events_cache: Dict[str, EconomicEventData] = {}
        self.active_events: List[EconomicEventData] = []
        self.recent_events: List[EconomicEventData] = []
        
        # Risk tracking
        self.event_risk_events: List[RiskEvent] = []
        self.last_check_time = datetime.utcnow()
        
        # Performance tracking
        self.events_processed = 0
        self.risk_events_generated = 0
        
        self.logger.info("EventMonitor initialized")
    
    async def start_monitoring(self, economic_calendar_service=None):
        """Start monitoring economic events"""
        self.logger.info("Starting event monitoring")
        
        try:
            while True:
                await self._check_events(economic_calendar_service)
                await asyncio.sleep(self.config.check_interval_seconds)
        except asyncio.CancelledError:
            self.logger.info("Event monitoring stopped")
        except Exception as e:
            self.logger.error(f"Error in event monitoring: {e}")
    
    async def _check_events(self, economic_calendar_service=None):
        """Check for new economic events"""
        try:
            current_time = datetime.utcnow()
            
            # Get events from service if available
            if economic_calendar_service:
                events = await self._fetch_events_from_service(economic_calendar_service)
            else:
                events = await self._get_mock_events()
            
            # Process events
            await self._process_events(events)
            
            # Update active events
            self._update_active_events(current_time)
            
            # Generate risk events if needed
            await self._generate_risk_events()
            
            self.last_check_time = current_time
            self.events_processed += len(events)
            
        except Exception as e:
            self.logger.error(f"Error checking events: {e}")
    
    async def _fetch_events_from_service(self, service) -> List[EconomicEventData]:
        """Fetch events from economic calendar service"""
        try:
            # This would integrate with the existing economic calendar service
            # For now, return empty list
            return []
        except Exception as e:
            self.logger.error(f"Error fetching events from service: {e}")
            return []
    
    async def _get_mock_events(self) -> List[EconomicEventData]:
        """Get mock events for testing"""
        current_time = datetime.utcnow()
        
        # Generate some mock events for demonstration
        mock_events = [
            EconomicEventData(
                event_id="mock_1",
                title="US Non-Farm Payrolls",
                event_time=current_time + timedelta(hours=2),
                impact=EventImpact.HIGH,
                currency="USD",
                currency_pairs=["EUR/USD", "GBP/USD", "USD/JPY"],
                actual=None,
                forecast=200000,
                previous=195000,
                country="US",
                category="Employment",
                relevance_score=0.9
            ),
            EconomicEventData(
                event_id="mock_2",
                title="ECB Interest Rate Decision",
                event_time=current_time + timedelta(hours=6),
                impact=EventImpact.CRITICAL,
                currency="EUR",
                currency_pairs=["EUR/USD", "EUR/GBP", "EUR/JPY"],
                actual=None,
                forecast=4.25,
                previous=4.25,
                country="EU",
                category="Central Bank",
                relevance_score=0.95
            ),
            EconomicEventData(
                event_id="mock_3",
                title="UK GDP Growth Rate",
                event_time=current_time + timedelta(hours=12),
                impact=EventImpact.MEDIUM,
                currency="GBP",
                currency_pairs=["GBP/USD", "EUR/GBP"],
                actual=None,
                forecast=0.3,
                previous=0.2,
                country="UK",
                category="GDP",
                relevance_score=0.7
            )
        ]
        
        return mock_events
    
    async def _process_events(self, events: List[EconomicEventData]):
        """Process and categorize events"""
        for event in events:
            # Check if event meets minimum impact threshold
            if self._is_relevant_event(event):
                # Add to cache
                self.events_cache[event.event_id] = event
                
                # Add to active events if within lookahead period
                if self._is_active_event(event):
                    self.active_events.append(event)
                
                # Add to recent events if within lookback period
                if self._is_recent_event(event):
                    self.recent_events.append(event)
                
                self.logger.debug(f"Processed event: {event.title} ({event.impact.value})")
    
    def _is_relevant_event(self, event: EconomicEventData) -> bool:
        """Check if event is relevant for risk monitoring"""
        # Check impact level
        if not self._is_impact_sufficient(event.impact, self.config.min_impact_level):
            return False
        
        # Check currency relevance
        if event.currency not in self.config.monitored_currencies:
            return False
        
        # Check relevance score
        if event.relevance_score < self.config.currency_relevance_threshold:
            return False
        
        return True
    
    def _is_impact_sufficient(self, event_impact: EventImpact, min_impact: EventImpact) -> bool:
        """Check if event impact meets minimum threshold"""
        impact_order = {
            EventImpact.LOW: 1,
            EventImpact.MEDIUM: 2,
            EventImpact.HIGH: 3,
            EventImpact.CRITICAL: 4
        }
        return impact_order[event_impact] >= impact_order[min_impact]
    
    def _is_active_event(self, event: EconomicEventData) -> bool:
        """Check if event is active (within lookahead period)"""
        current_time = datetime.utcnow()
        time_diff = (event.event_time - current_time).total_seconds() / 3600
        
        return 0 <= time_diff <= self.config.lookahead_hours
    
    def _is_recent_event(self, event: EconomicEventData) -> bool:
        """Check if event is recent (within lookback period)"""
        current_time = datetime.utcnow()
        time_diff = (current_time - event.event_time).total_seconds() / 3600
        
        return 0 <= time_diff <= self.config.lookback_hours
    
    def _update_active_events(self, current_time: datetime):
        """Update active events list"""
        # Remove events that are no longer active
        self.active_events = [
            event for event in self.active_events
            if self._is_active_event(event)
        ]
        
        # Remove old recent events
        self.recent_events = [
            event for event in self.recent_events
            if self._is_recent_event(event)
        ]
    
    async def _generate_risk_events(self):
        """Generate risk events based on active economic events"""
        if not self.active_events:
            return
        
        # Count events by impact level
        high_impact_count = sum(1 for e in self.active_events if e.impact == EventImpact.HIGH)
        critical_impact_count = sum(1 for e in self.active_events if e.impact == EventImpact.CRITICAL)
        total_events = len(self.active_events)
        
        # Generate risk events based on thresholds
        if critical_impact_count >= self.config.critical_impact_threshold:
            await self._create_risk_event(
                "critical_economic_event",
                RiskLevel.CRITICAL,
                f"Critical economic event detected: {critical_impact_count} events",
                {"critical_events": critical_impact_count, "total_events": total_events}
            )
        elif high_impact_count >= self.config.high_impact_threshold:
            await self._create_risk_event(
                "high_impact_economic_event",
                RiskLevel.HIGH,
                f"High impact economic event detected: {high_impact_count} events",
                {"high_impact_events": high_impact_count, "total_events": total_events}
            )
        elif total_events >= self.config.multiple_events_threshold:
            await self._create_risk_event(
                "multiple_economic_events",
                RiskLevel.MEDIUM,
                f"Multiple economic events detected: {total_events} events",
                {"total_events": total_events}
            )
    
    async def _create_risk_event(self, event_type: str, risk_level: RiskLevel, 
                               description: str, data: Dict[str, Any]):
        """Create a risk event"""
        risk_event = RiskEvent(
            event_id=f"event_{event_type}_{datetime.utcnow().timestamp()}",
            event_type=event_type,
            risk_level=risk_level,
            description=description,
            timestamp=datetime.utcnow(),
            source="event_monitor",
            data=data
        )
        
        self.event_risk_events.append(risk_event)
        self.risk_events_generated += 1
        
        self.logger.info(f"Generated risk event: {description}")
    
    def get_active_events(self) -> List[EconomicEventData]:
        """Get currently active events"""
        return self.active_events.copy()
    
    def get_recent_events(self) -> List[EconomicEventData]:
        """Get recently occurred events"""
        return self.recent_events.copy()
    
    def get_risk_events(self) -> List[RiskEvent]:
        """Get generated risk events"""
        return self.event_risk_events.copy()
    
    def get_event_risk_summary(self) -> Dict[str, Any]:
        """Get event risk summary"""
        high_impact = sum(1 for e in self.active_events if e.impact == EventImpact.HIGH)
        critical_impact = sum(1 for e in self.active_events if e.impact == EventImpact.CRITICAL)
        
        return {
            "active_events_count": len(self.active_events),
            "recent_events_count": len(self.recent_events),
            "high_impact_events": high_impact,
            "critical_impact_events": critical_impact,
            "risk_events_generated": self.risk_events_generated,
            "events_processed": self.events_processed,
            "last_check": self.last_check_time.isoformat()
        }
    
    def get_events_by_currency(self, currency: str) -> List[EconomicEventData]:
        """Get events for a specific currency"""
        return [
            event for event in self.active_events
            if event.currency == currency
        ]
    
    def get_events_by_impact(self, impact: EventImpact) -> List[EconomicEventData]:
        """Get events by impact level"""
        return [
            event for event in self.active_events
            if event.impact == impact
        ]
    
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[EconomicEventData]:
        """Get events within specified time window"""
        current_time = datetime.utcnow()
        cutoff_time = current_time + timedelta(hours=hours_ahead)
        
        return [
            event for event in self.active_events
            if event.event_time <= cutoff_time
        ]
    
    def clear_old_events(self, max_age_hours: int = 24):
        """Clear old events from cache"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        # Clear old events from cache
        old_event_ids = [
            event_id for event_id, event in self.events_cache.items()
            if event.event_time < cutoff_time
        ]
        
        for event_id in old_event_ids:
            del self.events_cache[event_id]
        
        # Clear old risk events
        self.event_risk_events = [
            event for event in self.event_risk_events
            if event.timestamp > cutoff_time
        ]
        
        self.logger.info(f"Cleared {len(old_event_ids)} old events")
