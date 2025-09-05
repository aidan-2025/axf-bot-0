#!/usr/bin/env python3
"""
Calendar Processor
Processes and integrates economic calendar data with market analysis
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .models import EconomicEvent, CalendarFilter, CalendarStats
from .calendar_service import EconomicCalendarService, CalendarServiceConfig
from .event_analyzer import EventAnalyzer, EventAnalysisConfig, EventAnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class CalendarProcessorConfig:
    """Configuration for calendar processor"""
    # Service configuration
    calendar_service_config: CalendarServiceConfig
    
    # Analyzer configuration
    event_analyzer_config: EventAnalysisConfig
    
    # Processing settings
    auto_analyze_events: bool = True
    analysis_batch_size: int = 50
    update_interval_minutes: int = 30
    
    # Integration settings
    enable_sentiment_integration: bool = True
    enable_market_data_integration: bool = True

class CalendarProcessor:
    """Processes and integrates economic calendar data"""
    
    def __init__(self, config: CalendarProcessorConfig):
        """Initialize calendar processor"""
        self.config = config
        
        # Initialize services
        self.calendar_service = EconomicCalendarService(config.calendar_service_config)
        self.event_analyzer = EventAnalyzer(config.event_analyzer_config)
        
        # Processing state
        self._initialized = False
        self._processing = False
        self._last_update = None
        
        # Analysis results cache
        self.analysis_results: Dict[str, EventAnalysisResult] = {}
        self.analysis_timestamps: Dict[str, datetime] = {}
    
    async def initialize(self):
        """Initialize the calendar processor"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing calendar processor")
            
            # Initialize services
            await self.calendar_service.initialize()
            await self.event_analyzer.initialize()
            
            # Start background processing if enabled
            if self.config.auto_analyze_events:
                asyncio.create_task(self._background_processing())
            
            self._initialized = True
            logger.info("Calendar processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize calendar processor: {e}")
            raise
    
    async def process_events(self, 
                           filter_criteria: Optional[CalendarFilter] = None,
                           force_analysis: bool = False) -> List[EventAnalysisResult]:
        """Process economic events and return analysis results"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get events from calendar service
            events = await self.calendar_service.get_events(filter_criteria)
            
            if not events:
                logger.info("No events found for processing")
                return []
            
            # Analyze events
            analysis_results = []
            
            for event in events:
                # Check if we already have analysis for this event
                if not force_analysis and event.event_id in self.analysis_results:
                    analysis_results.append(self.analysis_results[event.event_id])
                    continue
                
                # Analyze the event
                result = await self.event_analyzer.analyze_event(event)
                analysis_results.append(result)
                
                # Cache the result
                self.analysis_results[event.event_id] = result
                self.analysis_timestamps[event.event_id] = datetime.now()
            
            logger.info(f"Processed {len(analysis_results)} events")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error processing events: {e}")
            return []
    
    async def get_upcoming_analysis(self, 
                                   hours_ahead: int = 24,
                                   countries: Optional[List[str]] = None,
                                   impacts: Optional[List[str]] = None) -> List[EventAnalysisResult]:
        """Get analysis for upcoming events"""
        # Create filter criteria
        filter_criteria = CalendarFilter(
            countries=[self._parse_country(c) for c in countries] if countries else None,
            impacts=[self._parse_impact(i) for i in impacts] if impacts else None,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(hours=hours_ahead)
        )
        
        return await self.process_events(filter_criteria)
    
    async def get_high_impact_analysis(self, 
                                     hours_ahead: int = 24,
                                     countries: Optional[List[str]] = None) -> List[EventAnalysisResult]:
        """Get analysis for high impact events"""
        return await self.get_upcoming_analysis(
            hours_ahead=hours_ahead,
            countries=countries,
            impacts=['high', 'very_high']
        )
    
    async def get_event_analysis(self, event_id: str) -> Optional[EventAnalysisResult]:
        """Get analysis for a specific event"""
        if event_id in self.analysis_results:
            return self.analysis_results[event_id]
        
        # Try to find and analyze the event
        events = await self.calendar_service.get_events()
        event = next((e for e in events if e.event_id == event_id), None)
        
        if event:
            result = await self.event_analyzer.analyze_event(event)
            self.analysis_results[event_id] = result
            self.analysis_timestamps[event_id] = datetime.now()
            return result
        
        return None
    
    async def get_analysis_summary(self, 
                                 hours_ahead: int = 24,
                                 countries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get summary of event analysis"""
        results = await self.get_upcoming_analysis(hours_ahead, countries)
        return await self.event_analyzer.get_analysis_summary(results)
    
    async def get_calendar_stats(self) -> CalendarStats:
        """Get calendar statistics"""
        return await self.calendar_service.get_calendar_stats()
    
    async def update_events(self):
        """Update events from all sources"""
        if not self._initialized:
            await self.initialize()
        
        try:
            await self.calendar_service.update_events()
            self._last_update = datetime.now()
            logger.info("Events updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating events: {e}")
    
    async def _background_processing(self):
        """Background processing loop"""
        while self._initialized:
            try:
                if not self._processing:
                    self._processing = True
                    
                    # Update events
                    await self.update_events()
                    
                    # Process recent events
                    recent_events = await self.calendar_service.get_events(
                        CalendarFilter(
                            start_date=datetime.now() - timedelta(hours=1),
                            end_date=datetime.now() + timedelta(hours=24)
                        )
                    )
                    
                    if recent_events:
                        await self.process_events(force_analysis=True)
                    
                    self._processing = False
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                self._processing = False
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
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
    
    def _parse_impact(self, impact_str: str):
        """Parse impact string to EventImpact enum"""
        from .models import EventImpact
        impact_mapping = {
            'low': EventImpact.LOW,
            'medium': EventImpact.MEDIUM,
            'high': EventImpact.HIGH,
            'very_high': EventImpact.VERY_HIGH
        }
        return impact_mapping.get(impact_str.lower(), EventImpact.MEDIUM)
    
    async def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "processing": self._processing,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "cached_analyses": len(self.analysis_results),
            "auto_analyze_events": self.config.auto_analyze_events,
            "update_interval_minutes": self.config.update_interval_minutes,
            "calendar_service_initialized": self.calendar_service._initialized,
            "event_analyzer_initialized": self.event_analyzer._initialized
        }
    
    async def close(self):
        """Close the calendar processor"""
        self._initialized = False
        self._processing = False
        
        # Close services
        await self.calendar_service.close()
        await self.event_analyzer.close()
        
        # Clear caches
        self.analysis_results.clear()
        self.analysis_timestamps.clear()
        
        logger.info("Calendar processor closed")

