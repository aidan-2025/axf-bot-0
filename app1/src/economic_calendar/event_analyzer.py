#!/usr/bin/env python3
"""
Event Analyzer
Analyzes economic events for market impact and trading opportunities
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from .models import EconomicEvent, EventImpact, EventCategory, Country, Currency

logger = logging.getLogger(__name__)

@dataclass
class EventAnalysisResult:
    """Result of event analysis"""
    event: EconomicEvent
    market_impact_score: float
    volatility_expected: float
    affected_currency_pairs: List[str]
    trading_opportunities: List[str]
    risk_factors: List[str]
    confidence: float
    analysis_timestamp: datetime

@dataclass
class EventAnalysisConfig:
    """Configuration for event analysis"""
    # Impact scoring weights
    impact_weight: float = 0.4
    category_weight: float = 0.3
    country_weight: float = 0.2
    timing_weight: float = 0.1
    
    # Volatility calculation
    base_volatility: float = 0.1
    high_impact_multiplier: float = 2.0
    very_high_impact_multiplier: float = 3.0
    
    # Currency pair mapping
    currency_pair_mapping: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.currency_pair_mapping is None:
            self.currency_pair_mapping = {
                'USD': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
                'EUR': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD'],
                'GBP': ['GBPUSD', 'EURGBP', 'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD'],
                'JPY': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY'],
                'CHF': ['USDCHF', 'EURCHF', 'GBPCHF', 'AUDCHF', 'CADCHF', 'CHFJPY'],
                'AUD': ['AUDUSD', 'EURAUD', 'GBPAUD', 'AUDJPY', 'AUDCHF', 'AUDCAD'],
                'CAD': ['USDCAD', 'EURCAD', 'GBPCAD', 'CADJPY', 'AUDCAD', 'CADCHF'],
                'NZD': ['NZDUSD', 'EURNZD', 'GBPNZD', 'NZDJPY', 'NZDCHF', 'NZDCAD']
            }

class EventAnalyzer:
    """Analyzes economic events for market impact and trading opportunities"""
    
    def __init__(self, config: EventAnalysisConfig):
        """Initialize event analyzer"""
        self.config = config
        self._initialized = False
    
    async def initialize(self):
        """Initialize the event analyzer"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing event analyzer")
            self._initialized = True
            logger.info("Event analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize event analyzer: {e}")
            raise
    
    async def analyze_event(self, event: EconomicEvent) -> EventAnalysisResult:
        """Analyze a single economic event"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Calculate market impact score
            market_impact_score = self._calculate_market_impact_score(event)
            
            # Calculate expected volatility
            volatility_expected = self._calculate_expected_volatility(event)
            
            # Determine affected currency pairs
            affected_pairs = self._get_affected_currency_pairs(event)
            
            # Identify trading opportunities
            trading_opportunities = self._identify_trading_opportunities(event)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(event)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(event)
            
            return EventAnalysisResult(
                event=event,
                market_impact_score=market_impact_score,
                volatility_expected=volatility_expected,
                affected_currency_pairs=affected_pairs,
                trading_opportunities=trading_opportunities,
                risk_factors=risk_factors,
                confidence=confidence,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing event {event.event_id}: {e}")
            # Return minimal analysis result
            return EventAnalysisResult(
                event=event,
                market_impact_score=0.0,
                volatility_expected=0.0,
                affected_currency_pairs=[],
                trading_opportunities=[],
                risk_factors=["Analysis failed"],
                confidence=0.0,
                analysis_timestamp=datetime.now()
            )
    
    async def analyze_events(self, events: List[EconomicEvent]) -> List[EventAnalysisResult]:
        """Analyze multiple economic events"""
        if not self._initialized:
            await self.initialize()
        
        results = []
        
        for event in events:
            try:
                result = await self.analyze_event(event)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing event {event.event_id}: {e}")
                continue
        
        logger.info(f"Analyzed {len(results)} events")
        return results
    
    def _calculate_market_impact_score(self, event: EconomicEvent) -> float:
        """Calculate market impact score for an event"""
        score = 0.0
        
        # Impact level scoring
        impact_scores = {
            EventImpact.LOW: 0.2,
            EventImpact.MEDIUM: 0.5,
            EventImpact.HIGH: 0.8,
            EventImpact.VERY_HIGH: 1.0
        }
        score += impact_scores.get(event.impact, 0.5) * self.config.impact_weight
        
        # Category scoring
        category_scores = {
            EventCategory.INTEREST_RATE: 1.0,
            EventCategory.MONETARY_POLICY: 0.9,
            EventCategory.GDP: 0.8,
            EventCategory.INFLATION: 0.8,
            EventCategory.UNEMPLOYMENT: 0.7,
            EventCategory.RETAIL_SALES: 0.6,
            EventCategory.MANUFACTURING: 0.6,
            EventCategory.TRADE_BALANCE: 0.5,
            EventCategory.CONSUMER_CONFIDENCE: 0.4,
            EventCategory.CENTRAL_BANK_MEETING: 0.9,
            EventCategory.CENTRAL_BANK_SPEECH: 0.7,
            EventCategory.OTHER: 0.3
        }
        score += category_scores.get(event.category, 0.3) * self.config.category_weight
        
        # Country scoring (major economies have higher impact)
        country_scores = {
            Country.US: 1.0,
            Country.EU: 0.9,
            Country.UK: 0.8,
            Country.JP: 0.7,
            Country.CA: 0.6,
            Country.AU: 0.6,
            Country.NZ: 0.5,
            Country.CH: 0.5,
            Country.CN: 0.8,
            Country.OTHER: 0.3
        }
        score += country_scores.get(event.country, 0.3) * self.config.country_weight
        
        # Timing scoring (events closer to market hours have higher impact)
        now = datetime.now()
        time_diff = abs((event.event_time - now).total_seconds())
        hours_diff = time_diff / 3600
        
        if hours_diff <= 1:
            timing_score = 1.0
        elif hours_diff <= 4:
            timing_score = 0.8
        elif hours_diff <= 24:
            timing_score = 0.6
        else:
            timing_score = 0.4
        
        score += timing_score * self.config.timing_weight
        
        return min(1.0, max(0.0, score))
    
    def _calculate_expected_volatility(self, event: EconomicEvent) -> float:
        """Calculate expected volatility for an event"""
        base_volatility = self.config.base_volatility
        
        # Adjust based on impact level
        if event.impact == EventImpact.VERY_HIGH:
            multiplier = self.config.very_high_impact_multiplier
        elif event.impact == EventImpact.HIGH:
            multiplier = self.config.high_impact_multiplier
        elif event.impact == EventImpact.MEDIUM:
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        # Adjust based on category
        category_multipliers = {
            EventCategory.INTEREST_RATE: 2.0,
            EventCategory.MONETARY_POLICY: 1.8,
            EventCategory.GDP: 1.5,
            EventCategory.INFLATION: 1.5,
            EventCategory.UNEMPLOYMENT: 1.3,
            EventCategory.RETAIL_SALES: 1.2,
            EventCategory.MANUFACTURING: 1.2,
            EventCategory.TRADE_BALANCE: 1.1,
            EventCategory.CONSUMER_CONFIDENCE: 1.1,
            EventCategory.CENTRAL_BANK_MEETING: 1.8,
            EventCategory.CENTRAL_BANK_SPEECH: 1.3,
            EventCategory.OTHER: 1.0
        }
        
        category_multiplier = category_multipliers.get(event.category, 1.0)
        
        return base_volatility * multiplier * category_multiplier
    
    def _get_affected_currency_pairs(self, event: EconomicEvent) -> List[str]:
        """Get currency pairs affected by an event"""
        currency = event.currency.value
        return self.config.currency_pair_mapping.get(currency, [])
    
    def _identify_trading_opportunities(self, event: EconomicEvent) -> List[str]:
        """Identify trading opportunities from an event"""
        opportunities = []
        
        # High impact events create more opportunities
        if event.impact in [EventImpact.HIGH, EventImpact.VERY_HIGH]:
            opportunities.append("High volatility expected - good for scalping")
            
            if event.category == EventCategory.INTEREST_RATE:
                opportunities.append("Interest rate decision - potential for large moves")
            elif event.category == EventCategory.GDP:
                opportunities.append("GDP release - trend continuation likely")
            elif event.category == EventCategory.INFLATION:
                opportunities.append("Inflation data - central bank policy implications")
        
        # Specific opportunities based on category
        if event.category == EventCategory.UNEMPLOYMENT:
            opportunities.append("Employment data - USD strength/weakness potential")
        elif event.category == EventCategory.RETAIL_SALES:
            opportunities.append("Consumer spending data - economic health indicator")
        elif event.category == EventCategory.MANUFACTURING:
            opportunities.append("Manufacturing data - industrial sector impact")
        
        # Central bank events
        if event.category in [EventCategory.CENTRAL_BANK_MEETING, EventCategory.CENTRAL_BANK_SPEECH]:
            opportunities.append("Central bank event - policy direction signals")
        
        return opportunities
    
    def _identify_risk_factors(self, event: EconomicEvent) -> List[str]:
        """Identify risk factors for an event"""
        risk_factors = []
        
        # High impact events are inherently risky
        if event.impact == EventImpact.VERY_HIGH:
            risk_factors.append("Very high impact event - extreme volatility expected")
        elif event.impact == EventImpact.HIGH:
            risk_factors.append("High impact event - significant volatility expected")
        
        # Category-specific risks
        if event.category == EventCategory.INTEREST_RATE:
            risk_factors.append("Interest rate decision - potential for unexpected outcomes")
        elif event.category == EventCategory.GDP:
            risk_factors.append("GDP data - economic growth implications")
        elif event.category == EventCategory.INFLATION:
            risk_factors.append("Inflation data - central bank policy implications")
        
        # Timing risks
        now = datetime.now()
        time_diff = (event.event_time - now).total_seconds()
        if time_diff < 3600:  # Less than 1 hour
            risk_factors.append("Event very soon - limited preparation time")
        elif time_diff < 14400:  # Less than 4 hours
            risk_factors.append("Event approaching - monitor positions closely")
        
        # Data quality risks
        if not event.forecast:
            risk_factors.append("No forecast available - increased uncertainty")
        
        return risk_factors
    
    def _calculate_confidence(self, event: EconomicEvent) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for events with more data
        if event.forecast is not None:
            confidence += 0.2
        if event.previous is not None:
            confidence += 0.1
        if event.description:
            confidence += 0.1
        
        # Higher confidence for well-known categories
        high_confidence_categories = [
            EventCategory.INTEREST_RATE,
            EventCategory.GDP,
            EventCategory.INFLATION,
            EventCategory.UNEMPLOYMENT
        ]
        if event.category in high_confidence_categories:
            confidence += 0.1
        
        # Higher confidence for major economies
        major_economies = [Country.US, Country.EU, Country.UK, Country.JP]
        if event.country in major_economies:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    async def get_analysis_summary(self, results: List[EventAnalysisResult]) -> Dict[str, Any]:
        """Get summary of event analysis results"""
        if not results:
            return {"message": "No events analyzed"}
        
        # Calculate aggregate statistics
        avg_impact_score = np.mean([r.market_impact_score for r in results])
        avg_volatility = np.mean([r.volatility_expected for r in results])
        avg_confidence = np.mean([r.confidence for r in results])
        
        # Count by impact level
        high_impact_count = sum(1 for r in results if r.event.impact in [EventImpact.HIGH, EventImpact.VERY_HIGH])
        
        # Count by category
        category_counts = {}
        for result in results:
            category = result.event.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Get top opportunities
        all_opportunities = []
        for result in results:
            all_opportunities.extend(result.trading_opportunities)
        
        opportunity_counts = {}
        for opp in all_opportunities:
            opportunity_counts[opp] = opportunity_counts.get(opp, 0) + 1
        
        top_opportunities = sorted(opportunity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_events": len(results),
            "average_impact_score": round(avg_impact_score, 3),
            "average_volatility": round(avg_volatility, 3),
            "average_confidence": round(avg_confidence, 3),
            "high_impact_events": high_impact_count,
            "category_breakdown": category_counts,
            "top_opportunities": top_opportunities
        }
    
    async def close(self):
        """Close the event analyzer"""
        self._initialized = False
        logger.info("Event analyzer closed")

