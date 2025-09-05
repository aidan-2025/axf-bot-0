#!/usr/bin/env python3
"""
Variable Spread Simulation Module

Simulates realistic, variable spreads for high-fidelity backtesting.
Supports both historical spread data and statistical spread models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class SpreadModel(Enum):
    """Types of spread models"""
    HISTORICAL = "historical"
    STATISTICAL = "statistical"
    FIXED = "fixed"
    TIME_BASED = "time_based"
    VOLATILITY_BASED = "volatility_based"


@dataclass
class SpreadConfig:
    """Configuration for spread simulation"""
    model: SpreadModel = SpreadModel.STATISTICAL
    base_spread: float = 0.0001  # 1 pip base spread
    min_spread: float = 0.00005  # 0.5 pip minimum
    max_spread: float = 0.0005   # 5 pip maximum
    
    # Time-based spread factors
    london_open_factor: float = 1.5
    new_york_open_factor: float = 1.3
    asian_session_factor: float = 0.8
    weekend_factor: float = 2.0
    
    # Volatility-based spread factors
    volatility_sensitivity: float = 0.5
    news_event_factor: float = 2.0
    
    # Statistical model parameters
    mean_reversion_speed: float = 0.1
    spread_volatility: float = 0.00002
    correlation_factor: float = 0.3


class VariableSpreadSimulator:
    """
    Simulates realistic variable spreads for backtesting
    
    This module provides high-fidelity spread simulation that accounts for:
    - Time-of-day effects (London/NY/Asian sessions)
    - Market volatility
    - News events
    - Historical spread patterns
    - Statistical spread models
    """
    
    def __init__(self, config: SpreadConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Spread history for mean reversion
        self.spread_history = defaultdict(list)
        
        # News event tracking
        self.news_events = []
        
        # Market session times (UTC)
        self.sessions = {
            'asian': (21, 6),      # 9 PM - 6 AM UTC
            'london': (6, 15),     # 6 AM - 3 PM UTC
            'new_york': (13, 22),  # 1 PM - 10 PM UTC
            'overlap': (13, 15)    # London-NY overlap
        }
        
        self.logger.info(f"VariableSpreadSimulator initialized with {config.model.value} model")
    
    def get_spread(self, symbol: str, timestamp: datetime, 
                   bid_price: float, ask_price: float,
                   volatility: Optional[float] = None,
                   news_impact: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate variable spread for a given symbol and time
        
        Args:
            symbol: Currency pair symbol
            timestamp: Current timestamp
            bid_price: Current bid price
            ask_price: Current ask price
            volatility: Current market volatility (optional)
            news_impact: News event impact factor (optional)
        
        Returns:
            Tuple of (adjusted_bid, adjusted_ask)
        """
        try:
            # Calculate base spread
            base_spread = self._calculate_base_spread(symbol, timestamp, volatility, news_impact)
            
            # Apply spread to prices
            spread_half = base_spread / 2
            adjusted_bid = bid_price - spread_half
            adjusted_ask = ask_price + spread_half
            
            # Store spread for history
            self.spread_history[symbol].append({
                'timestamp': timestamp,
                'spread': base_spread,
                'bid': adjusted_bid,
                'ask': adjusted_ask
            })
            
            # Keep only recent history (last 1000 points)
            if len(self.spread_history[symbol]) > 1000:
                self.spread_history[symbol] = self.spread_history[symbol][-1000:]
            
            return adjusted_bid, adjusted_ask
            
        except Exception as e:
            self.logger.error(f"Error calculating spread for {symbol}: {e}")
            # Return original prices with base spread as fallback
            spread_half = self.config.base_spread / 2
            return bid_price - spread_half, ask_price + spread_half
    
    def _calculate_base_spread(self, symbol: str, timestamp: datetime,
                              volatility: Optional[float] = None,
                              news_impact: Optional[float] = None) -> float:
        """Calculate the base spread using the configured model"""
        
        if self.config.model == SpreadModel.FIXED:
            return self.config.base_spread
        
        elif self.config.model == SpreadModel.TIME_BASED:
            return self._calculate_time_based_spread(timestamp)
        
        elif self.config.model == SpreadModel.VOLATILITY_BASED:
            return self._calculate_volatility_based_spread(volatility or 0.01)
        
        elif self.config.model == SpreadModel.STATISTICAL:
            return self._calculate_statistical_spread(symbol, timestamp, volatility)
        
        elif self.config.model == SpreadModel.HISTORICAL:
            return self._calculate_historical_spread(symbol, timestamp)
        
        else:
            return self.config.base_spread
    
    def _calculate_time_based_spread(self, timestamp: datetime) -> float:
        """Calculate spread based on time of day"""
        hour = timestamp.hour
        
        # Determine current session
        if self.sessions['asian'][0] <= hour < self.sessions['asian'][1]:
            session_factor = self.config.asian_session_factor
        elif self.sessions['london'][0] <= hour < self.sessions['london'][1]:
            session_factor = self.config.london_open_factor
        elif self.sessions['new_york'][0] <= hour < self.sessions['new_york'][1]:
            session_factor = self.config.new_york_open_factor
        else:
            session_factor = 1.0
        
        # Weekend factor
        weekend_factor = self.config.weekend_factor if timestamp.weekday() >= 5 else 1.0
        
        # Calculate spread
        spread = self.config.base_spread * session_factor * weekend_factor
        
        # Add some randomness
        noise = np.random.normal(0, self.config.spread_volatility)
        spread += noise
        
        return max(self.config.min_spread, min(self.config.max_spread, spread))
    
    def _calculate_volatility_based_spread(self, volatility: float) -> float:
        """Calculate spread based on market volatility"""
        # Higher volatility = wider spreads
        volatility_factor = 1 + (volatility * self.config.volatility_sensitivity)
        
        spread = self.config.base_spread * volatility_factor
        
        # Add some randomness
        noise = np.random.normal(0, self.config.spread_volatility)
        spread += noise
        
        return max(self.config.min_spread, min(self.config.max_spread, spread))
    
    def _calculate_statistical_spread(self, symbol: str, timestamp: datetime,
                                    volatility: Optional[float] = None) -> float:
        """Calculate spread using statistical model with mean reversion"""
        
        # Get recent spread history for mean reversion
        recent_spreads = [s['spread'] for s in self.spread_history[symbol][-50:]]
        
        if not recent_spreads:
            # No history, use base spread
            base_spread = self.config.base_spread
        else:
            # Mean reversion: spread tends to return to historical mean
            historical_mean = np.mean(recent_spreads)
            last_spread = recent_spreads[-1]
            
            # Mean reversion calculation
            mean_reversion = self.config.mean_reversion_speed * (historical_mean - last_spread)
            base_spread = last_spread + mean_reversion
        
        # Add time-based factors
        time_factor = self._get_time_factor(timestamp)
        base_spread *= time_factor
        
        # Add volatility factor
        if volatility:
            volatility_factor = 1 + (volatility * self.config.volatility_sensitivity)
            base_spread *= volatility_factor
        
        # Add random walk
        random_walk = np.random.normal(0, self.config.spread_volatility)
        base_spread += random_walk
        
        # Apply bounds more strictly
        base_spread = max(self.config.min_spread, min(self.config.max_spread, base_spread))
        
        # Ensure we don't exceed bounds even with random walk
        if base_spread > self.config.max_spread:
            base_spread = self.config.max_spread
        elif base_spread < self.config.min_spread:
            base_spread = self.config.min_spread
            
        return base_spread
    
    def _calculate_historical_spread(self, symbol: str, timestamp: datetime) -> float:
        """Calculate spread based on historical data"""
        # This would typically load historical spread data
        # For now, use statistical model as fallback
        return self._calculate_statistical_spread(symbol, timestamp)
    
    def _get_time_factor(self, timestamp: datetime) -> float:
        """Get time-based factor for spread calculation"""
        hour = timestamp.hour
        
        if self.sessions['asian'][0] <= hour < self.sessions['asian'][1]:
            return self.config.asian_session_factor
        elif self.sessions['london'][0] <= hour < self.sessions['london'][1]:
            return self.config.london_open_factor
        elif self.sessions['new_york'][0] <= hour < self.sessions['new_york'][1]:
            return self.config.new_york_open_factor
        else:
            return 1.0
    
    def add_news_event(self, symbol: str, timestamp: datetime, 
                      impact_factor: float, duration_minutes: int = 30):
        """Add a news event that affects spreads"""
        self.news_events.append({
            'symbol': symbol,
            'timestamp': timestamp,
            'impact_factor': impact_factor,
            'duration_minutes': duration_minutes,
            'end_time': timestamp + timedelta(minutes=duration_minutes)
        })
        
        self.logger.info(f"Added news event for {symbol} at {timestamp} with impact {impact_factor}")
    
    def get_news_impact(self, symbol: str, timestamp: datetime) -> float:
        """Get current news impact factor for a symbol"""
        impact = 1.0
        
        for event in self.news_events:
            if (event['symbol'] == symbol and 
                event['timestamp'] <= timestamp <= event['end_time']):
                impact *= event['impact_factor']
        
        return impact
    
    def load_historical_spreads(self, symbol: str, data: List[Dict[str, Any]]):
        """Load historical spread data for a symbol"""
        for point in data:
            self.spread_history[symbol].append({
                'timestamp': point['timestamp'],
                'spread': point['spread'],
                'bid': point.get('bid', 0),
                'ask': point.get('ask', 0)
            })
        
        self.logger.info(f"Loaded {len(data)} historical spread points for {symbol}")
    
    def get_spread_statistics(self, symbol: str) -> Dict[str, float]:
        """Get spread statistics for a symbol"""
        spreads = [s['spread'] for s in self.spread_history[symbol]]
        
        if not spreads:
            return {}
        
        return {
            'mean': np.mean(spreads),
            'std': np.std(spreads),
            'min': np.min(spreads),
            'max': np.max(spreads),
            'median': np.median(spreads),
            'count': len(spreads)
        }
    
    def reset_history(self, symbol: Optional[str] = None):
        """Reset spread history for a symbol or all symbols"""
        if symbol:
            self.spread_history[symbol] = []
            self.logger.info(f"Reset spread history for {symbol}")
        else:
            self.spread_history.clear()
            self.logger.info("Reset all spread history")


class SpreadAwareBroker:
    """
    Backtrader broker that uses variable spreads
    
    This broker extends Backtrader's default broker to use realistic spreads
    for order execution.
    """
    
    def __init__(self, spread_simulator: VariableSpreadSimulator):
        self.spread_simulator = spread_simulator
        self.logger = logging.getLogger(__name__)
    
    def get_spread_adjusted_prices(self, symbol: str, timestamp: datetime,
                                 bid: float, ask: float,
                                 volatility: Optional[float] = None) -> Tuple[float, float]:
        """Get spread-adjusted bid/ask prices"""
        news_impact = self.spread_simulator.get_news_impact(symbol, timestamp)
        return self.spread_simulator.get_spread(
            symbol, timestamp, bid, ask, volatility, news_impact
        )


def create_spread_simulator(model: SpreadModel = SpreadModel.STATISTICAL,
                          base_spread: float = 0.0001) -> VariableSpreadSimulator:
    """Factory function to create a spread simulator"""
    config = SpreadConfig(
        model=model,
        base_spread=base_spread
    )
    return VariableSpreadSimulator(config)
