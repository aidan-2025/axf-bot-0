#!/usr/bin/env python3
"""
Range Validator
Validates data ranges, bounds, and domain constraints
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math

from ..brokers.broker_manager import PriceData, CandleData

logger = logging.getLogger(__name__)

@dataclass
class RangeValidationResult:
    """Result of range validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_time_ms: float = 0.0

@dataclass
class RangeValidationConfig:
    """Configuration for range validation"""
    # Price range validation
    price_range_multiplier: float = 2.0
    max_spread_pips: float = 50.0
    max_price_jump_pct: float = 5.0
    min_price: float = 0.1
    max_price: float = 1000.0
    
    # Candle range validation
    max_candle_range_pct: float = 10.0
    max_volume_multiplier: float = 10.0
    
    # Time range validation
    max_future_skew_seconds: int = 10
    max_past_skew_days: int = 365
    
    # Instrument-specific ranges
    instrument_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)

class RangeValidator:
    """Validates data ranges and domain constraints"""
    
    def __init__(self, config: Optional[RangeValidationConfig] = None):
        """Initialize range validator"""
        self.config = config or RangeValidationConfig()
        
        # Initialize instrument-specific ranges
        self._initialize_instrument_ranges()
        
        # Price history for trend analysis
        self.price_history: Dict[str, List[PriceData]] = {}
        self.candle_history: Dict[str, List[CandleData]] = {}
        self.max_history_size = 1000
        
        logger.info("RangeValidator initialized with config: %s", self.config)
    
    def _initialize_instrument_ranges(self):
        """Initialize instrument-specific price ranges"""
        # Common forex pairs with typical ranges
        self.config.instrument_ranges = {
            'EUR_USD': {'min': 0.8, 'max': 1.5, 'max_spread': 0.002},
            'GBP_USD': {'min': 1.0, 'max': 2.0, 'max_spread': 0.003},
            'USD_JPY': {'min': 80.0, 'max': 150.0, 'max_spread': 0.1},
            'USD_CHF': {'min': 0.7, 'max': 1.2, 'max_spread': 0.002},
            'AUD_USD': {'min': 0.5, 'max': 1.0, 'max_spread': 0.002},
            'USD_CAD': {'min': 1.0, 'max': 1.8, 'max_spread': 0.002},
            'NZD_USD': {'min': 0.4, 'max': 0.9, 'max_spread': 0.003},
            'EUR_GBP': {'min': 0.7, 'max': 1.0, 'max_spread': 0.002},
            'EUR_JPY': {'min': 100.0, 'max': 200.0, 'max_spread': 0.1},
            'GBP_JPY': {'min': 120.0, 'max': 250.0, 'max_spread': 0.2}
        }
    
    async def validate_price_data(self, price_data: PriceData, 
                                previous_data: Optional[PriceData] = None) -> RangeValidationResult:
        """Validate price data ranges"""
        import time
        start_time = time.time()
        
        result = RangeValidationResult(is_valid=True)
        
        try:
            # Basic range validation
            basic_errors = self._validate_basic_price_ranges(price_data)
            result.errors.extend(basic_errors)
            if basic_errors:
                result.is_valid = False
            
            # Instrument-specific range validation
            if result.is_valid:
                instrument_errors = self._validate_instrument_price_ranges(price_data)
                result.errors.extend(instrument_errors)
                if instrument_errors:
                    result.is_valid = False
            
            # Time range validation
            time_errors = self._validate_time_ranges(price_data.time)
            result.errors.extend(time_errors)
            if time_errors:
                result.is_valid = False
            
            # Trend-based validation
            if result.is_valid and previous_data:
                trend_errors = self._validate_price_trends(price_data, previous_data)
                result.errors.extend(trend_errors)
                if trend_errors:
                    result.is_valid = False
            
            # Historical context validation
            if result.is_valid:
                historical_errors = self._validate_historical_context(price_data)
                result.errors.extend(historical_errors)
                if historical_errors:
                    result.is_valid = False
            
            # Generate warnings
            warnings = self._generate_price_warnings(price_data, previous_data)
            result.warnings.extend(warnings)
            
            # Update price history
            self._update_price_history(price_data)
            
        except Exception as e:
            result.errors.append(f"Range validation error: {str(e)}")
            result.is_valid = False
            logger.error(f"Error validating price data ranges: {e}")
        
        finally:
            result.validation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    async def validate_candle_data(self, candle_data: CandleData,
                                 previous_data: Optional[CandleData] = None) -> RangeValidationResult:
        """Validate candle data ranges"""
        import time
        start_time = time.time()
        
        result = RangeValidationResult(is_valid=True)
        
        try:
            # Basic range validation
            basic_errors = self._validate_basic_candle_ranges(candle_data)
            result.errors.extend(basic_errors)
            if basic_errors:
                result.is_valid = False
            
            # OHLC consistency validation
            if result.is_valid:
                ohlc_errors = self._validate_ohlc_consistency(candle_data)
                result.errors.extend(ohlc_errors)
                if ohlc_errors:
                    result.is_valid = False
            
            # Volume validation
            volume_errors = self._validate_volume_ranges(candle_data)
            result.errors.extend(volume_errors)
            if volume_errors:
                result.is_valid = False
            
            # Time range validation
            time_errors = self._validate_time_ranges(candle_data.time)
            result.errors.extend(time_errors)
            if time_errors:
                result.is_valid = False
            
            # Generate warnings
            warnings = self._generate_candle_warnings(candle_data, previous_data)
            result.warnings.extend(warnings)
            
            # Update candle history
            self._update_candle_history(candle_data)
            
        except Exception as e:
            result.errors.append(f"Range validation error: {str(e)}")
            result.is_valid = False
            logger.error(f"Error validating candle data ranges: {e}")
        
        finally:
            result.validation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _validate_basic_price_ranges(self, price_data: PriceData) -> List[str]:
        """Validate basic price ranges"""
        errors = []
        
        # Check if prices are within absolute bounds
        if price_data.bid < self.config.min_price or price_data.bid > self.config.max_price:
            errors.append(f"Bid price {price_data.bid} outside absolute range [{self.config.min_price}, {self.config.max_price}]")
        
        if price_data.ask < self.config.min_price or price_data.ask > self.config.max_price:
            errors.append(f"Ask price {price_data.ask} outside absolute range [{self.config.min_price}, {self.config.max_price}]")
        
        # Check spread bounds
        max_spread = self.config.max_spread_pips / 10000  # Convert pips to price units
        if price_data.spread > max_spread:
            errors.append(f"Spread {price_data.spread} exceeds maximum {max_spread}")
        
        return errors
    
    def _validate_instrument_price_ranges(self, price_data: PriceData) -> List[str]:
        """Validate instrument-specific price ranges"""
        errors = []
        
        instrument = price_data.instrument
        if instrument not in self.config.instrument_ranges:
            return errors  # No specific validation for unknown instruments
        
        ranges = self.config.instrument_ranges[instrument]
        
        # Check price bounds
        if price_data.bid < ranges['min'] or price_data.bid > ranges['max']:
            errors.append(f"Bid price {price_data.bid} outside instrument range [{ranges['min']}, {ranges['max']}]")
        
        if price_data.ask < ranges['min'] or price_data.ask > ranges['max']:
            errors.append(f"Ask price {price_data.ask} outside instrument range [{ranges['min']}, {ranges['max']}]")
        
        # Check spread bounds
        if price_data.spread > ranges['max_spread']:
            errors.append(f"Spread {price_data.spread} exceeds instrument maximum {ranges['max_spread']}")
        
        return errors
    
    def _validate_time_ranges(self, timestamp: datetime) -> List[str]:
        """Validate timestamp ranges"""
        errors = []
        
        now = datetime.utcnow()
        
        # Check if timestamp is too far in the future
        if timestamp > now + timedelta(seconds=self.config.max_future_skew_seconds):
            errors.append(f"Timestamp {timestamp} is too far in the future (max {self.config.max_future_skew_seconds}s)")
        
        # Check if timestamp is too far in the past
        if timestamp < now - timedelta(days=self.config.max_past_skew_days):
            errors.append(f"Timestamp {timestamp} is too far in the past (max {self.config.max_past_skew_days} days)")
        
        return errors
    
    def _validate_price_trends(self, price_data: PriceData, previous_data: PriceData) -> List[str]:
        """Validate price trends and jumps"""
        errors = []
        
        # Calculate price change percentage
        prev_mid = (previous_data.bid + previous_data.ask) / 2
        curr_mid = (price_data.bid + price_data.ask) / 2
        
        if prev_mid > 0:
            price_change_pct = abs(curr_mid - prev_mid) / prev_mid * 100
            
            if price_change_pct > self.config.max_price_jump_pct:
                errors.append(f"Price jump {price_change_pct:.2f}% exceeds maximum {self.config.max_price_jump_pct}%")
        
        return errors
    
    def _validate_historical_context(self, price_data: PriceData) -> List[str]:
        """Validate against historical context"""
        errors = []
        
        instrument = price_data.instrument
        if instrument not in self.price_history or len(self.price_history[instrument]) < 10:
            return errors  # Not enough history for validation
        
        # Get recent price history
        recent_prices = self.price_history[instrument][-10:]
        prices = [(p.bid + p.ask) / 2 for p in recent_prices]
        
        if not prices:
            return errors
        
        # Calculate statistical bounds
        mean_price = sum(prices) / len(prices)
        price_std = math.sqrt(sum((p - mean_price) ** 2 for p in prices) / len(prices))
        
        # Check if current price is within reasonable bounds
        current_mid = (price_data.bid + price_data.ask) / 2
        z_score = abs(current_mid - mean_price) / max(price_std, 1e-8)
        
        if z_score > self.config.price_range_multiplier:
            errors.append(f"Price {current_mid} is {z_score:.2f} standard deviations from recent mean {mean_price}")
        
        return errors
    
    def _validate_basic_candle_ranges(self, candle_data: CandleData) -> List[str]:
        """Validate basic candle ranges"""
        errors = []
        
        # Check if all prices are within absolute bounds
        for field in ['open', 'high', 'low', 'close']:
            value = getattr(candle_data, field)
            if value < self.config.min_price or value > self.config.max_price:
                errors.append(f"{field.capitalize()} price {value} outside absolute range [{self.config.min_price}, {self.config.max_price}]")
        
        return errors
    
    def _validate_ohlc_consistency(self, candle_data: CandleData) -> List[str]:
        """Validate OHLC consistency"""
        errors = []
        
        # High must be >= all other prices
        if candle_data.high < max(candle_data.open, candle_data.close):
            errors.append(f"High {candle_data.high} must be >= max(open, close)")
        
        # Low must be <= all other prices
        if candle_data.low > min(candle_data.open, candle_data.close):
            errors.append(f"Low {candle_data.low} must be <= min(open, close)")
        
        # Check for reasonable range
        range_size = candle_data.high - candle_data.low
        if candle_data.close > 0:
            range_pct = range_size / candle_data.close * 100
            if range_pct > self.config.max_candle_range_pct:
                errors.append(f"Candle range {range_pct:.2f}% exceeds maximum {self.config.max_candle_range_pct}%")
        
        return errors
    
    def _validate_volume_ranges(self, candle_data: CandleData) -> List[str]:
        """Validate volume ranges"""
        errors = []
        
        if candle_data.volume < 0:
            errors.append(f"Volume {candle_data.volume} cannot be negative")
        
        # Check against historical volume if available
        instrument = candle_data.instrument
        if instrument in self.candle_history and len(self.candle_history[instrument]) > 0:
            recent_volumes = [c.volume for c in self.candle_history[instrument][-10:] if c.volume > 0]
            if recent_volumes:
                avg_volume = sum(recent_volumes) / len(recent_volumes)
                if candle_data.volume > avg_volume * self.config.max_volume_multiplier:
                    errors.append(f"Volume {candle_data.volume} is {candle_data.volume/avg_volume:.1f}x average volume")
        
        return errors
    
    def _generate_price_warnings(self, price_data: PriceData, previous_data: Optional[PriceData]) -> List[str]:
        """Generate warnings for price data"""
        warnings = []
        
        # Check for very small spreads
        if price_data.spread < 0.0001:
            warnings.append(f"Very small spread: {price_data.spread}")
        
        # Check for large spreads
        if price_data.spread > 0.01:
            warnings.append(f"Large spread: {price_data.spread}")
        
        # Check for price gaps
        if previous_data:
            prev_mid = (previous_data.bid + previous_data.ask) / 2
            curr_mid = (price_data.bid + price_data.ask) / 2
            gap_pct = abs(curr_mid - prev_mid) / prev_mid * 100
            if gap_pct > 1.0:  # 1% gap
                warnings.append(f"Price gap: {gap_pct:.2f}%")
        
        return warnings
    
    def _generate_candle_warnings(self, candle_data: CandleData, previous_data: Optional[CandleData]) -> List[str]:
        """Generate warnings for candle data"""
        warnings = []
        
        # Check for very small ranges
        range_size = candle_data.high - candle_data.low
        if range_size < 0.0001:
            warnings.append(f"Very small candle range: {range_size}")
        
        # Check for large ranges
        if candle_data.close > 0:
            range_pct = range_size / candle_data.close * 100
            if range_pct > 5.0:  # 5% range
                warnings.append(f"Large candle range: {range_pct:.2f}%")
        
        # Check for volume spikes
        if candle_data.volume > 0 and previous_data and previous_data.volume > 0:
            volume_change = candle_data.volume / previous_data.volume
            if volume_change > 5.0:  # 5x volume increase
                warnings.append(f"Volume spike: {volume_change:.1f}x previous volume")
        
        return warnings
    
    def _update_price_history(self, price_data: PriceData):
        """Update price history"""
        instrument = price_data.instrument
        if instrument not in self.price_history:
            self.price_history[instrument] = []
        
        self.price_history[instrument].append(price_data)
        
        # Maintain history size
        if len(self.price_history[instrument]) > self.max_history_size:
            self.price_history[instrument] = self.price_history[instrument][-self.max_history_size:]
    
    def _update_candle_history(self, candle_data: CandleData):
        """Update candle history"""
        instrument = candle_data.instrument
        if instrument not in self.candle_history:
            self.candle_history[instrument] = []
        
        self.candle_history[instrument].append(candle_data)
        
        # Maintain history size
        if len(self.candle_history[instrument]) > self.max_history_size:
            self.candle_history[instrument] = self.candle_history[instrument][-self.max_history_size:]

# Example usage and testing
async def test_range_validator():
    """Test the range validator"""
    from ..brokers.broker_manager import PriceData, CandleData
    
    config = RangeValidationConfig(
        max_price_jump_pct=2.0,
        max_spread_pips=20.0
    )
    
    validator = RangeValidator(config)
    
    # Test valid price data
    price_data = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=1.1000,
        ask=1.1002,
        spread=0.0002
    )
    
    result = await validator.validate_price_data(price_data)
    print(f"Valid price data: {result.is_valid}, Errors: {result.errors}")
    
    # Test invalid price data (out of range)
    invalid_price = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=0.5,  # Too low for EUR/USD
        ask=0.52,
        spread=0.02
    )
    
    result = await validator.validate_price_data(invalid_price)
    print(f"Invalid price data: {result.is_valid}, Errors: {result.errors}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_range_validator())

