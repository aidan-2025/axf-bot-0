#!/usr/bin/env python3
"""
Schema Validator
Validates data structure, types, and format compliance
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
import math

from ..brokers.broker_manager import PriceData, CandleData

logger = logging.getLogger(__name__)

@dataclass
class SchemaValidationResult:
    """Result of schema validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_time_ms: float = 0.0

class SchemaValidator:
    """Validates data schema and format compliance"""
    
    def __init__(self):
        """Initialize schema validator"""
        self.required_price_fields = ['instrument', 'time', 'bid', 'ask', 'spread']
        self.required_candle_fields = ['instrument', 'time', 'open', 'high', 'low', 'close', 'volume']
        
        # Data type validators
        self.type_validators = {
            'instrument': self._validate_instrument,
            'time': self._validate_timestamp,
            'bid': self._validate_price,
            'ask': self._validate_price,
            'spread': self._validate_spread,
            'open': self._validate_price,
            'high': self._validate_price,
            'low': self._validate_price,
            'close': self._validate_price,
            'volume': self._validate_volume
        }
        
        logger.info("SchemaValidator initialized")
    
    async def validate_price_data(self, price_data: PriceData) -> SchemaValidationResult:
        """Validate price data schema"""
        import time
        start_time = time.time()
        
        result = SchemaValidationResult(is_valid=True)
        
        try:
            # Check required fields
            for field in self.required_price_fields:
                if not hasattr(price_data, field):
                    result.errors.append(f"Missing required field: {field}")
                    result.is_valid = False
                elif getattr(price_data, field) is None:
                    result.errors.append(f"Required field is None: {field}")
                    result.is_valid = False
            
            if not result.is_valid:
                return result
            
            # Validate data types and values
            for field in self.required_price_fields:
                validator = self.type_validators.get(field)
                if validator:
                    field_value = getattr(price_data, field)
                    is_valid, error_msg = validator(field_value, field)
                    if not is_valid:
                        result.errors.append(f"{field}: {error_msg}")
                        result.is_valid = False
            
            # Validate business logic constraints
            if result.is_valid:
                business_errors = self._validate_price_business_logic(price_data)
                result.errors.extend(business_errors)
                if business_errors:
                    result.is_valid = False
            
            # Add warnings for potential issues
            warnings = self._generate_price_warnings(price_data)
            result.warnings.extend(warnings)
            
        except Exception as e:
            result.errors.append(f"Schema validation error: {str(e)}")
            result.is_valid = False
            logger.error(f"Error validating price data schema: {e}")
        
        finally:
            result.validation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    async def validate_candle_data(self, candle_data: CandleData) -> SchemaValidationResult:
        """Validate candle data schema"""
        import time
        start_time = time.time()
        
        result = SchemaValidationResult(is_valid=True)
        
        try:
            # Check required fields
            for field in self.required_candle_fields:
                if not hasattr(candle_data, field):
                    result.errors.append(f"Missing required field: {field}")
                    result.is_valid = False
                elif getattr(candle_data, field) is None:
                    result.errors.append(f"Required field is None: {field}")
                    result.is_valid = False
            
            if not result.is_valid:
                return result
            
            # Validate data types and values
            for field in self.required_candle_fields:
                validator = self.type_validators.get(field)
                if validator:
                    field_value = getattr(candle_data, field)
                    is_valid, error_msg = validator(field_value, field)
                    if not is_valid:
                        result.errors.append(f"{field}: {error_msg}")
                        result.is_valid = False
            
            # Validate business logic constraints
            if result.is_valid:
                business_errors = self._validate_candle_business_logic(candle_data)
                result.errors.extend(business_errors)
                if business_errors:
                    result.is_valid = False
            
            # Add warnings for potential issues
            warnings = self._generate_candle_warnings(candle_data)
            result.warnings.extend(warnings)
            
        except Exception as e:
            result.errors.append(f"Schema validation error: {str(e)}")
            result.is_valid = False
            logger.error(f"Error validating candle data schema: {e}")
        
        finally:
            result.validation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _validate_instrument(self, value: Any, field_name: str) -> tuple[bool, str]:
        """Validate instrument field"""
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        
        if not value.strip():
            return False, "Instrument cannot be empty"
        
        # Check for valid forex pair format (basic)
        if len(value) < 6 or '_' not in value:
            return False, "Invalid instrument format (expected format: XXX_YYY)"
        
        return True, ""
    
    def _validate_timestamp(self, value: Any, field_name: str) -> tuple[bool, str]:
        """Validate timestamp field"""
        if not isinstance(value, datetime):
            return False, f"Expected datetime, got {type(value).__name__}"
        
        # Check if timestamp is reasonable (not too far in past or future)
        now = datetime.utcnow()
        if value < datetime(2000, 1, 1):
            return False, "Timestamp too far in the past"
        
        if value > now.replace(year=now.year + 1):
            return False, "Timestamp too far in the future"
        
        return True, ""
    
    def _validate_price(self, value: Any, field_name: str) -> tuple[bool, str]:
        """Validate price field"""
        if not isinstance(value, (int, float)):
            return False, f"Expected number, got {type(value).__name__}"
        
        if math.isnan(value) or math.isinf(value):
            return False, "Price cannot be NaN or infinite"
        
        if value <= 0:
            return False, "Price must be positive"
        
        # Check for reasonable price range (basic forex validation)
        if value < 0.1 or value > 1000:
            return False, f"Price {value} is outside reasonable range (0.1-1000)"
        
        return True, ""
    
    def _validate_spread(self, value: Any, field_name: str) -> tuple[bool, str]:
        """Validate spread field"""
        if not isinstance(value, (int, float)):
            return False, f"Expected number, got {type(value).__name__}"
        
        if math.isnan(value) or math.isinf(value):
            return False, "Spread cannot be NaN or infinite"
        
        if value < 0:
            return False, "Spread cannot be negative"
        
        # Check for reasonable spread (max 1000 pips = 0.1)
        if value > 0.1:
            return False, f"Spread {value} is unreasonably large (max 0.1)"
        
        return True, ""
    
    def _validate_volume(self, value: Any, field_name: str) -> tuple[bool, str]:
        """Validate volume field"""
        if not isinstance(value, (int, float)):
            return False, f"Expected number, got {type(value).__name__}"
        
        if math.isnan(value) or math.isinf(value):
            return False, "Volume cannot be NaN or infinite"
        
        if value < 0:
            return False, "Volume cannot be negative"
        
        return True, ""
    
    def _validate_price_business_logic(self, price_data: PriceData) -> List[str]:
        """Validate price data business logic"""
        errors = []
        
        # Bid must be less than ask
        if price_data.bid >= price_data.ask:
            errors.append(f"Bid ({price_data.bid}) must be less than ask ({price_data.ask})")
        
        # Spread should match ask - bid
        expected_spread = price_data.ask - price_data.bid
        if abs(price_data.spread - expected_spread) > 1e-8:
            errors.append(f"Spread mismatch: expected {expected_spread}, got {price_data.spread}")
        
        return errors
    
    def _validate_candle_business_logic(self, candle_data: CandleData) -> List[str]:
        """Validate candle data business logic"""
        errors = []
        
        # High must be >= low
        if candle_data.high < candle_data.low:
            errors.append(f"High ({candle_data.high}) must be >= low ({candle_data.low})")
        
        # Open and close must be between high and low
        if not (candle_data.low <= candle_data.open <= candle_data.high):
            errors.append(f"Open ({candle_data.open}) must be between low and high")
        
        if not (candle_data.low <= candle_data.close <= candle_data.high):
            errors.append(f"Close ({candle_data.close}) must be between low and high")
        
        # All prices must be positive
        for field in ['open', 'high', 'low', 'close']:
            value = getattr(candle_data, field)
            if value <= 0:
                errors.append(f"{field.capitalize()} ({value}) must be positive")
        
        return errors
    
    def _generate_price_warnings(self, price_data: PriceData) -> List[str]:
        """Generate warnings for price data"""
        warnings = []
        
        # Check for very small spreads (might indicate data quality issues)
        if price_data.spread < 0.0001:
            warnings.append(f"Very small spread: {price_data.spread}")
        
        # Check for very large spreads (might indicate market stress)
        if price_data.spread > 0.01:
            warnings.append(f"Large spread: {price_data.spread}")
        
        # Check for round numbers (might indicate synthetic data)
        if (price_data.bid * 10000) % 1 == 0 and (price_data.ask * 10000) % 1 == 0:
            warnings.append("Both bid and ask are round numbers (possible synthetic data)")
        
        return warnings
    
    def _generate_candle_warnings(self, candle_data: CandleData) -> List[str]:
        """Generate warnings for candle data"""
        warnings = []
        
        # Check for very small ranges (might indicate low volatility or data issues)
        range_size = candle_data.high - candle_data.low
        if range_size < 0.0001:
            warnings.append(f"Very small candle range: {range_size}")
        
        # Check for very large ranges (might indicate market stress)
        if range_size > 0.1:
            warnings.append(f"Large candle range: {range_size}")
        
        # Check for doji candles (open == close)
        if abs(candle_data.open - candle_data.close) < 1e-8:
            warnings.append("Doji candle detected (open == close)")
        
        # Check for hammer/shooting star patterns (basic)
        body_size = abs(candle_data.close - candle_data.open)
        upper_shadow = candle_data.high - max(candle_data.open, candle_data.close)
        lower_shadow = min(candle_data.open, candle_data.close) - candle_data.low
        
        if lower_shadow > 2 * body_size and upper_shadow < body_size:
            warnings.append("Potential hammer pattern detected")
        elif upper_shadow > 2 * body_size and lower_shadow < body_size:
            warnings.append("Potential shooting star pattern detected")
        
        return warnings

# Example usage and testing
async def test_schema_validator():
    """Test the schema validator"""
    from ..brokers.broker_manager import PriceData, CandleData
    
    validator = SchemaValidator()
    
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
    
    # Test invalid price data
    invalid_price = PriceData(
        instrument="",  # Invalid instrument
        time=datetime.utcnow(),
        bid=1.1002,  # Bid > ask
        ask=1.1000,
        spread=0.0002
    )
    
    result = await validator.validate_price_data(invalid_price)
    print(f"Invalid price data: {result.is_valid}, Errors: {result.errors}")
    
    # Test valid candle data
    candle_data = CandleData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        open=1.1000,
        high=1.1005,
        low=1.0995,
        close=1.1002,
        volume=1000
    )
    
    result = await validator.validate_candle_data(candle_data)
    print(f"Valid candle data: {result.is_valid}, Errors: {result.errors}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_schema_validator())

