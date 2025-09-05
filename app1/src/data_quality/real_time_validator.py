#!/usr/bin/env python3
"""
Real-time Data Validator
Validates incoming data in real-time with comprehensive checks
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .quality_thresholds import QualityThresholds

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    confidence_score: float
    validation_timestamp: datetime

class RealTimeValidator:
    """Real-time data validator with comprehensive checks"""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self._validation_history = []
        self._pattern_cache = {}
        
    async def initialize(self):
        """Initialize the validator"""
        logger.info("Initializing real-time validator")
        # Load any cached patterns or models
        await self._load_validation_patterns()
        
    async def validate(self, data: Dict[str, Any], source: str) -> Tuple[bool, List[str]]:
        """Validate incoming data"""
        try:
            validation_result = await self._comprehensive_validation(data, source)
            
            # Store validation result
            self._validation_history.append({
                'timestamp': datetime.now(),
                'source': source,
                'result': validation_result
            })
            
            # Keep only recent history
            cutoff = datetime.now() - timedelta(hours=24)
            self._validation_history = [
                v for v in self._validation_history 
                if v['timestamp'] > cutoff
            ]
            
            return validation_result.is_valid, validation_result.issues
            
        except Exception as e:
            logger.error(f"Error validating data from {source}: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    async def _comprehensive_validation(self, data: Dict[str, Any], source: str) -> ValidationResult:
        """Perform comprehensive validation"""
        issues = []
        warnings = []
        confidence_scores = []
        
        # Schema validation
        schema_valid, schema_issues = await self._validate_schema(data, source)
        if not schema_valid:
            issues.extend(schema_issues)
        else:
            confidence_scores.append(0.9)
        
        # Data type validation
        type_valid, type_issues = await self._validate_data_types(data, source)
        if not type_valid:
            issues.extend(type_issues)
        else:
            confidence_scores.append(0.8)
        
        # Range validation
        range_valid, range_issues = await self._validate_ranges(data, source)
        if not range_valid:
            issues.extend(range_issues)
        else:
            confidence_scores.append(0.85)
        
        # Temporal validation
        temporal_valid, temporal_issues = await self._validate_temporal(data, source)
        if not temporal_valid:
            issues.extend(temporal_issues)
        else:
            confidence_scores.append(0.9)
        
        # Consistency validation
        consistency_valid, consistency_issues = await self._validate_consistency(data, source)
        if not consistency_valid:
            issues.extend(consistency_issues)
        else:
            confidence_scores.append(0.8)
        
        # Pattern validation
        pattern_valid, pattern_issues = await self._validate_patterns(data, source)
        if not pattern_valid:
            warnings.extend(pattern_issues)  # Patterns are warnings, not errors
        else:
            confidence_scores.append(0.7)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Determine if data is valid
        is_valid = len(issues) == 0 and overall_confidence > 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            confidence_score=overall_confidence,
            validation_timestamp=datetime.now()
        )
    
    async def _validate_schema(self, data: Dict[str, Any], source: str) -> Tuple[bool, List[str]]:
        """Validate data schema"""
        issues = []
        
        try:
            # Define expected schemas for different sources
            schemas = {
                'oanda': ['instrument', 'time', 'bid', 'ask', 'volume'],
                'alpha_vantage': ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'],
                'forex_factory': ['event_id', 'title', 'country', 'currency', 'impact', 'event_time'],
                'central_bank': ['article_id', 'title', 'content', 'published_at', 'source'],
                'twitter': ['tweet_id', 'text', 'created_at', 'user_id', 'sentiment']
            }
            
            expected_fields = schemas.get(source, [])
            if not expected_fields:
                warnings.append(f"No schema defined for source {source}")
                return True, []
            
            # Check for required fields
            missing_fields = [field for field in expected_fields if field not in data]
            if missing_fields:
                issues.append(f"Missing required fields: {missing_fields}")
            
            # Check for unexpected fields
            unexpected_fields = [field for field in data.keys() if field not in expected_fields]
            if unexpected_fields:
                warnings.append(f"Unexpected fields: {unexpected_fields}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Schema validation error: {str(e)}"]
    
    async def _validate_data_types(self, data: Dict[str, Any], source: str) -> Tuple[bool, List[str]]:
        """Validate data types"""
        issues = []
        
        try:
            # Define expected types for different sources
            type_schemas = {
                'oanda': {
                    'instrument': str,
                    'time': str,  # ISO datetime string
                    'bid': (int, float),
                    'ask': (int, float),
                    'volume': (int, float)
                },
                'alpha_vantage': {
                    'symbol': str,
                    'timestamp': str,
                    'open': (int, float),
                    'high': (int, float),
                    'low': (int, float),
                    'close': (int, float),
                    'volume': (int, float)
                },
                'forex_factory': {
                    'event_id': str,
                    'title': str,
                    'country': str,
                    'currency': str,
                    'impact': str,
                    'event_time': str
                }
            }
            
            type_schema = type_schemas.get(source, {})
            if not type_schema:
                return True, []
            
            for field, expected_type in type_schema.items():
                if field in data:
                    value = data[field]
                    if not isinstance(value, expected_type):
                        issues.append(f"Field {field} has wrong type. Expected {expected_type}, got {type(value)}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Data type validation error: {str(e)}"]
    
    async def _validate_ranges(self, data: Dict[str, Any], source: str) -> Tuple[bool, List[str]]:
        """Validate data ranges and bounds"""
        issues = []
        
        try:
            # Price data validation
            if source in ['oanda', 'alpha_vantage']:
                price_fields = ['bid', 'ask', 'open', 'high', 'low', 'close']
                for field in price_fields:
                    if field in data and isinstance(data[field], (int, float)):
                        value = data[field]
                        if value <= 0:
                            issues.append(f"Price field {field} must be positive, got {value}")
                        elif value > 1000:  # Reasonable upper bound for forex prices
                            issues.append(f"Price field {field} seems too high: {value}")
            
            # Volume validation
            if 'volume' in data and isinstance(data['volume'], (int, float)):
                volume = data['volume']
                if volume < 0:
                    issues.append(f"Volume must be non-negative, got {volume}")
            
            # Impact level validation
            if 'impact' in data and isinstance(data['impact'], str):
                valid_impacts = ['low', 'medium', 'high', 'very_high']
                if data['impact'] not in valid_impacts:
                    issues.append(f"Invalid impact level: {data['impact']}. Must be one of {valid_impacts}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Range validation error: {str(e)}"]
    
    async def _validate_temporal(self, data: Dict[str, Any], source: str) -> Tuple[bool, List[str]]:
        """Validate temporal aspects of data"""
        issues = []
        
        try:
            # Check timestamp fields
            timestamp_fields = ['time', 'timestamp', 'event_time', 'published_at', 'created_at']
            
            for field in timestamp_fields:
                if field in data and data[field]:
                    timestamp_str = data[field]
                    
                    # Try to parse timestamp
                    try:
                        if isinstance(timestamp_str, str):
                            # Try different formats
                            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                                try:
                                    parsed_time = datetime.strptime(timestamp_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                issues.append(f"Invalid timestamp format in {field}: {timestamp_str}")
                                continue
                        else:
                            parsed_time = timestamp_str
                        
                        # Check if timestamp is reasonable (not too far in past/future)
                        now = datetime.now()
                        if parsed_time > now + timedelta(days=1):
                            issues.append(f"Timestamp {field} is too far in future: {parsed_time}")
                        elif parsed_time < now - timedelta(days=365):
                            issues.append(f"Timestamp {field} is too old: {parsed_time}")
                        
                    except Exception as e:
                        issues.append(f"Error parsing timestamp {field}: {str(e)}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Temporal validation error: {str(e)}"]
    
    async def _validate_consistency(self, data: Dict[str, Any], source: str) -> Tuple[bool, List[str]]:
        """Validate data consistency"""
        issues = []
        
        try:
            # OHLC consistency for price data
            if source in ['alpha_vantage'] and all(field in data for field in ['open', 'high', 'low', 'close']):
                o, h, l, c = data['open'], data['high'], data['low'], data['close']
                
                if not (l <= o <= h and l <= c <= h):
                    issues.append("OHLC data inconsistent: low <= open,close <= high")
                
                if not (l <= min(o, c) and max(o, c) <= h):
                    issues.append("OHLC data inconsistent: low <= min(open,close) and max(open,close) <= high")
            
            # Bid/Ask consistency
            if source in ['oanda'] and all(field in data for field in ['bid', 'ask']):
                bid, ask = data['bid'], data['ask']
                if bid >= ask:
                    issues.append("Bid price must be less than ask price")
            
            # Currency pair consistency
            if 'instrument' in data and 'currency' in data:
                instrument = data['instrument']
                currency = data['currency']
                
                # Check if currency appears in instrument
                if currency not in instrument:
                    issues.append(f"Currency {currency} not found in instrument {instrument}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Consistency validation error: {str(e)}"]
    
    async def _validate_patterns(self, data: Dict[str, Any], source: str) -> Tuple[bool, List[str]]:
        """Validate data patterns and detect anomalies"""
        warnings = []
        
        try:
            # Load historical patterns for this source
            patterns = await self._get_patterns_for_source(source)
            
            # Check for unusual patterns
            if patterns:
                # Price pattern validation
                if source in ['oanda', 'alpha_vantage'] and 'bid' in data:
                    price = data['bid']
                    if price in patterns.get('unusual_prices', []):
                        warnings.append(f"Unusual price pattern detected: {price}")
                
                # Volume pattern validation
                if 'volume' in data:
                    volume = data['volume']
                    if volume in patterns.get('unusual_volumes', []):
                        warnings.append(f"Unusual volume pattern detected: {volume}")
            
            return len(warnings) == 0, warnings
            
        except Exception as e:
            return False, [f"Pattern validation error: {str(e)}"]
    
    async def _load_validation_patterns(self):
        """Load validation patterns from cache or database"""
        try:
            # This would typically load from a database or cache
            # For now, we'll use empty patterns
            self._pattern_cache = {}
            logger.info("Validation patterns loaded")
            
        except Exception as e:
            logger.error(f"Error loading validation patterns: {e}")
    
    async def _get_patterns_for_source(self, source: str) -> Dict[str, Any]:
        """Get validation patterns for a specific source"""
        return self._pattern_cache.get(source, {})
    
    async def shutdown(self):
        """Shutdown the validator"""
        logger.info("Shutting down real-time validator")
        # Clean up any resources

