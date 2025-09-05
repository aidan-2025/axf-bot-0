#!/usr/bin/env python3
"""
Data Corrector
Automatically corrects data issues and anomalies
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CorrectionResult:
    """Result of data correction"""
    corrected_data: Optional[Dict[str, Any]]
    corrections_applied: List[str]
    confidence: float
    correction_timestamp: datetime

class DataCorrector:
    """Automatic data correction and cleaning"""
    
    def __init__(self):
        self._correction_history = []
        self._correction_patterns = {}
        
    async def initialize(self):
        """Initialize the data corrector"""
        logger.info("Initializing data corrector")
        await self._load_correction_patterns()
        
    async def correct(self, data: Dict[str, Any], issues: List[str]) -> Optional[Dict[str, Any]]:
        """Correct data based on identified issues"""
        try:
            corrected_data = data.copy()
            corrections_applied = []
            confidence_scores = []
            
            for issue in issues:
                correction_result = await self._apply_correction(corrected_data, issue)
                if correction_result:
                    corrected_data = correction_result['data']
                    corrections_applied.append(correction_result['description'])
                    confidence_scores.append(correction_result['confidence'])
            
            if corrections_applied:
                overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
                
                result = CorrectionResult(
                    corrected_data=corrected_data,
                    corrections_applied=corrections_applied,
                    confidence=overall_confidence,
                    correction_timestamp=datetime.now()
                )
                
                # Store correction history
                self._correction_history.append({
                    'timestamp': datetime.now(),
                    'original_data': data,
                    'corrected_data': corrected_data,
                    'corrections': corrections_applied,
                    'confidence': overall_confidence
                })
                
                logger.info(f"Applied {len(corrections_applied)} corrections with confidence {overall_confidence:.2f}")
                return corrected_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error correcting data: {e}")
            return None
    
    async def _apply_correction(self, data: Dict[str, Any], issue: str) -> Optional[Dict[str, Any]]:
        """Apply a specific correction based on the issue"""
        try:
            # Missing fields correction
            if "Missing required fields" in issue:
                return await self._correct_missing_fields(data, issue)
            
            # Data type correction
            elif "wrong type" in issue:
                return await self._correct_data_types(data, issue)
            
            # Range correction
            elif "must be positive" in issue or "too high" in issue:
                return await self._correct_ranges(data, issue)
            
            # Temporal correction
            elif "timestamp" in issue.lower():
                return await self._correct_temporal(data, issue)
            
            # Consistency correction
            elif "inconsistent" in issue.lower():
                return await self._correct_consistency(data, issue)
            
            # Pattern correction
            elif "pattern" in issue.lower():
                return await self._correct_patterns(data, issue)
            
            return None
            
        except Exception as e:
            logger.error(f"Error applying correction for issue '{issue}': {e}")
            return None
    
    async def _correct_missing_fields(self, data: Dict[str, Any], issue: str) -> Optional[Dict[str, Any]]:
        """Correct missing required fields"""
        try:
            # Extract missing fields from issue
            if "Missing required fields:" in issue:
                missing_fields = issue.split("Missing required fields:")[1].strip().strip("[]").split(", ")
                missing_fields = [field.strip() for field in missing_fields]
            else:
                return None
            
            corrections_applied = []
            
            for field in missing_fields:
                if field not in data:
                    # Provide default values based on field type
                    if field in ['instrument', 'symbol']:
                        data[field] = "UNKNOWN"
                    elif field in ['time', 'timestamp', 'event_time', 'published_at', 'created_at']:
                        data[field] = datetime.now().isoformat()
                    elif field in ['bid', 'ask', 'open', 'high', 'low', 'close']:
                        data[field] = 0.0
                    elif field in ['volume']:
                        data[field] = 0
                    elif field in ['impact']:
                        data[field] = "low"
                    elif field in ['country', 'currency']:
                        data[field] = "UNKNOWN"
                    else:
                        data[field] = None
                    
                    corrections_applied.append(f"Added missing field '{field}' with default value")
            
            return {
                'data': data,
                'description': f"Added missing fields: {', '.join(missing_fields)}",
                'confidence': 0.6  # Lower confidence for default values
            }
            
        except Exception as e:
            logger.error(f"Error correcting missing fields: {e}")
            return None
    
    async def _correct_data_types(self, data: Dict[str, Any], issue: str) -> Optional[Dict[str, Any]]:
        """Correct data type issues"""
        try:
            # Extract field and type information from issue
            if "has wrong type" in issue:
                parts = issue.split("has wrong type")
                field_part = parts[0].strip().split()[-1]  # Get field name
                type_part = parts[1].strip().split("got")[0].strip()  # Get expected type
                
                field = field_part.strip("'\"")
                expected_type_str = type_part.strip("'\"")
                
                corrections_applied = []
                
                if field in data:
                    value = data[field]
                    
                    # Convert to expected type
                    if expected_type_str == "str":
                        data[field] = str(value)
                        corrections_applied.append(f"Converted {field} to string")
                    elif expected_type_str in ["int", "float"]:
                        try:
                            if expected_type_str == "int":
                                data[field] = int(float(value))
                            else:
                                data[field] = float(value)
                            corrections_applied.append(f"Converted {field} to {expected_type_str}")
                        except (ValueError, TypeError):
                            # If conversion fails, use default value
                            data[field] = 0 if expected_type_str == "int" else 0.0
                            corrections_applied.append(f"Set {field} to default {expected_type_str} value")
                
                return {
                    'data': data,
                    'description': f"Corrected data types for {field}",
                    'confidence': 0.8
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error correcting data types: {e}")
            return None
    
    async def _correct_ranges(self, data: Dict[str, Any], issue: str) -> Optional[Dict[str, Any]]:
        """Correct range and bound issues"""
        try:
            corrections_applied = []
            
            # Price corrections
            if "must be positive" in issue:
                price_fields = ['bid', 'ask', 'open', 'high', 'low', 'close']
                for field in price_fields:
                    if field in data and isinstance(data[field], (int, float)) and data[field] <= 0:
                        # Use previous valid value or reasonable default
                        data[field] = await self._get_reasonable_price(field)
                        corrections_applied.append(f"Corrected negative price for {field}")
            
            elif "too high" in issue:
                price_fields = ['bid', 'ask', 'open', 'high', 'low', 'close']
                for field in price_fields:
                    if field in data and isinstance(data[field], (int, float)) and data[field] > 1000:
                        # Cap at reasonable maximum
                        data[field] = min(data[field], 1000)
                        corrections_applied.append(f"Capped high price for {field}")
            
            # Volume corrections
            if 'volume' in data and isinstance(data['volume'], (int, float)) and data['volume'] < 0:
                data['volume'] = 0
                corrections_applied.append("Set negative volume to zero")
            
            if corrections_applied:
                return {
                    'data': data,
                    'description': f"Corrected range issues: {', '.join(corrections_applied)}",
                    'confidence': 0.7
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error correcting ranges: {e}")
            return None
    
    async def _correct_temporal(self, data: Dict[str, Any], issue: str) -> Optional[Dict[str, Any]]:
        """Correct temporal issues"""
        try:
            corrections_applied = []
            timestamp_fields = ['time', 'timestamp', 'event_time', 'published_at', 'created_at']
            
            for field in timestamp_fields:
                if field in data and data[field]:
                    timestamp_str = data[field]
                    
                    # Try to fix common timestamp issues
                    if isinstance(timestamp_str, str):
                        # Add timezone if missing
                        if 'T' in timestamp_str and not timestamp_str.endswith('Z') and '+' not in timestamp_str:
                            data[field] = timestamp_str + 'Z'
                            corrections_applied.append(f"Added UTC timezone to {field}")
                        
                        # Fix common format issues
                        if ' ' in timestamp_str and 'T' not in timestamp_str:
                            data[field] = timestamp_str.replace(' ', 'T')
                            corrections_applied.append(f"Fixed timestamp format for {field}")
            
            if corrections_applied:
                return {
                    'data': data,
                    'description': f"Corrected temporal issues: {', '.join(corrections_applied)}",
                    'confidence': 0.8
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error correcting temporal issues: {e}")
            return None
    
    async def _correct_consistency(self, data: Dict[str, Any], issue: str) -> Optional[Dict[str, Any]]:
        """Correct consistency issues"""
        try:
            corrections_applied = []
            
            # OHLC consistency
            if "OHLC data inconsistent" in issue:
                if all(field in data for field in ['open', 'high', 'low', 'close']):
                    o, h, l, c = data['open'], data['high'], data['low'], data['close']
                    
                    # Ensure low <= min(open, close) <= max(open, close) <= high
                    min_oc = min(o, c)
                    max_oc = max(o, c)
                    
                    if l > min_oc:
                        data['low'] = min_oc
                        corrections_applied.append("Adjusted low to min(open, close)")
                    
                    if h < max_oc:
                        data['high'] = max_oc
                        corrections_applied.append("Adjusted high to max(open, close)")
            
            # Bid/Ask consistency
            elif "Bid price must be less than ask price" in issue:
                if 'bid' in data and 'ask' in data:
                    bid, ask = data['bid'], data['ask']
                    if bid >= ask:
                        # Swap if they're equal, or adjust ask to be higher
                        data['ask'] = bid + 0.0001  # Small spread
                        corrections_applied.append("Adjusted ask price to be higher than bid")
            
            if corrections_applied:
                return {
                    'data': data,
                    'description': f"Corrected consistency issues: {', '.join(corrections_applied)}",
                    'confidence': 0.9
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error correcting consistency issues: {e}")
            return None
    
    async def _correct_patterns(self, data: Dict[str, Any], issue: str) -> Optional[Dict[str, Any]]:
        """Correct pattern issues"""
        try:
            corrections_applied = []
            
            # Unusual price patterns
            if "Unusual price pattern" in issue:
                if 'bid' in data:
                    # Use historical average or reasonable default
                    data['bid'] = await self._get_reasonable_price('bid')
                    corrections_applied.append("Adjusted unusual price pattern")
            
            # Unusual volume patterns
            elif "Unusual volume pattern" in issue:
                if 'volume' in data:
                    # Use historical average or reasonable default
                    data['volume'] = await self._get_reasonable_volume()
                    corrections_applied.append("Adjusted unusual volume pattern")
            
            if corrections_applied:
                return {
                    'data': data,
                    'description': f"Corrected pattern issues: {', '.join(corrections_applied)}",
                    'confidence': 0.6
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error correcting patterns: {e}")
            return None
    
    async def _get_reasonable_price(self, field: str) -> float:
        """Get a reasonable price value for a field"""
        # This would typically use historical data or market averages
        # For now, return reasonable defaults based on field type
        defaults = {
            'bid': 1.1000,
            'ask': 1.1001,
            'open': 1.1000,
            'high': 1.1005,
            'low': 1.0995,
            'close': 1.1000
        }
        return defaults.get(field, 1.1000)
    
    async def _get_reasonable_volume(self) -> int:
        """Get a reasonable volume value"""
        # This would typically use historical averages
        return 1000
    
    async def _load_correction_patterns(self):
        """Load correction patterns from cache or database"""
        try:
            # This would typically load from a database or cache
            self._correction_patterns = {}
            logger.info("Correction patterns loaded")
            
        except Exception as e:
            logger.error(f"Error loading correction patterns: {e}")
    
    async def shutdown(self):
        """Shutdown the data corrector"""
        logger.info("Shutting down data corrector")
        # Clean up any resources

