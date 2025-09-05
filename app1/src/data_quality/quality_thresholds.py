#!/usr/bin/env python3
"""
Quality Thresholds Configuration
Defines thresholds and limits for data quality validation
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class QualityThresholds:
    """Quality thresholds for different metrics"""
    # Data completeness thresholds
    min_completeness: float = 0.95
    max_missing_data: float = 0.05
    
    # Data accuracy thresholds
    max_outlier_ratio: float = 0.02
    max_price_deviation: float = 0.05  # 5% deviation from expected range
    
    # Timeliness thresholds
    max_delay_seconds: int = 30
    max_staleness_minutes: int = 5
    
    # Consistency thresholds
    min_cross_source_agreement: float = 0.90
    max_timezone_offset_hours: int = 1
    
    # Source health thresholds
    max_consecutive_failures: int = 3
    min_success_rate: float = 0.95
    max_response_time_seconds: int = 10
    
    # Data volume thresholds
    min_expected_volume: int = 100  # Minimum expected data points per hour
    max_expected_volume: int = 10000  # Maximum expected data points per hour
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'min_completeness': self.min_completeness,
            'max_missing_data': self.max_missing_data,
            'max_outlier_ratio': self.max_outlier_ratio,
            'max_price_deviation': self.max_price_deviation,
            'max_delay_seconds': self.max_delay_seconds,
            'max_staleness_minutes': self.max_staleness_minutes,
            'min_cross_source_agreement': self.min_cross_source_agreement,
            'max_timezone_offset_hours': self.max_timezone_offset_hours,
            'max_consecutive_failures': self.max_consecutive_failures,
            'min_success_rate': self.min_success_rate,
            'max_response_time_seconds': self.max_response_time_seconds,
            'min_expected_volume': self.min_expected_volume,
            'max_expected_volume': self.max_expected_volume
        }

