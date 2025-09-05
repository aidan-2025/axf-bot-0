#!/usr/bin/env python3
"""
Cross-Source Validator
Validates data consistency across multiple data sources
"""

import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math

from ..brokers.broker_manager import PriceData, CandleData

logger = logging.getLogger(__name__)

@dataclass
class CrossSourceValidationResult:
    """Result of cross-source validation"""
    is_valid: bool
    discrepancy_pct: float = 0.0
    description: str = ""
    warnings: List[str] = field(default_factory=list)
    source_agreement: float = 0.0
    validation_time_ms: float = 0.0

@dataclass
class CrossSourceValidationConfig:
    """Configuration for cross-source validation"""
    # Discrepancy thresholds
    max_price_discrepancy_pct: float = 0.1
    max_spread_discrepancy_pct: float = 10.0
    max_volume_discrepancy_pct: float = 50.0
    
    # Time synchronization
    max_time_difference_seconds: int = 5
    
    # Minimum sources for validation
    min_sources: int = 2
    min_agreement_pct: float = 70.0
    
    # Outlier detection
    outlier_threshold: float = 2.0  # Standard deviations
    enable_outlier_detection: bool = True
    
    # Source weighting
    enable_source_weighting: bool = True
    source_weights: Dict[str, float] = field(default_factory=dict)

class CrossSourceValidator:
    """Validates data consistency across multiple sources"""
    
    def __init__(self, config: Optional[CrossSourceValidationConfig] = None):
        """Initialize cross-source validator"""
        self.config = config or CrossSourceValidationConfig()
        
        # Source reliability tracking
        self.source_reliability: Dict[str, float] = {}
        self.source_error_counts: Dict[str, int] = {}
        self.source_success_counts: Dict[str, int] = {}
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'outlier_detections': 0,
            'source_disagreements': 0
        }
        
        logger.info("CrossSourceValidator initialized with config: %s", self.config)
    
    async def validate_price_data(self, primary_data: PriceData, 
                                cross_source_data: List[PriceData]) -> CrossSourceValidationResult:
        """Validate price data against cross-source data"""
        import time
        start_time = time.time()
        
        try:
            # Filter and validate cross-source data
            valid_sources = self._filter_valid_sources(primary_data, cross_source_data)
            
            if len(valid_sources) < self.config.min_sources:
                return CrossSourceValidationResult(
                    is_valid=True,  # Not enough sources to validate
                    description=f"Insufficient sources for validation ({len(valid_sources)} < {self.config.min_sources})"
                )
            
            # Calculate discrepancies
            price_discrepancy = self._calculate_price_discrepancy(primary_data, valid_sources)
            spread_discrepancy = self._calculate_spread_discrepancy(primary_data, valid_sources)
            
            # Detect outliers
            outliers = []
            if self.config.enable_outlier_detection:
                outliers = self._detect_outliers(primary_data, valid_sources)
            
            # Calculate source agreement
            agreement_pct = self._calculate_source_agreement(primary_data, valid_sources)
            
            # Determine validation result
            is_valid = True
            errors = []
            warnings = []
            
            # Check price discrepancy
            if price_discrepancy > self.config.max_price_discrepancy_pct:
                is_valid = False
                errors.append(f"Price discrepancy {price_discrepancy:.2f}% exceeds threshold {self.config.max_price_discrepancy_pct}%")
            
            # Check spread discrepancy
            if spread_discrepancy > self.config.max_spread_discrepancy_pct:
                warnings.append(f"Spread discrepancy {spread_discrepancy:.2f}% exceeds threshold {self.config.max_spread_discrepancy_pct}%")
            
            # Check source agreement
            if agreement_pct < self.config.min_agreement_pct:
                warnings.append(f"Source agreement {agreement_pct:.1f}% below threshold {self.config.min_agreement_pct}%")
            
            # Add outlier warnings
            for outlier in outliers:
                warnings.append(f"Outlier detected: {outlier['source']} - {outlier['description']}")
                self.validation_stats['outlier_detections'] += 1
            
            # Update source reliability
            self._update_source_reliability(primary_data, valid_sources, is_valid)
            
            # Update validation statistics
            self._update_validation_stats(is_valid)
            
            return CrossSourceValidationResult(
                is_valid=is_valid,
                discrepancy_pct=max(price_discrepancy, spread_discrepancy),
                description=f"Price discrepancy: {price_discrepancy:.2f}%, Spread discrepancy: {spread_discrepancy:.2f}%",
                warnings=warnings,
                source_agreement=agreement_pct,
                validation_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Error in cross-source validation: {e}")
            return CrossSourceValidationResult(
                is_valid=False,
                description=f"Cross-source validation error: {str(e)}",
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    async def validate_candle_data(self, primary_data: CandleData,
                                 cross_source_data: List[CandleData]) -> CrossSourceValidationResult:
        """Validate candle data against cross-source data"""
        import time
        start_time = time.time()
        
        try:
            # Filter and validate cross-source data
            valid_sources = self._filter_valid_candle_sources(primary_data, cross_source_data)
            
            if len(valid_sources) < self.config.min_sources:
                return CrossSourceValidationResult(
                    is_valid=True,
                    description=f"Insufficient sources for validation ({len(valid_sources)} < {self.config.min_sources})"
                )
            
            # Calculate OHLC discrepancies
            ohlc_discrepancies = self._calculate_ohlc_discrepancies(primary_data, valid_sources)
            volume_discrepancy = self._calculate_volume_discrepancy(primary_data, valid_sources)
            
            # Detect outliers
            outliers = []
            if self.config.enable_outlier_detection:
                outliers = self._detect_candle_outliers(primary_data, valid_sources)
            
            # Calculate source agreement
            agreement_pct = self._calculate_candle_agreement(primary_data, valid_sources)
            
            # Determine validation result
            is_valid = True
            warnings = []
            
            # Check OHLC discrepancies
            max_ohlc_discrepancy = max(ohlc_discrepancies.values())
            if max_ohlc_discrepancy > self.config.max_price_discrepancy_pct:
                is_valid = False
                warnings.append(f"OHLC discrepancy {max_ohlc_discrepancy:.2f}% exceeds threshold")
            
            # Check volume discrepancy
            if volume_discrepancy > self.config.max_volume_discrepancy_pct:
                warnings.append(f"Volume discrepancy {volume_discrepancy:.2f}% exceeds threshold")
            
            # Add outlier warnings
            for outlier in outliers:
                warnings.append(f"Outlier detected: {outlier['source']} - {outlier['description']}")
                self.validation_stats['outlier_detections'] += 1
            
            # Update validation statistics
            self._update_validation_stats(is_valid)
            
            return CrossSourceValidationResult(
                is_valid=is_valid,
                discrepancy_pct=max(max_ohlc_discrepancy, volume_discrepancy),
                description=f"OHLC discrepancy: {max_ohlc_discrepancy:.2f}%, Volume discrepancy: {volume_discrepancy:.2f}%",
                warnings=warnings,
                source_agreement=agreement_pct,
                validation_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Error in cross-source candle validation: {e}")
            return CrossSourceValidationResult(
                is_valid=False,
                description=f"Cross-source validation error: {str(e)}",
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _filter_valid_sources(self, primary_data: PriceData, cross_source_data: List[PriceData]) -> List[PriceData]:
        """Filter valid cross-source data based on time and instrument"""
        valid_sources = []
        
        for data in cross_source_data:
            # Check instrument match
            if data.instrument != primary_data.instrument:
                continue
            
            # Check time difference
            time_diff = abs((data.time - primary_data.time).total_seconds())
            if time_diff > self.config.max_time_difference_seconds:
                continue
            
            # Check data validity
            if (data.bid <= 0 or data.ask <= 0 or 
                data.spread < 0 or data.bid >= data.ask):
                continue
            
            valid_sources.append(data)
        
        return valid_sources
    
    def _filter_valid_candle_sources(self, primary_data: CandleData, cross_source_data: List[CandleData]) -> List[CandleData]:
        """Filter valid cross-source candle data"""
        valid_sources = []
        
        for data in cross_source_data:
            # Check instrument match
            if data.instrument != primary_data.instrument:
                continue
            
            # Check time difference
            time_diff = abs((data.time - primary_data.time).total_seconds())
            if time_diff > self.config.max_time_difference_seconds:
                continue
            
            # Check data validity
            if (data.open <= 0 or data.high <= 0 or data.low <= 0 or data.close <= 0 or
                data.volume < 0 or data.high < data.low or
                data.high < max(data.open, data.close) or
                data.low > min(data.open, data.close)):
                continue
            
            valid_sources.append(data)
        
        return valid_sources
    
    def _calculate_price_discrepancy(self, primary_data: PriceData, valid_sources: List[PriceData]) -> float:
        """Calculate price discrepancy percentage"""
        if not valid_sources:
            return 0.0
        
        primary_mid = (primary_data.bid + primary_data.ask) / 2
        source_mids = [(s.bid + s.ask) / 2 for s in valid_sources]
        
        # Calculate weighted average if source weighting is enabled
        if self.config.enable_source_weighting and self.config.source_weights:
            weighted_mid = self._calculate_weighted_average(source_mids, valid_sources)
        else:
            weighted_mid = statistics.mean(source_mids)
        
        if weighted_mid == 0:
            return 0.0
        
        discrepancy = abs(primary_mid - weighted_mid) / weighted_mid * 100
        return discrepancy
    
    def _calculate_spread_discrepancy(self, primary_data: PriceData, valid_sources: List[PriceData]) -> float:
        """Calculate spread discrepancy percentage"""
        if not valid_sources:
            return 0.0
        
        primary_spread = primary_data.spread
        source_spreads = [s.spread for s in valid_sources]
        
        # Calculate weighted average if source weighting is enabled
        if self.config.enable_source_weighting and self.config.source_weights:
            weighted_spread = self._calculate_weighted_average(source_spreads, valid_sources)
        else:
            weighted_spread = statistics.mean(source_spreads)
        
        if weighted_spread == 0:
            return 0.0
        
        discrepancy = abs(primary_spread - weighted_spread) / weighted_spread * 100
        return discrepancy
    
    def _calculate_ohlc_discrepancies(self, primary_data: CandleData, valid_sources: List[CandleData]) -> Dict[str, float]:
        """Calculate OHLC discrepancies"""
        discrepancies = {}
        
        for field in ['open', 'high', 'low', 'close']:
            primary_value = getattr(primary_data, field)
            source_values = [getattr(s, field) for s in valid_sources]
            
            if self.config.enable_source_weighting and self.config.source_weights:
                weighted_value = self._calculate_weighted_average(source_values, valid_sources)
            else:
                weighted_value = statistics.mean(source_values)
            
            if weighted_value == 0:
                discrepancies[field] = 0.0
            else:
                discrepancies[field] = abs(primary_value - weighted_value) / weighted_value * 100
        
        return discrepancies
    
    def _calculate_volume_discrepancy(self, primary_data: CandleData, valid_sources: List[CandleData]) -> float:
        """Calculate volume discrepancy percentage"""
        if not valid_sources:
            return 0.0
        
        primary_volume = primary_data.volume
        source_volumes = [s.volume for s in valid_sources]
        
        if self.config.enable_source_weighting and self.config.source_weights:
            weighted_volume = self._calculate_weighted_average(source_volumes, valid_sources)
        else:
            weighted_volume = statistics.mean(source_volumes)
        
        if weighted_volume == 0:
            return 0.0
        
        discrepancy = abs(primary_volume - weighted_volume) / weighted_volume * 100
        return discrepancy
    
    def _calculate_weighted_average(self, values: List[float], sources: List[Any]) -> float:
        """Calculate weighted average based on source reliability"""
        if not values or not self.config.source_weights:
            return statistics.mean(values)
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for i, value in enumerate(values):
            source_name = getattr(sources[i], 'source', f'source_{i}')
            weight = self.config.source_weights.get(source_name, 1.0)
            
            # Apply source reliability if available
            if source_name in self.source_reliability:
                weight *= self.source_reliability[source_name]
            
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else statistics.mean(values)
    
    def _detect_outliers(self, primary_data: PriceData, valid_sources: List[PriceData]) -> List[Dict[str, Any]]:
        """Detect outliers in cross-source data"""
        outliers = []
        
        if len(valid_sources) < 3:  # Need at least 3 sources for outlier detection
            return outliers
        
        primary_mid = (primary_data.bid + primary_data.ask) / 2
        source_mids = [(s.bid + s.ask) / 2 for s in valid_sources]
        
        # Calculate statistics
        mean_mid = statistics.mean(source_mids)
        std_mid = statistics.stdev(source_mids) if len(source_mids) > 1 else 0
        
        if std_mid == 0:
            return outliers
        
        # Check if primary data is an outlier
        z_score = abs(primary_mid - mean_mid) / std_mid
        if z_score > self.config.outlier_threshold:
            outliers.append({
                'source': 'primary',
                'description': f'Primary data Z-score {z_score:.2f} exceeds threshold {self.config.outlier_threshold}',
                'z_score': z_score
            })
        
        # Check each source for outliers
        for i, (source, mid) in enumerate(zip(valid_sources, source_mids)):
            z_score = abs(mid - mean_mid) / std_mid
            if z_score > self.config.outlier_threshold:
                outliers.append({
                    'source': getattr(source, 'source', f'source_{i}'),
                    'description': f'Source Z-score {z_score:.2f} exceeds threshold {self.config.outlier_threshold}',
                    'z_score': z_score
                })
        
        return outliers
    
    def _detect_candle_outliers(self, primary_data: CandleData, valid_sources: List[CandleData]) -> List[Dict[str, Any]]:
        """Detect outliers in cross-source candle data"""
        outliers = []
        
        if len(valid_sources) < 3:
            return outliers
        
        # Check each OHLC field for outliers
        for field in ['open', 'high', 'low', 'close']:
            primary_value = getattr(primary_data, field)
            source_values = [getattr(s, field) for s in valid_sources]
            
            mean_value = statistics.mean(source_values)
            std_value = statistics.stdev(source_values) if len(source_values) > 1 else 0
            
            if std_value == 0:
                continue
            
            # Check primary data
            z_score = abs(primary_value - mean_value) / std_value
            if z_score > self.config.outlier_threshold:
                outliers.append({
                    'source': 'primary',
                    'description': f'Primary {field} Z-score {z_score:.2f} exceeds threshold',
                    'z_score': z_score,
                    'field': field
                })
        
        return outliers
    
    def _calculate_source_agreement(self, primary_data: PriceData, valid_sources: List[PriceData]) -> float:
        """Calculate percentage of sources that agree with primary data"""
        if not valid_sources:
            return 100.0
        
        primary_mid = (primary_data.bid + primary_data.ask) / 2
        source_mids = [(s.bid + s.ask) / 2 for s in valid_sources]
        
        # Calculate how many sources are within acceptable range
        agreeing_sources = 0
        for mid in source_mids:
            if mid > 0:
                discrepancy = abs(primary_mid - mid) / mid * 100
                if discrepancy <= self.config.max_price_discrepancy_pct:
                    agreeing_sources += 1
        
        return (agreeing_sources / len(valid_sources)) * 100
    
    def _calculate_candle_agreement(self, primary_data: CandleData, valid_sources: List[CandleData]) -> float:
        """Calculate percentage of sources that agree with primary candle data"""
        if not valid_sources:
            return 100.0
        
        total_agreements = 0
        total_checks = 0
        
        for field in ['open', 'high', 'low', 'close']:
            primary_value = getattr(primary_data, field)
            source_values = [getattr(s, field) for s in valid_sources]
            
            for value in source_values:
                total_checks += 1
                if value > 0:
                    discrepancy = abs(primary_value - value) / value * 100
                    if discrepancy <= self.config.max_price_discrepancy_pct:
                        total_agreements += 1
        
        return (total_agreements / total_checks * 100) if total_checks > 0 else 100.0
    
    def _update_source_reliability(self, primary_data: PriceData, valid_sources: List[PriceData], is_valid: bool):
        """Update source reliability based on validation results"""
        for source in valid_sources:
            source_name = getattr(source, 'source', 'unknown')
            
            if source_name not in self.source_reliability:
                self.source_reliability[source_name] = 1.0
                self.source_error_counts[source_name] = 0
                self.source_success_counts[source_name] = 0
            
            if is_valid:
                self.source_success_counts[source_name] += 1
            else:
                self.source_error_counts[source_name] += 1
            
            # Calculate reliability as success rate
            total_attempts = self.source_success_counts[source_name] + self.source_error_counts[source_name]
            if total_attempts > 0:
                self.source_reliability[source_name] = self.source_success_counts[source_name] / total_attempts
    
    def _update_validation_stats(self, is_valid: bool):
        """Update validation statistics"""
        self.validation_stats['total_validations'] += 1
        if is_valid:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get cross-source validation statistics"""
        return {
            'validation_stats': self.validation_stats,
            'source_reliability': self.source_reliability,
            'source_error_counts': self.source_error_counts,
            'source_success_counts': self.source_success_counts,
            'config': {
                'max_price_discrepancy_pct': self.config.max_price_discrepancy_pct,
                'max_spread_discrepancy_pct': self.config.max_spread_discrepancy_pct,
                'min_sources': self.config.min_sources,
                'outlier_threshold': self.config.outlier_threshold
            }
        }

# Example usage and testing
async def test_cross_source_validator():
    """Test the cross-source validator"""
    from ..brokers.broker_manager import PriceData
    
    config = CrossSourceValidationConfig(
        max_price_discrepancy_pct=0.05,
        min_sources=2
    )
    
    validator = CrossSourceValidator(config)
    
    # Test with consistent data
    primary_data = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=1.1000,
        ask=1.1002,
        spread=0.0002
    )
    
    cross_source_data = [
        PriceData(
            instrument="EUR_USD",
            time=datetime.utcnow(),
            bid=1.1001,
            ask=1.1003,
            spread=0.0002
        ),
        PriceData(
            instrument="EUR_USD",
            time=datetime.utcnow(),
            bid=1.0999,
            ask=1.1001,
            spread=0.0002
        )
    ]
    
    result = await validator.validate_price_data(primary_data, cross_source_data)
    print(f"Cross-source validation: Valid={result.is_valid}, Discrepancy={result.discrepancy_pct:.2f}%")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_cross_source_validator())

