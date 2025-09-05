#!/usr/bin/env python3
"""
Validation Engine
Central coordinator for all data validation and quality checks
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict, deque

from ..brokers.broker_manager import PriceData, CandleData
from .schema_validator import SchemaValidator, SchemaValidationResult
from .range_validator import RangeValidator, RangeValidationResult
from .anomaly_detector import AnomalyDetector, AnomalyDetectionResult
from .cross_source_validator import CrossSourceValidator, CrossSourceValidationResult
from .quality_metrics import QualityMetrics, QualityReport
from .data_corrector import DataCorrector, CorrectionResult

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status"""
    VALID = "valid"
    INVALID = "invalid"
    QUARANTINED = "quarantined"
    CORRECTED = "corrected"
    PENDING = "pending"

@dataclass
class ValidationResult:
    """Result of data validation"""
    status: ValidationStatus
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: List[CorrectionResult] = field(default_factory=list)
    quality_score: float = 0.0
    validation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    data_type: str = ""

@dataclass
class ValidationConfig:
    """Configuration for validation engine"""
    # Schema validation
    enable_schema_validation: bool = True
    strict_schema: bool = True
    
    # Range validation
    enable_range_validation: bool = True
    price_range_multiplier: float = 2.0
    max_spread_pips: float = 50.0
    max_price_jump_pct: float = 5.0
    
    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 3.0
    window_size: int = 100
    enable_ml_anomaly_detection: bool = False
    
    # Cross-source validation
    enable_cross_source_validation: bool = True
    cross_source_tolerance_pct: float = 0.1
    min_sources_for_validation: int = 2
    
    # Quality metrics
    enable_quality_metrics: bool = True
    metrics_window_size: int = 1000
    quality_threshold: float = 0.8
    
    # Data correction
    enable_auto_correction: bool = True
    max_corrections_per_data_point: int = 3
    correction_timeout_ms: int = 100
    
    # Performance
    max_validation_time_ms: int = 50
    enable_async_validation: bool = True
    validation_batch_size: int = 10

class ValidationEngine:
    """Central validation engine for market data"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize validation engine"""
        self.config = config or ValidationConfig()
        self.status = ValidationStatus.PENDING
        
        # Initialize validators
        self.schema_validator = SchemaValidator()
        self.range_validator = RangeValidator(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.cross_source_validator = CrossSourceValidator(self.config)
        self.quality_metrics = QualityMetrics(self.config)
        self.data_corrector = DataCorrector(self.config)
        
        # Validation state
        self.validation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.quality_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.correction_history: List[CorrectionResult] = []
        
        # Performance metrics
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.avg_validation_time_ms = 0.0
        self.total_corrections = 0
        
        logger.info("ValidationEngine initialized with config: %s", self.config)
    
    async def validate_price_data(self, price_data: PriceData, 
                                previous_data: Optional[PriceData] = None,
                                cross_source_data: Optional[List[PriceData]] = None) -> ValidationResult:
        """Validate price data with comprehensive checks"""
        start_time = time.time()
        
        try:
            # Initialize result
            result = ValidationResult(
                status=ValidationStatus.PENDING,
                is_valid=True,
                source=price_data.instrument,
                data_type="price"
            )
            
            # Schema validation
            if self.config.enable_schema_validation:
                schema_result = await self.schema_validator.validate_price_data(price_data)
                if not schema_result.is_valid:
                    result.errors.extend(schema_result.errors)
                    result.is_valid = False
                    result.status = ValidationStatus.INVALID
                result.warnings.extend(schema_result.warnings)
            
            # Range validation
            if self.config.enable_range_validation and result.is_valid:
                range_result = await self.range_validator.validate_price_data(price_data, previous_data)
                if not range_result.is_valid:
                    result.errors.extend(range_result.errors)
                    result.is_valid = False
                    result.status = ValidationStatus.INVALID
                result.warnings.extend(range_result.warnings)
            
            # Anomaly detection
            if self.config.enable_anomaly_detection and result.is_valid:
                anomaly_result = await self.anomaly_detector.detect_price_anomaly(
                    price_data, self.validation_history[price_data.instrument]
                )
                if anomaly_result.is_anomaly:
                    result.warnings.extend(anomaly_result.warnings)
                    if anomaly_result.severity > self.config.anomaly_threshold:
                        result.errors.append(f"Severe anomaly detected: {anomaly_result.description}")
                        result.is_valid = False
                        result.status = ValidationStatus.QUARANTINED
            
            # Cross-source validation
            if (self.config.enable_cross_source_validation and 
                cross_source_data and 
                len(cross_source_data) >= self.config.min_sources_for_validation and
                result.is_valid):
                
                cross_source_result = await self.cross_source_validator.validate_price_data(
                    price_data, cross_source_data
                )
                if not cross_source_result.is_valid:
                    result.warnings.extend(cross_source_result.warnings)
                    if cross_source_result.discrepancy_pct > self.config.cross_source_tolerance_pct:
                        result.errors.append(f"Cross-source validation failed: {cross_source_result.description}")
                        result.is_valid = False
                        result.status = ValidationStatus.QUARANTINED
            
            # Calculate quality score
            if self.config.enable_quality_metrics:
                quality_score = await self.quality_metrics.calculate_price_quality_score(
                    price_data, result
                )
                result.quality_score = quality_score
                
                # Update quality history
                self.quality_scores[price_data.instrument].append(quality_score)
            
            # Auto-correction if enabled and data is invalid
            if (self.config.enable_auto_correction and 
                not result.is_valid and 
                result.status != ValidationStatus.QUARANTINED):
                
                correction_result = await self.data_corrector.correct_price_data(
                    price_data, result.errors, previous_data
                )
                
                if correction_result.is_corrected:
                    result.corrections.append(correction_result)
                    result.status = ValidationStatus.CORRECTED
                    result.is_valid = True
                    result.warnings.append(f"Data corrected: {correction_result.description}")
                    self.total_corrections += 1
                    self.correction_history.append(correction_result)
            
            # Update final status
            if result.is_valid and result.status == ValidationStatus.PENDING:
                result.status = ValidationStatus.VALID
            
            # Update validation history
            self.validation_history[price_data.instrument].append(result)
            
            # Update performance metrics
            validation_time_ms = (time.time() - start_time) * 1000
            result.validation_time_ms = validation_time_ms
            self._update_performance_metrics(validation_time_ms, result.is_valid)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating price data: {e}")
            return ValidationResult(
                status=ValidationStatus.INVALID,
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                source=price_data.instrument,
                data_type="price",
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    async def validate_candle_data(self, candle_data: CandleData,
                                 previous_data: Optional[CandleData] = None,
                                 cross_source_data: Optional[List[CandleData]] = None) -> ValidationResult:
        """Validate candle data with comprehensive checks"""
        start_time = time.time()
        
        try:
            # Initialize result
            result = ValidationResult(
                status=ValidationStatus.PENDING,
                is_valid=True,
                source=candle_data.instrument,
                data_type="candle"
            )
            
            # Schema validation
            if self.config.enable_schema_validation:
                schema_result = await self.schema_validator.validate_candle_data(candle_data)
                if not schema_result.is_valid:
                    result.errors.extend(schema_result.errors)
                    result.is_valid = False
                    result.status = ValidationStatus.INVALID
                result.warnings.extend(schema_result.warnings)
            
            # Range validation
            if self.config.enable_range_validation and result.is_valid:
                range_result = await self.range_validator.validate_candle_data(candle_data, previous_data)
                if not range_result.is_valid:
                    result.errors.extend(range_result.errors)
                    result.is_valid = False
                    result.status = ValidationStatus.INVALID
                result.warnings.extend(range_result.warnings)
            
            # Anomaly detection
            if self.config.enable_anomaly_detection and result.is_valid:
                anomaly_result = await self.anomaly_detector.detect_candle_anomaly(
                    candle_data, self.validation_history[candle_data.instrument]
                )
                if anomaly_result.is_anomaly:
                    result.warnings.extend(anomaly_result.warnings)
                    if anomaly_result.severity > self.config.anomaly_threshold:
                        result.errors.append(f"Severe anomaly detected: {anomaly_result.description}")
                        result.is_valid = False
                        result.status = ValidationStatus.QUARANTINED
            
            # Calculate quality score
            if self.config.enable_quality_metrics:
                quality_score = await self.quality_metrics.calculate_candle_quality_score(
                    candle_data, result
                )
                result.quality_score = quality_score
                self.quality_scores[candle_data.instrument].append(quality_score)
            
            # Update final status
            if result.is_valid and result.status == ValidationStatus.PENDING:
                result.status = ValidationStatus.VALID
            
            # Update validation history
            self.validation_history[candle_data.instrument].append(result)
            
            # Update performance metrics
            validation_time_ms = (time.time() - start_time) * 1000
            result.validation_time_ms = validation_time_ms
            self._update_performance_metrics(validation_time_ms, result.is_valid)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating candle data: {e}")
            return ValidationResult(
                status=ValidationStatus.INVALID,
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                source=candle_data.instrument,
                data_type="candle",
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    async def validate_batch(self, data_batch: List[Union[PriceData, CandleData]]) -> List[ValidationResult]:
        """Validate a batch of data points"""
        if not self.config.enable_async_validation:
            # Sequential validation
            results = []
            for data_point in data_batch:
                if isinstance(data_point, PriceData):
                    result = await self.validate_price_data(data_point)
                else:
                    result = await self.validate_candle_data(data_point)
                results.append(result)
            return results
        
        # Async validation
        tasks = []
        for data_point in data_batch:
            if isinstance(data_point, PriceData):
                task = self.validate_price_data(data_point)
            else:
                task = self.validate_candle_data(data_point)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_quality_report(self, instrument: Optional[str] = None) -> QualityReport:
        """Get comprehensive quality report"""
        return await self.quality_metrics.generate_report(
            self.validation_history,
            self.quality_scores,
            self.correction_history,
            instrument
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            'total_validations': self.total_validations,
            'successful_validations': self.successful_validations,
            'failed_validations': self.failed_validations,
            'success_rate': self.successful_validations / max(self.total_validations, 1),
            'avg_validation_time_ms': self.avg_validation_time_ms,
            'total_corrections': self.total_corrections,
            'instruments_tracked': len(self.validation_history),
            'total_quality_scores': sum(len(scores) for scores in self.quality_scores.values()),
            'config': {
                'schema_validation': self.config.enable_schema_validation,
                'range_validation': self.config.enable_range_validation,
                'anomaly_detection': self.config.enable_anomaly_detection,
                'cross_source_validation': self.config.enable_cross_source_validation,
                'auto_correction': self.config.enable_auto_correction
            }
        }
    
    def _update_performance_metrics(self, validation_time_ms: float, is_valid: bool):
        """Update performance metrics"""
        self.total_validations += 1
        if is_valid:
            self.successful_validations += 1
        else:
            self.failed_validations += 1
        
        # Update average validation time
        if self.total_validations == 1:
            self.avg_validation_time_ms = validation_time_ms
        else:
            self.avg_validation_time_ms = (
                (self.avg_validation_time_ms * (self.total_validations - 1) + validation_time_ms) 
                / self.total_validations
            )
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.avg_validation_time_ms = 0.0
        self.total_corrections = 0
        self.validation_history.clear()
        self.quality_scores.clear()
        self.correction_history.clear()
        logger.info("Validation metrics reset")

# Example usage and testing
async def test_validation_engine():
    """Test the validation engine"""
    from ..brokers.broker_manager import PriceData, CandleData
    
    # Create validation engine
    config = ValidationConfig(
        enable_schema_validation=True,
        enable_range_validation=True,
        enable_anomaly_detection=True,
        enable_cross_source_validation=True,
        enable_quality_metrics=True,
        enable_auto_correction=True
    )
    
    engine = ValidationEngine(config)
    
    # Test price data validation
    price_data = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=1.1000,
        ask=1.1002,
        spread=0.0002
    )
    
    result = await engine.validate_price_data(price_data)
    print(f"Price validation result: {result.status}, Valid: {result.is_valid}")
    print(f"Quality score: {result.quality_score:.2f}")
    print(f"Validation time: {result.validation_time_ms:.2f}ms")
    
    # Test candle data validation
    candle_data = CandleData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        open=1.1000,
        high=1.1005,
        low=1.0995,
        close=1.1002,
        volume=1000
    )
    
    result = await engine.validate_candle_data(candle_data)
    print(f"Candle validation result: {result.status}, Valid: {result.is_valid}")
    
    # Get quality report
    report = await engine.get_quality_report()
    print(f"Quality report: {report.overall_quality_score:.2f}")
    
    # Get validation stats
    stats = engine.get_validation_stats()
    print(f"Validation stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_validation_engine())

