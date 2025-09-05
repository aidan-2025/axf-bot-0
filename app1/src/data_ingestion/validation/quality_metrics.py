#!/usr/bin/env python3
"""
Quality Metrics
Tracks and reports data quality metrics and statistics
"""

import logging
import statistics
from typing import Dict, List, Optional, Any, Deque, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math

from ..brokers.broker_manager import PriceData, CandleData

logger = logging.getLogger(__name__)

@dataclass
class QualityMetricsConfig:
    """Configuration for quality metrics"""
    # Metrics collection
    metrics_window_size: int = 1000
    quality_threshold: float = 0.8
    
    # Latency tracking
    max_acceptable_latency_ms: float = 100.0
    latency_percentiles: List[float] = field(default_factory=lambda: [50.0, 95.0, 99.0])
    
    # Completeness tracking
    expected_frequency_seconds: int = 1
    max_gap_seconds: int = 300
    
    # Accuracy tracking
    accuracy_threshold: float = 0.95
    enable_accuracy_tracking: bool = True
    
    # Anomaly tracking
    anomaly_threshold: float = 0.05  # 5% anomaly rate
    enable_anomaly_tracking: bool = True

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    timestamp: datetime
    overall_quality_score: float
    completeness_score: float
    accuracy_score: float
    latency_score: float
    consistency_score: float
    anomaly_rate: float
    
    # Detailed metrics
    total_data_points: int
    valid_data_points: int
    invalid_data_points: int
    corrected_data_points: int
    
    # Latency metrics
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Completeness metrics
    expected_points: int
    received_points: int
    missing_points: int
    gap_count: int
    
    # Accuracy metrics
    validation_errors: int
    validation_warnings: int
    cross_source_disagreements: int
    
    # Anomaly metrics
    anomaly_count: int
    severe_anomaly_count: int
    
    # Instrument-specific metrics
    instrument_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)

class QualityMetrics:
    """Tracks and reports data quality metrics"""
    
    def __init__(self, config: Optional[QualityMetricsConfig] = None):
        """Initialize quality metrics tracker"""
        self.config = config or QualityMetricsConfig()
        
        # Data quality tracking
        self.validation_results: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=self.config.metrics_window_size))
        self.quality_scores: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=self.config.metrics_window_size))
        self.correction_history: List[Any] = []
        
        # Latency tracking
        self.latency_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=self.config.metrics_window_size))
        
        # Completeness tracking
        self.last_timestamps: Dict[str, datetime] = {}
        self.expected_points: Dict[str, int] = defaultdict(int)
        self.received_points: Dict[str, int] = defaultdict(int)
        self.gap_count: Dict[str, int] = defaultdict(int)
        
        # Accuracy tracking
        self.validation_errors: Dict[str, int] = defaultdict(int)
        self.validation_warnings: Dict[str, int] = defaultdict(int)
        self.cross_source_disagreements: Dict[str, int] = defaultdict(int)
        
        # Anomaly tracking
        self.anomaly_count: Dict[str, int] = defaultdict(int)
        self.severe_anomaly_count: Dict[str, int] = defaultdict(int)
        
        # Overall statistics
        self.total_data_points = 0
        self.total_valid_points = 0
        self.total_invalid_points = 0
        self.total_corrected_points = 0
        
        logger.info("QualityMetrics initialized with config: %s", self.config)
    
    async def calculate_price_quality_score(self, price_data: PriceData, 
                                          validation_result: Any) -> float:
        """Calculate quality score for price data"""
        try:
            score = 1.0
            
            # Base score from validation result
            if hasattr(validation_result, 'is_valid'):
                if not validation_result.is_valid:
                    score *= 0.5  # Significant penalty for invalid data
                elif validation_result.status == 'quarantined':
                    score *= 0.7  # Penalty for quarantined data
                elif validation_result.status == 'corrected':
                    score *= 0.9  # Small penalty for corrected data
            
            # Penalty for errors
            if hasattr(validation_result, 'errors') and validation_result.errors:
                error_penalty = min(len(validation_result.errors) * 0.1, 0.5)
                score -= error_penalty
            
            # Penalty for warnings
            if hasattr(validation_result, 'warnings') and validation_result.warnings:
                warning_penalty = min(len(validation_result.warnings) * 0.05, 0.2)
                score -= warning_penalty
            
            # Bonus for high quality data
            if hasattr(validation_result, 'quality_score'):
                score = (score + validation_result.quality_score) / 2
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
            
            # Update tracking
            self._update_price_metrics(price_data, validation_result, score)
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating price quality score: {e}")
            return 0.5  # Default medium quality score
    
    async def calculate_candle_quality_score(self, candle_data: CandleData,
                                           validation_result: Any) -> float:
        """Calculate quality score for candle data"""
        try:
            score = 1.0
            
            # Base score from validation result
            if hasattr(validation_result, 'is_valid'):
                if not validation_result.is_valid:
                    score *= 0.5
                elif validation_result.status == 'quarantined':
                    score *= 0.7
                elif validation_result.status == 'corrected':
                    score *= 0.9
            
            # Penalty for errors
            if hasattr(validation_result, 'errors') and validation_result.errors:
                error_penalty = min(len(validation_result.errors) * 0.1, 0.5)
                score -= error_penalty
            
            # Penalty for warnings
            if hasattr(validation_result, 'warnings') and validation_result.warnings:
                warning_penalty = min(len(validation_result.warnings) * 0.05, 0.2)
                score -= warning_penalty
            
            # OHLC consistency bonus
            ohlc_consistency = self._calculate_ohlc_consistency_score(candle_data)
            score = (score + ohlc_consistency) / 2
            
            # Volume reasonableness bonus
            volume_score = self._calculate_volume_reasonableness_score(candle_data)
            score = (score + volume_score) / 2
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
            
            # Update tracking
            self._update_candle_metrics(candle_data, validation_result, score)
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating candle quality score: {e}")
            return 0.5
    
    async def generate_report(self, validation_history: Dict[str, Deque],
                            quality_scores: Dict[str, Deque],
                            correction_history: List[Any],
                            instrument: Optional[str] = None) -> QualityReport:
        """Generate comprehensive quality report"""
        try:
            now = datetime.utcnow()
            
            # Calculate overall metrics
            total_points, valid_points, invalid_points, corrected_points = self._calculate_basic_metrics()
            
            # Calculate completeness
            completeness_score, expected_points, received_points, missing_points, gap_count = self._calculate_completeness_metrics()
            
            # Calculate accuracy
            accuracy_score, validation_errors, validation_warnings, cross_source_disagreements = self._calculate_accuracy_metrics()
            
            # Calculate latency
            latency_score, avg_latency, p95_latency, p99_latency = self._calculate_latency_metrics()
            
            # Calculate consistency
            consistency_score = self._calculate_consistency_score(quality_scores)
            
            # Calculate anomaly rate
            anomaly_rate, anomaly_count, severe_anomaly_count = self._calculate_anomaly_metrics()
            
            # Calculate overall quality score
            overall_quality = self._calculate_overall_quality_score(
                completeness_score, accuracy_score, latency_score, consistency_score, anomaly_rate
            )
            
            # Generate instrument-specific metrics
            instrument_metrics = self._generate_instrument_metrics(instrument)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                completeness_score, accuracy_score, latency_score, consistency_score, anomaly_rate
            )
            
            return QualityReport(
                timestamp=now,
                overall_quality_score=overall_quality,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                latency_score=latency_score,
                consistency_score=consistency_score,
                anomaly_rate=anomaly_rate,
                total_data_points=total_points,
                valid_data_points=valid_points,
                invalid_data_points=invalid_points,
                corrected_data_points=corrected_points,
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                expected_points=expected_points,
                received_points=received_points,
                missing_points=missing_points,
                gap_count=gap_count,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
                cross_source_disagreements=cross_source_disagreements,
                anomaly_count=anomaly_count,
                severe_anomaly_count=severe_anomaly_count,
                instrument_metrics=instrument_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return QualityReport(
                timestamp=datetime.utcnow(),
                overall_quality_score=0.0,
                completeness_score=0.0,
                accuracy_score=0.0,
                latency_score=0.0,
                consistency_score=0.0,
                anomaly_rate=1.0,
                total_data_points=0,
                valid_data_points=0,
                invalid_data_points=0,
                corrected_data_points=0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                expected_points=0,
                received_points=0,
                missing_points=0,
                gap_count=0,
                validation_errors=0,
                validation_warnings=0,
                cross_source_disagreements=0,
                anomaly_count=0,
                severe_anomaly_count=0,
                recommendations=["Error generating quality report"]
            )
    
    def _update_price_metrics(self, price_data: PriceData, validation_result: Any, quality_score: float):
        """Update price-specific metrics"""
        instrument = price_data.instrument
        
        # Update quality scores
        self.quality_scores[instrument].append(quality_score)
        
        # Update validation results
        self.validation_results[instrument].append(validation_result)
        
        # Update basic counts
        self.total_data_points += 1
        if hasattr(validation_result, 'is_valid') and validation_result.is_valid:
            self.total_valid_points += 1
        else:
            self.total_invalid_points += 1
        
        if hasattr(validation_result, 'status') and validation_result.status == 'corrected':
            self.total_corrected_points += 1
        
        # Update error/warning counts
        if hasattr(validation_result, 'errors'):
            self.validation_errors[instrument] += len(validation_result.errors)
        
        if hasattr(validation_result, 'warnings'):
            self.validation_warnings[instrument] += len(validation_result.warnings)
        
        # Update anomaly counts
        if hasattr(validation_result, 'status') and validation_result.status == 'quarantined':
            self.anomaly_count[instrument] += 1
            if hasattr(validation_result, 'severity') and validation_result.severity > 7.0:
                self.severe_anomaly_count[instrument] += 1
        
        # Update completeness tracking
        self._update_completeness_tracking(price_data)
    
    def _update_candle_metrics(self, candle_data: CandleData, validation_result: Any, quality_score: float):
        """Update candle-specific metrics"""
        instrument = candle_data.instrument
        
        # Update quality scores
        self.quality_scores[instrument].append(quality_score)
        
        # Update validation results
        self.validation_results[instrument].append(validation_result)
        
        # Update basic counts
        self.total_data_points += 1
        if hasattr(validation_result, 'is_valid') and validation_result.is_valid:
            self.total_valid_points += 1
        else:
            self.total_invalid_points += 1
        
        # Update completeness tracking
        self._update_completeness_tracking(candle_data)
    
    def _update_completeness_tracking(self, data: Union[PriceData, CandleData]):
        """Update completeness tracking"""
        instrument = data.instrument
        current_time = data.time
        
        # Update received points
        self.received_points[instrument] += 1
        
        # Check for gaps
        if instrument in self.last_timestamps:
            time_gap = (current_time - self.last_timestamps[instrument]).total_seconds()
            if time_gap > self.config.max_gap_seconds:
                self.gap_count[instrument] += 1
        
        # Update last timestamp
        self.last_timestamps[instrument] = current_time
        
        # Calculate expected points (simplified)
        if instrument in self.last_timestamps:
            time_span = (current_time - self.last_timestamps[instrument]).total_seconds()
            expected_increment = max(1, int(time_span / self.config.expected_frequency_seconds))
            self.expected_points[instrument] += expected_increment
    
    def _calculate_basic_metrics(self) -> tuple[int, int, int, int]:
        """Calculate basic data metrics"""
        return (
            self.total_data_points,
            self.total_valid_points,
            self.total_invalid_points,
            self.total_corrected_points
        )
    
    def _calculate_completeness_metrics(self) -> tuple[float, int, int, int, int]:
        """Calculate completeness metrics"""
        total_expected = sum(self.expected_points.values())
        total_received = sum(self.received_points.values())
        total_missing = max(0, total_expected - total_received)
        total_gaps = sum(self.gap_count.values())
        
        completeness_score = (total_received / total_expected) if total_expected > 0 else 1.0
        completeness_score = max(0.0, min(1.0, completeness_score))
        
        return completeness_score, total_expected, total_received, total_missing, total_gaps
    
    def _calculate_accuracy_metrics(self) -> tuple[float, int, int, int]:
        """Calculate accuracy metrics"""
        total_errors = sum(self.validation_errors.values())
        total_warnings = sum(self.validation_warnings.values())
        total_disagreements = sum(self.cross_source_disagreements.values())
        
        total_points = self.total_data_points
        if total_points == 0:
            return 1.0, 0, 0, 0
        
        # Accuracy based on error rate
        error_rate = total_errors / total_points
        accuracy_score = max(0.0, 1.0 - error_rate)
        
        return accuracy_score, total_errors, total_warnings, total_disagreements
    
    def _calculate_latency_metrics(self) -> tuple[float, float, float, float]:
        """Calculate latency metrics"""
        all_latencies = []
        for latencies in self.latency_history.values():
            all_latencies.extend(latencies)
        
        if not all_latencies:
            return 1.0, 0.0, 0.0, 0.0
        
        avg_latency = statistics.mean(all_latencies)
        p95_latency = self._calculate_percentile(all_latencies, 95.0)
        p99_latency = self._calculate_percentile(all_latencies, 99.0)
        
        # Latency score based on acceptable latency threshold
        latency_score = max(0.0, 1.0 - (avg_latency / self.config.max_acceptable_latency_ms))
        
        return latency_score, avg_latency, p95_latency, p99_latency
    
    def _calculate_consistency_score(self, quality_scores: Dict[str, Deque]) -> float:
        """Calculate consistency score based on quality score variance"""
        all_scores = []
        for scores in quality_scores.values():
            all_scores.extend(scores)
        
        if len(all_scores) < 2:
            return 1.0
        
        mean_score = statistics.mean(all_scores)
        std_score = statistics.stdev(all_scores)
        
        # Consistency score based on coefficient of variation
        cv = std_score / mean_score if mean_score > 0 else 0
        consistency_score = max(0.0, 1.0 - cv)
        
        return consistency_score
    
    def _calculate_anomaly_metrics(self) -> tuple[float, int, int]:
        """Calculate anomaly metrics"""
        total_anomalies = sum(self.anomaly_count.values())
        total_severe_anomalies = sum(self.severe_anomaly_count.values())
        
        total_points = self.total_data_points
        if total_points == 0:
            return 0.0, 0, 0
        
        anomaly_rate = total_anomalies / total_points
        return anomaly_rate, total_anomalies, total_severe_anomalies
    
    def _calculate_overall_quality_score(self, completeness: float, accuracy: float, 
                                       latency: float, consistency: float, anomaly_rate: float) -> float:
        """Calculate overall quality score"""
        # Weighted average of all quality dimensions
        weights = {
            'completeness': 0.25,
            'accuracy': 0.30,
            'latency': 0.20,
            'consistency': 0.15,
            'anomaly': 0.10
        }
        
        anomaly_score = max(0.0, 1.0 - anomaly_rate)
        
        overall_score = (
            weights['completeness'] * completeness +
            weights['accuracy'] * accuracy +
            weights['latency'] * latency +
            weights['consistency'] * consistency +
            weights['anomaly'] * anomaly_score
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _calculate_ohlc_consistency_score(self, candle_data: CandleData) -> float:
        """Calculate OHLC consistency score"""
        score = 1.0
        
        # Check basic OHLC relationships
        if candle_data.high < max(candle_data.open, candle_data.close):
            score -= 0.5
        if candle_data.low > min(candle_data.open, candle_data.close):
            score -= 0.5
        
        # Check for reasonable range
        range_size = candle_data.high - candle_data.low
        if candle_data.close > 0:
            range_pct = range_size / candle_data.close * 100
            if range_pct > 20.0:  # Extreme range
                score -= 0.3
        
        return max(0.0, score)
    
    def _calculate_volume_reasonableness_score(self, candle_data: CandleData) -> float:
        """Calculate volume reasonableness score"""
        if candle_data.volume <= 0:
            return 0.0
        
        # Check for reasonable volume (simplified)
        if candle_data.volume > 1e9:  # Unreasonably large volume
            return 0.5
        
        return 1.0
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        
        return sorted_data[index]
    
    def _generate_instrument_metrics(self, instrument: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Generate instrument-specific metrics"""
        metrics = {}
        
        if instrument:
            # Single instrument metrics
            metrics[instrument] = self._get_instrument_metrics(instrument)
        else:
            # All instruments metrics
            for inst in set(list(self.quality_scores.keys()) + list(self.validation_results.keys())):
                metrics[inst] = self._get_instrument_metrics(inst)
        
        return metrics
    
    def _get_instrument_metrics(self, instrument: str) -> Dict[str, Any]:
        """Get metrics for a specific instrument"""
        quality_scores = list(self.quality_scores.get(instrument, []))
        validation_results = list(self.validation_results.get(instrument, []))
        
        if not quality_scores:
            return {
                'avg_quality_score': 0.0,
                'total_points': 0,
                'valid_points': 0,
                'error_count': 0,
                'warning_count': 0,
                'anomaly_count': 0
            }
        
        valid_count = sum(1 for r in validation_results if hasattr(r, 'is_valid') and r.is_valid)
        error_count = sum(len(r.errors) for r in validation_results if hasattr(r, 'errors'))
        warning_count = sum(len(r.warnings) for r in validation_results if hasattr(r, 'warnings'))
        anomaly_count = sum(1 for r in validation_results if hasattr(r, 'status') and r.status == 'quarantined')
        
        return {
            'avg_quality_score': statistics.mean(quality_scores),
            'total_points': len(quality_scores),
            'valid_points': valid_count,
            'error_count': error_count,
            'warning_count': warning_count,
            'anomaly_count': anomaly_count
        }
    
    def _generate_recommendations(self, completeness: float, accuracy: float, 
                                latency: float, consistency: float, anomaly_rate: float) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if completeness < 0.9:
            recommendations.append("Data completeness is low - check for missing data points and gaps")
        
        if accuracy < 0.95:
            recommendations.append("Data accuracy is below threshold - review validation rules and data sources")
        
        if latency < 0.8:
            recommendations.append("Data latency is high - optimize data processing pipeline")
        
        if consistency < 0.8:
            recommendations.append("Data consistency is low - investigate data source variations")
        
        if anomaly_rate > 0.05:
            recommendations.append("Anomaly rate is high - review anomaly detection thresholds and data quality")
        
        if not recommendations:
            recommendations.append("Data quality is within acceptable parameters")
        
        return recommendations

# Example usage and testing
async def test_quality_metrics():
    """Test the quality metrics tracker"""
    from ..brokers.broker_manager import PriceData
    
    config = QualityMetricsConfig(
        metrics_window_size=100,
        quality_threshold=0.8
    )
    
    metrics = QualityMetrics(config)
    
    # Test quality score calculation
    price_data = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=1.1000,
        ask=1.1002,
        spread=0.0002
    )
    
    class MockValidationResult:
        is_valid = True
        status = 'valid'
        errors = []
        warnings = []
        quality_score = 0.9
    
    validation_result = MockValidationResult()
    quality_score = await metrics.calculate_price_quality_score(price_data, validation_result)
    print(f"Quality score: {quality_score:.2f}")
    
    # Generate report
    report = await metrics.generate_report({}, {}, [])
    print(f"Overall quality: {report.overall_quality_score:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_quality_metrics())

