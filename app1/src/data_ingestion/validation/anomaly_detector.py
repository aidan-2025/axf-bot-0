#!/usr/bin/env python3
"""
Anomaly Detector
Detects anomalies in market data using statistical and ML methods
"""

import logging
import math
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union, Deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from ..brokers.broker_manager import PriceData, CandleData

logger = logging.getLogger(__name__)

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    severity: float  # 0-10 scale
    description: str
    warnings: List[str] = field(default_factory=list)
    detection_method: str = ""
    confidence: float = 0.0

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    # Statistical thresholds
    z_score_threshold: float = 3.0
    mad_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # Window sizes
    price_window_size: int = 100
    candle_window_size: int = 50
    volume_window_size: int = 20
    
    # Price jump detection
    max_price_jump_pct: float = 5.0
    max_spread_jump_pct: float = 50.0
    
    # Volume anomaly detection
    volume_spike_multiplier: float = 3.0
    volume_drop_multiplier: float = 0.1
    
    # Time-based anomalies
    max_gap_seconds: int = 300  # 5 minutes
    min_expected_frequency_seconds: int = 1
    
    # ML-based detection (placeholder for future implementation)
    enable_ml_detection: bool = False
    ml_model_path: Optional[str] = None

class AnomalyDetector:
    """Detects anomalies in market data using multiple methods"""
    
    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        """Initialize anomaly detector"""
        self.config = config or AnomalyDetectionConfig()
        
        # Data windows for analysis
        self.price_windows: Dict[str, Deque[PriceData]] = {}
        self.candle_windows: Dict[str, Deque[CandleData]] = {}
        self.volume_windows: Dict[str, Deque[float]] = {}
        
        # Anomaly statistics
        self.anomaly_counts: Dict[str, int] = {
            'price_anomalies': 0,
            'volume_anomalies': 0,
            'spread_anomalies': 0,
            'time_anomalies': 0,
            'pattern_anomalies': 0
        }
        
        logger.info("AnomalyDetector initialized with config: %s", self.config)
    
    async def detect_price_anomaly(self, price_data: PriceData, 
                                 validation_history: Optional[Deque] = None) -> AnomalyDetectionResult:
        """Detect anomalies in price data"""
        try:
            instrument = price_data.instrument
            
            # Initialize window if needed
            if instrument not in self.price_windows:
                self.price_windows[instrument] = deque(maxlen=self.config.price_window_size)
            
            # Get current window
            window = self.price_windows[instrument]
            
            # Detect different types of anomalies
            anomalies = []
            
            # Statistical anomalies
            if len(window) >= 10:  # Need minimum data for statistical analysis
                stat_anomaly = self._detect_statistical_anomaly(price_data, window)
                if stat_anomaly.is_anomaly:
                    anomalies.append(stat_anomaly)
            
            # Price jump anomalies
            if len(window) >= 1:
                jump_anomaly = self._detect_price_jump_anomaly(price_data, window[-1])
                if jump_anomaly.is_anomaly:
                    anomalies.append(jump_anomaly)
            
            # Spread anomalies
            spread_anomaly = self._detect_spread_anomaly(price_data, window)
            if spread_anomaly.is_anomaly:
                anomalies.append(spread_anomaly)
            
            # Time-based anomalies
            time_anomaly = self._detect_time_anomaly(price_data, window)
            if time_anomaly.is_anomaly:
                anomalies.append(time_anomaly)
            
            # Pattern anomalies
            if len(window) >= 5:
                pattern_anomaly = self._detect_pattern_anomaly(price_data, window)
                if pattern_anomaly.is_anomaly:
                    anomalies.append(pattern_anomaly)
            
            # Combine results
            if anomalies:
                # Find the most severe anomaly
                max_severity = max(anomaly.severity for anomaly in anomalies)
                max_anomaly = next(anomaly for anomaly in anomalies if anomaly.severity == max_severity)
                
                # Update anomaly counts
                self.anomaly_counts['price_anomalies'] += 1
                
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=max_severity,
                    description=max_anomaly.description,
                    warnings=[a.description for a in anomalies if a.severity < max_severity],
                    detection_method=max_anomaly.detection_method,
                    confidence=max_anomaly.confidence
                )
            
            # Update window
            window.append(price_data)
            
            return AnomalyDetectionResult(
                is_anomaly=False,
                severity=0.0,
                description="No anomalies detected",
                detection_method="none",
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Error detecting price anomaly: {e}")
            return AnomalyDetectionResult(
                is_anomaly=True,
                severity=10.0,
                description=f"Anomaly detection error: {str(e)}",
                detection_method="error",
                confidence=0.0
            )
    
    async def detect_candle_anomaly(self, candle_data: CandleData,
                                  validation_history: Optional[Deque] = None) -> AnomalyDetectionResult:
        """Detect anomalies in candle data"""
        try:
            instrument = candle_data.instrument
            
            # Initialize window if needed
            if instrument not in self.candle_windows:
                self.candle_windows[instrument] = deque(maxlen=self.config.candle_window_size)
            
            window = self.candle_windows[instrument]
            
            # Detect different types of anomalies
            anomalies = []
            
            # OHLC consistency anomalies
            ohlc_anomaly = self._detect_ohlc_anomaly(candle_data)
            if ohlc_anomaly.is_anomaly:
                anomalies.append(ohlc_anomaly)
            
            # Volume anomalies
            volume_anomaly = self._detect_volume_anomaly(candle_data, window)
            if volume_anomaly.is_anomaly:
                anomalies.append(volume_anomaly)
            
            # Range anomalies
            if len(window) >= 5:
                range_anomaly = self._detect_range_anomaly(candle_data, window)
                if range_anomaly.is_anomaly:
                    anomalies.append(range_anomaly)
            
            # Combine results
            if anomalies:
                max_severity = max(anomaly.severity for anomaly in anomalies)
                max_anomaly = next(anomaly for anomaly in anomalies if anomaly.severity == max_severity)
                
                self.anomaly_counts['volume_anomalies'] += 1
                
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=max_severity,
                    description=max_anomaly.description,
                    warnings=[a.description for a in anomalies if a.severity < max_severity],
                    detection_method=max_anomaly.detection_method,
                    confidence=max_anomaly.confidence
                )
            
            # Update window
            window.append(candle_data)
            
            return AnomalyDetectionResult(
                is_anomaly=False,
                severity=0.0,
                description="No anomalies detected",
                detection_method="none",
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Error detecting candle anomaly: {e}")
            return AnomalyDetectionResult(
                is_anomaly=True,
                severity=10.0,
                description=f"Anomaly detection error: {str(e)}",
                detection_method="error",
                confidence=0.0
            )
    
    def _detect_statistical_anomaly(self, price_data: PriceData, window: Deque[PriceData]) -> AnomalyDetectionResult:
        """Detect statistical anomalies using Z-score and MAD"""
        if len(window) < 10:
            return AnomalyDetectionResult(False, 0.0, "", detection_method="statistical")
        
        # Calculate mid prices
        mid_prices = [(p.bid + p.ask) / 2 for p in window]
        current_mid = (price_data.bid + price_data.ask) / 2
        
        # Z-score method
        mean_price = statistics.mean(mid_prices)
        std_price = statistics.stdev(mid_prices) if len(mid_prices) > 1 else 0
        
        if std_price > 0:
            z_score = abs(current_mid - mean_price) / std_price
            if z_score > self.config.z_score_threshold:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=min(z_score / self.config.z_score_threshold, 10.0),
                    description=f"Statistical anomaly: Z-score {z_score:.2f} (threshold {self.config.z_score_threshold})",
                    detection_method="z_score",
                    confidence=min(z_score / self.config.z_score_threshold, 1.0)
                )
        
        # MAD (Median Absolute Deviation) method
        median_price = statistics.median(mid_prices)
        mad = statistics.median([abs(p - median_price) for p in mid_prices])
        
        if mad > 0:
            mad_score = abs(current_mid - median_price) / mad
            if mad_score > self.config.mad_threshold:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=min(mad_score / self.config.mad_threshold, 10.0),
                    description=f"Statistical anomaly: MAD score {mad_score:.2f} (threshold {self.config.mad_threshold})",
                    detection_method="mad",
                    confidence=min(mad_score / self.config.mad_threshold, 1.0)
                )
        
        return AnomalyDetectionResult(False, 0.0, "", detection_method="statistical")
    
    def _detect_price_jump_anomaly(self, price_data: PriceData, previous_data: PriceData) -> AnomalyDetectionResult:
        """Detect price jump anomalies"""
        prev_mid = (previous_data.bid + previous_data.ask) / 2
        curr_mid = (price_data.bid + price_data.ask) / 2
        
        if prev_mid > 0:
            jump_pct = abs(curr_mid - prev_mid) / prev_mid * 100
            
            if jump_pct > self.config.max_price_jump_pct:
                severity = min(jump_pct / self.config.max_price_jump_pct, 10.0)
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=severity,
                    description=f"Price jump anomaly: {jump_pct:.2f}% change (threshold {self.config.max_price_jump_pct}%)",
                    detection_method="price_jump",
                    confidence=min(jump_pct / self.config.max_price_jump_pct, 1.0)
                )
        
        return AnomalyDetectionResult(False, 0.0, "", detection_method="price_jump")
    
    def _detect_spread_anomaly(self, price_data: PriceData, window: Deque[PriceData]) -> AnomalyDetectionResult:
        """Detect spread anomalies"""
        if len(window) < 5:
            return AnomalyDetectionResult(False, 0.0, "", detection_method="spread")
        
        # Calculate recent spread statistics
        recent_spreads = [p.spread for p in list(window)[-10:]]
        mean_spread = statistics.mean(recent_spreads)
        std_spread = statistics.stdev(recent_spreads) if len(recent_spreads) > 1 else 0
        
        if std_spread > 0:
            z_score = abs(price_data.spread - mean_spread) / std_spread
            if z_score > self.config.z_score_threshold:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=min(z_score / self.config.z_score_threshold, 10.0),
                    description=f"Spread anomaly: Z-score {z_score:.2f} (mean {mean_spread:.6f})",
                    detection_method="spread",
                    confidence=min(z_score / self.config.z_score_threshold, 1.0)
                )
        
        # Check for spread jump
        if len(window) >= 1:
            prev_spread = window[-1].spread
            if prev_spread > 0:
                spread_jump_pct = abs(price_data.spread - prev_spread) / prev_spread * 100
                if spread_jump_pct > self.config.max_spread_jump_pct:
                    return AnomalyDetectionResult(
                        is_anomaly=True,
                        severity=min(spread_jump_pct / self.config.max_spread_jump_pct, 10.0),
                        description=f"Spread jump anomaly: {spread_jump_pct:.2f}% change",
                        detection_method="spread_jump",
                        confidence=min(spread_jump_pct / self.config.max_spread_jump_pct, 1.0)
                    )
        
        return AnomalyDetectionResult(False, 0.0, "", detection_method="spread")
    
    def _detect_time_anomaly(self, price_data: PriceData, window: Deque[PriceData]) -> AnomalyDetectionResult:
        """Detect time-based anomalies"""
        if len(window) < 1:
            return AnomalyDetectionResult(False, 0.0, "", detection_method="time")
        
        # Check for time gaps
        last_time = window[-1].time
        time_gap = (price_data.time - last_time).total_seconds()
        
        if time_gap > self.config.max_gap_seconds:
            severity = min(time_gap / self.config.max_gap_seconds, 10.0)
            return AnomalyDetectionResult(
                is_anomaly=True,
                severity=severity,
                description=f"Time gap anomaly: {time_gap:.0f}s gap (max {self.config.max_gap_seconds}s)",
                detection_method="time_gap",
                confidence=min(time_gap / self.config.max_gap_seconds, 1.0)
            )
        
        # Check for future timestamps
        now = datetime.utcnow()
        if price_data.time > now + timedelta(seconds=60):  # More than 1 minute in future
            return AnomalyDetectionResult(
                is_anomaly=True,
                severity=8.0,
                description=f"Future timestamp anomaly: {price_data.time} is in the future",
                detection_method="future_timestamp",
                confidence=0.9
            )
        
        return AnomalyDetectionResult(False, 0.0, "", detection_method="time")
    
    def _detect_pattern_anomaly(self, price_data: PriceData, window: Deque[PriceData]) -> AnomalyDetectionResult:
        """Detect pattern-based anomalies"""
        if len(window) < 5:
            return AnomalyDetectionResult(False, 0.0, "", detection_method="pattern")
        
        # Check for repeated identical prices (possible data feed issue)
        recent_prices = [(p.bid + p.ask) / 2 for p in list(window)[-5:]]
        current_mid = (price_data.bid + price_data.ask) / 2
        
        identical_count = sum(1 for p in recent_prices if abs(p - current_mid) < 1e-8)
        if identical_count >= 3:
            return AnomalyDetectionResult(
                is_anomaly=True,
                severity=7.0,
                description=f"Pattern anomaly: {identical_count} identical prices in recent history",
                detection_method="repeated_prices",
                confidence=0.8
            )
        
        # Check for unrealistic price patterns (e.g., perfect staircase)
        if len(recent_prices) >= 4:
            diffs = [recent_prices[i+1] - recent_prices[i] for i in range(len(recent_prices)-1)]
            if all(abs(d - diffs[0]) < 1e-8 for d in diffs) and abs(diffs[0]) > 1e-8:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=6.0,
                    description="Pattern anomaly: Perfect staircase pattern detected",
                    detection_method="staircase_pattern",
                    confidence=0.7
                )
        
        return AnomalyDetectionResult(False, 0.0, "", detection_method="pattern")
    
    def _detect_ohlc_anomaly(self, candle_data: CandleData) -> AnomalyDetectionResult:
        """Detect OHLC consistency anomalies"""
        # Check for impossible OHLC relationships
        if candle_data.high < max(candle_data.open, candle_data.close):
            return AnomalyDetectionResult(
                is_anomaly=True,
                severity=9.0,
                description=f"OHLC anomaly: High {candle_data.high} < max(open, close)",
                detection_method="ohlc_consistency",
                confidence=1.0
            )
        
        if candle_data.low > min(candle_data.open, candle_data.close):
            return AnomalyDetectionResult(
                is_anomaly=True,
                severity=9.0,
                description=f"OHLC anomaly: Low {candle_data.low} > min(open, close)",
                detection_method="ohlc_consistency",
                confidence=1.0
            )
        
        # Check for extreme ranges
        range_size = candle_data.high - candle_data.low
        if candle_data.close > 0:
            range_pct = range_size / candle_data.close * 100
            if range_pct > 20.0:  # 20% range is extreme
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=min(range_pct / 20.0, 10.0),
                    description=f"OHLC anomaly: Extreme range {range_pct:.2f}%",
                    detection_method="extreme_range",
                    confidence=min(range_pct / 20.0, 1.0)
                )
        
        return AnomalyDetectionResult(False, 0.0, "", detection_method="ohlc")
    
    def _detect_volume_anomaly(self, candle_data: CandleData, window: Deque[CandleData]) -> AnomalyDetectionResult:
        """Detect volume anomalies"""
        if len(window) < 5:
            return AnomalyDetectionResult(False, 0.0, "", detection_method="volume")
        
        # Calculate recent volume statistics
        recent_volumes = [c.volume for c in list(window)[-10:] if c.volume > 0]
        if not recent_volumes:
            return AnomalyDetectionResult(False, 0.0, "", detection_method="volume")
        
        mean_volume = statistics.mean(recent_volumes)
        std_volume = statistics.stdev(recent_volumes) if len(recent_volumes) > 1 else 0
        
        # Check for volume spikes
        if std_volume > 0:
            z_score = abs(candle_data.volume - mean_volume) / std_volume
            if z_score > self.config.z_score_threshold:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=min(z_score / self.config.z_score_threshold, 10.0),
                    description=f"Volume anomaly: Z-score {z_score:.2f} (mean {mean_volume:.0f})",
                    detection_method="volume_spike",
                    confidence=min(z_score / self.config.z_score_threshold, 1.0)
                )
        
        # Check for extreme volume changes
        if mean_volume > 0:
            volume_ratio = candle_data.volume / mean_volume
            if volume_ratio > self.config.volume_spike_multiplier:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=min(volume_ratio / self.config.volume_spike_multiplier, 10.0),
                    description=f"Volume spike: {volume_ratio:.1f}x average volume",
                    detection_method="volume_ratio",
                    confidence=min(volume_ratio / self.config.volume_spike_multiplier, 1.0)
                )
            elif volume_ratio < self.config.volume_drop_multiplier:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=min(1.0 / volume_ratio, 10.0),
                    description=f"Volume drop: {volume_ratio:.1f}x average volume",
                    detection_method="volume_drop",
                    confidence=min(1.0 / volume_ratio, 1.0)
                )
        
        return AnomalyDetectionResult(False, 0.0, "", detection_method="volume")
    
    def _detect_range_anomaly(self, candle_data: CandleData, window: Deque[CandleData]) -> AnomalyDetectionResult:
        """Detect range anomalies in candle data"""
        if len(window) < 5:
            return AnomalyDetectionResult(False, 0.0, "", detection_method="range")
        
        # Calculate recent range statistics
        recent_ranges = [c.high - c.low for c in list(window)[-10:]]
        mean_range = statistics.mean(recent_ranges)
        std_range = statistics.stdev(recent_ranges) if len(recent_ranges) > 1 else 0
        
        current_range = candle_data.high - candle_data.low
        
        if std_range > 0:
            z_score = abs(current_range - mean_range) / std_range
            if z_score > self.config.z_score_threshold:
                return AnomalyDetectionResult(
                    is_anomaly=True,
                    severity=min(z_score / self.config.z_score_threshold, 10.0),
                    description=f"Range anomaly: Z-score {z_score:.2f} (mean {mean_range:.6f})",
                    detection_method="range",
                    confidence=min(z_score / self.config.z_score_threshold, 1.0)
                )
        
        return AnomalyDetectionResult(False, 0.0, "", detection_method="range")
    
    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        return {
            'anomaly_counts': self.anomaly_counts,
            'total_anomalies': sum(self.anomaly_counts.values()),
            'instruments_tracked': len(self.price_windows),
            'config': {
                'z_score_threshold': self.config.z_score_threshold,
                'mad_threshold': self.config.mad_threshold,
                'max_price_jump_pct': self.config.max_price_jump_pct,
                'volume_spike_multiplier': self.config.volume_spike_multiplier
            }
        }

# Example usage and testing
async def test_anomaly_detector():
    """Test the anomaly detector"""
    from ..brokers.broker_manager import PriceData, CandleData
    
    config = AnomalyDetectionConfig(
        z_score_threshold=2.0,
        max_price_jump_pct=2.0
    )
    
    detector = AnomalyDetector(config)
    
    # Test normal price data
    price_data = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=1.1000,
        ask=1.1002,
        spread=0.0002
    )
    
    result = await detector.detect_price_anomaly(price_data)
    print(f"Normal price data: Anomaly={result.is_anomaly}, Severity={result.severity}")
    
    # Test anomalous price data (extreme jump)
    anomalous_price = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=2.0000,  # Extreme jump
        ask=2.0002,
        spread=0.0002
    )
    
    result = await detector.detect_price_anomaly(anomalous_price)
    print(f"Anomalous price data: Anomaly={result.is_anomaly}, Severity={result.severity}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_anomaly_detector())

