#!/usr/bin/env python3
"""
Data Corrector
Implements automatic data correction mechanisms
"""

import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math

from ..brokers.broker_manager import PriceData, CandleData

logger = logging.getLogger(__name__)

@dataclass
class CorrectionResult:
    """Result of data correction"""
    is_corrected: bool
    original_data: Union[PriceData, CandleData]
    corrected_data: Union[PriceData, CandleData, None]
    correction_method: str
    description: str
    confidence: float = 0.0
    correction_time_ms: float = 0.0

@dataclass
class DataCorrectorConfig:
    """Configuration for data correction"""
    # Correction limits
    max_corrections_per_data_point: int = 3
    correction_timeout_ms: int = 100
    
    # Price correction
    enable_price_correction: bool = True
    max_price_correction_pct: float = 10.0
    price_smoothing_window: int = 5
    
    # Spread correction
    enable_spread_correction: bool = True
    max_spread_correction_pct: float = 50.0
    min_spread_pips: float = 0.1
    max_spread_pips: float = 100.0
    
    # OHLC correction
    enable_ohlc_correction: bool = True
    max_ohlc_correction_pct: float = 5.0
    
    # Volume correction
    enable_volume_correction: bool = True
    max_volume_correction_pct: float = 100.0
    
    # Time correction
    enable_time_correction: bool = True
    max_time_correction_seconds: int = 60
    
    # Interpolation methods
    interpolation_methods: List[str] = field(default_factory=lambda: ['linear', 'last_known', 'average'])
    
    # Fallback strategies
    enable_fallback_sources: bool = True
    fallback_source_priority: List[str] = field(default_factory=lambda: ['oanda', 'fxcm', 'free_forex'])

class DataCorrector:
    """Implements automatic data correction mechanisms"""
    
    def __init__(self, config: Optional[DataCorrectorConfig] = None):
        """Initialize data corrector"""
        self.config = config or DataCorrectorConfig()
        
        # Correction history for learning
        self.correction_history: List[CorrectionResult] = []
        self.correction_success_rate: Dict[str, float] = {}
        
        # Data history for interpolation
        self.price_history: Dict[str, List[PriceData]] = {}
        self.candle_history: Dict[str, List[CandleData]] = {}
        self.max_history_size = 1000
        
        # Correction statistics
        self.correction_stats = {
            'total_attempts': 0,
            'successful_corrections': 0,
            'failed_corrections': 0,
            'corrections_by_method': {},
            'corrections_by_type': {}
        }
        
        logger.info("DataCorrector initialized with config: %s", self.config)
    
    async def correct_price_data(self, price_data: PriceData, errors: List[str],
                               previous_data: Optional[PriceData] = None) -> CorrectionResult:
        """Correct price data based on validation errors"""
        import time
        start_time = time.time()
        
        try:
            self.correction_stats['total_attempts'] += 1
            
            # Determine correction strategy based on errors
            correction_strategy = self._determine_correction_strategy(errors)
            
            if not correction_strategy:
                return CorrectionResult(
                    is_corrected=False,
                    original_data=price_data,
                    corrected_data=None,
                    correction_method="none",
                    description="No applicable correction strategy found",
                    correction_time_ms=(time.time() - start_time) * 1000
                )
            
            # Apply corrections
            corrected_data = price_data
            correction_methods = []
            correction_descriptions = []
            
            for strategy in correction_strategy:
                if strategy == 'price_correction' and self.config.enable_price_correction:
                    corrected_data, method, desc = await self._correct_price_values(corrected_data, previous_data)
                    correction_methods.append(method)
                    correction_descriptions.append(desc)
                
                elif strategy == 'spread_correction' and self.config.enable_spread_correction:
                    corrected_data, method, desc = await self._correct_spread(corrected_data, previous_data)
                    correction_methods.append(method)
                    correction_descriptions.append(desc)
                
                elif strategy == 'time_correction' and self.config.enable_time_correction:
                    corrected_data, method, desc = await self._correct_timestamp(corrected_data)
                    correction_methods.append(method)
                    correction_descriptions.append(desc)
                
                elif strategy == 'interpolation' and previous_data:
                    corrected_data, method, desc = await self._interpolate_price_data(corrected_data, previous_data)
                    correction_methods.append(method)
                    correction_descriptions.append(desc)
            
            # Validate corrected data
            is_valid = self._validate_corrected_data(corrected_data)
            
            if is_valid:
                self.correction_stats['successful_corrections'] += 1
                self._update_correction_stats(correction_methods, 'price')
                
                result = CorrectionResult(
                    is_corrected=True,
                    original_data=price_data,
                    corrected_data=corrected_data,
                    correction_method="+".join(correction_methods),
                    description="; ".join(correction_descriptions),
                    confidence=self._calculate_correction_confidence(corrected_data, price_data),
                    correction_time_ms=(time.time() - start_time) * 1000
                )
                
                self.correction_history.append(result)
                return result
            else:
                self.correction_stats['failed_corrections'] += 1
                return CorrectionResult(
                    is_corrected=False,
                    original_data=price_data,
                    corrected_data=None,
                    correction_method="+".join(correction_methods),
                    description="Correction failed validation",
                    correction_time_ms=(time.time() - start_time) * 1000
                )
            
        except Exception as e:
            logger.error(f"Error correcting price data: {e}")
            self.correction_stats['failed_corrections'] += 1
            return CorrectionResult(
                is_corrected=False,
                original_data=price_data,
                corrected_data=None,
                correction_method="error",
                description=f"Correction error: {str(e)}",
                correction_time_ms=(time.time() - start_time) * 1000
            )
    
    async def correct_candle_data(self, candle_data: CandleData, errors: List[str],
                                previous_data: Optional[CandleData] = None) -> CorrectionResult:
        """Correct candle data based on validation errors"""
        import time
        start_time = time.time()
        
        try:
            self.correction_stats['total_attempts'] += 1
            
            # Determine correction strategy
            correction_strategy = self._determine_candle_correction_strategy(errors)
            
            if not correction_strategy:
                return CorrectionResult(
                    is_corrected=False,
                    original_data=candle_data,
                    corrected_data=None,
                    correction_method="none",
                    description="No applicable correction strategy found",
                    correction_time_ms=(time.time() - start_time) * 1000
                )
            
            # Apply corrections
            corrected_data = candle_data
            correction_methods = []
            correction_descriptions = []
            
            for strategy in correction_strategy:
                if strategy == 'ohlc_correction' and self.config.enable_ohlc_correction:
                    corrected_data, method, desc = await self._correct_ohlc(corrected_data, previous_data)
                    correction_methods.append(method)
                    correction_descriptions.append(desc)
                
                elif strategy == 'volume_correction' and self.config.enable_volume_correction:
                    corrected_data, method, desc = await self._correct_volume(corrected_data, previous_data)
                    correction_methods.append(method)
                    correction_descriptions.append(desc)
                
                elif strategy == 'time_correction' and self.config.enable_time_correction:
                    corrected_data, method, desc = await self._correct_timestamp(corrected_data)
                    correction_methods.append(method)
                    correction_descriptions.append(desc)
            
            # Validate corrected data
            is_valid = self._validate_corrected_candle_data(corrected_data)
            
            if is_valid:
                self.correction_stats['successful_corrections'] += 1
                self._update_correction_stats(correction_methods, 'candle')
                
                result = CorrectionResult(
                    is_corrected=True,
                    original_data=candle_data,
                    corrected_data=corrected_data,
                    correction_method="+".join(correction_methods),
                    description="; ".join(correction_descriptions),
                    confidence=self._calculate_correction_confidence(corrected_data, candle_data),
                    correction_time_ms=(time.time() - start_time) * 1000
                )
                
                self.correction_history.append(result)
                return result
            else:
                self.correction_stats['failed_corrections'] += 1
                return CorrectionResult(
                    is_corrected=False,
                    original_data=candle_data,
                    corrected_data=None,
                    correction_method="+".join(correction_methods),
                    description="Correction failed validation",
                    correction_time_ms=(time.time() - start_time) * 1000
                )
            
        except Exception as e:
            logger.error(f"Error correcting candle data: {e}")
            self.correction_stats['failed_corrections'] += 1
            return CorrectionResult(
                is_corrected=False,
                original_data=candle_data,
                corrected_data=None,
                correction_method="error",
                description=f"Correction error: {str(e)}",
                correction_time_ms=(time.time() - start_time) * 1000
            )
    
    def _determine_correction_strategy(self, errors: List[str]) -> List[str]:
        """Determine correction strategy based on errors"""
        strategy = []
        
        for error in errors:
            error_lower = error.lower()
            
            if any(keyword in error_lower for keyword in ['price', 'bid', 'ask', 'mid']):
                strategy.append('price_correction')
            elif 'spread' in error_lower:
                strategy.append('spread_correction')
            elif 'timestamp' in error_lower or 'time' in error_lower:
                strategy.append('time_correction')
            elif any(keyword in error_lower for keyword in ['range', 'jump', 'anomaly']):
                strategy.append('interpolation')
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(strategy))
    
    def _determine_candle_correction_strategy(self, errors: List[str]) -> List[str]:
        """Determine candle correction strategy based on errors"""
        strategy = []
        
        for error in errors:
            error_lower = error.lower()
            
            if any(keyword in error_lower for keyword in ['ohlc', 'open', 'high', 'low', 'close']):
                strategy.append('ohlc_correction')
            elif 'volume' in error_lower:
                strategy.append('volume_correction')
            elif 'timestamp' in error_lower or 'time' in error_lower:
                strategy.append('time_correction')
        
        return list(dict.fromkeys(strategy))
    
    async def _correct_price_values(self, price_data: PriceData, 
                                  previous_data: Optional[PriceData]) -> Tuple[PriceData, str, str]:
        """Correct price values using various methods"""
        if not previous_data:
            return price_data, "no_correction", "No previous data for correction"
        
        # Calculate expected price based on previous data
        prev_mid = (previous_data.bid + previous_data.ask) / 2
        current_mid = (price_data.bid + price_data.ask) / 2
        
        # Check if correction is within limits
        if prev_mid > 0:
            price_change_pct = abs(current_mid - prev_mid) / prev_mid * 100
            if price_change_pct > self.config.max_price_correction_pct:
                return price_data, "no_correction", f"Price change {price_change_pct:.2f}% exceeds correction limit"
        
        # Apply smoothing correction
        corrected_mid = self._apply_price_smoothing(price_data, previous_data)
        
        # Calculate corrected bid/ask maintaining spread
        spread = price_data.spread
        corrected_bid = corrected_mid - spread / 2
        corrected_ask = corrected_mid + spread / 2
        
        corrected_data = PriceData(
            instrument=price_data.instrument,
            time=price_data.time,
            bid=corrected_bid,
            ask=corrected_ask,
            spread=spread
        )
        
        return corrected_data, "price_smoothing", f"Applied price smoothing correction"
    
    async def _correct_spread(self, price_data: PriceData, 
                            previous_data: Optional[PriceData]) -> Tuple[PriceData, str, str]:
        """Correct spread values"""
        if not previous_data:
            return price_data, "no_correction", "No previous data for spread correction"
        
        # Calculate expected spread based on previous data
        expected_spread = previous_data.spread
        
        # Check if current spread is reasonable
        if expected_spread > 0:
            spread_change_pct = abs(price_data.spread - expected_spread) / expected_spread * 100
            if spread_change_pct > self.config.max_spread_correction_pct:
                return price_data, "no_correction", f"Spread change {spread_change_pct:.2f}% exceeds correction limit"
        
        # Apply spread correction
        corrected_spread = max(self.config.min_spread_pips / 10000, 
                             min(expected_spread, self.config.max_spread_pips / 10000))
        
        # Adjust bid/ask to maintain mid price
        mid_price = (price_data.bid + price_data.ask) / 2
        corrected_bid = mid_price - corrected_spread / 2
        corrected_ask = mid_price + corrected_spread / 2
        
        corrected_data = PriceData(
            instrument=price_data.instrument,
            time=price_data.time,
            bid=corrected_bid,
            ask=corrected_ask,
            spread=corrected_spread
        )
        
        return corrected_data, "spread_correction", f"Corrected spread from {price_data.spread:.6f} to {corrected_spread:.6f}"
    
    async def _correct_ohlc(self, candle_data: CandleData, 
                          previous_data: Optional[CandleData]) -> Tuple[CandleData, str, str]:
        """Correct OHLC values"""
        if not previous_data:
            return candle_data, "no_correction", "No previous data for OHLC correction"
        
        # Fix basic OHLC consistency
        corrected_open = candle_data.open
        corrected_high = candle_data.high
        corrected_low = candle_data.low
        corrected_close = candle_data.close
        
        # Ensure high >= max(open, close)
        corrected_high = max(corrected_high, corrected_open, corrected_close)
        
        # Ensure low <= min(open, close)
        corrected_low = min(corrected_low, corrected_open, corrected_close)
        
        # Apply smoothing if values are too different from previous
        if previous_data:
            prev_values = [previous_data.open, previous_data.high, previous_data.low, previous_data.close]
            current_values = [corrected_open, corrected_high, corrected_low, corrected_close]
            
            for i, (prev_val, curr_val) in enumerate(zip(prev_values, current_values)):
                if prev_val > 0:
                    change_pct = abs(curr_val - prev_val) / prev_val * 100
                    if change_pct > self.config.max_ohlc_correction_pct:
                        # Apply smoothing
                        smoothed_val = prev_val * 0.7 + curr_val * 0.3
                        if i == 0:
                            corrected_open = smoothed_val
                        elif i == 1:
                            corrected_high = smoothed_val
                        elif i == 2:
                            corrected_low = smoothed_val
                        else:
                            corrected_close = smoothed_val
        
        corrected_data = CandleData(
            instrument=candle_data.instrument,
            time=candle_data.time,
            open=corrected_open,
            high=corrected_high,
            low=corrected_low,
            close=corrected_close,
            volume=candle_data.volume
        )
        
        return corrected_data, "ohlc_correction", "Applied OHLC consistency and smoothing corrections"
    
    async def _correct_volume(self, candle_data: CandleData, 
                            previous_data: Optional[CandleData]) -> Tuple[CandleData, str, str]:
        """Correct volume values"""
        if not previous_data or previous_data.volume <= 0:
            return candle_data, "no_correction", "No valid previous volume data"
        
        # Check if volume change is reasonable
        volume_change_pct = abs(candle_data.volume - previous_data.volume) / previous_data.volume * 100
        if volume_change_pct > self.config.max_volume_correction_pct:
            return candle_data, "no_correction", f"Volume change {volume_change_pct:.2f}% exceeds correction limit"
        
        # Apply volume smoothing
        corrected_volume = previous_data.volume * 0.8 + candle_data.volume * 0.2
        
        corrected_data = CandleData(
            instrument=candle_data.instrument,
            time=candle_data.time,
            open=candle_data.open,
            high=candle_data.high,
            low=candle_data.low,
            close=candle_data.close,
            volume=corrected_volume
        )
        
        return corrected_data, "volume_correction", f"Corrected volume from {candle_data.volume} to {corrected_volume:.0f}"
    
    async def _correct_timestamp(self, data: Union[PriceData, CandleData]) -> Tuple[Union[PriceData, CandleData], str, str]:
        """Correct timestamp values"""
        now = datetime.utcnow()
        
        # If timestamp is too far in the future, correct it
        if data.time > now + timedelta(seconds=self.config.max_time_correction_seconds):
            corrected_time = now
            description = f"Corrected future timestamp from {data.time} to {corrected_time}"
        else:
            return data, "no_correction", "Timestamp is within acceptable range"
        
        # Create corrected data
        if isinstance(data, PriceData):
            corrected_data = PriceData(
                instrument=data.instrument,
                time=corrected_time,
                bid=data.bid,
                ask=data.ask,
                spread=data.spread
            )
        else:
            corrected_data = CandleData(
                instrument=data.instrument,
                time=corrected_time,
                open=data.open,
                high=data.high,
                low=data.low,
                close=data.close,
                volume=data.volume
            )
        
        return corrected_data, "timestamp_correction", description
    
    async def _interpolate_price_data(self, price_data: PriceData, 
                                    previous_data: PriceData) -> Tuple[PriceData, str, str]:
        """Interpolate price data using historical context"""
        # Simple linear interpolation based on previous data
        # This is a placeholder for more sophisticated interpolation methods
        
        # Use previous data as base with small random variation
        variation = 0.0001  # 1 pip variation
        corrected_bid = previous_data.bid + variation
        corrected_ask = previous_data.ask + variation
        corrected_spread = corrected_ask - corrected_bid
        
        corrected_data = PriceData(
            instrument=price_data.instrument,
            time=price_data.time,
            bid=corrected_bid,
            ask=corrected_ask,
            spread=corrected_spread
        )
        
        return corrected_data, "interpolation", "Applied linear interpolation correction"
    
    def _apply_price_smoothing(self, price_data: PriceData, previous_data: PriceData) -> float:
        """Apply price smoothing using moving average"""
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        prev_mid = (previous_data.bid + previous_data.ask) / 2
        current_mid = (price_data.bid + price_data.ask) / 2
        
        smoothed_mid = alpha * current_mid + (1 - alpha) * prev_mid
        return smoothed_mid
    
    def _validate_corrected_data(self, data: Union[PriceData, CandleData]) -> bool:
        """Validate corrected data"""
        if isinstance(data, PriceData):
            return (data.bid > 0 and data.ask > 0 and 
                   data.spread >= 0 and data.bid < data.ask)
        else:
            return (data.open > 0 and data.high > 0 and data.low > 0 and data.close > 0 and
                   data.volume >= 0 and data.high >= data.low and
                   data.high >= max(data.open, data.close) and
                   data.low <= min(data.open, data.close))
    
    def _validate_corrected_candle_data(self, data: CandleData) -> bool:
        """Validate corrected candle data"""
        return self._validate_corrected_data(data)
    
    def _calculate_correction_confidence(self, corrected_data: Union[PriceData, CandleData], 
                                       original_data: Union[PriceData, CandleData]) -> float:
        """Calculate confidence in correction"""
        # Simple confidence based on how much the data changed
        if isinstance(corrected_data, PriceData) and isinstance(original_data, PriceData):
            original_mid = (original_data.bid + original_data.ask) / 2
            corrected_mid = (corrected_data.bid + corrected_data.ask) / 2
            
            if original_mid > 0:
                change_pct = abs(corrected_mid - original_mid) / original_mid * 100
                # Lower change = higher confidence
                confidence = max(0.0, 1.0 - change_pct / 10.0)
            else:
                confidence = 0.5
        else:
            confidence = 0.8  # Default confidence for candle data
        
        return min(1.0, max(0.0, confidence))
    
    def _update_correction_stats(self, methods: List[str], data_type: str):
        """Update correction statistics"""
        for method in methods:
            if method not in self.correction_stats['corrections_by_method']:
                self.correction_stats['corrections_by_method'][method] = 0
            self.correction_stats['corrections_by_method'][method] += 1
        
        if data_type not in self.correction_stats['corrections_by_type']:
            self.correction_stats['corrections_by_type'][data_type] = 0
        self.correction_stats['corrections_by_type'][data_type] += 1
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Get correction statistics"""
        total_attempts = self.correction_stats['total_attempts']
        success_rate = (self.correction_stats['successful_corrections'] / total_attempts) if total_attempts > 0 else 0.0
        
        return {
            'correction_stats': self.correction_stats,
            'success_rate': success_rate,
            'total_corrections': len(self.correction_history),
            'config': {
                'max_corrections_per_data_point': self.config.max_corrections_per_data_point,
                'max_price_correction_pct': self.config.max_price_correction_pct,
                'max_spread_correction_pct': self.config.max_spread_correction_pct,
                'enable_price_correction': self.config.enable_price_correction,
                'enable_spread_correction': self.config.enable_spread_correction
            }
        }

# Example usage and testing
async def test_data_corrector():
    """Test the data corrector"""
    from ..brokers.broker_manager import PriceData
    
    config = DataCorrectorConfig(
        max_price_correction_pct=5.0,
        enable_price_correction=True
    )
    
    corrector = DataCorrector(config)
    
    # Test price correction
    price_data = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=2.0000,  # Extreme price
        ask=2.0002,
        spread=0.0002
    )
    
    previous_data = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow() - timedelta(seconds=1),
        bid=1.1000,
        ask=1.1002,
        spread=0.0002
    )
    
    errors = ["Price jump anomaly: 81.82% change"]
    result = await corrector.correct_price_data(price_data, errors, previous_data)
    print(f"Correction result: {result.is_corrected}, Method: {result.correction_method}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_data_corrector())

