"""
Breakout trading strategy template
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import logging

from ..core.strategy_template import StrategyTemplate, StrategyType, Signal, StrategyParameters
from ..core.parameter_space import ParameterSpace, ParameterType, ParameterDefinition


class BreakoutStrategy(StrategyTemplate):
    """
    Breakout trading strategy using volume and price action
    """
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self.parameters.name
        
        # Strategy-specific attributes
        self.breakout_period = 20
        self.volume_threshold = 1.5  # 1.5x average volume
        self.price_threshold = 0.005  # 0.5% price breakout
        self.confirmation_periods = 2
        self.atr_period = 14
        self.atr_multiplier = 2.0
        
    def initialize(self) -> bool:
        """Initialize breakout strategy with parameters"""
        try:
            params = self.parameters.parameters
            
            # Extract parameters
            self.breakout_period = params.get('breakout_period', 20)
            self.volume_threshold = params.get('volume_threshold', 1.5)
            self.price_threshold = params.get('price_threshold', 0.005)
            self.confirmation_periods = params.get('confirmation_periods', 2)
            self.atr_period = params.get('atr_period', 14)
            self.atr_multiplier = params.get('atr_multiplier', 2.0)
            
            # Validate parameters
            is_valid, errors = self.validate_parameters()
            if not is_valid:
                self.logger.error(f"Invalid parameters: {errors}")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Breakout strategy initialized: {self}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing breakout strategy: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate breakout signals"""
        if not self.is_initialized:
            self.logger.warning("Strategy not initialized")
            return []
        
        try:
            signals = []
            
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv', {})
            if not ohlcv_data:
                return signals
            
            # Get price and volume data
            prices = ohlcv_data.get('close', [])
            highs = ohlcv_data.get('high', [])
            lows = ohlcv_data.get('low', [])
            volumes = ohlcv_data.get('volume', [])
            timestamps = ohlcv_data.get('timestamp', [])
            
            if len(prices) < max(self.breakout_period, self.atr_period):
                return signals
            
            # Calculate indicators
            atr = self._calculate_atr(highs, lows, prices, self.atr_period)
            avg_volume = self._calculate_average_volume(volumes, self.breakout_period)
            
            # Generate signals for each time period
            for i in range(max(self.breakout_period, self.atr_period), len(prices)):
                current_price = prices[i]
                current_high = highs[i]
                current_low = lows[i]
                current_volume = volumes[i] if i < len(volumes) else 0
                current_atr = atr[i]
                current_avg_volume = avg_volume[i]
                current_timestamp = timestamps[i] if i < len(timestamps) else datetime.now()
                
                # Check for breakout conditions
                signal = self._check_breakout(
                    timestamp=current_timestamp,
                    price=current_price,
                    high=current_high,
                    low=current_low,
                    volume=current_volume,
                    avg_volume=current_avg_volume,
                    atr=current_atr,
                    recent_highs=highs[i-self.breakout_period:i],
                    recent_lows=lows[i-self.breakout_period:i]
                )
                
                if signal:
                    signals.append(signal)
            
            self.logger.info(f"Generated {len(signals)} breakout signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating breakout signals: {e}")
            return []
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate breakout strategy parameters"""
        errors = []
        
        # Validate breakout period
        if self.breakout_period < 10 or self.breakout_period > 50:
            errors.append("Breakout period must be between 10 and 50")
        
        # Validate volume threshold
        if self.volume_threshold < 1.0 or self.volume_threshold > 5.0:
            errors.append("Volume threshold must be between 1.0 and 5.0")
        
        # Validate price threshold
        if not (0.001 <= self.price_threshold <= 0.02):
            errors.append("Price threshold must be between 0.001 and 0.02")
        
        # Validate confirmation periods
        if self.confirmation_periods < 1 or self.confirmation_periods > 5:
            errors.append("Confirmation periods must be between 1 and 5")
        
        return len(errors) == 0, errors
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization"""
        param_space = ParameterSpace()
        
        # Breakout period
        param_space.add_parameter(ParameterDefinition(
            name="breakout_period",
            param_type=ParameterType.INTEGER,
            min_value=10,
            max_value=30,
            default_value=20,
            description="Period for identifying breakout levels"
        ))
        
        # Volume threshold
        param_space.add_parameter(ParameterDefinition(
            name="volume_threshold",
            param_type=ParameterType.FLOAT,
            min_value=1.2,
            max_value=3.0,
            default_value=1.5,
            description="Volume multiplier threshold for breakout confirmation"
        ))
        
        # Price threshold
        param_space.add_parameter(ParameterDefinition(
            name="price_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.002,
            max_value=0.01,
            default_value=0.005,
            description="Price breakout threshold as percentage"
        ))
        
        # Confirmation periods
        param_space.add_parameter(ParameterDefinition(
            name="confirmation_periods",
            param_type=ParameterType.INTEGER,
            min_value=1,
            max_value=3,
            default_value=2,
            description="Number of periods for breakout confirmation"
        ))
        
        # ATR period
        param_space.add_parameter(ParameterDefinition(
            name="atr_period",
            param_type=ParameterType.INTEGER,
            min_value=10,
            max_value=20,
            default_value=14,
            description="ATR calculation period"
        ))
        
        # ATR multiplier
        param_space.add_parameter(ParameterDefinition(
            name="atr_multiplier",
            param_type=ParameterType.FLOAT,
            min_value=1.0,
            max_value=4.0,
            default_value=2.0,
            description="ATR multiplier for stop loss"
        ))
        
        return param_space
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """Update strategy parameters"""
        try:
            for key, value in new_parameters.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            self.parameters.parameters.update(new_parameters)
            self.parameters.updated_at = datetime.now()
            return self.initialize()
            
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")
            return False
    
    def _calculate_atr(self, highs: List[float], lows: List[float], 
                      closes: List[float], period: int) -> List[float]:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return [0.0] * len(highs)
        
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        atr = []
        for i in range(len(highs)):
            if i == 0:
                atr.append(0.0)
            elif i < period:
                atr.append(0.0)
            else:
                atr.append(np.mean(true_ranges[i-period:i]))
        
        return atr
    
    def _calculate_average_volume(self, volumes: List[float], period: int) -> List[float]:
        """Calculate average volume over period"""
        if len(volumes) < period:
            return [0.0] * len(volumes)
        
        avg_volumes = []
        for i in range(len(volumes)):
            if i < period - 1:
                avg_volumes.append(0.0)
            else:
                avg_volumes.append(np.mean(volumes[i-period+1:i+1]))
        
        return avg_volumes
    
    def _check_breakout(self, timestamp: datetime, price: float, high: float, low: float,
                       volume: float, avg_volume: float, atr: float,
                       recent_highs: List[float], recent_lows: List[float]) -> Signal:
        """Check for breakout conditions and create signal"""
        
        if not recent_highs or not recent_lows or avg_volume == 0:
            return None
        
        # Calculate breakout levels
        resistance_level = max(recent_highs)
        support_level = min(recent_lows)
        
        # Check volume condition
        volume_condition = volume >= (avg_volume * self.volume_threshold)
        
        # Check for upward breakout
        if high > resistance_level * (1 + self.price_threshold) and volume_condition:
            strength = min(1.0, (high - resistance_level) / resistance_level / self.price_threshold)
            confidence = min(1.0, volume / avg_volume / self.volume_threshold)
            
            return Signal(
                timestamp=timestamp,
                symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
                signal_type="buy",
                strength=strength,
                confidence=confidence,
                price=price,
                stop_loss=resistance_level * 0.98,  # Below breakout level
                take_profit=price + (atr * self.atr_multiplier * 2),
                metadata={
                    "strategy_type": "breakout",
                    "breakout_level": resistance_level,
                    "volume_ratio": volume / avg_volume,
                    "atr": atr
                }
            )
        
        # Check for downward breakout
        elif low < support_level * (1 - self.price_threshold) and volume_condition:
            strength = min(1.0, (support_level - low) / support_level / self.price_threshold)
            confidence = min(1.0, volume / avg_volume / self.volume_threshold)
            
            return Signal(
                timestamp=timestamp,
                symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
                signal_type="sell",
                strength=strength,
                confidence=confidence,
                price=price,
                stop_loss=support_level * 1.02,  # Above breakout level
                take_profit=price - (atr * self.atr_multiplier * 2),
                metadata={
                    "strategy_type": "breakout",
                    "breakout_level": support_level,
                    "volume_ratio": volume / avg_volume,
                    "atr": atr
                }
            )
        
        return None
