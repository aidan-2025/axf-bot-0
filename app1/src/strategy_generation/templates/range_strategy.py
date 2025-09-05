"""
Range trading strategy template
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import logging

from ..core.strategy_template import StrategyTemplate, StrategyType, Signal, StrategyParameters
from ..core.parameter_space import ParameterSpace, ParameterType, ParameterDefinition


class RangeStrategy(StrategyTemplate):
    """
    Range trading strategy using support/resistance levels and oscillators
    """
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self.parameters.name
        
        # Strategy-specific attributes
        self.lookback_period = 20
        self.support_threshold = 0.02  # 2% below recent low
        self.resistance_threshold = 0.02  # 2% above recent high
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        self.min_range_size = 0.005  # Minimum 0.5% range size
        
    def initialize(self) -> bool:
        """Initialize range strategy with parameters"""
        try:
            params = self.parameters.parameters
            
            # Extract parameters
            self.lookback_period = params.get('lookback_period', 20)
            self.support_threshold = params.get('support_threshold', 0.02)
            self.resistance_threshold = params.get('resistance_threshold', 0.02)
            self.rsi_period = params.get('rsi_period', 14)
            self.rsi_oversold = params.get('rsi_oversold', 30)
            self.rsi_overbought = params.get('rsi_overbought', 70)
            self.bollinger_period = params.get('bollinger_period', 20)
            self.bollinger_std = params.get('bollinger_std', 2.0)
            self.min_range_size = params.get('min_range_size', 0.005)
            
            # Validate parameters
            is_valid, errors = self.validate_parameters()
            if not is_valid:
                self.logger.error(f"Invalid parameters: {errors}")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Range strategy initialized: {self}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing range strategy: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate range trading signals"""
        if not self.is_initialized:
            self.logger.warning("Strategy not initialized")
            return []
        
        try:
            signals = []
            
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv', {})
            if not ohlcv_data:
                return signals
            
            # Get price data
            prices = ohlcv_data.get('close', [])
            highs = ohlcv_data.get('high', [])
            lows = ohlcv_data.get('low', [])
            timestamps = ohlcv_data.get('timestamp', [])
            
            if len(prices) < max(self.lookback_period, self.rsi_period, self.bollinger_period):
                return signals
            
            # Calculate indicators
            rsi = self._calculate_rsi(prices, self.rsi_period)
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(
                prices, self.bollinger_period, self.bollinger_std
            )
            
            # Generate signals for each time period
            for i in range(max(self.lookback_period, self.rsi_period, self.bollinger_period), len(prices)):
                current_price = prices[i]
                current_rsi = rsi[i]
                current_bb_upper = bollinger_upper[i]
                current_bb_lower = bollinger_lower[i]
                current_timestamp = timestamps[i] if i < len(timestamps) else datetime.now()
                
                # Identify support and resistance levels
                support_level, resistance_level = self._identify_range_levels(
                    highs[i-self.lookback_period:i+1],
                    lows[i-self.lookback_period:i+1]
                )
                
                # Check if we're in a valid range
                range_size = (resistance_level - support_level) / support_level
                if range_size < self.min_range_size:
                    continue
                
                # Generate signal based on range position and oscillators
                signal = self._create_range_signal(
                    timestamp=current_timestamp,
                    price=current_price,
                    support_level=support_level,
                    resistance_level=resistance_level,
                    rsi=current_rsi,
                    bb_upper=current_bb_upper,
                    bb_lower=current_bb_lower,
                    range_size=range_size
                )
                
                if signal:
                    signals.append(signal)
            
            self.logger.info(f"Generated {len(signals)} range signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating range signals: {e}")
            return []
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate range strategy parameters"""
        errors = []
        
        # Validate lookback period
        if self.lookback_period < 10 or self.lookback_period > 100:
            errors.append("Lookback period must be between 10 and 100")
        
        # Validate threshold values
        if not (0.001 <= self.support_threshold <= 0.1):
            errors.append("Support threshold must be between 0.001 and 0.1")
        
        if not (0.001 <= self.resistance_threshold <= 0.1):
            errors.append("Resistance threshold must be between 0.001 and 0.1")
        
        # Validate RSI parameters
        if self.rsi_oversold >= self.rsi_overbought:
            errors.append("RSI oversold must be less than overbought")
        
        # Validate Bollinger Bands parameters
        if self.bollinger_period < 5 or self.bollinger_period > 50:
            errors.append("Bollinger period must be between 5 and 50")
        
        if self.bollinger_std < 1.0 or self.bollinger_std > 4.0:
            errors.append("Bollinger standard deviation must be between 1.0 and 4.0")
        
        # Validate minimum range size
        if not (0.001 <= self.min_range_size <= 0.05):
            errors.append("Minimum range size must be between 0.001 and 0.05")
        
        return len(errors) == 0, errors
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization"""
        param_space = ParameterSpace()
        
        # Lookback period
        param_space.add_parameter(ParameterDefinition(
            name="lookback_period",
            param_type=ParameterType.INTEGER,
            min_value=10,
            max_value=50,
            default_value=20,
            description="Period for identifying support/resistance levels"
        ))
        
        # Support threshold
        param_space.add_parameter(ParameterDefinition(
            name="support_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.005,
            max_value=0.05,
            default_value=0.02,
            description="Support level threshold as percentage"
        ))
        
        # Resistance threshold
        param_space.add_parameter(ParameterDefinition(
            name="resistance_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.005,
            max_value=0.05,
            default_value=0.02,
            description="Resistance level threshold as percentage"
        ))
        
        # RSI period
        param_space.add_parameter(ParameterDefinition(
            name="rsi_period",
            param_type=ParameterType.INTEGER,
            min_value=5,
            max_value=30,
            default_value=14,
            description="RSI calculation period"
        ))
        
        # RSI oversold level
        param_space.add_parameter(ParameterDefinition(
            name="rsi_oversold",
            param_type=ParameterType.INTEGER,
            min_value=20,
            max_value=40,
            default_value=30,
            description="RSI oversold threshold"
        ))
        
        # RSI overbought level
        param_space.add_parameter(ParameterDefinition(
            name="rsi_overbought",
            param_type=ParameterType.INTEGER,
            min_value=60,
            max_value=80,
            default_value=70,
            description="RSI overbought threshold"
        ))
        
        # Bollinger Bands period
        param_space.add_parameter(ParameterDefinition(
            name="bollinger_period",
            param_type=ParameterType.INTEGER,
            min_value=10,
            max_value=30,
            default_value=20,
            description="Bollinger Bands calculation period"
        ))
        
        # Bollinger Bands standard deviation
        param_space.add_parameter(ParameterDefinition(
            name="bollinger_std",
            param_type=ParameterType.FLOAT,
            min_value=1.5,
            max_value=3.0,
            default_value=2.0,
            description="Bollinger Bands standard deviation multiplier"
        ))
        
        # Minimum range size
        param_space.add_parameter(ParameterDefinition(
            name="min_range_size",
            param_type=ParameterType.FLOAT,
            min_value=0.002,
            max_value=0.02,
            default_value=0.005,
            description="Minimum range size as percentage"
        ))
        
        # Add constraints
        param_space.add_constraint("rsi_oversold < rsi_overbought")
        param_space.add_constraint("support_threshold > 0")
        param_space.add_constraint("resistance_threshold > 0")
        
        return param_space
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """Update strategy parameters"""
        try:
            # Update internal parameters
            for key, value in new_parameters.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Update parameters object
            self.parameters.parameters.update(new_parameters)
            self.parameters.updated_at = datetime.now()
            
            # Re-initialize with new parameters
            return self.initialize()
            
        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")
            return False
    
    def _calculate_rsi(self, prices: List[float], period: int) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = []
        avg_losses = []
        
        for i in range(len(deltas)):
            if i < period - 1:
                avg_gains.append(0.0)
                avg_losses.append(0.0)
            else:
                avg_gains.append(np.mean(gains[i-period+1:i+1]))
                avg_losses.append(np.mean(losses[i-period+1:i+1]))
        
        rsi = []
        for i in range(len(prices)):
            if i == 0:
                rsi.append(50.0)
            elif i <= len(avg_gains):
                avg_gain = avg_gains[i-1]
                avg_loss = avg_losses[i-1]
                
                if avg_loss == 0:
                    rsi.append(100.0)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))
            else:
                rsi.append(50.0)
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int, std_mult: float) -> Tuple[List[float], List[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return [0.0] * len(prices), [0.0] * len(prices)
        
        sma = []
        upper_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(0.0)
                upper_band.append(0.0)
                lower_band.append(0.0)
            else:
                period_prices = prices[i-period+1:i+1]
                mean = np.mean(period_prices)
                std = np.std(period_prices)
                
                sma.append(mean)
                upper_band.append(mean + (std * std_mult))
                lower_band.append(mean - (std * std_mult))
        
        return upper_band, lower_band
    
    def _identify_range_levels(self, highs: List[float], lows: List[float]) -> Tuple[float, float]:
        """Identify support and resistance levels"""
        if not highs or not lows:
            return 0.0, 0.0
        
        # Find recent high and low
        recent_high = max(highs)
        recent_low = min(lows)
        
        # Calculate support and resistance levels
        support_level = recent_low * (1 - self.support_threshold)
        resistance_level = recent_high * (1 + self.resistance_threshold)
        
        return support_level, resistance_level
    
    def _create_range_signal(self, timestamp: datetime, price: float, support_level: float,
                           resistance_level: float, rsi: float, bb_upper: float, 
                           bb_lower: float, range_size: float) -> Signal:
        """Create a range trading signal"""
        
        # Check if price is near support (buy signal)
        if price <= support_level * (1 + self.support_threshold) and rsi < self.rsi_oversold:
            strength = (self.rsi_oversold - rsi) / self.rsi_oversold
            confidence = min(1.0, range_size / self.min_range_size)
            
            return Signal(
                timestamp=timestamp,
                symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
                signal_type="buy",
                strength=strength,
                confidence=confidence,
                price=price,
                stop_loss=support_level * 0.95,  # 5% below support
                take_profit=resistance_level * 0.98,  # Near resistance
                metadata={
                    "strategy_type": "range",
                    "support_level": support_level,
                    "resistance_level": resistance_level,
                    "rsi": rsi,
                    "range_size": range_size
                }
            )
        
        # Check if price is near resistance (sell signal)
        elif price >= resistance_level * (1 - self.resistance_threshold) and rsi > self.rsi_overbought:
            strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            confidence = min(1.0, range_size / self.min_range_size)
            
            return Signal(
                timestamp=timestamp,
                symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
                signal_type="sell",
                strength=strength,
                confidence=confidence,
                price=price,
                stop_loss=resistance_level * 1.05,  # 5% above resistance
                take_profit=support_level * 1.02,  # Near support
                metadata={
                    "strategy_type": "range",
                    "support_level": support_level,
                    "resistance_level": resistance_level,
                    "rsi": rsi,
                    "range_size": range_size
                }
            )
        
        return None
