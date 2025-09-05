"""
Multi-timeframe trading strategy template
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import logging

from ..core.strategy_template import StrategyTemplate, StrategyType, Signal, StrategyParameters
from ..core.parameter_space import ParameterSpace, ParameterType, ParameterDefinition


class MultiTimeframeStrategy(StrategyTemplate):
    """
    Multi-timeframe trading strategy using higher timeframe trend and lower timeframe entries
    """
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self.parameters.name
        
        # Strategy-specific attributes
        self.primary_timeframe = "H4"
        self.entry_timeframe = "H1"
        self.trend_period = 50
        self.entry_period = 20
        self.trend_threshold = 0.6
        self.entry_threshold = 0.7
        
    def initialize(self) -> bool:
        """Initialize multi-timeframe strategy with parameters"""
        try:
            params = self.parameters.parameters
            
            # Extract parameters
            self.primary_timeframe = params.get('primary_timeframe', 'H4')
            self.entry_timeframe = params.get('entry_timeframe', 'H1')
            self.trend_period = params.get('trend_period', 50)
            self.entry_period = params.get('entry_period', 20)
            self.trend_threshold = params.get('trend_threshold', 0.6)
            self.entry_threshold = params.get('entry_threshold', 0.7)
            
            # Validate parameters
            is_valid, errors = self.validate_parameters()
            if not is_valid:
                self.logger.error(f"Invalid parameters: {errors}")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Multi-timeframe strategy initialized: {self}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing multi-timeframe strategy: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate multi-timeframe signals"""
        if not self.is_initialized:
            self.logger.warning("Strategy not initialized")
            return []
        
        try:
            signals = []
            
            # Extract multi-timeframe data
            primary_data = market_data.get('timeframes', {}).get(self.primary_timeframe, {})
            entry_data = market_data.get('timeframes', {}).get(self.entry_timeframe, {})
            
            if not primary_data or not entry_data:
                return signals
            
            # Determine primary trend
            primary_trend = self._determine_primary_trend(primary_data)
            
            if primary_trend == "neutral":
                return signals
            
            # Generate entry signals on lower timeframe
            entry_signals = self._generate_entry_signals(entry_data, primary_trend)
            signals.extend(entry_signals)
            
            self.logger.info(f"Generated {len(signals)} multi-timeframe signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating multi-timeframe signals: {e}")
            return []
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate multi-timeframe strategy parameters"""
        errors = []
        
        # Validate timeframes
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
        if self.primary_timeframe not in valid_timeframes:
            errors.append(f"Primary timeframe must be one of {valid_timeframes}")
        
        if self.entry_timeframe not in valid_timeframes:
            errors.append(f"Entry timeframe must be one of {valid_timeframes}")
        
        # Validate periods
        if self.trend_period < 20 or self.trend_period > 200:
            errors.append("Trend period must be between 20 and 200")
        
        if self.entry_period < 10 or self.entry_period > 100:
            errors.append("Entry period must be between 10 and 100")
        
        # Validate thresholds
        if not (0.1 <= self.trend_threshold <= 1.0):
            errors.append("Trend threshold must be between 0.1 and 1.0")
        
        if not (0.1 <= self.entry_threshold <= 1.0):
            errors.append("Entry threshold must be between 0.1 and 1.0")
        
        return len(errors) == 0, errors
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization"""
        param_space = ParameterSpace()
        
        # Primary timeframe
        param_space.add_parameter(ParameterDefinition(
            name="primary_timeframe",
            param_type=ParameterType.CATEGORICAL,
            categories=['H4', 'D1', 'W1'],
            default_value='H4',
            description="Primary timeframe for trend analysis"
        ))
        
        # Entry timeframe
        param_space.add_parameter(ParameterDefinition(
            name="entry_timeframe",
            param_type=ParameterType.CATEGORICAL,
            categories=['M15', 'M30', 'H1'],
            default_value='H1',
            description="Entry timeframe for signal generation"
        ))
        
        # Trend period
        param_space.add_parameter(ParameterDefinition(
            name="trend_period",
            param_type=ParameterType.INTEGER,
            min_value=30,
            max_value=100,
            default_value=50,
            description="Period for primary trend calculation"
        ))
        
        # Entry period
        param_space.add_parameter(ParameterDefinition(
            name="entry_period",
            param_type=ParameterType.INTEGER,
            min_value=10,
            max_value=50,
            default_value=20,
            description="Period for entry signal calculation"
        ))
        
        # Trend threshold
        param_space.add_parameter(ParameterDefinition(
            name="trend_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.4,
            max_value=0.8,
            default_value=0.6,
            description="Minimum trend strength threshold"
        ))
        
        # Entry threshold
        param_space.add_parameter(ParameterDefinition(
            name="entry_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.5,
            max_value=0.9,
            default_value=0.7,
            description="Minimum entry signal strength threshold"
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
    
    def _determine_primary_trend(self, primary_data: Dict[str, Any]) -> str:
        """Determine primary trend direction"""
        prices = primary_data.get('close', [])
        if len(prices) < self.trend_period:
            return "neutral"
        
        # Calculate trend strength
        recent_prices = prices[-self.trend_period:]
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        # Normalize trend strength
        price_range = max(recent_prices) - min(recent_prices)
        trend_strength = abs(trend_slope) / (price_range / len(recent_prices)) if price_range > 0 else 0
        
        if trend_strength < self.trend_threshold:
            return "neutral"
        
        return "up" if trend_slope > 0 else "down"
    
    def _generate_entry_signals(self, entry_data: Dict[str, Any], primary_trend: str) -> List[Signal]:
        """Generate entry signals on lower timeframe"""
        signals = []
        
        prices = entry_data.get('close', [])
        highs = entry_data.get('high', [])
        lows = entry_data.get('low', [])
        timestamps = entry_data.get('timestamp', [])
        
        if len(prices) < self.entry_period:
            return signals
        
        # Calculate entry indicators
        rsi = self._calculate_rsi(prices, 14)
        sma = self._calculate_sma(prices, self.entry_period)
        
        for i in range(self.entry_period, len(prices)):
            current_price = prices[i]
            current_rsi = rsi[i]
            current_sma = sma[i]
            current_timestamp = timestamps[i] if i < len(timestamps) else datetime.now()
            
            # Generate signal based on primary trend and entry conditions
            signal = self._create_entry_signal(
                timestamp=current_timestamp,
                price=current_price,
                primary_trend=primary_trend,
                rsi=current_rsi,
                sma=current_sma
            )
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _create_entry_signal(self, timestamp: datetime, price: float, primary_trend: str,
                           rsi: float, sma: float) -> Signal:
        """Create entry signal based on primary trend and entry conditions"""
        
        if primary_trend == "up" and price > sma and rsi < 70:
            strength = min(1.0, (price - sma) / sma * 10)  # Normalize strength
            confidence = min(1.0, (70 - rsi) / 70)
            
            return Signal(
                timestamp=timestamp,
                symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
                signal_type="buy",
                strength=strength,
                confidence=confidence,
                price=price,
                metadata={
                    "strategy_type": "multi_timeframe",
                    "primary_trend": primary_trend,
                    "primary_timeframe": self.primary_timeframe,
                    "entry_timeframe": self.entry_timeframe,
                    "rsi": rsi,
                    "sma": sma
                }
            )
        
        elif primary_trend == "down" and price < sma and rsi > 30:
            strength = min(1.0, (sma - price) / sma * 10)  # Normalize strength
            confidence = min(1.0, (rsi - 30) / 70)
            
            return Signal(
                timestamp=timestamp,
                symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
                signal_type="sell",
                strength=strength,
                confidence=confidence,
                price=price,
                metadata={
                    "strategy_type": "multi_timeframe",
                    "primary_trend": primary_trend,
                    "primary_timeframe": self.primary_timeframe,
                    "entry_timeframe": self.entry_timeframe,
                    "rsi": rsi,
                    "sma": sma
                }
            )
        
        return None
    
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
    
    def _calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return [0.0] * len(prices)
        
        sma = []
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(0.0)
            else:
                sma.append(np.mean(prices[i-period+1:i+1]))
        
        return sma
