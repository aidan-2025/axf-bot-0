"""
Trend following strategy template
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime, timedelta
import logging

from ..core.strategy_template import StrategyTemplate, StrategyType, Signal, StrategyParameters
from ..core.parameter_space import ParameterSpace, ParameterType, ParameterDefinition


class TrendStrategy(StrategyTemplate):
    """
    Trend following strategy using moving averages and momentum indicators
    """
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.logger = logging.getLogger(__name__)
        
        # Strategy-specific attributes
        self.fast_ma_period = 20
        self.slow_ma_period = 50
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.atr_period = 14
        self.atr_multiplier = 2.0
        self.min_trend_strength = 0.6
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self.parameters.name
        
    def initialize(self) -> bool:
        """Initialize trend strategy with parameters"""
        try:
            params = self.parameters.parameters
            
            # Extract parameters
            self.fast_ma_period = params.get('fast_ma_period', 20)
            self.slow_ma_period = params.get('slow_ma_period', 50)
            self.rsi_period = params.get('rsi_period', 14)
            self.rsi_oversold = params.get('rsi_oversold', 30)
            self.rsi_overbought = params.get('rsi_overbought', 70)
            self.atr_period = params.get('atr_period', 14)
            self.atr_multiplier = params.get('atr_multiplier', 2.0)
            self.min_trend_strength = params.get('min_trend_strength', 0.6)
            
            # Validate parameters
            is_valid, errors = self.validate_parameters()
            if not is_valid:
                self.logger.error(f"Invalid parameters: {errors}")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Trend strategy initialized: {self}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing trend strategy: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trend following signals"""
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
            
            if len(prices) < max(self.slow_ma_period, self.rsi_period, self.atr_period):
                return signals
            
            # Calculate indicators
            fast_ma = self._calculate_sma(prices, self.fast_ma_period)
            slow_ma = self._calculate_sma(prices, self.slow_ma_period)
            rsi = self._calculate_rsi(prices, self.rsi_period)
            atr = self._calculate_atr(highs, lows, prices, self.atr_period)
            
            # Generate signals for each time period
            for i in range(max(self.slow_ma_period, self.rsi_period, self.atr_period), len(prices)):
                current_price = prices[i]
                current_fast_ma = fast_ma[i]
                current_slow_ma = slow_ma[i]
                current_rsi = rsi[i]
                current_atr = atr[i]
                current_timestamp = timestamps[i] if i < len(timestamps) else datetime.now()
                
                # Determine trend direction
                trend_direction = self._determine_trend_direction(
                    current_fast_ma, current_slow_ma, current_rsi
                )
                
                # Calculate trend strength
                trend_strength = self._calculate_trend_strength(
                    fast_ma[i-10:i+1] if i >= 10 else fast_ma[:i+1],
                    slow_ma[i-10:i+1] if i >= 10 else slow_ma[:i+1]
                )
                
                # Generate signal if trend is strong enough
                if trend_strength >= self.min_trend_strength:
                    signal = self._create_signal(
                        timestamp=current_timestamp,
                        price=current_price,
                        trend_direction=trend_direction,
                        atr=current_atr,
                        rsi=current_rsi,
                        trend_strength=trend_strength
                    )
                    
                    if signal:
                        signals.append(signal)
            
            self.logger.info(f"Generated {len(signals)} trend signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trend signals: {e}")
            return []
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate trend strategy parameters"""
        errors = []
        
        # Validate moving average periods
        if self.fast_ma_period >= self.slow_ma_period:
            errors.append("Fast MA period must be less than slow MA period")
        
        if self.fast_ma_period < 5 or self.slow_ma_period < 10:
            errors.append("MA periods too small")
        
        if self.fast_ma_period > 100 or self.slow_ma_period > 200:
            errors.append("MA periods too large")
        
        # Validate RSI parameters
        if self.rsi_oversold >= self.rsi_overbought:
            errors.append("RSI oversold must be less than overbought")
        
        if not (0 <= self.rsi_oversold <= 50):
            errors.append("RSI oversold must be between 0 and 50")
        
        if not (50 <= self.rsi_overbought <= 100):
            errors.append("RSI overbought must be between 50 and 100")
        
        # Validate ATR parameters
        if self.atr_period < 5 or self.atr_period > 50:
            errors.append("ATR period must be between 5 and 50")
        
        if self.atr_multiplier < 0.5 or self.atr_multiplier > 5.0:
            errors.append("ATR multiplier must be between 0.5 and 5.0")
        
        # Validate trend strength threshold
        if not (0.0 <= self.min_trend_strength <= 1.0):
            errors.append("Min trend strength must be between 0.0 and 1.0")
        
        return len(errors) == 0, errors
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization"""
        param_space = ParameterSpace()
        
        # Fast MA period
        param_space.add_parameter(ParameterDefinition(
            name="fast_ma_period",
            param_type=ParameterType.INTEGER,
            min_value=5,
            max_value=50,
            default_value=20,
            description="Fast moving average period"
        ))
        
        # Slow MA period
        param_space.add_parameter(ParameterDefinition(
            name="slow_ma_period",
            param_type=ParameterType.INTEGER,
            min_value=20,
            max_value=100,
            default_value=50,
            description="Slow moving average period"
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
            min_value=10,
            max_value=40,
            default_value=30,
            description="RSI oversold threshold"
        ))
        
        # RSI overbought level
        param_space.add_parameter(ParameterDefinition(
            name="rsi_overbought",
            param_type=ParameterType.INTEGER,
            min_value=60,
            max_value=90,
            default_value=70,
            description="RSI overbought threshold"
        ))
        
        # ATR period
        param_space.add_parameter(ParameterDefinition(
            name="atr_period",
            param_type=ParameterType.INTEGER,
            min_value=5,
            max_value=30,
            default_value=14,
            description="ATR calculation period"
        ))
        
        # ATR multiplier
        param_space.add_parameter(ParameterDefinition(
            name="atr_multiplier",
            param_type=ParameterType.FLOAT,
            min_value=0.5,
            max_value=5.0,
            default_value=2.0,
            description="ATR multiplier for stop loss"
        ))
        
        # Min trend strength
        param_space.add_parameter(ParameterDefinition(
            name="min_trend_strength",
            param_type=ParameterType.FLOAT,
            min_value=0.3,
            max_value=0.9,
            default_value=0.6,
            description="Minimum trend strength threshold"
        ))
        
        # Add constraints
        param_space.add_constraint("fast_ma_period < slow_ma_period")
        param_space.add_constraint("rsi_oversold < rsi_overbought")
        
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
    
    def _determine_trend_direction(self, fast_ma: float, slow_ma: float, rsi: float) -> str:
        """Determine trend direction based on indicators"""
        if fast_ma > slow_ma and rsi < self.rsi_overbought:
            return "buy"
        elif fast_ma < slow_ma and rsi > self.rsi_oversold:
            return "sell"
        else:
            return "hold"
    
    def _calculate_trend_strength(self, fast_ma_values: List[float], 
                                 slow_ma_values: List[float]) -> float:
        """Calculate trend strength based on MA alignment"""
        if len(fast_ma_values) < 2 or len(slow_ma_values) < 2:
            return 0.0
        
        # Calculate correlation between fast and slow MA
        correlation = np.corrcoef(fast_ma_values, slow_ma_values)[0, 1]
        
        # Calculate slope consistency
        fast_slope = np.polyfit(range(len(fast_ma_values)), fast_ma_values, 1)[0]
        slow_slope = np.polyfit(range(len(slow_ma_values)), slow_ma_values, 1)[0]
        
        # Combine correlation and slope consistency
        slope_consistency = 1.0 - abs(fast_slope - slow_slope) / (abs(fast_slope) + abs(slow_slope) + 1e-8)
        trend_strength = (correlation + slope_consistency) / 2
        
        return max(0.0, min(1.0, trend_strength))
    
    def _create_signal(self, timestamp: datetime, price: float, trend_direction: str,
                      atr: float, rsi: float, trend_strength: float) -> Signal:
        """Create a trading signal"""
        if trend_direction == "hold":
            return None
        
        # Calculate signal strength based on trend strength and RSI position
        if trend_direction == "buy":
            strength = trend_strength * (1.0 - (rsi - self.rsi_oversold) / (self.rsi_overbought - self.rsi_oversold))
        else:  # sell
            strength = trend_strength * ((rsi - self.rsi_oversold) / (self.rsi_overbought - self.rsi_oversold))
        
        strength = max(0.1, min(1.0, strength))
        
        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None
        
        if atr > 0:
            atr_distance = atr * self.atr_multiplier
            if trend_direction == "buy":
                stop_loss = price - atr_distance
                take_profit = price + (atr_distance * 2)  # 1:2 risk-reward ratio
            else:  # sell
                stop_loss = price + atr_distance
                take_profit = price - (atr_distance * 2)
        
        return Signal(
            timestamp=timestamp,
            symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
            signal_type=trend_direction,
            strength=strength,
            confidence=trend_strength,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy_type": "trend",
                "fast_ma": self.fast_ma_period,
                "slow_ma": self.slow_ma_period,
                "rsi": rsi,
                "atr": atr,
                "trend_strength": trend_strength
            }
        )
