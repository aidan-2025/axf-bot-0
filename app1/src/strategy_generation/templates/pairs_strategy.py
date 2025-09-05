"""
Pairs trading strategy template
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import logging

from ..core.strategy_template import StrategyTemplate, StrategyType, Signal, StrategyParameters
from ..core.parameter_space import ParameterSpace, ParameterType, ParameterDefinition


class PairsStrategy(StrategyTemplate):
    """
    Pairs trading strategy using correlation and mean reversion
    """
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self.parameters.name
        
        # Strategy-specific attributes
        self.lookback_period = 60
        self.correlation_threshold = 0.7
        self.zscore_threshold = 2.0
        self.mean_reversion_period = 20
        self.min_correlation = 0.5
        
    def initialize(self) -> bool:
        """Initialize pairs strategy with parameters"""
        try:
            params = self.parameters.parameters
            
            # Extract parameters
            self.lookback_period = params.get('lookback_period', 60)
            self.correlation_threshold = params.get('correlation_threshold', 0.7)
            self.zscore_threshold = params.get('zscore_threshold', 2.0)
            self.mean_reversion_period = params.get('mean_reversion_period', 20)
            self.min_correlation = params.get('min_correlation', 0.5)
            
            # Validate parameters
            is_valid, errors = self.validate_parameters()
            if not is_valid:
                self.logger.error(f"Invalid parameters: {errors}")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Pairs strategy initialized: {self}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing pairs strategy: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate pairs trading signals"""
        if not self.is_initialized:
            self.logger.warning("Strategy not initialized")
            return []
        
        try:
            signals = []
            
            # Extract pairs data
            pairs_data = market_data.get('pairs', {})
            if len(pairs_data) < 2:
                return signals
            
            # Get available pairs
            pair_symbols = list(pairs_data.keys())
            
            # Generate signals for each pair combination
            for i in range(len(pair_symbols)):
                for j in range(i + 1, len(pair_symbols)):
                    pair_signals = self._analyze_pair(
                        pair_symbols[i], pair_symbols[j],
                        pairs_data[pair_symbols[i]], pairs_data[pair_symbols[j]],
                        market_data
                    )
                    signals.extend(pair_signals)
            
            self.logger.info(f"Generated {len(signals)} pairs signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating pairs signals: {e}")
            return []
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate pairs strategy parameters"""
        errors = []
        
        # Validate lookback period
        if self.lookback_period < 20 or self.lookback_period > 200:
            errors.append("Lookback period must be between 20 and 200")
        
        # Validate correlation thresholds
        if not (0.1 <= self.correlation_threshold <= 1.0):
            errors.append("Correlation threshold must be between 0.1 and 1.0")
        
        if not (0.1 <= self.min_correlation <= 1.0):
            errors.append("Min correlation must be between 0.1 and 1.0")
        
        if self.min_correlation >= self.correlation_threshold:
            errors.append("Min correlation must be less than correlation threshold")
        
        # Validate z-score threshold
        if self.zscore_threshold < 1.0 or self.zscore_threshold > 5.0:
            errors.append("Z-score threshold must be between 1.0 and 5.0")
        
        # Validate mean reversion period
        if self.mean_reversion_period < 10 or self.mean_reversion_period > 50:
            errors.append("Mean reversion period must be between 10 and 50")
        
        return len(errors) == 0, errors
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization"""
        param_space = ParameterSpace()
        
        # Lookback period
        param_space.add_parameter(ParameterDefinition(
            name="lookback_period",
            param_type=ParameterType.INTEGER,
            min_value=30,
            max_value=120,
            default_value=60,
            description="Period for correlation calculation"
        ))
        
        # Correlation threshold
        param_space.add_parameter(ParameterDefinition(
            name="correlation_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.5,
            max_value=0.9,
            default_value=0.7,
            description="Minimum correlation for pair selection"
        ))
        
        # Z-score threshold
        param_space.add_parameter(ParameterDefinition(
            name="zscore_threshold",
            param_type=ParameterType.FLOAT,
            min_value=1.5,
            max_value=3.0,
            default_value=2.0,
            description="Z-score threshold for signal generation"
        ))
        
        # Mean reversion period
        param_space.add_parameter(ParameterDefinition(
            name="mean_reversion_period",
            param_type=ParameterType.INTEGER,
            min_value=10,
            max_value=30,
            default_value=20,
            description="Period for mean reversion calculation"
        ))
        
        # Min correlation
        param_space.add_parameter(ParameterDefinition(
            name="min_correlation",
            param_type=ParameterType.FLOAT,
            min_value=0.3,
            max_value=0.7,
            default_value=0.5,
            description="Minimum correlation to consider pair"
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
    
    def _analyze_pair(self, symbol1: str, symbol2: str, data1: Dict[str, Any], 
                     data2: Dict[str, Any], market_data: Dict[str, Any]) -> List[Signal]:
        """Analyze a pair for trading opportunities"""
        signals = []
        
        # Extract price data
        prices1 = data1.get('close', [])
        prices2 = data2.get('close', [])
        timestamps1 = data1.get('timestamp', [])
        timestamps2 = data2.get('timestamp', [])
        
        if len(prices1) < self.lookback_period or len(prices2) < self.lookback_period:
            return signals
        
        # Calculate correlation
        correlation = self._calculate_correlation(prices1, prices2)
        
        if abs(correlation) < self.min_correlation:
            return signals
        
        # Calculate spread and z-score
        spread = self._calculate_spread(prices1, prices2)
        z_scores = self._calculate_z_scores(spread, self.mean_reversion_period)
        
        # Generate signals based on z-score
        for i in range(len(z_scores)):
            if i < self.mean_reversion_period:
                continue
            
            z_score = z_scores[i]
            current_price1 = prices1[i]
            current_price2 = prices2[i]
            current_timestamp = timestamps1[i] if i < len(timestamps1) else datetime.now()
            
            signal = self._create_pairs_signal(
                timestamp=current_timestamp,
                symbol1=symbol1,
                symbol2=symbol2,
                price1=current_price1,
                price2=current_price2,
                z_score=z_score,
                correlation=correlation
            )
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """Calculate correlation between two price series"""
        if len(prices1) != len(prices2) or len(prices1) < 2:
            return 0.0
        
        # Use recent data for correlation
        recent_prices1 = prices1[-self.lookback_period:]
        recent_prices2 = prices2[-self.lookback_period:]
        
        return np.corrcoef(recent_prices1, recent_prices2)[0, 1]
    
    def _calculate_spread(self, prices1: List[float], prices2: List[float]) -> List[float]:
        """Calculate spread between two price series"""
        if len(prices1) != len(prices2):
            return []
        
        # Calculate ratio spread
        spread = [p1 / p2 for p1, p2 in zip(prices1, prices2)]
        return spread
    
    def _calculate_z_scores(self, spread: List[float], period: int) -> List[float]:
        """Calculate z-scores for mean reversion"""
        if len(spread) < period:
            return [0.0] * len(spread)
        
        z_scores = []
        for i in range(len(spread)):
            if i < period - 1:
                z_scores.append(0.0)
            else:
                period_spread = spread[i-period+1:i+1]
                mean = np.mean(period_spread)
                std = np.std(period_spread)
                
                if std == 0:
                    z_scores.append(0.0)
                else:
                    z_scores.append((spread[i] - mean) / std)
        
        return z_scores
    
    def _create_pairs_signal(self, timestamp: datetime, symbol1: str, symbol2: str,
                           price1: float, price2: float, z_score: float, 
                           correlation: float) -> Signal:
        """Create pairs trading signal"""
        
        # Check if z-score exceeds threshold
        if abs(z_score) < self.zscore_threshold:
            return None
        
        # Determine signal direction based on z-score
        if z_score > self.zscore_threshold:
            # Spread is too high, expect mean reversion
            signal_type = "sell"  # Sell the overpriced asset
            strength = min(1.0, (z_score - self.zscore_threshold) / self.zscore_threshold)
        elif z_score < -self.zscore_threshold:
            # Spread is too low, expect mean reversion
            signal_type = "buy"  # Buy the underpriced asset
            strength = min(1.0, (abs(z_score) - self.zscore_threshold) / self.zscore_threshold)
        else:
            return None
        
        # Calculate confidence based on correlation
        confidence = min(1.0, abs(correlation))
        
        return Signal(
            timestamp=timestamp,
            symbol=f"{symbol1}/{symbol2}",
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price=price1 / price2,  # Ratio price
            metadata={
                "strategy_type": "pairs",
                "symbol1": symbol1,
                "symbol2": symbol2,
                "price1": price1,
                "price2": price2,
                "z_score": z_score,
                "correlation": correlation,
                "spread_ratio": price1 / price2
            }
        )
