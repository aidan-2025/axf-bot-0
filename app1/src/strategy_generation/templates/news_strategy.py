"""
News-based trading strategy template
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import logging

from ..core.strategy_template import StrategyTemplate, StrategyType, Signal, StrategyParameters
from ..core.parameter_space import ParameterSpace, ParameterType, ParameterDefinition


class NewsStrategy(StrategyTemplate):
    """
    News-based trading strategy using economic calendar events
    """
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self.parameters.name
        
        # Strategy-specific attributes
        self.high_impact_threshold = 0.8
        self.medium_impact_threshold = 0.5
        self.pre_event_hours = 2
        self.post_event_hours = 4
        self.volatility_multiplier = 1.5
        
    def initialize(self) -> bool:
        """Initialize news strategy with parameters"""
        try:
            params = self.parameters.parameters
            
            # Extract parameters
            self.high_impact_threshold = params.get('high_impact_threshold', 0.8)
            self.medium_impact_threshold = params.get('medium_impact_threshold', 0.5)
            self.pre_event_hours = params.get('pre_event_hours', 2)
            self.post_event_hours = params.get('post_event_hours', 4)
            self.volatility_multiplier = params.get('volatility_multiplier', 1.5)
            
            # Validate parameters
            is_valid, errors = self.validate_parameters()
            if not is_valid:
                self.logger.error(f"Invalid parameters: {errors}")
                return False
            
            self.is_initialized = True
            self.logger.info(f"News strategy initialized: {self}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing news strategy: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate news-based signals"""
        if not self.is_initialized:
            self.logger.warning("Strategy not initialized")
            return []
        
        try:
            signals = []
            
            # Extract economic events data
            events_data = market_data.get('economic_events', [])
            
            if not events_data:
                return signals
            
            # Generate signals for upcoming events
            for event in events_data:
                signal = self._create_news_signal(event, market_data)
                if signal:
                    signals.append(signal)
            
            self.logger.info(f"Generated {len(signals)} news signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating news signals: {e}")
            return []
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Validate news strategy parameters"""
        errors = []
        
        # Validate impact thresholds
        if not (0.1 <= self.high_impact_threshold <= 1.0):
            errors.append("High impact threshold must be between 0.1 and 1.0")
        
        if not (0.1 <= self.medium_impact_threshold <= 1.0):
            errors.append("Medium impact threshold must be between 0.1 and 1.0")
        
        if self.high_impact_threshold <= self.medium_impact_threshold:
            errors.append("High impact threshold must be greater than medium impact threshold")
        
        # Validate time parameters
        if self.pre_event_hours < 0 or self.pre_event_hours > 24:
            errors.append("Pre-event hours must be between 0 and 24")
        
        if self.post_event_hours < 0 or self.post_event_hours > 24:
            errors.append("Post-event hours must be between 0 and 24")
        
        return len(errors) == 0, errors
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization"""
        param_space = ParameterSpace()
        
        # High impact threshold
        param_space.add_parameter(ParameterDefinition(
            name="high_impact_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.6,
            max_value=1.0,
            default_value=0.8,
            description="Threshold for high impact events"
        ))
        
        # Medium impact threshold
        param_space.add_parameter(ParameterDefinition(
            name="medium_impact_threshold",
            param_type=ParameterType.FLOAT,
            min_value=0.3,
            max_value=0.7,
            default_value=0.5,
            description="Threshold for medium impact events"
        ))
        
        # Pre-event hours
        param_space.add_parameter(ParameterDefinition(
            name="pre_event_hours",
            param_type=ParameterType.INTEGER,
            min_value=0,
            max_value=12,
            default_value=2,
            description="Hours before event to generate signal"
        ))
        
        # Post-event hours
        param_space.add_parameter(ParameterDefinition(
            name="post_event_hours",
            param_type=ParameterType.INTEGER,
            min_value=1,
            max_value=12,
            default_value=4,
            description="Hours after event to maintain signal"
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
    
    def _create_news_signal(self, event: Dict[str, Any], market_data: Dict[str, Any]) -> Signal:
        """Create news-based signal for an event"""
        
        # Check if event is relevant
        impact_score = event.get('market_impact_score', 0.0)
        if impact_score < self.medium_impact_threshold:
            return None
        
        # Determine signal direction based on forecast vs previous
        forecast = event.get('forecast', 0.0)
        previous = event.get('previous', 0.0)
        
        if forecast is None or previous is None:
            return None
        
        # Simple direction logic - can be enhanced
        if forecast > previous:
            signal_type = "buy"
            strength = min(1.0, impact_score)
        elif forecast < previous:
            signal_type = "sell"
            strength = min(1.0, impact_score)
        else:
            return None
        
        # Calculate confidence based on impact and volatility
        volatility = event.get('volatility_expected', 0.0)
        confidence = min(1.0, impact_score * (1 + volatility))
        
        # Get current price
        ohlcv_data = market_data.get('ohlcv', {})
        current_price = ohlcv_data.get('close', [0])[-1] if ohlcv_data.get('close') else 0.0
        
        return Signal(
            timestamp=datetime.now(),
            symbol=self.parameters.symbols[0] if self.parameters.symbols else "EURUSD",
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price=current_price,
            metadata={
                "strategy_type": "news",
                "event_title": event.get('title', ''),
                "impact_score": impact_score,
                "volatility_expected": volatility,
                "forecast": forecast,
                "previous": previous
            }
        )
