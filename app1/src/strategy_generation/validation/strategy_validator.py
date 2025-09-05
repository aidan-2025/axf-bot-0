"""
Strategy validator for logical consistency and requirements
"""

from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

from ..core.strategy_template import StrategyTemplate


class StrategyValidator:
    """
    Validates strategies for logical consistency and requirements
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.min_signals = self.config.get('min_signals', 10)
        self.max_signals = self.config.get('max_signals', 1000)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.min_strength = self.config.get('min_strength', 0.2)
        
    def validate(self, strategy: StrategyTemplate) -> Tuple[bool, List[str]]:
        """
        Validate a strategy for logical consistency and requirements
        
        Args:
            strategy: Strategy to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Validate strategy parameters
            param_valid, param_errors = strategy.validate_parameters()
            if not param_valid:
                errors.extend(param_errors)
            
            # Validate strategy configuration
            config_errors = self._validate_configuration(strategy)
            errors.extend(config_errors)
            
            # Validate strategy logic
            logic_errors = self._validate_strategy_logic(strategy)
            errors.extend(logic_errors)
            
            # Validate performance requirements
            perf_errors = self._validate_performance_requirements(strategy)
            errors.extend(perf_errors)
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            self.logger.error(f"Error validating strategy: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def _validate_configuration(self, strategy: StrategyTemplate) -> List[str]:
        """Validate strategy configuration"""
        errors = []
        
        # Check required fields
        if not strategy.parameters.strategy_id:
            errors.append("Strategy ID is required")
        
        if not strategy.parameters.name:
            errors.append("Strategy name is required")
        
        if not strategy.parameters.description:
            errors.append("Strategy description is required")
        
        # Check symbols
        if not strategy.parameters.symbols:
            errors.append("At least one trading symbol is required")
        
        # Check timeframes
        if not strategy.parameters.timeframes:
            errors.append("At least one timeframe is required")
        
        # Check risk level
        valid_risk_levels = ['low', 'medium', 'high']
        if strategy.parameters.risk_level not in valid_risk_levels:
            errors.append(f"Risk level must be one of {valid_risk_levels}")
        
        return errors
    
    def _validate_strategy_logic(self, strategy: StrategyTemplate) -> List[str]:
        """Validate strategy logic consistency"""
        errors = []
        
        # Check if strategy is initialized
        if not strategy.is_initialized:
            errors.append("Strategy must be initialized before validation")
            return errors
        
        # Validate parameter space
        try:
            param_space = strategy.get_parameter_space()
            if param_space.get_parameter_count() == 0:
                errors.append("Strategy must have at least one parameter")
        except Exception as e:
            errors.append(f"Error getting parameter space: {str(e)}")
        
        # Check for logical parameter relationships
        param_errors = self._validate_parameter_relationships(strategy)
        errors.extend(param_errors)
        
        return errors
    
    def _validate_parameter_relationships(self, strategy: StrategyTemplate) -> List[str]:
        """Validate parameter relationships for logical consistency"""
        errors = []
        params = strategy.parameters.parameters
        
        # Strategy-specific validations
        if strategy.parameters.strategy_type.value == "trend":
            # Check MA periods
            fast_ma = params.get('fast_ma_period', 0)
            slow_ma = params.get('slow_ma_period', 0)
            if fast_ma >= slow_ma:
                errors.append("Fast MA period must be less than slow MA period")
        
        elif strategy.parameters.strategy_type.value == "range":
            # Check RSI levels
            rsi_oversold = params.get('rsi_oversold', 0)
            rsi_overbought = params.get('rsi_overbought', 0)
            if rsi_oversold >= rsi_overbought:
                errors.append("RSI oversold must be less than RSI overbought")
        
        elif strategy.parameters.strategy_type.value == "breakout":
            # Check volume threshold
            volume_threshold = params.get('volume_threshold', 0)
            if volume_threshold < 1.0:
                errors.append("Volume threshold must be at least 1.0")
        
        elif strategy.parameters.strategy_type.value == "sentiment":
            # Check weights sum to 1
            news_weight = params.get('news_weight', 0)
            social_weight = params.get('social_weight', 0)
            technical_weight = params.get('technical_weight', 0)
            total_weight = news_weight + social_weight + technical_weight
            if abs(total_weight - 1.0) > 0.01:
                errors.append("Sentiment weights must sum to 1.0")
        
        return errors
    
    def _validate_performance_requirements(self, strategy: StrategyTemplate) -> List[str]:
        """Validate performance requirements"""
        errors = []
        
        # Check if strategy has performance history
        if not strategy.performance_history:
            # This is OK for new strategies
            return errors
        
        # Get latest performance
        latest_performance = strategy.get_latest_performance()
        if not latest_performance:
            return errors
        
        # Validate performance metrics
        if latest_performance.total_trades < self.min_signals:
            errors.append(f"Strategy must generate at least {self.min_signals} signals")
        
        if latest_performance.total_trades > self.max_signals:
            errors.append(f"Strategy generates too many signals (>{self.max_signals})")
        
        # Check win rate
        if latest_performance.win_rate < 0.3:
            errors.append("Strategy win rate is too low (<30%)")
        
        # Check profit factor
        if latest_performance.profit_factor < 1.0:
            errors.append("Strategy profit factor is below 1.0")
        
        # Check max drawdown
        if latest_performance.max_drawdown > 0.5:
            errors.append("Strategy max drawdown is too high (>50%)")
        
        return errors
    
    def validate_signal_quality(self, signals: List[Any]) -> Tuple[bool, List[str]]:
        """Validate signal quality"""
        errors = []
        
        if not signals:
            errors.append("No signals generated")
            return len(errors) == 0, errors
        
        # Check signal count
        if len(signals) < self.min_signals:
            errors.append(f"Too few signals generated: {len(signals)} < {self.min_signals}")
        
        if len(signals) > self.max_signals:
            errors.append(f"Too many signals generated: {len(signals)} > {self.max_signals}")
        
        # Check signal quality
        low_confidence_count = sum(1 for s in signals if s.confidence < self.min_confidence)
        if low_confidence_count > len(signals) * 0.5:
            errors.append(f"Too many low-confidence signals: {low_confidence_count}/{len(signals)}")
        
        low_strength_count = sum(1 for s in signals if s.strength < self.min_strength)
        if low_strength_count > len(signals) * 0.3:
            errors.append(f"Too many low-strength signals: {low_strength_count}/{len(signals)}")
        
        # Check signal diversity
        signal_types = set(s.signal_type for s in signals)
        if len(signal_types) < 2:
            errors.append("Strategy generates only one type of signal")
        
        return len(errors) == 0, errors
    
    def get_validation_summary(self, strategy: StrategyTemplate) -> Dict[str, Any]:
        """Get validation summary for a strategy"""
        is_valid, errors = self.validate(strategy)
        
        return {
            "is_valid": is_valid,
            "error_count": len(errors),
            "errors": errors,
            "strategy_id": strategy.parameters.strategy_id,
            "strategy_type": strategy.parameters.strategy_type.value,
            "is_initialized": strategy.is_initialized,
            "performance_history_count": len(strategy.performance_history),
            "timestamp": datetime.now().isoformat()
        }

