"""
Enhanced Circuit Breakers

Advanced circuit breaker implementation with adaptive thresholds, 
machine learning integration, and sophisticated risk assessment.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from ..models import RiskLevel, RiskEvent, RiskAction, PortfolioData, RiskMetrics


class BreakerType(Enum):
    """Types of circuit breakers"""
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    CONSECUTIVE_LOSS = "consecutive_loss"
    POSITION_SIZE = "position_size"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    EVENT_RISK = "event_risk"


class BreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    DISABLED = "disabled"


@dataclass
class BreakerThreshold:
    """Dynamic threshold configuration for circuit breakers"""
    base_value: float
    min_value: float
    max_value: float
    volatility_factor: float = 0.1
    trend_factor: float = 0.05
    market_condition_factor: float = 0.2
    adaptive_enabled: bool = True
    
    def get_current_threshold(self, market_volatility: float = 0.0, 
                            trend_direction: float = 0.0, 
                            market_condition: str = "normal") -> float:
        """Calculate current threshold based on market conditions"""
        if not self.adaptive_enabled:
            return self.base_value
        
        # Adjust based on volatility
        volatility_adjustment = self.base_value * self.volatility_factor * market_volatility
        
        # Adjust based on trend
        trend_adjustment = self.base_value * self.trend_factor * trend_direction
        
        # Adjust based on market condition
        condition_multiplier = {
            "normal": 1.0,
            "stressed": 0.7,
            "crisis": 0.5,
            "recovery": 0.8
        }.get(market_condition, 1.0)
        
        condition_adjustment = self.base_value * self.market_condition_factor * (1 - condition_multiplier)
        
        # Calculate final threshold
        final_threshold = self.base_value + volatility_adjustment + trend_adjustment + condition_adjustment
        
        # Ensure within bounds
        return max(self.min_value, min(self.max_value, final_threshold))


@dataclass
class BreakerConfig:
    """Configuration for individual circuit breaker"""
    breaker_type: BreakerType
    threshold: BreakerThreshold
    cooldown_minutes: int = 30
    max_trigger_count: int = 5
    reset_period_hours: int = 24
    severity_weight: float = 1.0
    dependencies: List[BreakerType] = field(default_factory=list)
    enabled: bool = True


@dataclass
class BreakerResult:
    """Result of circuit breaker check"""
    breaker_type: BreakerType
    state: BreakerState
    triggered: bool
    current_value: float
    threshold_value: float
    severity: float
    message: str
    timestamp: datetime
    cooldown_remaining: int = 0
    trigger_count: int = 0


class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with adaptive thresholds and ML integration"""
    
    def __init__(self, configs: List[BreakerConfig] = None):
        self.configs = configs or self._create_default_configs()
        self.breaker_states: Dict[BreakerType, BreakerState] = {
            config.breaker_type: BreakerState.CLOSED for config in self.configs
        }
        self.breaker_timestamps: Dict[BreakerType, datetime] = {}
        self.trigger_counts: Dict[BreakerType, int] = {
            config.breaker_type: 0 for config in self.configs
        }
        self.market_conditions: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.adaptive_learning_enabled = True
    
    def check_all_breakers(self, portfolio_data: PortfolioData, 
                          risk_metrics: RiskMetrics,
                          market_data: Dict[str, Any] = None) -> Dict[BreakerType, BreakerResult]:
        """Check all circuit breakers"""
        results = {}
        market_data = market_data or {}
        
        # Update market conditions
        self._update_market_conditions(market_data)
        
        for config in self.configs:
            if not config.enabled:
                results[config.breaker_type] = BreakerResult(
                    breaker_type=config.breaker_type,
                    state=BreakerState.DISABLED,
                    triggered=False,
                    current_value=0.0,
                    threshold_value=0.0,
                    severity=0.0,
                    message="Breaker disabled",
                    timestamp=datetime.utcnow()
                )
                continue
            
            # Check dependencies
            if not self._check_dependencies(config, results):
                continue
            
            # Check individual breaker
            result = self._check_breaker(config, portfolio_data, risk_metrics, market_data)
            results[config.breaker_type] = result
            
            # Update state
            self._update_breaker_state(config, result)
        
        # Record performance
        self._record_performance(results)
        
        return results
    
    def _check_breaker(self, config: BreakerConfig, portfolio_data: PortfolioData,
                      risk_metrics: RiskMetrics, market_data: Dict[str, Any]) -> BreakerResult:
        """Check individual circuit breaker"""
        current_time = datetime.utcnow()
        
        # Get current value and threshold
        current_value = self._get_current_value(config.breaker_type, portfolio_data, risk_metrics, market_data)
        threshold_value = config.threshold.get_current_threshold(
            market_volatility=market_data.get("volatility", 0.0),
            trend_direction=market_data.get("trend", 0.0),
            market_condition=market_data.get("condition", "normal")
        )
        
        # Check if breaker should trigger
        should_trigger = self._should_trigger(config, current_value, threshold_value)
        
        # Calculate severity
        severity = self._calculate_severity(config, current_value, threshold_value)
        
        # Check cooldown
        cooldown_remaining = self._get_cooldown_remaining(config)
        
        # Determine state
        if should_trigger and cooldown_remaining == 0:
            state = BreakerState.OPEN
            triggered = True
            message = f"{config.breaker_type.value} breaker triggered: {current_value:.4f} > {threshold_value:.4f}"
        elif self.breaker_states[config.breaker_type] == BreakerState.OPEN:
            if cooldown_remaining > 0:
                state = BreakerState.OPEN
                triggered = True
                message = f"{config.breaker_type.value} breaker open (cooldown: {cooldown_remaining}min)"
            else:
                state = BreakerState.HALF_OPEN
                triggered = False
                message = f"{config.breaker_type.value} breaker half-open (testing)"
        else:
            state = BreakerState.CLOSED
            triggered = False
            message = f"{config.breaker_type.value} breaker closed"
        
        return BreakerResult(
            breaker_type=config.breaker_type,
            state=state,
            triggered=triggered,
            current_value=current_value,
            threshold_value=threshold_value,
            severity=severity,
            message=message,
            timestamp=current_time,
            cooldown_remaining=cooldown_remaining,
            trigger_count=self.trigger_counts[config.breaker_type]
        )
    
    def _get_current_value(self, breaker_type: BreakerType, portfolio_data: PortfolioData,
                          risk_metrics: RiskMetrics, market_data: Dict[str, Any]) -> float:
        """Get current value for breaker type"""
        if breaker_type == BreakerType.DAILY_LOSS:
            return abs(portfolio_data.total_pnl / portfolio_data.total_value) if portfolio_data.total_value > 0 else 0
        
        elif breaker_type == BreakerType.DRAWDOWN:
            return risk_metrics.current_drawdown
        
        elif breaker_type == BreakerType.CONSECUTIVE_LOSS:
            return risk_metrics.max_consecutive_losses
        
        elif breaker_type == BreakerType.POSITION_SIZE:
            if not portfolio_data.positions:
                return 0
            max_position = max(pos.get("size", 0) for pos in portfolio_data.positions)
            return max_position / portfolio_data.total_value if portfolio_data.total_value > 0 else 0
        
        elif breaker_type == BreakerType.VOLATILITY:
            return market_data.get("volatility", 0.0)
        
        elif breaker_type == BreakerType.CORRELATION:
            return market_data.get("correlation", 0.0)
        
        elif breaker_type == BreakerType.LIQUIDITY:
            return market_data.get("liquidity", 1.0)
        
        elif breaker_type == BreakerType.SENTIMENT:
            return abs(market_data.get("sentiment", 0.0))
        
        elif breaker_type == BreakerType.EVENT_RISK:
            return market_data.get("event_risk", 0.0)
        
        return 0.0
    
    def _should_trigger(self, config: BreakerConfig, current_value: float, threshold_value: float) -> bool:
        """Determine if breaker should trigger"""
        # Check if value exceeds threshold
        if current_value <= threshold_value:
            return False
        
        # Check if max trigger count reached
        if self.trigger_counts[config.breaker_type] >= config.max_trigger_count:
            return False
        
        # Check cooldown
        if self._get_cooldown_remaining(config) > 0:
            return False
        
        return True
    
    def _calculate_severity(self, config: BreakerConfig, current_value: float, threshold_value: float) -> float:
        """Calculate severity of breaker trigger"""
        if current_value <= threshold_value:
            return 0.0
        
        # Calculate severity as ratio of excess
        excess_ratio = (current_value - threshold_value) / threshold_value
        severity = min(1.0, excess_ratio * config.severity_weight)
        
        return severity
    
    def _get_cooldown_remaining(self, config: BreakerConfig) -> int:
        """Get remaining cooldown time in minutes"""
        if config.breaker_type not in self.breaker_timestamps:
            return 0
        
        last_trigger = self.breaker_timestamps[config.breaker_type]
        elapsed = datetime.utcnow() - last_trigger
        remaining = config.cooldown_minutes - int(elapsed.total_seconds() / 60)
        
        return max(0, remaining)
    
    def _check_dependencies(self, config: BreakerConfig, results: Dict[BreakerType, BreakerResult]) -> bool:
        """Check if breaker dependencies are met"""
        for dependency in config.dependencies:
            if dependency not in results:
                continue
            
            dep_result = results[dependency]
            if dep_result.state == BreakerState.OPEN:
                return False
        
        return True
    
    def _update_breaker_state(self, config: BreakerConfig, result: BreakerResult) -> None:
        """Update breaker state based on result"""
        if result.triggered and result.state == BreakerState.OPEN:
            self.breaker_states[config.breaker_type] = BreakerState.OPEN
            self.breaker_timestamps[config.breaker_type] = result.timestamp
            self.trigger_counts[config.breaker_type] += 1
        elif result.state == BreakerState.HALF_OPEN:
            self.breaker_states[config.breaker_type] = BreakerState.HALF_OPEN
        elif result.state == BreakerState.CLOSED:
            self.breaker_states[config.breaker_type] = BreakerState.CLOSED
    
    def _update_market_conditions(self, market_data: Dict[str, Any]) -> None:
        """Update market conditions for adaptive thresholds"""
        self.market_conditions.update(market_data)
    
    def _record_performance(self, results: Dict[BreakerType, BreakerResult]) -> None:
        """Record breaker performance for learning"""
        performance_data = {
            "timestamp": datetime.utcnow(),
            "breakers": {bt.value: {
                "state": result.state.value,
                "triggered": result.triggered,
                "severity": result.severity,
                "current_value": result.current_value,
                "threshold_value": result.threshold_value
            } for bt, result in results.items()},
            "market_conditions": self.market_conditions.copy()
        }
        
        self.performance_history.append(performance_data)
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
    
    def get_breaker_status(self) -> Dict[str, Any]:
        """Get current status of all breakers"""
        status = {}
        current_time = datetime.utcnow()
        
        for config in self.configs:
            cooldown_remaining = self._get_cooldown_remaining(config)
            
            status[config.breaker_type.value] = {
                "state": self.breaker_states[config.breaker_type].value,
                "enabled": config.enabled,
                "cooldown_remaining": cooldown_remaining,
                "trigger_count": self.trigger_counts[config.breaker_type],
                "max_triggers": config.max_trigger_count,
                "threshold": {
                    "base": config.threshold.base_value,
                    "min": config.threshold.min_value,
                    "max": config.threshold.max_value,
                    "adaptive": config.threshold.adaptive_enabled
                }
            }
        
        return status
    
    def reset_breaker(self, breaker_type: BreakerType) -> bool:
        """Reset specific breaker"""
        if breaker_type not in self.breaker_states:
            return False
        
        self.breaker_states[breaker_type] = BreakerState.CLOSED
        self.trigger_counts[breaker_type] = 0
        if breaker_type in self.breaker_timestamps:
            del self.breaker_timestamps[breaker_type]
        
        self.logger.info(f"Reset breaker: {breaker_type.value}")
        return True
    
    def reset_all_breakers(self) -> None:
        """Reset all breakers"""
        for breaker_type in self.breaker_states:
            self.reset_breaker(breaker_type)
        
        self.logger.info("Reset all breakers")
    
    def _create_default_configs(self) -> List[BreakerConfig]:
        """Create default breaker configurations"""
        return [
            BreakerConfig(
                breaker_type=BreakerType.DAILY_LOSS,
                threshold=BreakerThreshold(0.05, 0.01, 0.20),
                cooldown_minutes=30,
                max_trigger_count=3,
                severity_weight=1.0
            ),
            BreakerConfig(
                breaker_type=BreakerType.DRAWDOWN,
                threshold=BreakerThreshold(0.10, 0.05, 0.30),
                cooldown_minutes=60,
                max_trigger_count=2,
                severity_weight=1.2
            ),
            BreakerConfig(
                breaker_type=BreakerType.CONSECUTIVE_LOSS,
                threshold=BreakerThreshold(5, 3, 10),
                cooldown_minutes=120,
                max_trigger_count=2,
                severity_weight=0.8
            ),
            BreakerConfig(
                breaker_type=BreakerType.POSITION_SIZE,
                threshold=BreakerThreshold(0.20, 0.05, 0.50),
                cooldown_minutes=15,
                max_trigger_count=5,
                severity_weight=0.6
            ),
            BreakerConfig(
                breaker_type=BreakerType.VOLATILITY,
                threshold=BreakerThreshold(0.30, 0.10, 0.80),
                cooldown_minutes=45,
                max_trigger_count=3,
                severity_weight=0.7
            ),
            BreakerConfig(
                breaker_type=BreakerType.SENTIMENT,
                threshold=BreakerThreshold(0.80, 0.50, 1.00),
                cooldown_minutes=30,
                max_trigger_count=4,
                severity_weight=0.5
            )
        ]

