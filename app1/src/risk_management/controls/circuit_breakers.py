#!/usr/bin/env python3
"""
Circuit Breakers

Implements circuit breaker patterns for risk management, including
automatic trading suspension, position reduction, and recovery protocols.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..models import (
    RiskLevel, RiskEvent, RiskAction, TradingState, PortfolioData, RiskMetrics
)

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, trading suspended
    HALF_OPEN = "half_open"  # Testing if conditions have improved


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers"""
    # Drawdown circuit breaker
    max_drawdown_threshold: float = 0.15  # 15%
    critical_drawdown_threshold: float = 0.20  # 20%
    drawdown_recovery_threshold: float = 0.05  # 5% recovery
    
    # Daily loss circuit breaker
    max_daily_loss_threshold: float = 0.05  # 5%
    critical_daily_loss_threshold: float = 0.10  # 10%
    daily_loss_reset_hours: int = 24
    
    # Consecutive losses circuit breaker
    max_consecutive_losses: int = 5
    critical_consecutive_losses: int = 8
    consecutive_loss_reset_hours: int = 48
    
    # Position size circuit breaker
    max_position_size_threshold: float = 0.20  # 20%
    max_total_exposure_threshold: float = 0.50  # 50%
    
    # Recovery settings
    recovery_test_period_hours: int = 24
    gradual_recovery_enabled: bool = True
    recovery_position_reduction: float = 0.5  # 50% of normal size
    
    # Circuit breaker timeouts
    open_duration_minutes: int = 60
    half_open_test_duration_minutes: int = 30
    max_open_duration_hours: int = 24


class CircuitBreaker:
    """
    Circuit breaker implementation for risk management.
    
    Monitors various risk metrics and automatically triggers circuit breakers
    to protect trading capital during adverse conditions.
    """
    
    def __init__(self, config: CircuitBreakerConfig = None):
        """Initialize circuit breaker"""
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Circuit breaker states
        self.drawdown_breaker = CircuitBreakerState.CLOSED
        self.daily_loss_breaker = CircuitBreakerState.CLOSED
        self.consecutive_loss_breaker = CircuitBreakerState.CLOSED
        self.position_size_breaker = CircuitBreakerState.CLOSED
        
        # State tracking
        self.breaker_states: Dict[str, CircuitBreakerState] = {
            "drawdown": self.drawdown_breaker,
            "daily_loss": self.daily_loss_breaker,
            "consecutive_loss": self.consecutive_loss_breaker,
            "position_size": self.position_size_breaker
        }
        
        # Timing tracking
        self.breaker_timestamps: Dict[str, datetime] = {}
        self.recovery_attempts: Dict[str, int] = {}
        
        # Event tracking
        self.breaker_events: List[RiskEvent] = []
        self.daily_pnl_history: List[float] = []
        self.consecutive_loss_count = 0
        
        # Callbacks
        self.on_breaker_open: Optional[Callable] = None
        self.on_breaker_close: Optional[Callable] = None
        self.on_recovery_test: Optional[Callable] = None
        
        self.logger.info("CircuitBreaker initialized")
    
    def check_circuit_breakers(self, portfolio_data: PortfolioData, 
                             risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """
        Check all circuit breakers and return current state.
        
        Args:
            portfolio_data: Current portfolio state
            risk_metrics: Current risk metrics
            
        Returns:
            Dictionary with circuit breaker status and recommendations
        """
        try:
            self.logger.debug("Checking circuit breakers")
            
            # Update tracking data
            self._update_tracking_data(portfolio_data, risk_metrics)
            
            # Check each circuit breaker
            results = {
                "overall_state": self._get_overall_state(),
                "breakers": {},
                "recommendations": [],
                "actions": []
            }
            
            # Drawdown circuit breaker
            drawdown_result = self._check_drawdown_breaker(portfolio_data, risk_metrics)
            results["breakers"]["drawdown"] = drawdown_result
            
            # Daily loss circuit breaker
            daily_loss_result = self._check_daily_loss_breaker(portfolio_data)
            results["breakers"]["daily_loss"] = daily_loss_result
            
            # Consecutive losses circuit breaker
            consecutive_loss_result = self._check_consecutive_loss_breaker()
            results["breakers"]["consecutive_loss"] = consecutive_loss_result
            
            # Position size circuit breaker
            position_size_result = self._check_position_size_breaker(portfolio_data)
            results["breakers"]["position_size"] = position_size_result
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results["breakers"])
            results["actions"] = self._determine_actions(results["breakers"])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error checking circuit breakers: {e}")
            return {
                "overall_state": "error",
                "breakers": {},
                "recommendations": ["Error in circuit breaker check"],
                "actions": [RiskAction.EMERGENCY_STOP]
            }
    
    def _update_tracking_data(self, portfolio_data: PortfolioData, risk_metrics: RiskMetrics):
        """Update internal tracking data"""
        # Update daily PnL history
        current_pnl = portfolio_data.total_pnl
        self.daily_pnl_history.append(current_pnl)
        
        # Keep only recent history (last 30 days)
        if len(self.daily_pnl_history) > 30:
            self.daily_pnl_history = self.daily_pnl_history[-30:]
        
        # Update consecutive loss count
        if current_pnl < 0:
            self.consecutive_loss_count += 1
        else:
            self.consecutive_loss_count = 0
    
    def _check_drawdown_breaker(self, portfolio_data: PortfolioData, 
                              risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """Check drawdown circuit breaker"""
        current_drawdown = risk_metrics.current_drawdown
        max_drawdown = risk_metrics.max_drawdown
        
        result = {
            "state": self.drawdown_breaker.value,
            "threshold": self.config.max_drawdown_threshold,
            "current_value": current_drawdown,
            "triggered": False,
            "message": ""
        }
        
        # Check if circuit should be opened
        if (self.drawdown_breaker == CircuitBreakerState.CLOSED and 
            current_drawdown >= self.config.critical_drawdown_threshold):
            
            self.drawdown_breaker = CircuitBreakerState.OPEN
            self.breaker_timestamps["drawdown"] = datetime.utcnow()
            result["state"] = "open"
            result["triggered"] = True
            result["message"] = f"Critical drawdown threshold exceeded: {current_drawdown:.2%}"
            
            self._create_breaker_event("drawdown", "open", result["message"])
            
        elif (self.drawdown_breaker == CircuitBreakerState.CLOSED and 
              current_drawdown >= self.config.max_drawdown_threshold):
            
            self.drawdown_breaker = CircuitBreakerState.OPEN
            self.breaker_timestamps["drawdown"] = datetime.utcnow()
            result["state"] = "open"
            result["triggered"] = True
            result["message"] = f"Drawdown threshold exceeded: {current_drawdown:.2%}"
            
            self._create_breaker_event("drawdown", "open", result["message"])
        
        # Check if circuit should be closed
        elif (self.drawdown_breaker == CircuitBreakerState.OPEN and 
              current_drawdown <= self.config.drawdown_recovery_threshold):
            
            self.drawdown_breaker = CircuitBreakerState.CLOSED
            result["state"] = "closed"
            result["message"] = f"Drawdown recovered: {current_drawdown:.2%}"
            
            self._create_breaker_event("drawdown", "closed", result["message"])
        
        # Check if circuit should move to half-open for testing
        elif (self.drawdown_breaker == CircuitBreakerState.OPEN and 
              self._should_test_recovery("drawdown")):
            
            self.drawdown_breaker = CircuitBreakerState.HALF_OPEN
            result["state"] = "half_open"
            result["message"] = "Testing recovery conditions"
            
            self._create_breaker_event("drawdown", "half_open", result["message"])
        
        return result
    
    def _check_daily_loss_breaker(self, portfolio_data: PortfolioData) -> Dict[str, Any]:
        """Check daily loss circuit breaker"""
        # Calculate daily loss percentage
        if not self.daily_pnl_history:
            daily_loss_pct = 0.0
        else:
            # Get today's PnL (assuming last entry is today)
            today_pnl = self.daily_pnl_history[-1]
            daily_loss_pct = abs(today_pnl) / portfolio_data.total_value if portfolio_data.total_value > 0 else 0.0
        
        result = {
            "state": self.daily_loss_breaker.value,
            "threshold": self.config.max_daily_loss_threshold,
            "current_value": daily_loss_pct,
            "triggered": False,
            "message": ""
        }
        
        # Check if circuit should be opened
        if (self.daily_loss_breaker == CircuitBreakerState.CLOSED and 
            daily_loss_pct >= self.config.critical_daily_loss_threshold):
            
            self.daily_loss_breaker = CircuitBreakerState.OPEN
            self.breaker_timestamps["daily_loss"] = datetime.utcnow()
            result["state"] = "open"
            result["triggered"] = True
            result["message"] = f"Critical daily loss threshold exceeded: {daily_loss_pct:.2%}"
            
            self._create_breaker_event("daily_loss", "open", result["message"])
            
        elif (self.daily_loss_breaker == CircuitBreakerState.CLOSED and 
              daily_loss_pct >= self.config.max_daily_loss_threshold):
            
            self.daily_loss_breaker = CircuitBreakerState.OPEN
            self.breaker_timestamps["daily_loss"] = datetime.utcnow()
            result["state"] = "open"
            result["triggered"] = True
            result["message"] = f"Daily loss threshold exceeded: {daily_loss_pct:.2%}"
            
            self._create_breaker_event("daily_loss", "open", result["message"])
        
        # Check if circuit should be closed (reset at start of new day)
        elif (self.daily_loss_breaker == CircuitBreakerState.OPEN and 
              self._is_new_trading_day()):
            
            self.daily_loss_breaker = CircuitBreakerState.CLOSED
            result["state"] = "closed"
            result["message"] = "Daily loss circuit breaker reset for new trading day"
            
            self._create_breaker_event("daily_loss", "closed", result["message"])
        
        return result
    
    def _check_consecutive_loss_breaker(self) -> Dict[str, Any]:
        """Check consecutive losses circuit breaker"""
        result = {
            "state": self.consecutive_loss_breaker.value,
            "threshold": self.config.max_consecutive_losses,
            "current_value": self.consecutive_loss_count,
            "triggered": False,
            "message": ""
        }
        
        # Check if circuit should be opened
        if (self.consecutive_loss_breaker == CircuitBreakerState.CLOSED and 
            self.consecutive_loss_count >= self.config.critical_consecutive_losses):
            
            self.consecutive_loss_breaker = CircuitBreakerState.OPEN
            self.breaker_timestamps["consecutive_loss"] = datetime.utcnow()
            result["state"] = "open"
            result["triggered"] = True
            result["message"] = f"Critical consecutive losses threshold exceeded: {self.consecutive_loss_count}"
            
            self._create_breaker_event("consecutive_loss", "open", result["message"])
            
        elif (self.consecutive_loss_breaker == CircuitBreakerState.CLOSED and 
              self.consecutive_loss_count >= self.config.max_consecutive_losses):
            
            self.consecutive_loss_breaker = CircuitBreakerState.OPEN
            self.breaker_timestamps["consecutive_loss"] = datetime.utcnow()
            result["state"] = "open"
            result["triggered"] = True
            result["message"] = f"Consecutive losses threshold exceeded: {self.consecutive_loss_count}"
            
            self._create_breaker_event("consecutive_loss", "open", result["message"])
        
        # Check if circuit should be closed (reset after successful trade)
        elif (self.consecutive_loss_breaker == CircuitBreakerState.OPEN and 
              self.consecutive_loss_count == 0):
            
            self.consecutive_loss_breaker = CircuitBreakerState.CLOSED
            result["state"] = "closed"
            result["message"] = "Consecutive losses circuit breaker reset after successful trade"
            
            self._create_breaker_event("consecutive_loss", "closed", result["message"])
        
        return result
    
    def _check_position_size_breaker(self, portfolio_data: PortfolioData) -> Dict[str, Any]:
        """Check position size circuit breaker"""
        # Calculate maximum position size
        max_position_size = 0.0
        total_exposure = 0.0
        
        for position in portfolio_data.positions:
            position_size_pct = abs(position.size * position.current_price) / portfolio_data.total_value
            max_position_size = max(max_position_size, position_size_pct)
            total_exposure += position_size_pct
        
        result = {
            "state": self.position_size_breaker.value,
            "max_position_threshold": self.config.max_position_size_threshold,
            "total_exposure_threshold": self.config.max_total_exposure_threshold,
            "max_position_size": max_position_size,
            "total_exposure": total_exposure,
            "triggered": False,
            "message": ""
        }
        
        # Check if circuit should be opened
        if (self.position_size_breaker == CircuitBreakerState.CLOSED and 
            (max_position_size >= self.config.max_position_size_threshold or 
             total_exposure >= self.config.max_total_exposure_threshold)):
            
            self.position_size_breaker = CircuitBreakerState.OPEN
            self.breaker_timestamps["position_size"] = datetime.utcnow()
            result["state"] = "open"
            result["triggered"] = True
            result["message"] = f"Position size threshold exceeded: max={max_position_size:.2%}, total={total_exposure:.2%}"
            
            self._create_breaker_event("position_size", "open", result["message"])
        
        # Check if circuit should be closed
        elif (self.position_size_breaker == CircuitBreakerState.OPEN and 
              max_position_size < self.config.max_position_size_threshold and 
              total_exposure < self.config.max_total_exposure_threshold):
            
            self.position_size_breaker = CircuitBreakerState.CLOSED
            result["state"] = "closed"
            result["message"] = "Position size within acceptable limits"
            
            self._create_breaker_event("position_size", "closed", result["message"])
        
        return result
    
    def _should_test_recovery(self, breaker_name: str) -> bool:
        """Check if circuit breaker should test recovery"""
        if breaker_name not in self.breaker_timestamps:
            return False
        
        time_since_open = datetime.utcnow() - self.breaker_timestamps[breaker_name]
        return time_since_open >= timedelta(minutes=self.config.open_duration_minutes)
    
    def _is_new_trading_day(self) -> bool:
        """Check if it's a new trading day (simplified)"""
        # This is a simplified check - in production, you'd check against actual trading calendar
        current_hour = datetime.utcnow().hour
        return current_hour == 0  # Assume new day starts at midnight UTC
    
    def _create_breaker_event(self, breaker_name: str, action: str, message: str):
        """Create a circuit breaker event"""
        event = RiskEvent(
            event_id=f"circuit_breaker_{breaker_name}_{datetime.utcnow().timestamp()}",
            event_type="circuit_breaker",
            risk_level=RiskLevel.HIGH if action == "open" else RiskLevel.LOW,
            description=f"Circuit breaker {breaker_name} {action}: {message}",
            timestamp=datetime.utcnow(),
            source="circuit_breaker",
            data={
                "breaker_name": breaker_name,
                "action": action,
                "message": message
            }
        )
        
        self.breaker_events.append(event)
        
        # Trigger callbacks
        if action == "open" and self.on_breaker_open:
            self.on_breaker_open(breaker_name, event)
        elif action == "closed" and self.on_breaker_close:
            self.on_breaker_close(breaker_name, event)
        elif action == "half_open" and self.on_recovery_test:
            self.on_recovery_test(breaker_name, event)
    
    def _get_overall_state(self) -> str:
        """Get overall circuit breaker state"""
        if any(state == CircuitBreakerState.OPEN for state in self.breaker_states.values()):
            return "open"
        elif any(state == CircuitBreakerState.HALF_OPEN for state in self.breaker_states.values()):
            return "half_open"
        else:
            return "closed"
    
    def _generate_recommendations(self, breaker_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on circuit breaker states"""
        recommendations = []
        
        for breaker_name, result in breaker_results.items():
            if result["state"] == "open":
                if breaker_name == "drawdown":
                    recommendations.append("Reduce position sizes and consider closing losing positions")
                elif breaker_name == "daily_loss":
                    recommendations.append("Suspend new trades for the remainder of the day")
                elif breaker_name == "consecutive_loss":
                    recommendations.append("Review trading strategy and reduce position sizes")
                elif breaker_name == "position_size":
                    recommendations.append("Reduce position sizes to within acceptable limits")
            elif result["state"] == "half_open":
                recommendations.append(f"Test recovery conditions for {breaker_name} circuit breaker")
        
        return recommendations
    
    def _determine_actions(self, breaker_results: Dict[str, Any]) -> List[RiskAction]:
        """Determine required actions based on circuit breaker states"""
        actions = []
        
        for breaker_name, result in breaker_results.items():
            if result["state"] == "open":
                if breaker_name == "drawdown":
                    actions.append(RiskAction.REDUCE_POSITION)
                elif breaker_name == "daily_loss":
                    actions.append(RiskAction.SUSPEND_NEW_TRADES)
                elif breaker_name == "consecutive_loss":
                    actions.append(RiskAction.REDUCE_POSITION)
                elif breaker_name == "position_size":
                    actions.append(RiskAction.REDUCE_POSITION)
        
        # If multiple critical breakers are open, consider emergency stop
        open_breakers = sum(1 for result in breaker_results.values() if result["state"] == "open")
        if open_breakers >= 2:
            actions.append(RiskAction.EMERGENCY_STOP)
        
        return list(set(actions))  # Remove duplicates
    
    def get_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "overall_state": self._get_overall_state(),
            "breakers": {
                "drawdown": self.drawdown_breaker.value,
                "daily_loss": self.daily_loss_breaker.value,
                "consecutive_loss": self.consecutive_loss_breaker.value,
                "position_size": self.position_size_breaker.value
            },
            "consecutive_losses": self.consecutive_loss_count,
            "events_generated": len(self.breaker_events),
            "last_check": datetime.utcnow().isoformat()
        }
    
    def reset_breaker(self, breaker_name: str):
        """Manually reset a circuit breaker"""
        if breaker_name in self.breaker_states:
            self.breaker_states[breaker_name] = CircuitBreakerState.CLOSED
            if breaker_name in self.breaker_timestamps:
                del self.breaker_timestamps[breaker_name]
            
            self._create_breaker_event(breaker_name, "manual_reset", "Circuit breaker manually reset")
            self.logger.info(f"Circuit breaker {breaker_name} manually reset")
    
    def reset_all_breakers(self):
        """Reset all circuit breakers"""
        for breaker_name in self.breaker_states:
            self.reset_breaker(breaker_name)
        
        self.logger.info("All circuit breakers reset")

