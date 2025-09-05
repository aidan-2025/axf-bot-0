"""
Trading Suspension System

Implements automated trading suspension mechanisms based on risk conditions,
market events, and system health.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

from ..models import RiskLevel, RiskEvent, PortfolioData, RiskMetrics, TradingState


class SuspensionReason(Enum):
    """Reasons for trading suspension"""
    RISK_THRESHOLD_BREACH = "risk_threshold_breach"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    HIGH_IMPACT_EVENT = "high_impact_event"
    SYSTEM_ERROR = "system_error"
    MANUAL_OVERRIDE = "manual_override"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"


class SuspensionLevel(Enum):
    """Levels of trading suspension"""
    NONE = "none"
    PARTIAL = "partial"  # Reduce position sizes
    FULL = "full"        # Stop all new trades
    EMERGENCY = "emergency"  # Close all positions


@dataclass
class SuspensionRule:
    """Rule for automatic trading suspension"""
    reason: SuspensionReason
    condition: Callable[[Dict[str, Any]], bool]
    suspension_level: SuspensionLevel
    duration_minutes: int = 60
    cooldown_minutes: int = 30
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority


@dataclass
class SuspensionConfig:
    """Configuration for trading suspension system"""
    # Risk-based suspension thresholds
    max_risk_threshold: float = 0.15
    critical_risk_threshold: float = 0.25
    
    # Drawdown thresholds
    max_drawdown_threshold: float = 0.10
    critical_drawdown_threshold: float = 0.20
    
    # Daily loss thresholds
    max_daily_loss_threshold: float = 0.05
    critical_daily_loss_threshold: float = 0.10
    
    # Consecutive loss thresholds
    max_consecutive_losses: int = 5
    critical_consecutive_losses: int = 8
    
    # Volatility thresholds
    max_volatility_threshold: float = 0.30
    critical_volatility_threshold: float = 0.50
    
    # Correlation thresholds
    max_correlation_threshold: float = 0.80
    critical_correlation_threshold: float = 0.95
    
    # Event impact thresholds
    high_impact_event_threshold: int = 1
    critical_impact_event_threshold: int = 1
    
    # System health thresholds
    max_system_errors_per_hour: int = 10
    max_latency_ms: int = 1000
    
    # Suspension rules
    suspension_rules: List[SuspensionRule] = field(default_factory=list)
    
    # Recovery settings
    auto_recovery_enabled: bool = True
    recovery_check_interval: int = 300  # 5 minutes
    min_recovery_time: int = 60  # 1 minute
    max_suspension_time: int = 1440  # 24 hours


@dataclass
class SuspensionEvent:
    """Record of a trading suspension event"""
    suspension_id: str
    reason: SuspensionReason
    level: SuspensionLevel
    timestamp: datetime
    duration_minutes: int
    triggered_by: str
    description: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    recovery_actions: List[str] = field(default_factory=list)


class TradingSuspensionManager:
    """Manages automated trading suspension and recovery"""
    
    def __init__(self, config: SuspensionConfig = None):
        self.config = config or SuspensionConfig()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.current_suspension: Optional[SuspensionEvent] = None
        self.suspension_history: List[SuspensionEvent] = []
        self.system_health: Dict[str, Any] = {}
        self.last_health_check: datetime = datetime.utcnow()
        
        # Initialize default suspension rules
        self._initialize_default_rules()
        
        # Recovery monitoring
        self.recovery_monitoring = False
        self.recovery_task: Optional[asyncio.Task] = None
    
    def _initialize_default_rules(self):
        """Initialize default suspension rules"""
        if not self.config.suspension_rules:
            self.config.suspension_rules = [
                SuspensionRule(
                    reason=SuspensionReason.RISK_THRESHOLD_BREACH,
                    condition=self._check_risk_threshold,
                    suspension_level=SuspensionLevel.PARTIAL,
                    duration_minutes=60,
                    priority=3
                ),
                SuspensionRule(
                    reason=SuspensionReason.RISK_THRESHOLD_BREACH,
                    condition=self._check_critical_risk_threshold,
                    suspension_level=SuspensionLevel.FULL,
                    duration_minutes=120,
                    priority=5
                ),
                SuspensionRule(
                    reason=SuspensionReason.DRAWDOWN_LIMIT,
                    condition=self._check_drawdown_limit,
                    suspension_level=SuspensionLevel.PARTIAL,
                    duration_minutes=90,
                    priority=4
                ),
                SuspensionRule(
                    reason=SuspensionReason.DRAWDOWN_LIMIT,
                    condition=self._check_critical_drawdown,
                    suspension_level=SuspensionLevel.EMERGENCY,
                    duration_minutes=180,
                    priority=6
                ),
                SuspensionRule(
                    reason=SuspensionReason.CONSECUTIVE_LOSSES,
                    condition=self._check_consecutive_losses,
                    suspension_level=SuspensionLevel.PARTIAL,
                    duration_minutes=60,
                    priority=3
                ),
                SuspensionRule(
                    reason=SuspensionReason.VOLATILITY_SPIKE,
                    condition=self._check_volatility_spike,
                    suspension_level=SuspensionLevel.PARTIAL,
                    duration_minutes=45,
                    priority=2
                ),
                SuspensionRule(
                    reason=SuspensionReason.CORRELATION_SPIKE,
                    condition=self._check_correlation_spike,
                    suspension_level=SuspensionLevel.PARTIAL,
                    duration_minutes=30,
                    priority=2
                ),
                SuspensionRule(
                    reason=SuspensionReason.HIGH_IMPACT_EVENT,
                    condition=self._check_high_impact_event,
                    suspension_level=SuspensionLevel.PARTIAL,
                    duration_minutes=120,
                    priority=4
                ),
                SuspensionRule(
                    reason=SuspensionReason.SYSTEM_ERROR,
                    condition=self._check_system_errors,
                    suspension_level=SuspensionLevel.FULL,
                    duration_minutes=30,
                    priority=5
                )
            ]
    
    def check_suspension_conditions(self, portfolio_data: PortfolioData,
                                  risk_metrics: RiskMetrics,
                                  market_data: Dict[str, Any] = None,
                                  active_events: List[RiskEvent] = None) -> Optional[SuspensionEvent]:
        """Check if trading should be suspended"""
        market_data = market_data or {}
        active_events = active_events or []
        
        # Update system health
        self._update_system_health(market_data)
        
        # Check suspension rules in priority order
        triggered_rules = []
        for rule in sorted(self.config.suspension_rules, key=lambda r: r.priority, reverse=True):
            if not rule.enabled:
                continue
            
            try:
                context = {
                    'portfolio_data': portfolio_data,
                    'risk_metrics': risk_metrics,
                    'market_data': market_data,
                    'active_events': active_events,
                    'system_health': self.system_health
                }
                
                if rule.condition(context):
                    triggered_rules.append(rule)
            except Exception as e:
                self.logger.error(f"Error checking suspension rule {rule.reason}: {e}")
        
        # If no rules triggered, check if we should recover
        if not triggered_rules:
            if self.current_suspension and self._should_recover():
                self._recover_from_suspension()
            return None
        
        # Get highest priority rule
        highest_priority_rule = max(triggered_rules, key=lambda r: r.priority)
        
        # Check if we're already suspended for this reason
        if (self.current_suspension and 
            self.current_suspension.reason == highest_priority_rule.reason and
            not self.current_suspension.resolved):
            return self.current_suspension
        
        # Create new suspension event
        suspension_event = self._create_suspension_event(highest_priority_rule, context)
        
        # Apply suspension
        self._apply_suspension(suspension_event)
        
        return suspension_event
    
    def _check_risk_threshold(self, context: Dict[str, Any]) -> bool:
        """Check if risk threshold is breached"""
        risk_metrics = context['risk_metrics']
        return risk_metrics.total_risk > self.config.max_risk_threshold
    
    def _check_critical_risk_threshold(self, context: Dict[str, Any]) -> bool:
        """Check if critical risk threshold is breached"""
        risk_metrics = context['risk_metrics']
        return risk_metrics.total_risk > self.config.critical_risk_threshold
    
    def _check_drawdown_limit(self, context: Dict[str, Any]) -> bool:
        """Check if drawdown limit is reached"""
        risk_metrics = context['risk_metrics']
        return risk_metrics.current_drawdown > self.config.max_drawdown_threshold
    
    def _check_critical_drawdown(self, context: Dict[str, Any]) -> bool:
        """Check if critical drawdown is reached"""
        risk_metrics = context['risk_metrics']
        return risk_metrics.current_drawdown > self.config.critical_drawdown_threshold
    
    def _check_consecutive_losses(self, context: Dict[str, Any]) -> bool:
        """Check if consecutive loss limit is reached"""
        risk_metrics = context['risk_metrics']
        return risk_metrics.max_consecutive_losses >= self.config.max_consecutive_losses
    
    def _check_volatility_spike(self, context: Dict[str, Any]) -> bool:
        """Check if volatility spike occurred"""
        market_data = context['market_data']
        volatility = market_data.get('volatility', 0.0)
        return volatility > self.config.max_volatility_threshold
    
    def _check_correlation_spike(self, context: Dict[str, Any]) -> bool:
        """Check if correlation spike occurred"""
        market_data = context['market_data']
        correlation = market_data.get('max_correlation', 0.0)
        return correlation > self.config.max_correlation_threshold
    
    def _check_high_impact_event(self, context: Dict[str, Any]) -> bool:
        """Check if high impact event is active"""
        active_events = context['active_events']
        high_impact_count = sum(1 for event in active_events if event.risk_level == RiskLevel.HIGH)
        return high_impact_count >= self.config.high_impact_event_threshold
    
    def _check_system_errors(self, context: Dict[str, Any]) -> bool:
        """Check if system error threshold is exceeded"""
        system_health = context['system_health']
        errors_per_hour = system_health.get('errors_per_hour', 0)
        return errors_per_hour > self.config.max_system_errors_per_hour
    
    def _create_suspension_event(self, rule: SuspensionRule, context: Dict[str, Any]) -> SuspensionEvent:
        """Create a new suspension event"""
        suspension_id = f"suspension_{datetime.utcnow().timestamp()}"
        
        # Generate description based on rule
        description = self._generate_suspension_description(rule, context)
        
        return SuspensionEvent(
            suspension_id=suspension_id,
            reason=rule.reason,
            level=rule.suspension_level,
            timestamp=datetime.utcnow(),
            duration_minutes=rule.duration_minutes,
            triggered_by=rule.reason.value,
            description=description
        )
    
    def _generate_suspension_description(self, rule: SuspensionRule, context: Dict[str, Any]) -> str:
        """Generate human-readable suspension description"""
        risk_metrics = context['risk_metrics']
        market_data = context['market_data']
        
        descriptions = {
            SuspensionReason.RISK_THRESHOLD_BREACH: f"Risk level {risk_metrics.total_risk:.2%} exceeds threshold {self.config.max_risk_threshold:.2%}",
            SuspensionReason.DRAWDOWN_LIMIT: f"Drawdown {risk_metrics.current_drawdown:.2%} exceeds limit {self.config.max_drawdown_threshold:.2%}",
            SuspensionReason.CONSECUTIVE_LOSSES: f"Consecutive losses {risk_metrics.max_consecutive_losses} exceeds limit {self.config.max_consecutive_losses}",
            SuspensionReason.VOLATILITY_SPIKE: f"Volatility {market_data.get('volatility', 0):.2%} exceeds threshold {self.config.max_volatility_threshold:.2%}",
            SuspensionReason.CORRELATION_SPIKE: f"Correlation {market_data.get('max_correlation', 0):.2%} exceeds threshold {self.config.max_correlation_threshold:.2%}",
            SuspensionReason.HIGH_IMPACT_EVENT: f"High impact events detected: {len([e for e in context['active_events'] if e.risk_level == RiskLevel.HIGH])}",
            SuspensionReason.SYSTEM_ERROR: f"System errors {self.system_health.get('errors_per_hour', 0)} exceed threshold {self.config.max_system_errors_per_hour}"
        }
        
        return descriptions.get(rule.reason, f"Suspension triggered by {rule.reason.value}")
    
    def _apply_suspension(self, suspension_event: SuspensionEvent):
        """Apply trading suspension"""
        self.current_suspension = suspension_event
        self.suspension_history.append(suspension_event)
        
        # Log suspension
        self.logger.warning(f"Trading suspended: {suspension_event.description}")
        
        # Start recovery monitoring if enabled
        if self.config.auto_recovery_enabled and not self.recovery_monitoring:
            self._start_recovery_monitoring()
    
    def _should_recover(self) -> bool:
        """Check if suspension should be recovered"""
        if not self.current_suspension or self.current_suspension.resolved:
            return False
        
        # Check if minimum recovery time has passed
        min_recovery_time = timedelta(minutes=self.config.min_recovery_time)
        if datetime.utcnow() - self.current_suspension.timestamp < min_recovery_time:
            return False
        
        # Check if maximum suspension time has been reached
        max_suspension_time = timedelta(minutes=self.config.max_suspension_time)
        if datetime.utcnow() - self.current_suspension.timestamp > max_suspension_time:
            self.logger.warning("Maximum suspension time reached, forcing recovery")
            return True
        
        # Check if suspension duration has passed
        suspension_duration = timedelta(minutes=self.current_suspension.duration_minutes)
        if datetime.utcnow() - self.current_suspension.timestamp > suspension_duration:
            return True
        
        return False
    
    def _recover_from_suspension(self):
        """Recover from trading suspension"""
        if not self.current_suspension or self.current_suspension.resolved:
            return
        
        # Mark suspension as resolved
        self.current_suspension.resolved = True
        self.current_suspension.resolved_at = datetime.utcnow()
        
        # Log recovery
        self.logger.info(f"Trading suspension recovered: {self.current_suspension.suspension_id}")
        
        # Clear current suspension
        self.current_suspension = None
        
        # Stop recovery monitoring
        self._stop_recovery_monitoring()
    
    def _start_recovery_monitoring(self):
        """Start recovery monitoring task"""
        if self.recovery_monitoring:
            return
        
        self.recovery_monitoring = True
        try:
            self.recovery_task = asyncio.create_task(self._recovery_monitor())
        except RuntimeError:
            # No event loop running (e.g., in test environment)
            # Recovery monitoring will be handled manually
            self.recovery_task = None
    
    def _stop_recovery_monitoring(self):
        """Stop recovery monitoring task"""
        self.recovery_monitoring = False
        if self.recovery_task:
            self.recovery_task.cancel()
            self.recovery_task = None
    
    async def _recovery_monitor(self):
        """Monitor for recovery conditions"""
        while self.recovery_monitoring:
            try:
                await asyncio.sleep(self.config.recovery_check_interval)
                
                if self._should_recover():
                    self._recover_from_suspension()
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in recovery monitoring: {e}")
    
    def _update_system_health(self, market_data: Dict[str, Any]):
        """Update system health metrics"""
        current_time = datetime.utcnow()
        
        # Update basic health metrics
        self.system_health.update({
            'last_update': current_time,
            'latency_ms': market_data.get('latency_ms', 0),
            'errors_per_hour': market_data.get('errors_per_hour', 0),
            'memory_usage': market_data.get('memory_usage', 0),
            'cpu_usage': market_data.get('cpu_usage', 0)
        })
        
        self.last_health_check = current_time
    
    def get_suspension_status(self) -> Dict[str, Any]:
        """Get current suspension status"""
        if not self.current_suspension:
            return {
                'suspended': False,
                'level': SuspensionLevel.NONE.value,
                'reason': None,
                'description': None,
                'timestamp': None,
                'duration_minutes': None
            }
        
        return {
            'suspended': True,
            'level': self.current_suspension.level.value,
            'reason': self.current_suspension.reason.value,
            'description': self.current_suspension.description,
            'timestamp': self.current_suspension.timestamp.isoformat(),
            'duration_minutes': self.current_suspension.duration_minutes,
            'resolved': self.current_suspension.resolved
        }
    
    def get_suspension_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get suspension history"""
        recent_suspensions = self.suspension_history[-limit:]
        
        return [
            {
                'suspension_id': s.suspension_id,
                'reason': s.reason.value,
                'level': s.level.value,
                'timestamp': s.timestamp.isoformat(),
                'duration_minutes': s.duration_minutes,
                'description': s.description,
                'resolved': s.resolved,
                'resolved_at': s.resolved_at.isoformat() if s.resolved_at else None
            }
            for s in recent_suspensions
        ]
    
    def manual_suspend(self, reason: str, level: SuspensionLevel, duration_minutes: int = 60) -> SuspensionEvent:
        """Manually suspend trading"""
        suspension_event = SuspensionEvent(
            suspension_id=f"manual_{datetime.utcnow().timestamp()}",
            reason=SuspensionReason.MANUAL_OVERRIDE,
            level=level,
            timestamp=datetime.utcnow(),
            duration_minutes=duration_minutes,
            triggered_by="manual",
            description=f"Manual suspension: {reason}"
        )
        
        self._apply_suspension(suspension_event)
        return suspension_event
    
    def manual_recover(self) -> bool:
        """Manually recover from suspension"""
        if not self.current_suspension or self.current_suspension.resolved:
            return False
        
        self._recover_from_suspension()
        return True
    
    def update_config(self, new_config: SuspensionConfig):
        """Update suspension configuration"""
        self.config = new_config
        self._initialize_default_rules()
        self.logger.info("Suspension configuration updated")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        return {
            'last_check': self.last_health_check.isoformat(),
            'health_metrics': self.system_health,
            'recovery_monitoring': self.recovery_monitoring
        }
