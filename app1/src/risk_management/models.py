#!/usr/bin/env python3
"""
Risk Management Models

Data models and enums for the risk management and event avoidance system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventImpact(Enum):
    """Economic event impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SentimentLevel(Enum):
    """Market sentiment levels"""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class TradingState(Enum):
    """Trading system states"""
    ACTIVE = "active"
    REDUCED = "reduced"
    SUSPENDED = "suspended"
    EMERGENCY_STOP = "emergency_stop"


class RiskAction(Enum):
    """Risk management actions"""
    NONE = "none"
    REDUCE_POSITION = "reduce_position"
    SUSPEND_NEW_TRADES = "suspend_new_trades"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskThreshold:
    """Risk threshold configuration"""
    name: str
    threshold_value: float
    risk_level: RiskLevel
    action: RiskAction
    enabled: bool = True
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None
    
    def is_triggered(self, current_value: float) -> bool:
        """Check if threshold is triggered"""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                return False
        
        return current_value >= self.threshold_value
    
    def trigger(self):
        """Mark threshold as triggered"""
        self.last_triggered = datetime.utcnow()


@dataclass
class RiskEvent:
    """Risk event data model"""
    event_id: str
    event_type: str
    risk_level: RiskLevel
    description: str
    timestamp: datetime
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'risk_level': self.risk_level.value,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class EconomicEventData:
    """Economic event data for risk assessment"""
    event_id: str
    title: str
    event_time: datetime
    impact: EventImpact
    currency: str
    currency_pairs: List[str]
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    country: Optional[str] = None
    category: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class SentimentData:
    """Sentiment data for risk assessment"""
    currency_pair: str
    sentiment_level: SentimentLevel
    sentiment_score: float
    confidence: float
    timestamp: datetime
    source: str
    factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionData:
    """Position data for risk assessment"""
    currency_pair: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    risk_value: float = 0.0


@dataclass
class PortfolioData:
    """Portfolio data for risk assessment"""
    total_value: float
    available_margin: float
    used_margin: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    max_drawdown: float
    current_drawdown: float
    positions: List[PositionData] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskMetrics:
    """Risk metrics data model"""
    portfolio_value: float
    total_risk: float
    risk_per_trade: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    var_99: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_consecutive_losses: int
    win_rate: float
    profit_factor: float
    recovery_factor: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'portfolio_value': self.portfolio_value,
            'total_risk': self.total_risk,
            'risk_per_trade': self.risk_per_trade,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_consecutive_losses': self.max_consecutive_losses,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'recovery_factor': self.recovery_factor,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RiskAlert:
    """Risk alert data model"""
    alert_id: str
    alert_type: str
    severity: RiskLevel
    message: str
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'data': self.data
        }


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # General settings
    enabled: bool = True
    update_interval_seconds: int = 60
    max_events_history: int = 1000
    
    # Risk thresholds
    max_drawdown_threshold: float = 0.15  # 15%
    max_risk_per_trade: float = 0.02  # 2%
    max_portfolio_risk: float = 0.10  # 10%
    max_position_size: float = 0.20  # 20%
    
    # Event-based controls
    high_impact_event_reduction: float = 0.5  # 50% position reduction
    critical_event_suspension: bool = True
    event_lookahead_hours: int = 24
    
    # Sentiment-based controls
    bearish_sentiment_reduction: float = 0.3  # 30% position reduction
    very_bearish_sentiment_suspension: bool = True
    sentiment_threshold: float = -0.7  # -0.7 sentiment score
    
    # Circuit breakers
    max_daily_loss: float = 0.05  # 5% daily loss
    max_consecutive_losses: int = 5
    emergency_stop_threshold: float = 0.20  # 20% loss
    
    # Recovery settings
    recovery_threshold: float = 0.05  # 5% recovery
    min_recovery_time_hours: int = 24
    gradual_recovery: bool = True
    
    # Alerting
    enable_alerts: bool = True
    alert_cooldown_minutes: int = 15
    critical_alert_immediate: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'enabled': self.enabled,
            'update_interval_seconds': self.update_interval_seconds,
            'max_events_history': self.max_events_history,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_position_size': self.max_position_size,
            'high_impact_event_reduction': self.high_impact_event_reduction,
            'critical_event_suspension': self.critical_event_suspension,
            'event_lookahead_hours': self.event_lookahead_hours,
            'bearish_sentiment_reduction': self.bearish_sentiment_reduction,
            'very_bearish_sentiment_suspension': self.very_bearish_sentiment_suspension,
            'sentiment_threshold': self.sentiment_threshold,
            'max_daily_loss': self.max_daily_loss,
            'max_consecutive_losses': self.max_consecutive_losses,
            'emergency_stop_threshold': self.emergency_stop_threshold,
            'recovery_threshold': self.recovery_threshold,
            'min_recovery_time_hours': self.min_recovery_time_hours,
            'gradual_recovery': self.gradual_recovery,
            'enable_alerts': self.enable_alerts,
            'alert_cooldown_minutes': self.alert_cooldown_minutes,
            'critical_alert_immediate': self.critical_alert_immediate
        }


@dataclass
class RiskState:
    """Current risk management state"""
    trading_state: TradingState
    current_risk_level: RiskLevel
    active_events: List[RiskEvent] = field(default_factory=list)
    active_alerts: List[RiskAlert] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.utcnow)
    risk_metrics: Optional[RiskMetrics] = None
    portfolio_data: Optional[PortfolioData] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'trading_state': self.trading_state.value,
            'current_risk_level': self.current_risk_level.value,
            'active_events': [event.to_dict() for event in self.active_events],
            'active_alerts': [alert.to_dict() for alert in self.active_alerts],
            'last_update': self.last_update.isoformat(),
            'risk_metrics': self.risk_metrics.to_dict() if self.risk_metrics else None,
            'portfolio_data': self.portfolio_data.__dict__ if self.portfolio_data else None
        }

