#!/usr/bin/env python3
"""
Risk Dashboard

Real-time risk monitoring dashboard that provides comprehensive
visualization of risk metrics, alerts, and system status.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..models import RiskState, RiskLevel, RiskEvent, RiskAlert, RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class RiskDashboardConfig:
    """Configuration for risk dashboard"""
    # Update intervals
    update_interval_seconds: int = 30
    data_retention_hours: int = 24
    
    # Display settings
    max_events_display: int = 50
    max_alerts_display: int = 20
    max_metrics_history: int = 100
    
    # Alert thresholds for dashboard highlighting
    high_risk_threshold: float = 0.10  # 10%
    critical_risk_threshold: float = 0.15  # 15%
    
    # Dashboard features
    enable_real_time_updates: bool = True
    enable_historical_charts: bool = True
    enable_alert_notifications: bool = True


class RiskDashboard:
    """
    Real-time risk monitoring dashboard.
    
    Provides comprehensive visualization of:
    - Current risk metrics and portfolio status
    - Active risk events and alerts
    - Historical risk trends and performance
    - Circuit breaker status
    - Economic events and sentiment data
    """
    
    def __init__(self, config: RiskDashboardConfig = None):
        """Initialize risk dashboard"""
        self.config = config or RiskDashboardConfig()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.risk_metrics_history: List[RiskMetrics] = []
        self.risk_events_history: List[RiskEvent] = []
        self.alerts_history: List[RiskAlert] = []
        
        # Current state
        self.current_state: Optional[RiskState] = None
        self.last_update = datetime.utcnow()
        
        # Dashboard data
        self.dashboard_data: Dict[str, Any] = {}
        
        # Performance tracking
        self.updates_count = 0
        self.data_points_collected = 0
        
        self.logger.info("RiskDashboard initialized")
    
    async def update_dashboard(self, risk_state: RiskState):
        """Update dashboard with current risk state"""
        try:
            self.current_state = risk_state
            self.last_update = datetime.utcnow()
            
            # Update metrics history
            if risk_state.risk_metrics:
                self.risk_metrics_history.append(risk_state.risk_metrics)
                self._trim_metrics_history()
            
            # Update events history
            self.risk_events_history.extend(risk_state.active_events)
            self._trim_events_history()
            
            # Generate dashboard data
            await self._generate_dashboard_data()
            
            self.updates_count += 1
            self.data_points_collected += len(risk_state.active_events)
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard: {e}")
    
    def _trim_metrics_history(self):
        """Trim metrics history to keep only recent data"""
        if len(self.risk_metrics_history) > self.config.max_metrics_history:
            self.risk_metrics_history = self.risk_metrics_history[-self.config.max_metrics_history:]
    
    def _trim_events_history(self):
        """Trim events history to keep only recent data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.data_retention_hours)
        
        self.risk_events_history = [
            event for event in self.risk_events_history
            if event.timestamp > cutoff_time
        ]
        
        # Keep only most recent events for display
        if len(self.risk_events_history) > self.config.max_events_display:
            self.risk_events_history = self.risk_events_history[-self.config.max_events_display:]
    
    async def _generate_dashboard_data(self):
        """Generate comprehensive dashboard data"""
        if not self.current_state:
            return
        
        # Basic status
        self.dashboard_data = {
            "timestamp": self.last_update.isoformat(),
            "risk_level": self.current_state.current_risk_level.value,
            "trading_state": self.current_state.trading_state.value,
            "last_update": self.current_state.last_update.isoformat()
        }
        
        # Risk metrics
        if self.current_state.risk_metrics:
            self.dashboard_data["risk_metrics"] = self._format_risk_metrics(self.current_state.risk_metrics)
        
        # Portfolio data
        if self.current_state.portfolio_data:
            self.dashboard_data["portfolio"] = self._format_portfolio_data(self.current_state.portfolio_data)
        
        # Active events
        self.dashboard_data["active_events"] = self._format_active_events(self.current_state.active_events)
        
        # Active alerts
        self.dashboard_data["active_alerts"] = self._format_active_alerts(self.current_state.active_alerts)
        
        # Risk trends
        self.dashboard_data["risk_trends"] = self._calculate_risk_trends()
        
        # Performance summary
        self.dashboard_data["performance"] = self._calculate_performance_summary()
        
        # System status
        self.dashboard_data["system_status"] = self._get_system_status()
    
    def _format_risk_metrics(self, metrics: RiskMetrics) -> Dict[str, Any]:
        """Format risk metrics for dashboard display"""
        return {
            "portfolio_value": {
                "value": metrics.portfolio_value,
                "formatted": f"${metrics.portfolio_value:,.2f}",
                "status": "normal"
            },
            "current_drawdown": {
                "value": metrics.current_drawdown,
                "formatted": f"{metrics.current_drawdown:.2%}",
                "status": self._get_drawdown_status(metrics.current_drawdown)
            },
            "max_drawdown": {
                "value": metrics.max_drawdown,
                "formatted": f"{metrics.max_drawdown:.2%}",
                "status": self._get_drawdown_status(metrics.max_drawdown)
            },
            "var_95": {
                "value": metrics.var_95,
                "formatted": f"{metrics.var_95:.2%}",
                "status": self._get_var_status(metrics.var_95)
            },
            "sharpe_ratio": {
                "value": metrics.sharpe_ratio,
                "formatted": f"{metrics.sharpe_ratio:.2f}",
                "status": self._get_sharpe_status(metrics.sharpe_ratio)
            },
            "win_rate": {
                "value": metrics.win_rate,
                "formatted": f"{metrics.win_rate:.1%}",
                "status": self._get_win_rate_status(metrics.win_rate)
            },
            "profit_factor": {
                "value": metrics.profit_factor,
                "formatted": f"{metrics.profit_factor:.2f}",
                "status": self._get_profit_factor_status(metrics.profit_factor)
            }
        }
    
    def _format_portfolio_data(self, portfolio_data) -> Dict[str, Any]:
        """Format portfolio data for dashboard display"""
        return {
            "total_value": {
                "value": portfolio_data.total_value,
                "formatted": f"${portfolio_data.total_value:,.2f}"
            },
            "total_pnl": {
                "value": portfolio_data.total_pnl,
                "formatted": f"${portfolio_data.total_pnl:,.2f}",
                "status": "positive" if portfolio_data.total_pnl >= 0 else "negative"
            },
            "unrealized_pnl": {
                "value": portfolio_data.unrealized_pnl,
                "formatted": f"${portfolio_data.unrealized_pnl:,.2f}",
                "status": "positive" if portfolio_data.unrealized_pnl >= 0 else "negative"
            },
            "realized_pnl": {
                "value": portfolio_data.realized_pnl,
                "formatted": f"${portfolio_data.realized_pnl:,.2f}",
                "status": "positive" if portfolio_data.realized_pnl >= 0 else "negative"
            },
            "positions_count": len(portfolio_data.positions),
            "margin_usage": {
                "used": portfolio_data.used_margin,
                "available": portfolio_data.available_margin,
                "percentage": (portfolio_data.used_margin / portfolio_data.total_value) * 100 if portfolio_data.total_value > 0 else 0
            }
        }
    
    def _format_active_events(self, events: List[RiskEvent]) -> List[Dict[str, Any]]:
        """Format active events for dashboard display"""
        formatted_events = []
        
        for event in events[-self.config.max_events_display:]:  # Show most recent events
            formatted_events.append({
                "id": event.event_id,
                "type": event.event_type,
                "risk_level": event.risk_level.value,
                "description": event.description,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "resolved": event.resolved,
                "age_minutes": (datetime.utcnow() - event.timestamp).total_seconds() / 60
            })
        
        return formatted_events
    
    def _format_active_alerts(self, alerts: List[RiskAlert]) -> List[Dict[str, Any]]:
        """Format active alerts for dashboard display"""
        formatted_alerts = []
        
        for alert in alerts[-self.config.max_alerts_display:]:  # Show most recent alerts
            formatted_alerts.append({
                "id": alert.alert_id,
                "type": alert.alert_type,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged,
                "acknowledged_by": alert.acknowledged_by,
                "age_minutes": (datetime.utcnow() - alert.timestamp).total_seconds() / 60
            })
        
        return formatted_alerts
    
    def _calculate_risk_trends(self) -> Dict[str, Any]:
        """Calculate risk trends from historical data"""
        if len(self.risk_metrics_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_metrics = self.risk_metrics_history[-10:]  # Last 10 data points
        
        # Calculate trends
        drawdown_trend = self._calculate_trend([m.current_drawdown for m in recent_metrics])
        var_trend = self._calculate_trend([m.var_95 for m in recent_metrics])
        sharpe_trend = self._calculate_trend([m.sharpe_ratio for m in recent_metrics])
        
        return {
            "drawdown_trend": drawdown_trend,
            "var_trend": var_trend,
            "sharpe_trend": sharpe_trend,
            "data_points": len(recent_metrics),
            "time_period_hours": (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 3600
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_pct = (second_avg - first_avg) / first_avg if first_avg != 0 else 0
        
        if change_pct > 0.05:  # 5% increase
            return "increasing"
        elif change_pct < -0.05:  # 5% decrease
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate performance summary"""
        return {
            "updates_count": self.updates_count,
            "data_points_collected": self.data_points_collected,
            "events_tracked": len(self.risk_events_history),
            "metrics_tracked": len(self.risk_metrics_history),
            "uptime_hours": (datetime.utcnow() - self.last_update).total_seconds() / 3600,
            "last_update": self.last_update.isoformat()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            "dashboard_running": True,
            "real_time_updates": self.config.enable_real_time_updates,
            "historical_charts": self.config.enable_historical_charts,
            "alert_notifications": self.config.enable_alert_notifications,
            "data_retention_hours": self.config.data_retention_hours,
            "update_interval_seconds": self.config.update_interval_seconds
        }
    
    def _get_drawdown_status(self, drawdown: float) -> str:
        """Get status color for drawdown value"""
        if drawdown >= self.config.critical_risk_threshold:
            return "critical"
        elif drawdown >= self.config.high_risk_threshold:
            return "high"
        else:
            return "normal"
    
    def _get_var_status(self, var: float) -> str:
        """Get status color for VaR value"""
        if var >= 0.05:  # 5% VaR
            return "high"
        elif var >= 0.03:  # 3% VaR
            return "medium"
        else:
            return "normal"
    
    def _get_sharpe_status(self, sharpe: float) -> str:
        """Get status color for Sharpe ratio"""
        if sharpe >= 2.0:
            return "excellent"
        elif sharpe >= 1.0:
            return "good"
        elif sharpe >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _get_win_rate_status(self, win_rate: float) -> str:
        """Get status color for win rate"""
        if win_rate >= 0.6:  # 60%
            return "excellent"
        elif win_rate >= 0.5:  # 50%
            return "good"
        elif win_rate >= 0.4:  # 40%
            return "fair"
        else:
            return "poor"
    
    def _get_profit_factor_status(self, profit_factor: float) -> str:
        """Get status color for profit factor"""
        if profit_factor >= 2.0:
            return "excellent"
        elif profit_factor >= 1.5:
            return "good"
        elif profit_factor >= 1.0:
            return "fair"
        else:
            return "poor"
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()
    
    def get_risk_metrics_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get risk metrics history for charts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        recent_metrics = [
            m for m in self.risk_metrics_history
            if m.timestamp > cutoff_time
        ]
        
        return [m.to_dict() for m in recent_metrics]
    
    def get_events_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get events history for charts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        recent_events = [
            e for e in self.risk_events_history
            if e.timestamp > cutoff_time
        ]
        
        return [e.to_dict() for e in recent_events]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dashboard summary"""
        return {
            "current_risk_level": self.current_state.current_risk_level.value if self.current_state else "unknown",
            "trading_state": self.current_state.trading_state.value if self.current_state else "unknown",
            "active_events_count": len(self.current_state.active_events) if self.current_state else 0,
            "active_alerts_count": len(self.current_state.active_alerts) if self.current_state else 0,
            "metrics_history_count": len(self.risk_metrics_history),
            "events_history_count": len(self.risk_events_history),
            "last_update": self.last_update.isoformat(),
            "updates_count": self.updates_count,
            "data_points_collected": self.data_points_collected
        }

