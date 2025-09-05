#!/usr/bin/env python3
"""
Risk Manager

Main risk management orchestrator that integrates all risk management components
and provides a unified interface for risk assessment, monitoring, and control.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from ..models import (
    RiskConfig, RiskState, RiskLevel, RiskEvent, RiskAction, TradingState,
    PortfolioData, RiskMetrics, EconomicEventData, SentimentData
)
from .risk_engine import RiskEngine, RiskEngineConfig
from ..event_integration.event_monitor import EventMonitor, EventMonitorConfig
from ..event_integration.sentiment_monitor import SentimentMonitor, SentimentMonitorConfig
from ..controls.circuit_breakers import CircuitBreaker, CircuitBreakerConfig
from ..monitoring.risk_dashboard import RiskDashboard, RiskDashboardConfig
from ..monitoring.alerting import RiskAlerting, AlertingConfig

logger = logging.getLogger(__name__)


@dataclass
class RiskManagerConfig:
    """Configuration for the risk manager"""
    # Core configuration
    risk_config: RiskConfig = None
    risk_engine_config: RiskEngineConfig = None
    event_monitor_config: EventMonitorConfig = None
    sentiment_monitor_config: SentimentMonitorConfig = None
    circuit_breaker_config: CircuitBreakerConfig = None
    dashboard_config: RiskDashboardConfig = None
    alerting_config: AlertingConfig = None
    
    # Integration settings
    enable_event_monitoring: bool = True
    enable_sentiment_monitoring: bool = True
    enable_circuit_breakers: bool = True
    enable_dashboard: bool = True
    enable_alerting: bool = True
    
    # Update intervals
    risk_assessment_interval: int = 60  # seconds
    monitoring_interval: int = 300  # seconds
    dashboard_update_interval: int = 30  # seconds
    
    # External service integration
    economic_calendar_service: Optional[Any] = None
    sentiment_service: Optional[Any] = None
    trading_service: Optional[Any] = None
    
    def __post_init__(self):
        if self.risk_config is None:
            self.risk_config = RiskConfig()
        if self.risk_engine_config is None:
            self.risk_engine_config = RiskEngineConfig()
        if self.event_monitor_config is None:
            self.event_monitor_config = EventMonitorConfig()
        if self.sentiment_monitor_config is None:
            self.sentiment_monitor_config = SentimentMonitorConfig()
        if self.circuit_breaker_config is None:
            self.circuit_breaker_config = CircuitBreakerConfig()
        if self.dashboard_config is None:
            self.dashboard_config = RiskDashboardConfig()
        if self.alerting_config is None:
            self.alerting_config = AlertingConfig()


class RiskManager:
    """
    Main risk management orchestrator.
    
    Coordinates all risk management components to provide:
    - Comprehensive risk assessment
    - Real-time monitoring and alerting
    - Automatic risk controls and circuit breakers
    - Integration with economic calendar and sentiment analysis
    - Risk dashboard and reporting
    """
    
    def __init__(self, config: RiskManagerConfig = None):
        """Initialize risk manager"""
        self.config = config or RiskManagerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.risk_engine = RiskEngine(self.config.risk_engine_config)
        
        self.event_monitor = None
        if self.config.enable_event_monitoring:
            self.event_monitor = EventMonitor(self.config.event_monitor_config)
        
        self.sentiment_monitor = None
        if self.config.enable_sentiment_monitoring:
            self.sentiment_monitor = SentimentMonitor(self.config.sentiment_monitor_config)
        
        self.circuit_breaker = None
        if self.config.enable_circuit_breakers:
            self.circuit_breaker = CircuitBreaker(self.config.circuit_breaker_config)
        
        self.dashboard = None
        if self.config.enable_dashboard:
            self.dashboard = RiskDashboard(self.config.dashboard_config)
        
        self.alerting = None
        if self.config.enable_alerting:
            self.alerting = RiskAlerting(self.config.alerting_config)
        
        # State tracking
        self.current_state = RiskState(
            trading_state=TradingState.ACTIVE,
            current_risk_level=RiskLevel.LOW
        )
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._dashboard_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Performance tracking
        self.risk_assessments_count = 0
        self.alerts_generated = 0
        self.actions_taken = 0
        
        # Callbacks
        self.on_risk_level_change: Optional[Callable] = None
        self.on_trading_state_change: Optional[Callable] = None
        self.on_emergency_stop: Optional[Callable] = None
        
        self.logger.info("RiskManager initialized")
    
    async def start(self):
        """Start the risk management system"""
        if self._is_running:
            self.logger.warning("Risk management system already running")
            return
        
        self.logger.info("Starting risk management system")
        self._is_running = True
        
        try:
            # Start monitoring tasks
            tasks = []
            
            if self.event_monitor:
                task = asyncio.create_task(
                    self.event_monitor.start_monitoring(self.config.economic_calendar_service)
                )
                tasks.append(task)
            
            if self.sentiment_monitor:
                task = asyncio.create_task(
                    self.sentiment_monitor.start_monitoring(self.config.sentiment_service)
                )
                tasks.append(task)
            
            # Start main monitoring loop
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            tasks.append(self._monitoring_task)
            
            # Start dashboard updates
            if self.dashboard:
                self._dashboard_task = asyncio.create_task(self._dashboard_loop())
                tasks.append(self._dashboard_task)
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
            
        except asyncio.CancelledError:
            self.logger.info("Risk management system stopped")
        except Exception as e:
            self.logger.error(f"Error in risk management system: {e}")
        finally:
            self._is_running = False
    
    async def stop(self):
        """Stop the risk management system"""
        self.logger.info("Stopping risk management system")
        self._is_running = False
        
        # Cancel monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        if self._dashboard_task:
            self._dashboard_task.cancel()
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._is_running:
            try:
                await self._perform_risk_assessment()
                await asyncio.sleep(self.config.risk_assessment_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _dashboard_loop(self):
        """Dashboard update loop"""
        while self._is_running:
            try:
                if self.dashboard:
                    await self.dashboard.update_dashboard(self.current_state)
                await asyncio.sleep(self.config.dashboard_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in dashboard loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _perform_risk_assessment(self):
        """Perform comprehensive risk assessment"""
        try:
            # Get current portfolio data (this would come from trading service)
            portfolio_data = await self._get_current_portfolio_data()
            
            # Get economic events
            economic_events = []
            if self.event_monitor:
                economic_events = self.event_monitor.get_active_events()
            
            # Get sentiment data
            sentiment_data = []
            if self.sentiment_monitor:
                sentiment_data = list(self.sentiment_monitor.get_current_sentiment().values())
            
            # Perform risk assessment
            risk_state = await self.risk_engine.assess_risk(
                portfolio_data, economic_events, sentiment_data
            )
            
            # Check circuit breakers
            if self.circuit_breaker and risk_state.risk_metrics:
                breaker_results = self.circuit_breaker.check_circuit_breakers(
                    portfolio_data, risk_state.risk_metrics
                )
                
                # Update risk state based on circuit breaker results
                if breaker_results["overall_state"] == "open":
                    risk_state.trading_state = TradingState.SUSPENDED
                    risk_state.current_risk_level = RiskLevel.HIGH
                
                # Add circuit breaker events
                risk_state.active_events.extend(self.circuit_breaker.breaker_events)
            
            # Update current state
            await self._update_risk_state(risk_state)
            
            # Generate alerts
            await self._generate_alerts(risk_state)
            
            # Take actions
            await self._execute_actions(risk_state)
            
            self.risk_assessments_count += 1
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
    
    async def _get_current_portfolio_data(self) -> PortfolioData:
        """Get current portfolio data from trading service"""
        # This would integrate with the actual trading service
        # For now, return mock data
        from ..models import PositionData
        
        mock_positions = [
            PositionData(
                currency_pair="EUR/USD",
                size=10000,
                entry_price=1.0850,
                current_price=1.0820,
                unrealized_pnl=-300,
                realized_pnl=0,
                timestamp=datetime.utcnow()
            ),
            PositionData(
                currency_pair="GBP/USD",
                size=5000,
                entry_price=1.2650,
                current_price=1.2620,
                unrealized_pnl=-150,
                realized_pnl=0,
                timestamp=datetime.utcnow()
            )
        ]
        
        total_value = 100000
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in mock_positions)
        
        return PortfolioData(
            total_value=total_value,
            available_margin=total_value * 0.5,
            used_margin=total_value * 0.1,
            total_pnl=total_pnl,
            unrealized_pnl=sum(p.unrealized_pnl for p in mock_positions),
            realized_pnl=sum(p.realized_pnl for p in mock_positions),
            max_drawdown=0.05,
            current_drawdown=abs(total_pnl) / total_value,
            positions=mock_positions
        )
    
    async def _update_risk_state(self, new_state: RiskState):
        """Update current risk state and trigger callbacks"""
        old_risk_level = self.current_state.current_risk_level
        old_trading_state = self.current_state.trading_state
        
        self.current_state = new_state
        
        # Trigger callbacks for significant changes
        if old_risk_level != new_state.current_risk_level:
            if self.on_risk_level_change:
                await self.on_risk_level_change(old_risk_level, new_state.current_risk_level)
        
        if old_trading_state != new_state.trading_state:
            if self.on_trading_state_change:
                await self.on_trading_state_change(old_trading_state, new_state.trading_state)
            
            # Special handling for emergency stop
            if new_state.trading_state == TradingState.EMERGENCY_STOP:
                if self.on_emergency_stop:
                    await self.on_emergency_stop(new_state)
    
    async def _generate_alerts(self, risk_state: RiskState):
        """Generate risk alerts"""
        if not self.alerting:
            return
        
        # Generate alerts for high-risk events
        for event in risk_state.active_events:
            if event.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                await self.alerting.create_alert(
                    alert_type="risk_event",
                    severity=event.risk_level,
                    message=event.description,
                    data=event.data
                )
                self.alerts_generated += 1
        
        # Generate alerts for trading state changes
        if risk_state.trading_state in [TradingState.SUSPENDED, TradingState.EMERGENCY_STOP]:
            await self.alerting.create_alert(
                alert_type="trading_state_change",
                severity=RiskLevel.HIGH,
                message=f"Trading state changed to {risk_state.trading_state.value}",
                data={"trading_state": risk_state.trading_state.value}
            )
            self.alerts_generated += 1
    
    async def _execute_actions(self, risk_state: RiskState):
        """Execute risk management actions"""
        # This would integrate with the actual trading service
        # For now, just log the actions
        
        if risk_state.trading_state == TradingState.EMERGENCY_STOP:
            self.logger.critical("EMERGENCY STOP: All trading suspended")
            self.actions_taken += 1
        
        elif risk_state.trading_state == TradingState.SUSPENDED:
            self.logger.warning("Trading suspended due to risk conditions")
            self.actions_taken += 1
        
        elif risk_state.trading_state == TradingState.REDUCED:
            self.logger.info("Position sizes reduced due to risk conditions")
            self.actions_taken += 1
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        summary = {
            "current_state": self.current_state.to_dict(),
            "risk_engine_summary": self.risk_engine.get_risk_summary(),
            "performance": {
                "risk_assessments_count": self.risk_assessments_count,
                "alerts_generated": self.alerts_generated,
                "actions_taken": self.actions_taken,
                "is_running": self._is_running
            }
        }
        
        # Add component summaries
        if self.event_monitor:
            summary["event_monitor"] = self.event_monitor.get_event_risk_summary()
        
        if self.sentiment_monitor:
            summary["sentiment_monitor"] = self.sentiment_monitor.get_sentiment_summary()
        
        if self.circuit_breaker:
            summary["circuit_breakers"] = self.circuit_breaker.get_breaker_status()
        
        return summary
    
    async def force_risk_assessment(self) -> RiskState:
        """Force an immediate risk assessment"""
        await self._perform_risk_assessment()
        return self.current_state
    
    async def get_active_events(self) -> List[RiskEvent]:
        """Get all active risk events"""
        return self.current_state.active_events.copy()
    
    async def get_active_alerts(self) -> List[Any]:
        """Get all active alerts"""
        if self.alerting:
            return await self.alerting.get_active_alerts()
        return []
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge a risk alert"""
        if self.alerting:
            await self.alerting.acknowledge_alert(alert_id, acknowledged_by)
    
    async def reset_circuit_breakers(self):
        """Reset all circuit breakers"""
        if self.circuit_breaker:
            self.circuit_breaker.reset_all_breakers()
            self.logger.info("All circuit breakers reset")
    
    async def set_risk_config(self, config: RiskConfig):
        """Update risk configuration"""
        self.config.risk_config = config
        self.logger.info("Risk configuration updated")
    
    def is_running(self) -> bool:
        """Check if risk management system is running"""
        return self._is_running
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "is_running": self._is_running,
            "components": {
                "event_monitor": self.event_monitor is not None,
                "sentiment_monitor": self.sentiment_monitor is not None,
                "circuit_breaker": self.circuit_breaker is not None,
                "dashboard": self.dashboard is not None,
                "alerting": self.alerting is not None
            },
            "current_risk_level": self.current_state.current_risk_level.value,
            "trading_state": self.current_state.trading_state.value,
            "last_update": self.current_state.last_update.isoformat()
        }
