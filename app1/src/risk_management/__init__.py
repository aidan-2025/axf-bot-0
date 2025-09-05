#!/usr/bin/env python3
"""
Risk Management and Event Avoidance Package

Comprehensive risk management system that integrates with economic calendar
and sentiment analysis to provide automatic trading suspension, position
size reduction, and risk controls during high-impact events.
"""

from .core.risk_manager import RiskManager, RiskManagerConfig
from .core.risk_engine import RiskEngine, RiskEngineConfig
from .core.risk_metrics import RiskMetrics, RiskMetricsConfig
from .event_integration.event_monitor import EventMonitor, EventMonitorConfig
from .event_integration.sentiment_monitor import SentimentMonitor, SentimentMonitorConfig
from .controls.circuit_breakers import CircuitBreaker, CircuitBreakerConfig
from .monitoring.risk_dashboard import RiskDashboard, RiskDashboardConfig
from .monitoring.alerting import RiskAlerting, AlertingConfig
from .models import (
    RiskLevel, RiskEvent, RiskThreshold, RiskAction, 
    EventImpact, SentimentLevel, TradingState
)

__all__ = [
    # Core risk management
    'RiskManager',
    'RiskManagerConfig',
    'RiskEngine',
    'RiskEngineConfig',
    'RiskMetrics',
    'RiskMetricsConfig',
    
    # Event integration
    'EventMonitor',
    'EventMonitorConfig',
    'SentimentMonitor',
    'SentimentMonitorConfig',
    
    # Risk controls
    'CircuitBreaker',
    'CircuitBreakerConfig',
    
    # Monitoring and alerting
    'RiskDashboard',
    'RiskDashboardConfig',
    'RiskAlerting',
    'AlertingConfig',
    
    # Models
    'RiskLevel',
    'RiskEvent',
    'RiskThreshold',
    'RiskAction',
    'EventImpact',
    'SentimentLevel',
    'TradingState'
]

__version__ = "1.0.0"
__author__ = "AXF Bot Development Team"
__description__ = "Comprehensive risk management and event avoidance system for forex trading"
