#!/usr/bin/env python3
"""
Failover Package
Advanced failover and redundancy mechanisms for data ingestion
"""

from .failover_manager import (
    FailoverManager,
    CircuitBreakerConfig,
    CircuitState,
    FailoverEvent,
    FailoverMetrics
)

from .data_continuity_manager import (
    DataContinuityManager,
    BufferConfig,
    DataType,
    BufferState,
    ContinuityMetrics
)

from .redundancy_manager import (
    RedundancyManager,
    RedundancyConfig,
    ValidationStrategy,
    DataSource,
    DataPoint,
    ValidationResult,
    RedundancyMetrics
)

from .health_monitoring_manager import (
    HealthMonitoringManager,
    HealthConfig,
    HealthStatus,
    AlertLevel,
    MetricType,
    HealthCheck,
    Metric,
    Alert,
    HealthMetrics
)

from .failover_orchestrator import (
    FailoverOrchestrator,
    FailoverOrchestratorConfig,
    OrchestratorStatus,
    OrchestratorMetrics
)

__all__ = [
    # Failover Manager
    'FailoverManager',
    'CircuitBreakerConfig',
    'CircuitState',
    'FailoverEvent',
    'FailoverMetrics',
    
    # Data Continuity Manager
    'DataContinuityManager',
    'BufferConfig',
    'DataType',
    'BufferState',
    'ContinuityMetrics',
    
    # Redundancy Manager
    'RedundancyManager',
    'RedundancyConfig',
    'ValidationStrategy',
    'DataSource',
    'DataPoint',
    'ValidationResult',
    'RedundancyMetrics',
    
    # Health Monitoring Manager
    'HealthMonitoringManager',
    'HealthConfig',
    'HealthStatus',
    'AlertLevel',
    'MetricType',
    'HealthCheck',
    'Metric',
    'Alert',
    'HealthMetrics',
    
    # Failover Orchestrator
    'FailoverOrchestrator',
    'FailoverOrchestratorConfig',
    'OrchestratorStatus',
    'OrchestratorMetrics'
]

