#!/usr/bin/env python3
"""
Failover Orchestrator
Main orchestrator that integrates all failover and redundancy components
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import json

from .failover_manager import FailoverManager, CircuitBreakerConfig
from .data_continuity_manager import DataContinuityManager, BufferConfig, DataType
from .redundancy_manager import RedundancyManager, RedundancyConfig
from .health_monitoring_manager import HealthMonitoringManager, HealthConfig
from ..brokers.broker_manager import BrokerManager
from ..cache.redis_cache import RedisCacheManager
from ..storage.storage_manager import StorageManager
from ..config import CONFIG

logger = logging.getLogger(__name__)

class OrchestratorStatus(Enum):
    """Orchestrator status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class FailoverOrchestratorConfig:
    """Configuration for failover orchestrator"""
    enable_failover: bool = True
    enable_data_continuity: bool = True
    enable_redundancy: bool = True
    enable_health_monitoring: bool = True
    failover_timeout_seconds: int = 30
    data_continuity_timeout_seconds: int = 60
    redundancy_timeout_seconds: int = 10
    health_check_interval_seconds: int = 30
    enable_automatic_recovery: bool = True
    recovery_attempts: int = 3
    recovery_delay_seconds: int = 5

@dataclass
class OrchestratorMetrics:
    """Metrics for failover orchestrator"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    failover_events: int = 0
    data_replays: int = 0
    redundancy_validations: int = 0
    health_checks: int = 0
    avg_operation_time_ms: float = 0.0
    max_operation_time_ms: float = 0.0
    min_operation_time_ms: float = float('inf')
    last_operation: Optional[datetime] = None

class FailoverOrchestrator:
    """Main orchestrator for failover and redundancy mechanisms"""
    
    def __init__(self, 
                 broker_manager: BrokerManager,
                 cache_manager: Optional[RedisCacheManager] = None,
                 storage_manager: Optional[StorageManager] = None,
                 config: Optional[FailoverOrchestratorConfig] = None):
        """
        Initialize failover orchestrator
        
        Args:
            broker_manager: Broker manager instance
            cache_manager: Redis cache manager
            storage_manager: Storage manager
            config: Orchestrator configuration
        """
        self.broker_manager = broker_manager
        self.cache_manager = cache_manager
        self.storage_manager = storage_manager
        self.config = config or FailoverOrchestratorConfig()
        
        # Orchestrator state
        self.status = OrchestratorStatus.STOPPED
        self.metrics = OrchestratorMetrics()
        
        # Component managers
        self.failover_manager: Optional[FailoverManager] = None
        self.continuity_manager: Optional[DataContinuityManager] = None
        self.redundancy_manager: Optional[RedundancyManager] = None
        self.health_manager: Optional[HealthMonitoringManager] = None
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
        
        # Recovery state
        self.recovery_in_progress = False
        self.last_recovery_attempt: Optional[datetime] = None
    
    async def start(self) -> None:
        """Start the failover orchestrator"""
        if self.status != OrchestratorStatus.STOPPED:
            logger.warning("Orchestrator is already running or starting")
            return
        
        logger.info("Starting failover orchestrator...")
        self.status = OrchestratorStatus.STARTING
        
        try:
            # Initialize component managers
            await self._initialize_components()
            
            # Start component managers
            await self._start_components()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            self.status = OrchestratorStatus.RUNNING
            logger.info("Failover orchestrator started successfully")
            
        except Exception as e:
            self.status = OrchestratorStatus.ERROR
            logger.error(f"Failed to start orchestrator: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the failover orchestrator"""
        if self.status == OrchestratorStatus.STOPPED:
            return
        
        logger.info("Stopping failover orchestrator...")
        self.status = OrchestratorStatus.STOPPING
        
        try:
            # Stop component managers
            await self._stop_components()
            
            self.status = OrchestratorStatus.STOPPED
            logger.info("Failover orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")
            self.status = OrchestratorStatus.ERROR
    
    async def _initialize_components(self) -> None:
        """Initialize all component managers"""
        # Initialize failover manager
        if self.config.enable_failover:
            circuit_config = CircuitBreakerConfig()
            self.failover_manager = FailoverManager(
                self.broker_manager,
                self.cache_manager,
                circuit_config
            )
            logger.info("Failover manager initialized")
        
        # Initialize data continuity manager
        if self.config.enable_data_continuity:
            buffer_config = BufferConfig()
            self.continuity_manager = DataContinuityManager(
                self.cache_manager,
                self.storage_manager,
                buffer_config
            )
            logger.info("Data continuity manager initialized")
        
        # Initialize redundancy manager
        if self.config.enable_redundancy:
            redundancy_config = RedundancyConfig()
            self.redundancy_manager = RedundancyManager(
                self.broker_manager,
                self.cache_manager,
                redundancy_config
            )
            logger.info("Redundancy manager initialized")
        
        # Initialize health monitoring manager
        if self.config.enable_health_monitoring:
            health_config = HealthConfig()
            self.health_manager = HealthMonitoringManager(
                self.broker_manager,
                self.cache_manager,
                health_config
            )
            logger.info("Health monitoring manager initialized")
    
    async def _start_components(self) -> None:
        """Start all component managers"""
        # Start failover manager
        if self.failover_manager:
            await self.failover_manager.start()
        
        # Start data continuity manager
        if self.continuity_manager:
            await self.continuity_manager.start()
        
        # Start redundancy manager
        if self.redundancy_manager:
            # Redundancy manager doesn't have a start method
            pass
        
        # Start health monitoring manager
        if self.health_manager:
            await self.health_manager.start()
    
    async def _stop_components(self) -> None:
        """Stop all component managers"""
        # Stop failover manager
        if self.failover_manager:
            await self.failover_manager.stop()
        
        # Stop data continuity manager
        if self.continuity_manager:
            await self.continuity_manager.stop()
        
        # Stop health monitoring manager
        if self.health_manager:
            await self.health_manager.stop()
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers between components"""
        # Setup failover event handlers
        if self.failover_manager:
            self.failover_manager.add_event_callback(self._handle_failover_event)
        
        # Setup continuity event handlers
        if self.continuity_manager:
            self.continuity_manager.add_event_callback(self._handle_continuity_event)
        
        # Setup health monitoring event handlers
        if self.health_manager:
            self.health_manager.add_event_callback(self._handle_health_event)
    
    async def _handle_failover_event(self, event: Dict[str, Any]) -> None:
        """Handle failover events"""
        logger.info(f"Failover event: {event['type']}")
        
        # Update metrics
        self.metrics.failover_events += 1
        
        # Handle specific event types
        if event['type'] == 'failover_triggered':
            # Start data buffering during failover
            if self.continuity_manager:
                await self.continuity_manager.buffer_data(
                    event['data'], 
                    event['data'].get('instrument', 'unknown'),
                    DataType.PRICE
                )
        
        # Emit orchestrator event
        await self._emit_event("failover_event", event)
    
    async def _handle_continuity_event(self, event: Dict[str, Any]) -> None:
        """Handle data continuity events"""
        logger.info(f"Continuity event: {event['type']}")
        
        # Update metrics
        if event['type'] == 'data_replay_started':
            self.metrics.data_replays += 1
        
        # Emit orchestrator event
        await self._emit_event("continuity_event", event)
    
    async def _handle_health_event(self, event: Dict[str, Any]) -> None:
        """Handle health monitoring events"""
        logger.info(f"Health event: {event['type']}")
        
        # Update metrics
        self.metrics.health_checks += 1
        
        # Handle critical health events
        if event['type'] == 'alert_triggered':
            alert_data = event['data']
            if alert_data.get('level') == 'critical':
                await self._handle_critical_alert(alert_data)
        
        # Emit orchestrator event
        await self._emit_event("health_event", event)
    
    async def _handle_critical_alert(self, alert_data: Dict[str, Any]) -> None:
        """Handle critical health alerts"""
        logger.warning(f"Critical alert: {alert_data['message']}")
        
        # Trigger automatic recovery if enabled
        if self.config.enable_automatic_recovery:
            await self._trigger_automatic_recovery(alert_data)
    
    async def _trigger_automatic_recovery(self, alert_data: Dict[str, Any]) -> None:
        """Trigger automatic recovery procedures"""
        if self.recovery_in_progress:
            logger.warning("Recovery already in progress, skipping")
            return
        
        # Check recovery cooldown
        if (self.last_recovery_attempt and 
            (datetime.now() - self.last_recovery_attempt).total_seconds() < 300):
            logger.warning("Recovery cooldown active, skipping")
            return
        
        self.recovery_in_progress = True
        self.last_recovery_attempt = datetime.now()
        
        try:
            logger.info("Starting automatic recovery...")
            
            # Attempt recovery procedures
            for attempt in range(self.config.recovery_attempts):
                try:
                    await self._perform_recovery_procedures(alert_data)
                    logger.info(f"Recovery attempt {attempt + 1} successful")
                    break
                except Exception as e:
                    logger.error(f"Recovery attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.recovery_attempts - 1:
                        await asyncio.sleep(self.config.recovery_delay_seconds)
            
        finally:
            self.recovery_in_progress = False
    
    async def _perform_recovery_procedures(self, alert_data: Dict[str, Any]) -> None:
        """Perform recovery procedures based on alert type"""
        component = alert_data.get('component', '')
        
        if component == 'broker_health':
            # Try to recover broker connections
            await self._recover_broker_connections()
        elif component == 'cache_health':
            # Try to recover cache connection
            await self._recover_cache_connection()
        elif component == 'ingestion_health':
            # Try to recover ingestion
            await self._recover_ingestion()
        else:
            # Generic recovery
            await self._generic_recovery()
    
    async def _recover_broker_connections(self) -> None:
        """Recover broker connections"""
        logger.info("Attempting to recover broker connections...")
        
        # Restart broker health monitoring
        if self.broker_manager:
            await self.broker_manager.stop_health_monitoring()
            await asyncio.sleep(1)
            await self.broker_manager.start_health_monitoring()
        
        # Reset circuit breakers
        if self.failover_manager:
            # Reset all circuit breakers
            for name in self.failover_manager.circuit_breakers:
                self.failover_manager.circuit_breakers[name].state = 'closed'
                self.failover_manager.circuit_breakers[name].failure_count = 0
    
    async def _recover_cache_connection(self) -> None:
        """Recover cache connection"""
        logger.info("Attempting to recover cache connection...")
        
        if self.cache_manager:
            await self.cache_manager.disconnect()
            await asyncio.sleep(1)
            await self.cache_manager.connect()
    
    async def _recover_ingestion(self) -> None:
        """Recover data ingestion"""
        logger.info("Attempting to recover data ingestion...")
        
        # This would restart the ingestion engine
        # Implementation depends on the specific ingestion engine
        pass
    
    async def _generic_recovery(self) -> None:
        """Generic recovery procedures"""
        logger.info("Performing generic recovery...")
        
        # Restart all components
        await self._stop_components()
        await asyncio.sleep(2)
        await self._start_components()
    
    async def execute_with_failover(self, 
                                  operation: str,
                                  operation_func: Callable,
                                  *args, **kwargs) -> Any:
        """Execute an operation with full failover and redundancy support"""
        start_time = time.time()
        
        try:
            # Use redundancy manager if available
            if self.redundancy_manager:
                result = await self.redundancy_manager.get_data_with_redundancy(
                    operation, operation_func, *args, **kwargs
                )
                validated_data = result.validated_data
            else:
                # Use failover manager if available
                if self.failover_manager:
                    validated_data = await self.failover_manager.execute_with_failover(
                        operation, operation_func, *args, **kwargs
                    )
                else:
                    # Direct execution
                    validated_data = await operation_func(*args, **kwargs)
            
            # Update metrics
            self.metrics.total_operations += 1
            self.metrics.successful_operations += 1
            self.metrics.last_operation = datetime.now()
            
            operation_time_ms = (time.time() - start_time) * 1000
            self._update_operation_timing_metrics(operation_time_ms)
            
            return validated_data
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error(f"Operation {operation} failed: {e}")
            raise
    
    def _update_operation_timing_metrics(self, operation_time_ms: float) -> None:
        """Update operation timing metrics"""
        self.metrics.avg_operation_time_ms = (
            (self.metrics.avg_operation_time_ms * (self.metrics.total_operations - 1) + operation_time_ms) /
            self.metrics.total_operations
        )
        self.metrics.max_operation_time_ms = max(self.metrics.max_operation_time_ms, operation_time_ms)
        self.metrics.min_operation_time_ms = min(self.metrics.min_operation_time_ms, operation_time_ms)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an orchestrator event"""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        for callback in self.event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in orchestrator event callback: {e}")
    
    def add_event_callback(self, callback: Callable) -> None:
        """Add a callback for orchestrator events"""
        self.event_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        status = {
            'orchestrator_status': self.status.value,
            'components': {
                'failover_manager': self.failover_manager is not None,
                'continuity_manager': self.continuity_manager is not None,
                'redundancy_manager': self.redundancy_manager is not None,
                'health_manager': self.health_manager is not None
            },
            'recovery_in_progress': self.recovery_in_progress,
            'last_recovery_attempt': self.last_recovery_attempt.isoformat() if self.last_recovery_attempt else None
        }
        
        # Add component statuses
        if self.health_manager:
            status['health_status'] = self.health_manager.get_health_status()
        
        if self.failover_manager:
            status['failover_metrics'] = self.failover_manager.get_metrics()
        
        if self.continuity_manager:
            status['continuity_metrics'] = self.continuity_manager.get_metrics()
        
        if self.redundancy_manager:
            status['redundancy_metrics'] = self.redundancy_manager.get_metrics()
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            'total_operations': self.metrics.total_operations,
            'successful_operations': self.metrics.successful_operations,
            'failed_operations': self.metrics.failed_operations,
            'failover_events': self.metrics.failover_events,
            'data_replays': self.metrics.data_replays,
            'redundancy_validations': self.metrics.redundancy_validations,
            'health_checks': self.metrics.health_checks,
            'avg_operation_time_ms': round(self.metrics.avg_operation_time_ms, 2),
            'max_operation_time_ms': round(self.metrics.max_operation_time_ms, 2),
            'min_operation_time_ms': round(self.metrics.min_operation_time_ms, 2),
            'success_rate': (
                self.metrics.successful_operations / max(self.metrics.total_operations, 1) * 100
            ),
            'last_operation': self.metrics.last_operation.isoformat() if self.metrics.last_operation else None
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

# Example usage and testing
async def test_failover_orchestrator():
    """Test the failover orchestrator"""
    from ..brokers.broker_manager import BrokerManager
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Create orchestrator
    orchestrator = FailoverOrchestrator(broker_manager)
    
    # Add event callback
    async def event_callback(event):
        print(f"Orchestrator event: {event['type']}")
    
    orchestrator.add_event_callback(event_callback)
    
    # Start orchestrator
    await orchestrator.start()
    
    try:
        # Get status
        status = orchestrator.get_status()
        print(f"Orchestrator status: {status}")
        
        # Get metrics
        metrics = orchestrator.get_metrics()
        print(f"Orchestrator metrics: {metrics}")
        
        # Wait a bit
        await asyncio.sleep(5)
        
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(test_failover_orchestrator())

