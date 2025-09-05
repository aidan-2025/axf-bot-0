#!/usr/bin/env python3
"""
Health Monitoring Manager
Comprehensive health monitoring and alerting for data ingestion
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict, deque

from ..brokers.broker_manager import BrokerManager, BrokerStatus
from ..cache.redis_cache import RedisCacheManager
from ..config import CONFIG

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_func: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 10
    alert_threshold: float = 0.8
    critical_threshold: float = 0.5
    enabled: bool = True

@dataclass
class Metric:
    """Metric definition"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """Alert definition"""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthConfig:
    """Configuration for health monitoring"""
    check_interval_seconds: int = 30
    metrics_retention_hours: int = 24
    alert_cooldown_seconds: int = 300  # 5 minutes
    enable_prometheus_export: bool = True
    enable_alerting: bool = True
    max_alerts_per_hour: int = 100

@dataclass
class HealthMetrics:
    """Health monitoring metrics"""
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    alerts_triggered: int = 0
    alerts_resolved: int = 0
    avg_check_time_ms: float = 0.0
    max_check_time_ms: float = 0.0
    min_check_time_ms: float = float('inf')
    last_check: Optional[datetime] = None

class HealthMonitoringManager:
    """Comprehensive health monitoring and alerting manager"""
    
    def __init__(self, 
                 broker_manager: BrokerManager,
                 cache_manager: Optional[RedisCacheManager] = None,
                 config: Optional[HealthConfig] = None):
        """
        Initialize health monitoring manager
        
        Args:
            broker_manager: Broker manager instance
            cache_manager: Redis cache manager for persistence
            config: Health monitoring configuration
        """
        self.broker_manager = broker_manager
        self.cache_manager = cache_manager
        self.config = config or HealthConfig()
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Health status
        self.overall_status = HealthStatus.UNKNOWN
        self.component_status: Dict[str, HealthStatus] = {}
        
        # Monitoring tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
        
        # Initialize default health checks
        self._initialize_default_health_checks()
    
    def _initialize_default_health_checks(self) -> None:
        """Initialize default health checks"""
        # Broker health check
        self.add_health_check(
            "broker_health",
            self._check_broker_health,
            interval_seconds=30,
            alert_threshold=0.8,
            critical_threshold=0.5
        )
        
        # Cache health check
        self.add_health_check(
            "cache_health",
            self._check_cache_health,
            interval_seconds=60,
            alert_threshold=0.9,
            critical_threshold=0.7
        )
        
        # Data ingestion health check
        self.add_health_check(
            "ingestion_health",
            self._check_ingestion_health,
            interval_seconds=30,
            alert_threshold=0.8,
            critical_threshold=0.6
        )
        
        # System resources health check
        self.add_health_check(
            "system_resources",
            self._check_system_resources,
            interval_seconds=60,
            alert_threshold=0.8,
            critical_threshold=0.9
        )
    
    async def start(self) -> None:
        """Start the health monitoring manager"""
        if self.is_running:
            return
        
        logger.info("Starting health monitoring manager...")
        self.is_running = True
        
        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info("Health monitoring manager started successfully")
    
    async def stop(self) -> None:
        """Stop the health monitoring manager"""
        if not self.is_running:
            return
        
        logger.info("Stopping health monitoring manager...")
        self.is_running = False
        
        # Cancel monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self.monitoring_task, self.metrics_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Health monitoring manager stopped")
    
    def add_health_check(self, 
                        name: str,
                        check_func: Callable,
                        interval_seconds: int = 30,
                        timeout_seconds: int = 10,
                        alert_threshold: float = 0.8,
                        critical_threshold: float = 0.5,
                        enabled: bool = True) -> None:
        """Add a health check"""
        health_check = HealthCheck(
            name=name,
            check_func=check_func,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            alert_threshold=alert_threshold,
            critical_threshold=critical_threshold,
            enabled=enabled
        )
        
        self.health_checks[name] = health_check
        logger.info(f"Added health check: {name}")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self._run_health_checks()
                await self._update_overall_status()
                await self._process_alerts()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_loop(self) -> None:
        """Metrics collection loop"""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(5)
    
    async def _run_health_checks(self) -> None:
        """Run all enabled health checks"""
        for name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            try:
                start_time = time.time()
                
                # Run health check with timeout
                result = await asyncio.wait_for(
                    health_check.check_func(),
                    timeout=health_check.timeout_seconds
                )
                
                check_time_ms = (time.time() - start_time) * 1000
                
                # Update metrics
                self.metrics["health_check_duration"].append(Metric(
                    name="health_check_duration",
                    value=check_time_ms,
                    timestamp=datetime.now(),
                    labels={"check_name": name},
                    metric_type=MetricType.HISTOGRAM
                ))
                
                # Update component status
                if isinstance(result, dict) and 'status' in result:
                    status_value = result['status']
                    if status_value >= health_check.alert_threshold:
                        self.component_status[name] = HealthStatus.HEALTHY
                    elif status_value >= health_check.critical_threshold:
                        self.component_status[name] = HealthStatus.WARNING
                    else:
                        self.component_status[name] = HealthStatus.CRITICAL
                else:
                    self.component_status[name] = HealthStatus.UNKNOWN
                
                # Record metrics
                self.metrics["health_check_success"].append(Metric(
                    name="health_check_success",
                    value=1.0,
                    timestamp=datetime.now(),
                    labels={"check_name": name},
                    metric_type=MetricType.COUNTER
                ))
                
                self.metrics["health_check_value"].append(Metric(
                    name="health_check_value",
                    value=result.get('status', 0.0) if isinstance(result, dict) else 1.0,
                    timestamp=datetime.now(),
                    labels={"check_name": name},
                    metric_type=MetricType.GAUGE
                ))
                
            except asyncio.TimeoutError:
                logger.warning(f"Health check {name} timed out")
                self.component_status[name] = HealthStatus.CRITICAL
                self._record_failed_check(name, "timeout")
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                self.component_status[name] = HealthStatus.CRITICAL
                self._record_failed_check(name, str(e))
    
    def _record_failed_check(self, check_name: str, error: str) -> None:
        """Record a failed health check"""
        self.metrics["health_check_failure"].append(Metric(
            name="health_check_failure",
            value=1.0,
            timestamp=datetime.now(),
            labels={"check_name": check_name, "error": error},
            metric_type=MetricType.COUNTER
        ))
    
    async def _update_overall_status(self) -> None:
        """Update overall health status"""
        if not self.component_status:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        # Determine overall status based on component statuses
        statuses = list(self.component_status.values())
        
        if HealthStatus.CRITICAL in statuses:
            self.overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            self.overall_status = HealthStatus.WARNING
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            self.overall_status = HealthStatus.HEALTHY
        else:
            self.overall_status = HealthStatus.UNKNOWN
        
        # Record overall status metric
        status_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.WARNING: 0.5,
            HealthStatus.CRITICAL: 0.0,
            HealthStatus.UNKNOWN: 0.0
        }[self.overall_status]
        
        self.metrics["overall_health"].append(Metric(
            name="overall_health",
            value=status_value,
            timestamp=datetime.now(),
            labels={"status": self.overall_status.value},
            metric_type=MetricType.GAUGE
        ))
    
    async def _process_alerts(self) -> None:
        """Process alerts based on health status"""
        if not self.config.enable_alerting:
            return
        
        for component, status in self.component_status.items():
            alert_id = f"{component}_{status.value}"
            
            # Check if alert should be triggered
            if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                if alert_id not in self.active_alerts:
                    await self._trigger_alert(component, status)
            else:
                # Resolve alert if component is healthy
                if alert_id in self.active_alerts:
                    await self._resolve_alert(alert_id)
    
    async def _trigger_alert(self, component: str, status: HealthStatus) -> None:
        """Trigger an alert"""
        alert_id = f"{component}_{status.value}"
        
        alert_level = AlertLevel.WARNING if status == HealthStatus.WARNING else AlertLevel.CRITICAL
        
        alert = Alert(
            id=alert_id,
            level=alert_level,
            message=f"{component} is {status.value}",
            timestamp=datetime.now(),
            source=component,
            metadata={"component": component, "status": status.value}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {alert.message}")
        
        # Emit event
        await self._emit_event("alert_triggered", {
            "alert_id": alert_id,
            "component": component,
            "level": alert_level.value,
            "message": alert.message
        })
    
    async def _resolve_alert(self, alert_id: str) -> None:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.message}")
            
            # Emit event
            await self._emit_event("alert_resolved", {
                "alert_id": alert_id,
                "component": alert.source,
                "message": alert.message
            })
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics"""
        try:
            # CPU usage (simplified)
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics["cpu_usage"].append(Metric(
                name="cpu_usage",
                value=cpu_percent,
                timestamp=datetime.now(),
                metric_type=MetricType.GAUGE
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["memory_usage"].append(Metric(
                name="memory_usage",
                value=memory.percent,
                timestamp=datetime.now(),
                metric_type=MetricType.GAUGE
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics["disk_usage"].append(Metric(
                name="disk_usage",
                value=disk_percent,
                timestamp=datetime.now(),
                metric_type=MetricType.GAUGE
            ))
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    # Health check implementations
    async def _check_broker_health(self) -> Dict[str, Any]:
        """Check broker health"""
        try:
            status = await self.broker_manager.get_status()
            healthy_brokers = status.get('healthy_brokers', 0)
            total_brokers = status.get('total_brokers', 0)
            
            if total_brokers == 0:
                return {'status': 0.0, 'message': 'No brokers configured'}
            
            health_ratio = healthy_brokers / total_brokers
            return {
                'status': health_ratio,
                'message': f'{healthy_brokers}/{total_brokers} brokers healthy',
                'healthy_brokers': healthy_brokers,
                'total_brokers': total_brokers
            }
        except Exception as e:
            return {'status': 0.0, 'message': f'Broker health check failed: {e}'}
    
    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health"""
        if not self.cache_manager:
            return {'status': 1.0, 'message': 'Cache not configured'}
        
        try:
            is_healthy = await self.cache_manager.is_healthy()
            if is_healthy:
                stats = await self.cache_manager.get_cache_stats()
                return {
                    'status': 1.0,
                    'message': 'Cache healthy',
                    'stats': stats
                }
            else:
                return {'status': 0.0, 'message': 'Cache unhealthy'}
        except Exception as e:
            return {'status': 0.0, 'message': f'Cache health check failed: {e}'}
    
    async def _check_ingestion_health(self) -> Dict[str, Any]:
        """Check data ingestion health"""
        # This would check ingestion engine metrics
        # For now, return a placeholder
        return {'status': 1.0, 'message': 'Ingestion healthy'}
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Calculate overall resource health
            resource_health = 1.0
            if cpu_percent > 80:
                resource_health -= 0.3
            if memory.percent > 80:
                resource_health -= 0.3
            
            return {
                'status': max(0.0, resource_health),
                'message': f'CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%',
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent
            }
        except ImportError:
            return {'status': 1.0, 'message': 'System metrics not available'}
        except Exception as e:
            return {'status': 0.0, 'message': f'System resource check failed: {e}'}
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a monitoring event"""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        for callback in self.event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in monitoring event callback: {e}")
    
    def add_event_callback(self, callback: Callable) -> None:
        """Add a callback for monitoring events"""
        self.event_callbacks.append(callback)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            'overall_status': self.overall_status.value,
            'component_status': {k: v.value for k, v in self.component_status.items()},
            'active_alerts': len(self.active_alerts),
            'total_alerts': len(self.alert_history),
            'last_check': datetime.now().isoformat()
        }
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics data"""
        if metric_name:
            if metric_name in self.metrics:
                return {
                    'name': metric_name,
                    'data': [
                        {
                            'value': m.value,
                            'timestamp': m.timestamp.isoformat(),
                            'labels': m.labels
                        }
                        for m in self.metrics[metric_name]
                    ]
                }
            else:
                return {'name': metric_name, 'data': []}
        else:
            return {
                name: [
                    {
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'labels': m.labels
                    }
                    for m in metrics
                ]
                for name, metrics in self.metrics.items()
            }
    
    def get_alerts(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get alerts"""
        if active_only:
            return [
                {
                    'id': alert.id,
                    'level': alert.level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'source': alert.source,
                    'resolved': alert.resolved,
                    'metadata': alert.metadata
                }
                for alert in self.active_alerts.values()
            ]
        else:
            return [
                {
                    'id': alert.id,
                    'level': alert.level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'source': alert.source,
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                    'metadata': alert.metadata
                }
                for alert in self.alert_history
            ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

# Example usage and testing
async def test_health_monitoring_manager():
    """Test the health monitoring manager"""
    from ..brokers.broker_manager import BrokerManager
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Create health monitoring manager
    health_manager = HealthMonitoringManager(broker_manager)
    
    # Add event callback
    async def event_callback(event):
        print(f"Health event: {event}")
    
    health_manager.add_event_callback(event_callback)
    
    # Start manager
    await health_manager.start()
    
    try:
        # Wait for some checks
        await asyncio.sleep(5)
        
        # Get health status
        status = health_manager.get_health_status()
        print(f"Health status: {status}")
        
        # Get metrics
        metrics = health_manager.get_metrics()
        print(f"Metrics: {list(metrics.keys())}")
        
        # Get alerts
        alerts = health_manager.get_alerts()
        print(f"Active alerts: {len(alerts)}")
        
    finally:
        await health_manager.stop()

if __name__ == "__main__":
    asyncio.run(test_health_monitoring_manager())

