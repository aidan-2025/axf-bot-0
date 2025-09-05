#!/usr/bin/env python3
"""
Performance Monitor
Central performance monitoring system for data ingestion
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

from ..brokers.broker_manager import PriceData, CandleData

logger = logging.getLogger(__name__)

class PerformanceStatus(Enum):
    """Performance status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Latency metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    
    # Throughput metrics
    data_points_per_second: float = 0.0
    peak_throughput: float = 0.0
    avg_throughput: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Data quality metrics
    data_completeness: float = 1.0
    validation_success_rate: float = 1.0
    quality_score: float = 1.0
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    
    # Queue metrics
    queue_depth: int = 0
    queue_utilization: float = 0.0
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.utcnow)
    measurement_duration_ms: float = 0.0

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""
    # Monitoring intervals
    measurement_interval: float = 1.0  # seconds
    aggregation_window: int = 60  # measurements
    alert_check_interval: float = 5.0  # seconds
    
    # Latency thresholds
    latency_warning_threshold_ms: float = 50.0
    latency_critical_threshold_ms: float = 100.0
    
    # Throughput thresholds
    min_throughput_threshold: float = 10.0  # data points per second
    max_throughput_threshold: float = 10000.0  # data points per second
    
    # Error rate thresholds
    error_rate_warning_threshold: float = 0.01  # 1%
    error_rate_critical_threshold: float = 0.05  # 5%
    
    # System resource thresholds
    cpu_warning_threshold: float = 70.0  # %
    cpu_critical_threshold: float = 90.0  # %
    memory_warning_threshold: float = 80.0  # %
    memory_critical_threshold: float = 95.0  # %
    
    # Data quality thresholds
    completeness_warning_threshold: float = 0.95  # 95%
    completeness_critical_threshold: float = 0.90  # 90%
    quality_warning_threshold: float = 0.80  # 80%
    quality_critical_threshold: float = 0.70  # 70%
    
    # Queue thresholds
    queue_warning_threshold: float = 0.70  # 70%
    queue_critical_threshold: float = 0.90  # 90%
    
    # Alerting
    enable_alerts: bool = True
    alert_cooldown: float = 60.0  # seconds
    max_alerts_per_hour: int = 10

class PerformanceMonitor:
    """Central performance monitoring system"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize performance monitor"""
        self.config = config or PerformanceConfig()
        
        # Monitoring state
        self.status = PerformanceStatus.UNKNOWN
        self.is_running = False
        
        # Metrics collection
        self.latency_history: deque = deque(maxlen=self.config.aggregation_window)
        self.throughput_history: deque = deque(maxlen=self.config.aggregation_window)
        self.error_history: deque = deque(maxlen=self.config.aggregation_window)
        self.system_history: deque = deque(maxlen=self.config.aggregation_window)
        
        # Current metrics
        self.current_metrics = PerformanceMetrics()
        
        # Performance tracking
        self.data_point_count = 0
        self.error_count = 0
        self.start_time = None
        self.last_measurement_time = None
        
        # Alerting
        self.alert_history: List[Dict[str, Any]] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        # Task management
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alerting_task: Optional[asyncio.Task] = None
        
        logger.info("PerformanceMonitor initialized with config: %s", self.config)
    
    async def start(self) -> None:
        """Start performance monitoring"""
        try:
            if self.is_running:
                logger.warning("Performance monitoring is already running")
                return
            
            self.is_running = True
            self.start_time = datetime.utcnow()
            self.status = PerformanceStatus.HEALTHY
            
            # Start monitoring tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.alerting_task = asyncio.create_task(self._alerting_loop())
            
            logger.info("Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {e}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop performance monitoring"""
        try:
            self.is_running = False
            
            # Cancel monitoring tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.alerting_task:
                self.alerting_task.cancel()
            
            # Wait for tasks to complete
            tasks = [t for t in [self.monitoring_task, self.alerting_task] if t]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.monitoring_task = None
            self.alerting_task = None
            
            logger.info("Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                # Calculate performance metrics
                await self._calculate_performance_metrics(system_metrics)
                
                # Update status
                await self._update_performance_status()
                
                # Call metrics callbacks
                for callback in self.metrics_callbacks:
                    try:
                        await callback(self.current_metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics callback: {e}")
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.measurement_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _alerting_loop(self) -> None:
        """Alerting loop"""
        while self.is_running:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.config.alert_check_interval)
                
            except Exception as e:
                logger.error(f"Error in alerting loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_rate = 0.0
            if disk_io:
                disk_io_rate = (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024  # MB/s
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_rate = 0.0
            if network_io:
                network_io_rate = (network_io.bytes_sent + network_io.bytes_recv) / 1024 / 1024  # MB/s
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_io': disk_io_rate,
                'network_io': network_io_rate
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_io': 0.0,
                'network_io': 0.0
            }
    
    async def _calculate_performance_metrics(self, system_metrics: Dict[str, float]) -> None:
        """Calculate comprehensive performance metrics"""
        try:
            current_time = datetime.utcnow()
            
            # Calculate throughput
            if self.start_time:
                time_elapsed = (current_time - self.start_time).total_seconds()
                if time_elapsed > 0:
                    self.current_metrics.data_points_per_second = self.data_point_count / time_elapsed
                    self.current_metrics.avg_throughput = self.current_metrics.data_points_per_second
            
            # Calculate error rate
            if self.data_point_count > 0:
                self.current_metrics.error_rate = self.error_count / self.data_point_count
                self.current_metrics.total_errors = self.error_count
            
            # Calculate latency metrics
            if self.latency_history:
                latencies = list(self.latency_history)
                self.current_metrics.avg_latency_ms = sum(latencies) / len(latencies)
                self.current_metrics.p50_latency_ms = self._calculate_percentile(latencies, 50)
                self.current_metrics.p95_latency_ms = self._calculate_percentile(latencies, 95)
                self.current_metrics.p99_latency_ms = self._calculate_percentile(latencies, 99)
                self.current_metrics.max_latency_ms = max(latencies)
                self.current_metrics.min_latency_ms = min(latencies)
            
            # Update system metrics
            self.current_metrics.cpu_usage = system_metrics['cpu_usage']
            self.current_metrics.memory_usage = system_metrics['memory_usage']
            self.current_metrics.disk_io = system_metrics['disk_io']
            self.current_metrics.network_io = system_metrics['network_io']
            
            # Update timestamp
            self.current_metrics.timestamp = current_time
            if self.last_measurement_time:
                self.current_metrics.measurement_duration_ms = (current_time - self.last_measurement_time).total_seconds() * 1000
            
            self.last_measurement_time = current_time
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    async def _update_performance_status(self) -> None:
        """Update performance status based on metrics"""
        try:
            old_status = self.status
            
            # Check latency
            if self.current_metrics.avg_latency_ms > self.config.latency_critical_threshold_ms:
                self.status = PerformanceStatus.CRITICAL
            elif self.current_metrics.avg_latency_ms > self.config.latency_warning_threshold_ms:
                self.status = PerformanceStatus.WARNING
            else:
                self.status = PerformanceStatus.HEALTHY
            
            # Check error rate
            if self.current_metrics.error_rate > self.config.error_rate_critical_threshold:
                self.status = PerformanceStatus.CRITICAL
            elif self.current_metrics.error_rate > self.config.error_rate_warning_threshold:
                if self.status == PerformanceStatus.HEALTHY:
                    self.status = PerformanceStatus.WARNING
            
            # Check system resources
            if (self.current_metrics.cpu_usage > self.config.cpu_critical_threshold or
                self.current_metrics.memory_usage > self.config.memory_critical_threshold):
                self.status = PerformanceStatus.CRITICAL
            elif (self.current_metrics.cpu_usage > self.config.cpu_warning_threshold or
                  self.current_metrics.memory_usage > self.config.memory_warning_threshold):
                if self.status == PerformanceStatus.HEALTHY:
                    self.status = PerformanceStatus.WARNING
            
            # Log status changes
            if old_status != self.status:
                logger.info(f"Performance status changed: {old_status.value} -> {self.status.value}")
                
        except Exception as e:
            logger.error(f"Error updating performance status: {e}")
    
    async def _check_alerts(self) -> None:
        """Check for alert conditions"""
        if not self.config.enable_alerts:
            return
        
        try:
            current_time = datetime.utcnow()
            
            # Check if we're within alert cooldown
            if self._is_alert_cooldown_active(current_time):
                return
            
            # Check if we've exceeded max alerts per hour
            if self._has_exceeded_max_alerts(current_time):
                return
            
            # Check alert conditions
            alerts = []
            
            # Latency alerts
            if self.current_metrics.avg_latency_ms > self.config.latency_critical_threshold_ms:
                alerts.append({
                    'type': 'latency_critical',
                    'severity': 'critical',
                    'message': f"Critical latency: {self.current_metrics.avg_latency_ms:.2f}ms (threshold: {self.config.latency_critical_threshold_ms}ms)",
                    'value': self.current_metrics.avg_latency_ms,
                    'threshold': self.config.latency_critical_threshold_ms
                })
            elif self.current_metrics.avg_latency_ms > self.config.latency_warning_threshold_ms:
                alerts.append({
                    'type': 'latency_warning',
                    'severity': 'warning',
                    'message': f"High latency: {self.current_metrics.avg_latency_ms:.2f}ms (threshold: {self.config.latency_warning_threshold_ms}ms)",
                    'value': self.current_metrics.avg_latency_ms,
                    'threshold': self.config.latency_warning_threshold_ms
                })
            
            # Error rate alerts
            if self.current_metrics.error_rate > self.config.error_rate_critical_threshold:
                alerts.append({
                    'type': 'error_rate_critical',
                    'severity': 'critical',
                    'message': f"Critical error rate: {self.current_metrics.error_rate:.2%} (threshold: {self.config.error_rate_critical_threshold:.2%})",
                    'value': self.current_metrics.error_rate,
                    'threshold': self.config.error_rate_critical_threshold
                })
            elif self.current_metrics.error_rate > self.config.error_rate_warning_threshold:
                alerts.append({
                    'type': 'error_rate_warning',
                    'severity': 'warning',
                    'message': f"High error rate: {self.current_metrics.error_rate:.2%} (threshold: {self.config.error_rate_warning_threshold:.2%})",
                    'value': self.current_metrics.error_rate,
                    'threshold': self.config.error_rate_warning_threshold
                })
            
            # System resource alerts
            if self.current_metrics.cpu_usage > self.config.cpu_critical_threshold:
                alerts.append({
                    'type': 'cpu_critical',
                    'severity': 'critical',
                    'message': f"Critical CPU usage: {self.current_metrics.cpu_usage:.1f}% (threshold: {self.config.cpu_critical_threshold}%)",
                    'value': self.current_metrics.cpu_usage,
                    'threshold': self.config.cpu_critical_threshold
                })
            
            if self.current_metrics.memory_usage > self.config.memory_critical_threshold:
                alerts.append({
                    'type': 'memory_critical',
                    'severity': 'critical',
                    'message': f"Critical memory usage: {self.current_metrics.memory_usage:.1f}% (threshold: {self.config.memory_critical_threshold}%)",
                    'value': self.current_metrics.memory_usage,
                    'threshold': self.config.memory_critical_threshold
                })
            
            # Send alerts
            for alert in alerts:
                await self._send_alert(alert, current_time)
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _is_alert_cooldown_active(self, current_time: datetime) -> bool:
        """Check if alert cooldown is active"""
        if not self.alert_history:
            return False
        
        last_alert_time = self.alert_history[-1]['timestamp']
        return (current_time - last_alert_time).total_seconds() < self.config.alert_cooldown
    
    def _has_exceeded_max_alerts(self, current_time: datetime) -> bool:
        """Check if max alerts per hour has been exceeded"""
        if not self.alert_history:
            return False
        
        # Count alerts in the last hour
        hour_ago = current_time - timedelta(hours=1)
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > hour_ago]
        
        return len(recent_alerts) >= self.config.max_alerts_per_hour
    
    async def _send_alert(self, alert: Dict[str, Any], timestamp: datetime) -> None:
        """Send alert to callbacks"""
        try:
            alert['timestamp'] = timestamp
            self.alert_history.append(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            logger.warning(f"Alert sent: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        
        return sorted_data[index]
    
    # Public methods for metrics collection
    def record_latency(self, latency_ms: float) -> None:
        """Record latency measurement"""
        self.latency_history.append(latency_ms)
    
    def record_data_point(self) -> None:
        """Record data point processed"""
        self.data_point_count += 1
    
    def record_error(self, error_type: str) -> None:
        """Record error occurrence"""
        self.error_count += 1
        if error_type not in self.current_metrics.errors_by_type:
            self.current_metrics.errors_by_type[error_type] = 0
        self.current_metrics.errors_by_type[error_type] += 1
    
    def update_data_quality(self, completeness: float, quality_score: float) -> None:
        """Update data quality metrics"""
        self.current_metrics.data_completeness = completeness
        self.current_metrics.quality_score = quality_score
    
    def update_validation_success_rate(self, success_rate: float) -> None:
        """Update validation success rate"""
        self.current_metrics.validation_success_rate = success_rate
    
    def update_queue_metrics(self, queue_depth: int, queue_utilization: float) -> None:
        """Update queue metrics"""
        self.current_metrics.queue_depth = queue_depth
        self.current_metrics.queue_utilization = queue_utilization
    
    # Callback management
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable) -> None:
        """Add metrics callback"""
        self.metrics_callbacks.append(callback)
    
    # Status and metrics access
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.current_metrics
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get comprehensive performance status"""
        return {
            'status': self.status.value,
            'is_running': self.is_running,
            'metrics': {
                'avg_latency_ms': self.current_metrics.avg_latency_ms,
                'p95_latency_ms': self.current_metrics.p95_latency_ms,
                'p99_latency_ms': self.current_metrics.p99_latency_ms,
                'data_points_per_second': self.current_metrics.data_points_per_second,
                'error_rate': self.current_metrics.error_rate,
                'cpu_usage': self.current_metrics.cpu_usage,
                'memory_usage': self.current_metrics.memory_usage,
                'data_completeness': self.current_metrics.data_completeness,
                'quality_score': self.current_metrics.quality_score
            },
            'config': {
                'latency_warning_threshold_ms': self.config.latency_warning_threshold_ms,
                'latency_critical_threshold_ms': self.config.latency_critical_threshold_ms,
                'error_rate_warning_threshold': self.config.error_rate_warning_threshold,
                'error_rate_critical_threshold': self.config.error_rate_critical_threshold,
                'cpu_warning_threshold': self.config.cpu_warning_threshold,
                'cpu_critical_threshold': self.config.cpu_critical_threshold
            },
            'alert_count': len(self.alert_history),
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        }

# Example usage and testing
async def test_performance_monitor():
    """Test the performance monitor"""
    config = PerformanceConfig(
        measurement_interval=1.0,
        latency_warning_threshold_ms=50.0,
        latency_critical_threshold_ms=100.0
    )
    
    monitor = PerformanceMonitor(config)
    
    # Add callbacks
    async def alert_callback(alert):
        print(f"Alert: {alert['message']}")
    
    async def metrics_callback(metrics):
        print(f"Metrics: Latency={metrics.avg_latency_ms:.2f}ms, Throughput={metrics.data_points_per_second:.2f}/s")
    
    monitor.add_alert_callback(alert_callback)
    monitor.add_metrics_callback(metrics_callback)
    
    # Start monitoring
    await monitor.start()
    
    # Simulate some data
    for i in range(10):
        monitor.record_latency(30.0 + i * 5.0)  # Increasing latency
        monitor.record_data_point()
        await asyncio.sleep(0.1)
    
    # Get status
    status = monitor.get_performance_status()
    print(f"Performance status: {status}")
    
    # Stop monitoring
    await monitor.stop()

if __name__ == "__main__":
    asyncio.run(test_performance_monitor())

