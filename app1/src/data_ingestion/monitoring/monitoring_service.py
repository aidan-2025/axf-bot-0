#!/usr/bin/env python3
"""
Monitoring Service
Integrated monitoring service that coordinates all monitoring components
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from .performance_monitor import PerformanceMonitor, MetricType
from .alert_manager import AlertManager, AlertRule, AlertSeverity, AlertChannel, NotificationConfig
from .benchmark_runner import BenchmarkRunner, BenchmarkConfig, BenchmarkType
from .monitoring_dashboard import MonitoringDashboard, DashboardConfig

logger = logging.getLogger(__name__)

@dataclass
class MonitoringServiceConfig:
    """Monitoring service configuration"""
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_metrics_retention_days: int = 30
    performance_metrics_cleanup_interval_hours: int = 24
    
    # Alerting
    enable_alerting: bool = True
    alert_retention_days: int = 90
    alert_cleanup_interval_hours: int = 24
    
    # Benchmarking
    enable_benchmarking: bool = True
    benchmark_results_retention_days: int = 30
    benchmark_cleanup_interval_hours: int = 24
    
    # Dashboard
    enable_dashboard: bool = True
    dashboard_auto_refresh: bool = True
    dashboard_refresh_interval_seconds: float = 5.0
    
    # System monitoring
    enable_system_monitoring: bool = True
    system_metrics_interval_seconds: float = 30.0
    
    # Data quality monitoring
    enable_data_quality_monitoring: bool = True
    data_quality_check_interval_seconds: float = 60.0
    
    # Notification configuration
    notification_config: Optional[NotificationConfig] = None
    
    # Dashboard configuration
    dashboard_config: Optional[DashboardConfig] = None
    
    # Benchmark configuration
    benchmark_config: Optional[BenchmarkConfig] = None

class MonitoringService:
    """Integrated monitoring service"""
    
    def __init__(self, config: Optional[MonitoringServiceConfig] = None):
        """Initialize monitoring service"""
        self.config = config or MonitoringServiceConfig()
        
        # Core components
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.alert_manager: Optional[AlertManager] = None
        self.benchmark_runner: Optional[BenchmarkRunner] = None
        self.dashboard: Optional[MonitoringDashboard] = None
        
        # Service state
        self.running: bool = False
        self.start_time: Optional[datetime] = None
        
        # Task management
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Callbacks
        self.service_callbacks: List[Callable] = []
        
        logger.info("MonitoringService initialized")
    
    async def start(self) -> None:
        """Start monitoring service"""
        if self.running:
            logger.warning("MonitoringService already running")
            return
        
        try:
            self.start_time = datetime.utcnow()
            self.running = True
            
            # Initialize performance monitor
            if self.config.enable_performance_monitoring:
                self.performance_monitor = PerformanceMonitor()
                await self.performance_monitor.start()
                logger.info("Performance monitor started")
            
            # Initialize alert manager
            if self.config.enable_alerting:
                self.alert_manager = AlertManager(self.config.notification_config)
                await self.alert_manager.start()
                
                # Add default alert rules
                await self._setup_default_alert_rules()
                logger.info("Alert manager started")
            
            # Initialize benchmark runner
            if self.config.enable_benchmarking:
                self.benchmark_runner = BenchmarkRunner(self.config.benchmark_config)
                logger.info("Benchmark runner initialized")
            
            # Initialize dashboard
            if self.config.enable_dashboard and self.performance_monitor and self.alert_manager:
                self.dashboard = MonitoringDashboard(
                    self.performance_monitor,
                    self.alert_manager,
                    self.config.dashboard_config
                )
                await self.dashboard.start()
                logger.info("Dashboard started")
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("MonitoringService started successfully")
            
        except Exception as e:
            logger.error(f"Error starting MonitoringService: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop monitoring service"""
        if not self.running:
            return
        
        try:
            self.running = False
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            # Stop components
            if self.dashboard:
                await self.dashboard.stop()
                logger.info("Dashboard stopped")
            
            if self.alert_manager:
                await self.alert_manager.stop()
                logger.info("Alert manager stopped")
            
            if self.performance_monitor:
                await self.performance_monitor.stop()
                logger.info("Performance monitor stopped")
            
            logger.info("MonitoringService stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MonitoringService: {e}")
    
    async def _start_monitoring_tasks(self) -> None:
        """Start monitoring tasks"""
        # System monitoring task
        if self.config.enable_system_monitoring:
            task = asyncio.create_task(self._system_monitoring_loop())
            self.monitoring_tasks.append(task)
        
        # Data quality monitoring task
        if self.config.enable_data_quality_monitoring:
            task = asyncio.create_task(self._data_quality_monitoring_loop())
            self.monitoring_tasks.append(task)
        
        # Cleanup tasks
        if self.config.enable_performance_monitoring:
            task = asyncio.create_task(self._performance_cleanup_loop())
            self.monitoring_tasks.append(task)
        
        if self.config.enable_alerting:
            task = asyncio.create_task(self._alert_cleanup_loop())
            self.monitoring_tasks.append(task)
        
        if self.config.enable_benchmarking:
            task = asyncio.create_task(self._benchmark_cleanup_loop())
            self.monitoring_tasks.append(task)
    
    async def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules"""
        if not self.alert_manager:
            return
        
        # High latency alert
        high_latency_rule = AlertRule(
            rule_id="high_latency",
            name="High Data Ingestion Latency",
            description="Data ingestion latency is above acceptable threshold",
            metric_name="ingestion_latency_ms",
            condition=">",
            threshold_value=1000.0,  # 1 second
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            evaluation_interval_seconds=30.0,
            min_duration_seconds=60.0,
            cooldown_seconds=300.0
        )
        
        # Critical latency alert
        critical_latency_rule = AlertRule(
            rule_id="critical_latency",
            name="Critical Data Ingestion Latency",
            description="Data ingestion latency is critically high",
            metric_name="ingestion_latency_ms",
            condition=">",
            threshold_value=5000.0,  # 5 seconds
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK],
            evaluation_interval_seconds=30.0,
            min_duration_seconds=30.0,
            cooldown_seconds=600.0
        )
        
        # High error rate alert
        high_error_rate_rule = AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            description="Data ingestion error rate is above threshold",
            metric_name="ingestion_error_rate",
            condition=">",
            threshold_value=0.05,  # 5%
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK],
            evaluation_interval_seconds=30.0,
            min_duration_seconds=60.0,
            cooldown_seconds=300.0
        )
        
        # Critical error rate alert
        critical_error_rate_rule = AlertRule(
            rule_id="critical_error_rate",
            name="Critical Error Rate",
            description="Data ingestion error rate is critically high",
            metric_name="ingestion_error_rate",
            condition=">",
            threshold_value=0.20,  # 20%
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK],
            evaluation_interval_seconds=30.0,
            min_duration_seconds=30.0,
            cooldown_seconds=600.0
        )
        
        # Low throughput alert
        low_throughput_rule = AlertRule(
            rule_id="low_throughput",
            name="Low Throughput",
            description="Data ingestion throughput is below threshold",
            metric_name="ingestion_throughput_ops_per_second",
            condition="<",
            threshold_value=10.0,  # 10 ops/s
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            evaluation_interval_seconds=60.0,
            min_duration_seconds=120.0,
            cooldown_seconds=600.0
        )
        
        # System resource alerts
        high_cpu_rule = AlertRule(
            rule_id="high_cpu_usage",
            name="High CPU Usage",
            description="System CPU usage is above threshold",
            metric_name="system_cpu_usage",
            condition=">",
            threshold_value=80.0,  # 80%
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            evaluation_interval_seconds=60.0,
            min_duration_seconds=300.0,  # 5 minutes
            cooldown_seconds=1800.0  # 30 minutes
        )
        
        high_memory_rule = AlertRule(
            rule_id="high_memory_usage",
            name="High Memory Usage",
            description="System memory usage is above threshold",
            metric_name="system_memory_usage",
            condition=">",
            threshold_value=85.0,  # 85%
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            evaluation_interval_seconds=60.0,
            min_duration_seconds=300.0,  # 5 minutes
            cooldown_seconds=1800.0  # 30 minutes
        )
        
        # Add rules to alert manager
        rules = [
            high_latency_rule,
            critical_latency_rule,
            high_error_rate_rule,
            critical_error_rate_rule,
            low_throughput_rule,
            high_cpu_rule,
            high_memory_rule
        ]
        
        for rule in rules:
            self.alert_manager.add_alert_rule(rule)
        
        logger.info(f"Added {len(rules)} default alert rules")
    
    async def _system_monitoring_loop(self) -> None:
        """System monitoring loop"""
        while self.running:
            try:
                if self.performance_monitor:
                    # Monitor system metrics
                    await self._collect_system_metrics()
                
                await asyncio.sleep(self.config.system_metrics_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _data_quality_monitoring_loop(self) -> None:
        """Data quality monitoring loop"""
        while self.running:
            try:
                if self.performance_monitor:
                    # Monitor data quality metrics
                    await self._collect_data_quality_metrics()
                
                await asyncio.sleep(self.config.data_quality_check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data quality monitoring loop: {e}")
                await asyncio.sleep(60.0)
    
    async def _performance_cleanup_loop(self) -> None:
        """Performance metrics cleanup loop"""
        while self.running:
            try:
                if self.performance_monitor:
                    # Cleanup old metrics
                    await self.performance_monitor.cleanup_old_metrics(
                        self.config.performance_metrics_retention_days
                    )
                
                # Wait for next cleanup
                await asyncio.sleep(self.config.performance_metrics_cleanup_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance cleanup loop: {e}")
                await asyncio.sleep(3600.0)  # 1 hour
    
    async def _alert_cleanup_loop(self) -> None:
        """Alert cleanup loop"""
        while self.running:
            try:
                if self.alert_manager:
                    # Cleanup old alerts (implement in AlertManager if needed)
                    pass
                
                # Wait for next cleanup
                await asyncio.sleep(self.config.alert_cleanup_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert cleanup loop: {e}")
                await asyncio.sleep(3600.0)  # 1 hour
    
    async def _benchmark_cleanup_loop(self) -> None:
        """Benchmark cleanup loop"""
        while self.running:
            try:
                if self.benchmark_runner:
                    # Cleanup old benchmark results (implement in BenchmarkRunner if needed)
                    pass
                
                # Wait for next cleanup
                await asyncio.sleep(self.config.benchmark_cleanup_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in benchmark cleanup loop: {e}")
                await asyncio.sleep(3600.0)  # 1 hour
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.performance_monitor.record_metric("system_cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            await self.performance_monitor.record_metric("system_memory_usage", memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.performance_monitor.record_metric("system_disk_usage", disk_percent)
            
            # Network I/O
            network = psutil.net_io_counters()
            network_usage = (network.bytes_sent + network.bytes_recv) / (1024 * 1024)  # MB
            await self.performance_monitor.record_metric("system_network_usage", network_usage)
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_data_quality_metrics(self) -> None:
        """Collect data quality metrics"""
        try:
            # This would integrate with the data validation system
            # For now, we'll simulate some metrics
            
            # Data completeness (simulated)
            completeness = 0.95 + (time.time() % 0.1)  # 95-100%
            await self.performance_monitor.record_metric("data_completeness", completeness)
            
            # Data accuracy (simulated)
            accuracy = 0.98 + (time.time() % 0.02)  # 98-100%
            await self.performance_monitor.record_metric("data_accuracy", accuracy)
            
            # Data freshness (simulated)
            freshness = 0.99 + (time.time() % 0.01)  # 99-100%
            await self.performance_monitor.record_metric("data_freshness", freshness)
            
        except Exception as e:
            logger.error(f"Error collecting data quality metrics: {e}")
    
    # Public API methods
    async def record_metric(self, metric_name: str, value: float, 
                          metric_type: MetricType = MetricType.GAUGE) -> None:
        """Record a metric"""
        if self.performance_monitor:
            await self.performance_monitor.record_metric(metric_name, value, metric_type)
    
    async def evaluate_metric(self, metric_name: str, value: float, 
                            context: Dict[str, Any] = None) -> None:
        """Evaluate metric against alert rules"""
        if self.alert_manager:
            await self.alert_manager.evaluate_metric(metric_name, value, context)
    
    async def run_benchmark(self, benchmark_type: BenchmarkType, 
                          operation_func: Callable, **kwargs) -> Any:
        """Run benchmark test"""
        if self.benchmark_runner:
            if benchmark_type == BenchmarkType.LATENCY:
                return await self.benchmark_runner.run_latency_benchmark(operation_func, **kwargs)
            elif benchmark_type == BenchmarkType.THROUGHPUT:
                return await self.benchmark_runner.run_throughput_benchmark(operation_func, **kwargs)
            elif benchmark_type == BenchmarkType.LOAD:
                return await self.benchmark_runner.run_load_benchmark(operation_func, **kwargs)
            elif benchmark_type == BenchmarkType.STRESS:
                return await self.benchmark_runner.run_stress_benchmark(operation_func, **kwargs)
        return None
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        if self.dashboard:
            return self.dashboard.get_dashboard_data()
        return {}
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        if self.alert_manager:
            return self.alert_manager.get_alert_summary()
        return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if self.performance_monitor:
            return self.performance_monitor.get_performance_summary()
        return {}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            'components': {
                'performance_monitor': self.performance_monitor is not None,
                'alert_manager': self.alert_manager is not None,
                'benchmark_runner': self.benchmark_runner is not None,
                'dashboard': self.dashboard is not None
            },
            'active_tasks': len(self.monitoring_tasks)
        }
    
    # Callback management
    def add_service_callback(self, callback: Callable) -> None:
        """Add service callback"""
        self.service_callbacks.append(callback)
    
    async def _notify_service_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify service callbacks"""
        for callback in self.service_callbacks:
            try:
                await callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in service callback: {e}")

# Example usage and testing
async def test_monitoring_service():
    """Test the monitoring service"""
    
    # Create configuration
    config = MonitoringServiceConfig(
        enable_performance_monitoring=True,
        enable_alerting=True,
        enable_benchmarking=True,
        enable_dashboard=True,
        enable_system_monitoring=True,
        enable_data_quality_monitoring=True
    )
    
    # Create monitoring service
    service = MonitoringService(config)
    
    # Add callbacks
    async def service_callback(event_type, data):
        print(f"Service event: {event_type} - {data}")
    
    service.add_service_callback(service_callback)
    
    # Start service
    await service.start()
    
    # Simulate some metrics
    print("Simulating metrics...")
    for i in range(20):
        await service.record_metric("ingestion_latency_ms", 100 + i * 10)
        await service.record_metric("ingestion_throughput_ops_per_second", 50 + i * 5)
        await service.record_metric("ingestion_error_rate", 0.01 + i * 0.001)
        await asyncio.sleep(1.0)
    
    # Get status
    status = service.get_service_status()
    print(f"Service status: {status}")
    
    # Get dashboard data
    dashboard_data = service.get_dashboard_data()
    print(f"Dashboard widgets: {len(dashboard_data.get('widgets', {}))}")
    
    # Get alert summary
    alert_summary = service.get_alert_summary()
    print(f"Alert summary: {alert_summary}")
    
    # Stop service
    await service.stop()

if __name__ == "__main__":
    asyncio.run(test_monitoring_service())

