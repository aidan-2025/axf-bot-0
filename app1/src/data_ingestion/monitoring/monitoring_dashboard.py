#!/usr/bin/env python3
"""
Monitoring Dashboard
Real-time monitoring dashboard and visualization system
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math

from .performance_monitor import PerformanceMonitor, MetricType
from .alert_manager import AlertManager, Alert, AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)

class DashboardWidget(Enum):
    """Dashboard widget types"""
    METRIC_CARD = "metric_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    TABLE = "table"
    ALERT_LIST = "alert_list"
    STATUS_INDICATOR = "status_indicator"

class DashboardLayout(Enum):
    """Dashboard layout types"""
    GRID = "grid"
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    CUSTOM = "custom"

@dataclass
class WidgetConfig:
    """Widget configuration"""
    widget_id: str
    widget_type: DashboardWidget
    title: str
    description: str
    
    # Data configuration
    metric_name: Optional[str] = None
    data_source: Optional[str] = None
    refresh_interval_seconds: float = 5.0
    
    # Display configuration
    width: int = 1
    height: int = 1
    position_x: int = 0
    position_y: int = 0
    
    # Chart configuration
    chart_type: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    color_scheme: Optional[str] = None
    
    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Filters
    time_range_minutes: int = 60
    aggregation_function: str = "avg"  # avg, sum, min, max, count
    
    # Additional configuration
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    layout: DashboardLayout = DashboardLayout.GRID
    
    # Grid configuration
    grid_columns: int = 4
    grid_rows: int = 6
    
    # Auto-refresh
    auto_refresh: bool = True
    refresh_interval_seconds: float = 5.0
    
    # Widgets
    widgets: List[WidgetConfig] = field(default_factory=list)
    
    # Theme
    theme: str = "dark"
    color_scheme: str = "default"
    
    # Additional configuration
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardData:
    """Dashboard data point"""
    timestamp: datetime
    widget_id: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MonitoringDashboard:
    """Real-time monitoring dashboard system"""
    
    def __init__(self, 
                 performance_monitor: PerformanceMonitor,
                 alert_manager: AlertManager,
                 config: Optional[DashboardConfig] = None):
        """Initialize monitoring dashboard"""
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager
        self.config = config or self._create_default_config()
        
        # Dashboard state
        self.dashboard_data: Dict[str, List[DashboardData]] = {}
        self.widget_configs: Dict[str, WidgetConfig] = {}
        self.last_refresh_times: Dict[str, datetime] = {}
        
        # Real-time data
        self.real_time_metrics: Dict[str, Any] = {}
        self.alert_summary: Dict[str, Any] = {}
        
        # Task management
        self.refresh_task: Optional[asyncio.Task] = None
        self.running: bool = False
        
        # Callbacks
        self.dashboard_callbacks: List[Callable] = []
        self.data_callbacks: List[Callable] = []
        
        # Initialize widgets
        self._initialize_widgets()
        
        logger.info("MonitoringDashboard initialized")
    
    def _create_default_config(self) -> DashboardConfig:
        """Create default dashboard configuration"""
        return DashboardConfig(
            dashboard_id="main_dashboard",
            name="Forex Bot Monitoring Dashboard",
            description="Real-time monitoring dashboard for forex trading system",
            layout=DashboardLayout.GRID,
            grid_columns=4,
            grid_rows=6,
            auto_refresh=True,
            refresh_interval_seconds=5.0,
            theme="dark",
            color_scheme="default"
        )
    
    def _initialize_widgets(self) -> None:
        """Initialize default widgets"""
        # System overview widgets
        self._add_widget(WidgetConfig(
            widget_id="system_status",
            widget_type=DashboardWidget.STATUS_INDICATOR,
            title="System Status",
            description="Overall system health status",
            width=1,
            height=1,
            position_x=0,
            position_y=0
        ))
        
        # Performance metrics
        self._add_widget(WidgetConfig(
            widget_id="latency_metric",
            widget_type=DashboardWidget.METRIC_CARD,
            title="Avg Latency",
            description="Average data ingestion latency",
            metric_name="ingestion_latency_ms",
            width=1,
            height=1,
            position_x=1,
            position_y=0,
            warning_threshold=500.0,
            critical_threshold=1000.0
        ))
        
        self._add_widget(WidgetConfig(
            widget_id="throughput_metric",
            widget_type=DashboardWidget.METRIC_CARD,
            title="Throughput",
            description="Data ingestion throughput",
            metric_name="ingestion_throughput_ops_per_second",
            width=1,
            height=1,
            position_x=2,
            position_y=0
        ))
        
        self._add_widget(WidgetConfig(
            widget_id="error_rate_metric",
            widget_type=DashboardWidget.METRIC_CARD,
            title="Error Rate",
            description="Data ingestion error rate",
            metric_name="ingestion_error_rate",
            width=1,
            height=1,
            position_x=3,
            position_y=0,
            warning_threshold=0.01,
            critical_threshold=0.05
        ))
        
        # Charts
        self._add_widget(WidgetConfig(
            widget_id="latency_chart",
            widget_type=DashboardWidget.LINE_CHART,
            title="Latency Trend",
            description="Data ingestion latency over time",
            metric_name="ingestion_latency_ms",
            width=2,
            height=2,
            position_x=0,
            position_y=1,
            chart_type="line",
            x_axis_label="Time",
            y_axis_label="Latency (ms)",
            time_range_minutes=60
        ))
        
        self._add_widget(WidgetConfig(
            widget_id="throughput_chart",
            widget_type=DashboardWidget.LINE_CHART,
            title="Throughput Trend",
            description="Data ingestion throughput over time",
            metric_name="ingestion_throughput_ops_per_second",
            width=2,
            height=2,
            position_x=2,
            position_y=1,
            chart_type="line",
            x_axis_label="Time",
            y_axis_label="Throughput (ops/s)",
            time_range_minutes=60
        ))
        
        # Alerts
        self._add_widget(WidgetConfig(
            widget_id="active_alerts",
            widget_type=DashboardWidget.ALERT_LIST,
            title="Active Alerts",
            description="Currently active alerts",
            width=4,
            height=2,
            position_x=0,
            position_y=3,
            refresh_interval_seconds=1.0
        ))
        
        # System metrics
        self._add_widget(WidgetConfig(
            widget_id="cpu_usage",
            widget_type=DashboardWidget.GAUGE,
            title="CPU Usage",
            description="System CPU usage percentage",
            metric_name="system_cpu_usage",
            width=1,
            height=2,
            position_x=0,
            position_y=5,
            warning_threshold=70.0,
            critical_threshold=90.0
        ))
        
        self._add_widget(WidgetConfig(
            widget_id="memory_usage",
            widget_type=DashboardWidget.GAUGE,
            title="Memory Usage",
            description="System memory usage percentage",
            metric_name="system_memory_usage",
            width=1,
            height=2,
            position_x=1,
            position_y=5,
            warning_threshold=80.0,
            critical_threshold=95.0
        ))
        
        self._add_widget(WidgetConfig(
            widget_id="disk_usage",
            widget_type=DashboardWidget.GAUGE,
            title="Disk Usage",
            description="System disk usage percentage",
            metric_name="system_disk_usage",
            width=1,
            height=2,
            position_x=2,
            position_y=5,
            warning_threshold=85.0,
            critical_threshold=95.0
        ))
        
        self._add_widget(WidgetConfig(
            widget_id="network_usage",
            widget_type=DashboardWidget.GAUGE,
            title="Network Usage",
            description="Network bandwidth usage percentage",
            metric_name="system_network_usage",
            width=1,
            height=2,
            position_x=3,
            position_y=5
        ))
    
    def _add_widget(self, widget_config: WidgetConfig) -> None:
        """Add widget to dashboard"""
        self.widget_configs[widget_config.widget_id] = widget_config
        self.dashboard_data[widget_config.widget_id] = []
        logger.info(f"Added widget: {widget_config.title}")
    
    async def start(self) -> None:
        """Start dashboard"""
        if self.running:
            logger.warning("Dashboard already running")
            return
        
        self.running = True
        
        # Start refresh task
        if self.config.auto_refresh:
            self.refresh_task = asyncio.create_task(self._refresh_loop())
        
        logger.info("Dashboard started")
    
    async def stop(self) -> None:
        """Stop dashboard"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel refresh task
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dashboard stopped")
    
    async def _refresh_loop(self) -> None:
        """Main refresh loop"""
        while self.running:
            try:
                await self.refresh_dashboard()
                await asyncio.sleep(self.config.refresh_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                await asyncio.sleep(5.0)
    
    async def refresh_dashboard(self) -> None:
        """Refresh dashboard data"""
        try:
            current_time = datetime.utcnow()
            
            # Refresh each widget
            for widget_id, widget_config in self.widget_configs.items():
                try:
                    await self._refresh_widget(widget_id, widget_config, current_time)
                except Exception as e:
                    logger.error(f"Error refreshing widget {widget_id}: {e}")
            
            # Update real-time metrics
            await self._update_real_time_metrics()
            
            # Update alert summary
            await self._update_alert_summary()
            
            # Call dashboard callbacks
            await self._notify_dashboard_callbacks()
            
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {e}")
    
    async def _refresh_widget(self, widget_id: str, widget_config: WidgetConfig, 
                            current_time: datetime) -> None:
        """Refresh individual widget"""
        # Check if widget needs refresh
        last_refresh = self.last_refresh_times.get(widget_id)
        if last_refresh and (current_time - last_refresh).total_seconds() < widget_config.refresh_interval_seconds:
            return
        
        # Get data for widget
        data = await self._get_widget_data(widget_id, widget_config, current_time)
        
        if data is not None:
            # Store data
            dashboard_data = DashboardData(
                timestamp=current_time,
                widget_id=widget_id,
                metric_name=widget_config.metric_name or "unknown",
                value=data,
                metadata={
                    'widget_type': widget_config.widget_type.value,
                    'title': widget_config.title
                }
            )
            
            self.dashboard_data[widget_id].append(dashboard_data)
            
            # Keep only recent data
            cutoff_time = current_time - timedelta(minutes=widget_config.time_range_minutes)
            self.dashboard_data[widget_id] = [
                d for d in self.dashboard_data[widget_id] 
                if d.timestamp >= cutoff_time
            ]
            
            # Update last refresh time
            self.last_refresh_times[widget_id] = current_time
            
            # Call data callbacks
            await self._notify_data_callbacks(widget_id, dashboard_data)
    
    async def _get_widget_data(self, widget_id: str, widget_config: WidgetConfig, 
                             current_time: datetime) -> Optional[float]:
        """Get data for widget"""
        try:
            if widget_config.metric_name:
                # Get metric from performance monitor
                metric_data = await self.performance_monitor.get_metric(
                    widget_config.metric_name,
                    widget_config.time_range_minutes
                )
                
                if metric_data:
                    # Apply aggregation
                    values = [point['value'] for point in metric_data]
                    if values:
                        return self._apply_aggregation(values, widget_config.aggregation_function)
            
            elif widget_id == "system_status":
                # Calculate overall system status
                return await self._calculate_system_status()
            
            elif widget_id == "active_alerts":
                # Get active alerts count
                active_alerts = self.alert_manager.get_active_alerts()
                return len(active_alerts)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting widget data for {widget_id}: {e}")
            return None
    
    def _apply_aggregation(self, values: List[float], aggregation_function: str) -> float:
        """Apply aggregation function to values"""
        if not values:
            return 0.0
        
        try:
            if aggregation_function == "avg":
                return statistics.mean(values)
            elif aggregation_function == "sum":
                return sum(values)
            elif aggregation_function == "min":
                return min(values)
            elif aggregation_function == "max":
                return max(values)
            elif aggregation_function == "count":
                return len(values)
            else:
                return statistics.mean(values)
        except Exception as e:
            logger.error(f"Error applying aggregation {aggregation_function}: {e}")
            return 0.0
    
    async def _calculate_system_status(self) -> float:
        """Calculate overall system status (0-100)"""
        try:
            # Get key metrics
            latency_data = await self.performance_monitor.get_metric("ingestion_latency_ms", 5)
            error_rate_data = await self.performance_monitor.get_metric("ingestion_error_rate", 5)
            
            # Calculate status score
            status_score = 100.0
            
            # Penalize high latency
            if latency_data:
                avg_latency = statistics.mean([point['value'] for point in latency_data])
                if avg_latency > 1000:  # 1 second
                    status_score -= 30
                elif avg_latency > 500:  # 500ms
                    status_score -= 15
            
            # Penalize high error rate
            if error_rate_data:
                avg_error_rate = statistics.mean([point['value'] for point in error_rate_data])
                if avg_error_rate > 0.05:  # 5%
                    status_score -= 40
                elif avg_error_rate > 0.01:  # 1%
                    status_score -= 20
            
            # Check active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            warning_alerts = [a for a in active_alerts if a.severity == AlertSeverity.WARNING]
            
            if critical_alerts:
                status_score -= 50
            elif warning_alerts:
                status_score -= 20
            
            return max(0.0, min(100.0, status_score))
            
        except Exception as e:
            logger.error(f"Error calculating system status: {e}")
            return 50.0  # Unknown status
    
    async def _update_real_time_metrics(self) -> None:
        """Update real-time metrics"""
        try:
            # Get current metrics
            current_time = datetime.utcnow()
            
            # System metrics
            self.real_time_metrics = {
                'timestamp': current_time.isoformat(),
                'system_status': await self._calculate_system_status(),
                'active_alerts': len(self.alert_manager.get_active_alerts()),
                'total_alerts': len(self.alert_manager.get_alert_history()),
                'uptime_seconds': time.time() - self.performance_monitor.start_time.timestamp() if hasattr(self.performance_monitor, 'start_time') else 0
            }
            
            # Add performance metrics
            for metric_name in ['ingestion_latency_ms', 'ingestion_throughput_ops_per_second', 'ingestion_error_rate']:
                metric_data = await self.performance_monitor.get_metric(metric_name, 1)
                if metric_data:
                    self.real_time_metrics[metric_name] = metric_data[-1]['value']
            
        except Exception as e:
            logger.error(f"Error updating real-time metrics: {e}")
    
    async def _update_alert_summary(self) -> None:
        """Update alert summary"""
        try:
            summary = self.alert_manager.get_alert_summary()
            self.alert_summary = summary
        except Exception as e:
            logger.error(f"Error updating alert summary: {e}")
    
    async def _notify_dashboard_callbacks(self) -> None:
        """Notify dashboard callbacks"""
        for callback in self.dashboard_callbacks:
            try:
                await callback(self.real_time_metrics, self.alert_summary)
            except Exception as e:
                logger.error(f"Error in dashboard callback: {e}")
    
    async def _notify_data_callbacks(self, widget_id: str, data: DashboardData) -> None:
        """Notify data callbacks"""
        for callback in self.data_callbacks:
            try:
                await callback(widget_id, data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    # Public API methods
    def get_dashboard_data(self, widget_id: Optional[str] = None) -> Dict[str, Any]:
        """Get dashboard data"""
        if widget_id:
            return {
                'widget_id': widget_id,
                'data': self.dashboard_data.get(widget_id, []),
                'config': self.widget_configs.get(widget_id)
            }
        else:
            return {
                'dashboard_config': self.config,
                'widgets': {
                    widget_id: {
                        'config': config,
                        'data': self.dashboard_data.get(widget_id, [])
                    }
                    for widget_id, config in self.widget_configs.items()
                },
                'real_time_metrics': self.real_time_metrics,
                'alert_summary': self.alert_summary
            }
    
    def get_widget_data(self, widget_id: str, time_range_minutes: int = 60) -> List[DashboardData]:
        """Get widget data for specific time range"""
        if widget_id not in self.dashboard_data:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_range_minutes)
        return [
            data for data in self.dashboard_data[widget_id]
            if data.timestamp >= cutoff_time
        ]
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics"""
        return self.real_time_metrics.copy()
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        return self.alert_summary.copy()
    
    def add_widget(self, widget_config: WidgetConfig) -> None:
        """Add widget to dashboard"""
        self._add_widget(widget_config)
    
    def remove_widget(self, widget_id: str) -> None:
        """Remove widget from dashboard"""
        if widget_id in self.widget_configs:
            del self.widget_configs[widget_id]
            if widget_id in self.dashboard_data:
                del self.dashboard_data[widget_id]
            logger.info(f"Removed widget: {widget_id}")
    
    def update_widget(self, widget_config: WidgetConfig) -> None:
        """Update widget configuration"""
        self.widget_configs[widget_config.widget_id] = widget_config
        logger.info(f"Updated widget: {widget_config.title}")
    
    # Callback management
    def add_dashboard_callback(self, callback: Callable) -> None:
        """Add dashboard callback"""
        self.dashboard_callbacks.append(callback)
    
    def add_data_callback(self, callback: Callable) -> None:
        """Add data callback"""
        self.data_callbacks.append(callback)
    
    # Export methods
    def export_dashboard_config(self) -> Dict[str, Any]:
        """Export dashboard configuration"""
        return {
            'dashboard_id': self.config.dashboard_id,
            'name': self.config.name,
            'description': self.config.description,
            'layout': self.config.layout.value,
            'grid_columns': self.config.grid_columns,
            'grid_rows': self.config.grid_rows,
            'auto_refresh': self.config.auto_refresh,
            'refresh_interval_seconds': self.config.refresh_interval_seconds,
            'theme': self.config.theme,
            'color_scheme': self.config.color_scheme,
            'widgets': [
                {
                    'widget_id': config.widget_id,
                    'widget_type': config.widget_type.value,
                    'title': config.title,
                    'description': config.description,
                    'metric_name': config.metric_name,
                    'width': config.width,
                    'height': config.height,
                    'position_x': config.position_x,
                    'position_y': config.position_y,
                    'refresh_interval_seconds': config.refresh_interval_seconds,
                    'warning_threshold': config.warning_threshold,
                    'critical_threshold': config.critical_threshold,
                    'time_range_minutes': config.time_range_minutes,
                    'aggregation_function': config.aggregation_function
                }
                for config in self.widget_configs.values()
            ]
        }
    
    def export_dashboard_data(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Export dashboard data"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(minutes=time_range_minutes)
        
        return {
            'export_timestamp': current_time.isoformat(),
            'time_range_minutes': time_range_minutes,
            'dashboard_config': self.export_dashboard_config(),
            'widget_data': {
                widget_id: [
                    {
                        'timestamp': data.timestamp.isoformat(),
                        'metric_name': data.metric_name,
                        'value': data.value,
                        'metadata': data.metadata
                    }
                    for data in data_list
                    if data.timestamp >= cutoff_time
                ]
                for widget_id, data_list in self.dashboard_data.items()
            },
            'real_time_metrics': self.real_time_metrics,
            'alert_summary': self.alert_summary
        }

# Example usage and testing
async def test_monitoring_dashboard():
    """Test the monitoring dashboard"""
    
    # Create performance monitor
    performance_monitor = PerformanceMonitor()
    await performance_monitor.start()
    
    # Create alert manager
    alert_manager = AlertManager()
    await alert_manager.start()
    
    # Create dashboard
    dashboard = MonitoringDashboard(performance_monitor, alert_manager)
    await dashboard.start()
    
    # Add callbacks
    async def dashboard_callback(metrics, alerts):
        print(f"Dashboard update: {metrics}")
    
    async def data_callback(widget_id, data):
        print(f"Data update for {widget_id}: {data.value}")
    
    dashboard.add_dashboard_callback(dashboard_callback)
    dashboard.add_data_callback(data_callback)
    
    # Simulate some metrics
    print("Simulating metrics...")
    for i in range(10):
        await performance_monitor.record_metric("ingestion_latency_ms", 100 + i * 10)
        await performance_monitor.record_metric("ingestion_throughput_ops_per_second", 50 + i * 5)
        await performance_monitor.record_metric("ingestion_error_rate", 0.01 + i * 0.001)
        await asyncio.sleep(1.0)
    
    # Get dashboard data
    dashboard_data = dashboard.get_dashboard_data()
    print(f"Dashboard data: {len(dashboard_data['widgets'])} widgets")
    
    # Export configuration
    config = dashboard.export_dashboard_config()
    print(f"Dashboard config: {config['name']}")
    
    # Stop dashboard
    await dashboard.stop()
    await alert_manager.stop()
    await performance_monitor.stop()

if __name__ == "__main__":
    asyncio.run(test_monitoring_dashboard())

