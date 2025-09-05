#!/usr/bin/env python3
"""
Alert Manager
Real-time alerting and notification system for data ingestion monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"
    DASHBOARD = "dashboard"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., ">", "<", "==", "!=", ">=", "<="
    threshold_value: float
    severity: AlertSeverity
    channels: List[AlertChannel]
    
    # Timing configuration
    evaluation_interval_seconds: float = 30.0
    min_duration_seconds: float = 0.0  # Minimum duration before alert triggers
    cooldown_seconds: float = 300.0  # Cooldown between alerts
    
    # Suppression configuration
    suppress_during_maintenance: bool = True
    maintenance_windows: List[Dict[str, str]] = field(default_factory=list)
    
    # Escalation configuration
    escalation_enabled: bool = False
    escalation_delay_seconds: float = 1800.0  # 30 minutes
    escalation_severity: AlertSeverity = AlertSeverity.CRITICAL
    escalation_channels: List[AlertChannel] = field(default_factory=list)
    
    # Additional configuration
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    
    # Alert data
    metric_name: str
    current_value: float
    threshold_value: float
    condition: str
    
    # Context
    tags: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    # Notification
    channels: List[AlertChannel] = field(default_factory=list)
    notification_sent: bool = False
    notification_attempts: int = 0
    last_notification_at: Optional[datetime] = None
    
    # Escalation
    escalated: bool = False
    escalation_triggered_at: Optional[datetime] = None

@dataclass
class NotificationConfig:
    """Notification configuration"""
    # Email configuration
    smtp_server: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    from_email: str = "alerts@forex-bot.com"
    to_emails: List[str] = field(default_factory=list)
    
    # Slack configuration
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    slack_username: str = "Forex Bot Alert"
    
    # Webhook configuration
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    # Rate limiting
    max_notifications_per_minute: int = 60
    max_notifications_per_hour: int = 1000

class AlertManager:
    """Real-time alerting and notification system"""
    
    def __init__(self, notification_config: Optional[NotificationConfig] = None):
        """Initialize alert manager"""
        self.notification_config = notification_config or NotificationConfig()
        
        # Alert state
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Rule evaluation state
        self.rule_states: Dict[str, Dict[str, Any]] = {}
        self.last_evaluation_times: Dict[str, datetime] = {}
        
        # Notification state
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.notification_rates: Dict[str, List[datetime]] = {}
        
        # Task management
        self.evaluation_task: Optional[asyncio.Task] = None
        self.notification_task: Optional[asyncio.Task] = None
        self.running: bool = False
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.notification_callbacks: List[Callable] = []
        
        logger.info("AlertManager initialized")
    
    async def start(self) -> None:
        """Start alert manager"""
        if self.running:
            logger.warning("AlertManager already running")
            return
        
        self.running = True
        
        # Start evaluation task
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        
        # Start notification task
        self.notification_task = asyncio.create_task(self._notification_loop())
        
        logger.info("AlertManager started")
    
    async def stop(self) -> None:
        """Stop alert manager"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel tasks
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        
        if self.notification_task:
            self.notification_task.cancel()
            try:
                await self.notification_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AlertManager stopped")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule"""
        self.alert_rules[rule.rule_id] = rule
        self.rule_states[rule.rule_id] = {
            'last_triggered': None,
            'consecutive_violations': 0,
            'in_cooldown': False,
            'cooldown_until': None
        }
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> None:
        """Remove alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            if rule_id in self.rule_states:
                del self.rule_states[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def update_alert_rule(self, rule: AlertRule) -> None:
        """Update alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Updated alert rule: {rule.name}")
    
    async def evaluate_metric(self, metric_name: str, value: float, 
                            context: Dict[str, Any] = None) -> None:
        """Evaluate metric against alert rules"""
        if not self.running:
            return
        
        context = context or {}
        current_time = datetime.utcnow()
        
        # Find rules for this metric
        matching_rules = [
            rule for rule in self.alert_rules.values()
            if rule.enabled and rule.metric_name == metric_name
        ]
        
        for rule in matching_rules:
            try:
                await self._evaluate_rule(rule, value, context, current_time)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule, value: float, 
                           context: Dict[str, Any], current_time: datetime) -> None:
        """Evaluate a single rule"""
        rule_state = self.rule_states[rule.rule_id]
        
        # Check if rule is in cooldown
        if rule_state['in_cooldown'] and rule_state['cooldown_until']:
            if current_time < rule_state['cooldown_until']:
                return
            else:
                rule_state['in_cooldown'] = False
                rule_state['cooldown_until'] = None
        
        # Check if in maintenance window
        if rule.suppress_during_maintenance and self._is_in_maintenance_window(rule, current_time):
            return
        
        # Evaluate condition
        condition_met = self._evaluate_condition(value, rule.condition, rule.threshold_value)
        
        if condition_met:
            # Increment consecutive violations
            rule_state['consecutive_violations'] += 1
            
            # Check if minimum duration is met
            if rule_state['consecutive_violations'] == 1:
                rule_state['first_violation_time'] = current_time
            
            first_violation_time = rule_state.get('first_violation_time', current_time)
            violation_duration = (current_time - first_violation_time).total_seconds()
            
            if violation_duration >= rule.min_duration_seconds:
                # Trigger alert
                await self._trigger_alert(rule, value, context, current_time)
                
                # Set cooldown
                rule_state['in_cooldown'] = True
                rule_state['cooldown_until'] = current_time + timedelta(seconds=rule.cooldown_seconds)
                rule_state['last_triggered'] = current_time
        else:
            # Reset consecutive violations
            rule_state['consecutive_violations'] = 0
            rule_state['first_violation_time'] = None
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate condition"""
        try:
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return abs(value - threshold) < 0.0001  # Float comparison
            elif condition == "!=":
                return abs(value - threshold) >= 0.0001
            else:
                logger.error(f"Unknown condition: {condition}")
                return False
        except Exception as e:
            logger.error(f"Error evaluating condition {condition}: {e}")
            return False
    
    def _is_in_maintenance_window(self, rule: AlertRule, current_time: datetime) -> bool:
        """Check if current time is in maintenance window"""
        if not rule.maintenance_windows:
            return False
        
        current_weekday = current_time.weekday()  # 0 = Monday, 6 = Sunday
        current_time_str = current_time.strftime("%H:%M")
        
        for window in rule.maintenance_windows:
            start_day = int(window.get('start_day', 0))
            end_day = int(window.get('end_day', 6))
            start_time = window.get('start_time', '00:00')
            end_time = window.get('end_time', '23:59')
            
            # Check if current day is in range
            if start_day <= current_weekday <= end_day:
                # Check if current time is in range
                if start_time <= current_time_str <= end_time:
                    return True
        
        return False
    
    async def _trigger_alert(self, rule: AlertRule, value: float, 
                           context: Dict[str, Any], current_time: datetime) -> None:
        """Trigger alert"""
        # Check if alert already exists for this rule
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.rule_id and alert.status == AlertStatus.ACTIVE:
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.updated_at = current_time
            existing_alert.current_value = value
            existing_alert.context_data.update(context)
            logger.info(f"Updated existing alert: {rule.name}")
        else:
            # Create new alert
            alert_id = f"{rule.rule_id}_{int(current_time.timestamp())}"
            
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                title=f"{rule.name} - {rule.severity.value.upper()}",
                description=f"{rule.description}\nCurrent value: {value}, Threshold: {rule.threshold_value}",
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                created_at=current_time,
                updated_at=current_time,
                metric_name=rule.metric_name,
                current_value=value,
                threshold_value=rule.threshold_value,
                condition=rule.condition,
                tags=rule.tags.copy(),
                context_data=context.copy(),
                channels=rule.channels.copy()
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            logger.warning(f"Alert triggered: {rule.name} - {value} {rule.condition} {rule.threshold_value}")
            
            # Queue notification
            await self.notification_queue.put(alert)
            
            # Call alert callbacks
            await self._notify_alert_callbacks(alert)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = None) -> bool:
        """Acknowledge alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.updated_at = datetime.utcnow()
        
        if acknowledged_by:
            alert.context_data['acknowledged_by'] = acknowledged_by
        
        logger.info(f"Alert acknowledged: {alert_id}")
        return True
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """Resolve alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        
        if resolved_by:
            alert.context_data['resolved_by'] = resolved_by
        
        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert_id}")
        return True
    
    async def _evaluation_loop(self) -> None:
        """Main evaluation loop"""
        while self.running:
            try:
                # Process any pending evaluations
                await asyncio.sleep(1.0)  # Small delay to prevent busy waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _notification_loop(self) -> None:
        """Notification processing loop"""
        while self.running:
            try:
                # Get next notification
                alert = await asyncio.wait_for(self.notification_queue.get(), timeout=1.0)
                
                # Send notification
                await self._send_notification(alert)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in notification loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _send_notification(self, alert: Alert) -> None:
        """Send notification for alert"""
        try:
            # Check rate limits
            if not self._check_rate_limits(alert):
                logger.warning(f"Rate limit exceeded for alert {alert.alert_id}")
                return
            
            # Send to each channel
            for channel in alert.channels:
                try:
                    if channel == AlertChannel.EMAIL:
                        await self._send_email_notification(alert)
                    elif channel == AlertChannel.SLACK:
                        await self._send_slack_notification(alert)
                    elif channel == AlertChannel.WEBHOOK:
                        await self._send_webhook_notification(alert)
                    elif channel == AlertChannel.LOG:
                        await self._send_log_notification(alert)
                    elif channel == AlertChannel.DASHBOARD:
                        await self._send_dashboard_notification(alert)
                    
                except Exception as e:
                    logger.error(f"Error sending notification via {channel.value}: {e}")
            
            # Update notification status
            alert.notification_sent = True
            alert.notification_attempts += 1
            alert.last_notification_at = datetime.utcnow()
            
            # Call notification callbacks
            await self._notify_notification_callbacks(alert)
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def _check_rate_limits(self, alert: Alert) -> bool:
        """Check rate limits for notifications"""
        current_time = datetime.utcnow()
        
        # Check per-minute limit
        minute_key = f"minute_{current_time.strftime('%Y%m%d%H%M')}"
        if minute_key not in self.notification_rates:
            self.notification_rates[minute_key] = []
        
        minute_notifications = self.notification_rates[minute_key]
        minute_notifications = [t for t in minute_notifications if (current_time - t).total_seconds() < 60]
        
        if len(minute_notifications) >= self.notification_config.max_notifications_per_minute:
            return False
        
        # Check per-hour limit
        hour_key = f"hour_{current_time.strftime('%Y%m%d%H')}"
        if hour_key not in self.notification_rates:
            self.notification_rates[hour_key] = []
        
        hour_notifications = self.notification_rates[hour_key]
        hour_notifications = [t for t in hour_notifications if (current_time - t).total_seconds() < 3600]
        
        if len(hour_notifications) >= self.notification_config.max_notifications_per_hour:
            return False
        
        # Add current notification
        minute_notifications.append(current_time)
        hour_notifications.append(current_time)
        
        self.notification_rates[minute_key] = minute_notifications
        self.notification_rates[hour_key] = hour_notifications
        
        return True
    
    async def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification"""
        if not self.notification_config.to_emails:
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.notification_config.from_email
            msg['To'] = ', '.join(self.notification_config.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create body
            body = f"""
Alert Details:
- Title: {alert.title}
- Description: {alert.description}
- Severity: {alert.severity.value.upper()}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold_value} {alert.condition}
- Created: {alert.created_at.isoformat()}
- Alert ID: {alert.alert_id}

Context Data:
{json.dumps(alert.context_data, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.notification_config.smtp_server, self.notification_config.smtp_port) as server:
                if self.notification_config.smtp_use_tls:
                    server.starttls()
                
                if self.notification_config.smtp_username:
                    server.login(self.notification_config.smtp_username, self.notification_config.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert) -> None:
        """Send Slack notification"""
        if not self.notification_config.slack_webhook_url:
            return
        
        try:
            # Create Slack message
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            payload = {
                "channel": self.notification_config.slack_channel,
                "username": self.notification_config.slack_username,
                "attachments": [{
                    "color": color_map.get(alert.severity, "good"),
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold_value} {alert.condition}", "short": True},
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {"title": "Created", "value": alert.created_at.isoformat(), "short": True}
                    ],
                    "footer": "Forex Bot Alert System",
                    "ts": int(alert.created_at.timestamp())
                }]
            }
            
            # Send to Slack
            response = requests.post(
                self.notification_config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert) -> None:
        """Send webhook notification"""
        if not self.notification_config.webhook_url:
            return
        
        try:
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "condition": alert.condition,
                "created_at": alert.created_at.isoformat(),
                "updated_at": alert.updated_at.isoformat(),
                "tags": alert.tags,
                "context_data": alert.context_data
            }
            
            # Send webhook
            response = requests.post(
                self.notification_config.webhook_url,
                json=payload,
                headers=self.notification_config.webhook_headers,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    async def _send_log_notification(self, alert: Alert) -> None:
        """Send log notification"""
        logger.warning(f"ALERT: {alert.title} - {alert.description}")
    
    async def _send_dashboard_notification(self, alert: Alert) -> None:
        """Send dashboard notification (placeholder)"""
        # This would typically update a real-time dashboard
        logger.info(f"Dashboard notification: {alert.title}")
    
    async def _notify_alert_callbacks(self, alert: Alert) -> None:
        """Notify alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def _notify_notification_callbacks(self, alert: Alert) -> None:
        """Notify notification callbacks"""
        for callback in self.notification_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    # Callback management
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def add_notification_callback(self, callback: Callable) -> None:
        """Add notification callback"""
        self.notification_callbacks.append(callback)
    
    # Status and query methods
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return self.alert_history[-limit:]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        return [alert for alert in self.active_alerts.values() if alert.severity == severity]
    
    def get_alerts_by_rule(self, rule_id: str) -> List[Alert]:
        """Get alerts by rule ID"""
        return [alert for alert in self.active_alerts.values() if alert.rule_id == rule_id]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_count = len(self.active_alerts)
        total_count = len(self.alert_history)
        
        severity_counts = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'active_alerts': active_count,
            'total_alerts': total_count,
            'severity_breakdown': severity_counts,
            'rules_configured': len(self.alert_rules),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled])
        }

# Example usage and testing
async def test_alert_manager():
    """Test the alert manager"""
    
    # Create notification config
    notification_config = NotificationConfig(
        to_emails=["admin@forex-bot.com"],
        slack_webhook_url="https://hooks.slack.com/services/...",
        webhook_url="https://api.example.com/alerts"
    )
    
    # Create alert manager
    alert_manager = AlertManager(notification_config)
    
    # Add alert rules
    high_latency_rule = AlertRule(
        rule_id="high_latency",
        name="High Data Ingestion Latency",
        description="Data ingestion latency is above threshold",
        metric_name="ingestion_latency_ms",
        condition=">",
        threshold_value=1000.0,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.LOG, AlertChannel.EMAIL],
        evaluation_interval_seconds=30.0,
        min_duration_seconds=60.0,
        cooldown_seconds=300.0
    )
    
    error_rate_rule = AlertRule(
        rule_id="high_error_rate",
        name="High Error Rate",
        description="Error rate is above threshold",
        metric_name="error_rate",
        condition=">",
        threshold_value=0.05,  # 5%
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.EMAIL],
        evaluation_interval_seconds=30.0,
        min_duration_seconds=30.0,
        cooldown_seconds=600.0
    )
    
    alert_manager.add_alert_rule(high_latency_rule)
    alert_manager.add_alert_rule(error_rate_rule)
    
    # Add callbacks
    async def alert_callback(alert):
        print(f"Alert callback: {alert.title}")
    
    async def notification_callback(alert):
        print(f"Notification callback: {alert.title}")
    
    alert_manager.add_alert_callback(alert_callback)
    alert_manager.add_notification_callback(notification_callback)
    
    # Start alert manager
    await alert_manager.start()
    
    # Simulate metrics
    print("Simulating metrics...")
    
    # Normal latency
    await alert_manager.evaluate_metric("ingestion_latency_ms", 500.0)
    await alert_manager.evaluate_metric("error_rate", 0.01)
    
    # High latency (should trigger alert)
    await alert_manager.evaluate_metric("ingestion_latency_ms", 1500.0)
    await alert_manager.evaluate_metric("error_rate", 0.08)  # Should trigger critical alert
    
    # Wait for processing
    await asyncio.sleep(2.0)
    
    # Get summary
    summary = alert_manager.get_alert_summary()
    print(f"Alert summary: {summary}")
    
    # Get active alerts
    active_alerts = alert_manager.get_active_alerts()
    print(f"Active alerts: {len(active_alerts)}")
    
    # Stop alert manager
    await alert_manager.stop()

if __name__ == "__main__":
    asyncio.run(test_alert_manager())

