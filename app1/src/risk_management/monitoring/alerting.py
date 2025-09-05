#!/usr/bin/env python3
"""
Risk Alerting System

Comprehensive alerting system for risk management that provides
real-time notifications, escalation, and alert management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import uuid

from ..models import RiskLevel, RiskAlert, RiskEvent

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Alert delivery channels"""
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


class AlertStatus(Enum):
    """Alert status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class AlertingConfig:
    """Configuration for alerting system"""
    # Alert thresholds
    enable_alerts: bool = True
    min_severity: RiskLevel = RiskLevel.MEDIUM
    critical_immediate: bool = True
    
    # Rate limiting
    max_alerts_per_minute: int = 10
    max_alerts_per_hour: int = 50
    cooldown_minutes: int = 15
    
    # Delivery channels
    enabled_channels: List[AlertChannel] = None
    email_recipients: List[str] = None
    sms_recipients: List[str] = None
    webhook_urls: List[str] = None
    
    # Alert management
    max_alerts_history: int = 1000
    auto_acknowledge_hours: int = 24
    escalation_minutes: int = 30
    
    # Templates
    alert_templates: Dict[str, str] = None
    
    def __post_init__(self):
        if self.enabled_channels is None:
            self.enabled_channels = [AlertChannel.LOG, AlertChannel.DASHBOARD]
        if self.alert_templates is None:
            self.alert_templates = {
                "risk_event": "Risk Event: {description} (Severity: {severity})",
                "trading_state_change": "Trading State Changed: {trading_state}",
                "circuit_breaker": "Circuit Breaker {action}: {message}",
                "emergency_stop": "EMERGENCY STOP: {reason}",
                "recovery": "System Recovery: {message}"
            }


class RiskAlerting:
    """
    Comprehensive risk alerting system.
    
    Provides:
    - Real-time alert generation and delivery
    - Multiple delivery channels (log, email, SMS, webhook, dashboard)
    - Alert escalation and acknowledgment
    - Rate limiting and cooldown management
    - Alert templates and customization
    """
    
    def __init__(self, config: AlertingConfig = None):
        """Initialize alerting system"""
        self.config = config or AlertingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Alert storage
        self.alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        
        # Rate limiting
        self.alert_timestamps: List[datetime] = []
        self.hourly_alert_count = 0
        self.last_hour_reset = datetime.utcnow()
        
        # Delivery tracking
        self.delivery_status: Dict[str, AlertStatus] = {}
        self.failed_deliveries: List[str] = []
        
        # Callbacks
        self.on_alert_created: Optional[Callable] = None
        self.on_alert_acknowledged: Optional[Callable] = None
        self.on_alert_escalated: Optional[Callable] = None
        
        # Performance tracking
        self.alerts_created = 0
        self.alerts_delivered = 0
        self.alerts_acknowledged = 0
        self.alerts_failed = 0
        
        self.logger.info("RiskAlerting initialized")
    
    async def create_alert(self, alert_type: str, severity: RiskLevel, 
                          message: str, data: Dict[str, Any] = None) -> str:
        """
        Create a new risk alert.
        
        Args:
            alert_type: Type of alert
            severity: Risk severity level
            message: Alert message
            data: Additional alert data
            
        Returns:
            Alert ID
        """
        try:
            # Check if alerting is enabled
            if not self.config.enable_alerts:
                return None
            
            # Check severity threshold
            if severity.value < self.config.min_severity.value:
                return None
            
            # Check rate limiting
            if not self._check_rate_limits():
                self.logger.warning("Alert rate limit exceeded, skipping alert")
                return None
            
            # Generate alert ID
            alert_id = str(uuid.uuid4())
            
            # Create alert
            alert = RiskAlert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                timestamp=datetime.utcnow(),
                data=data or {}
            )
            
            # Store alert
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.delivery_status[alert_id] = AlertStatus.PENDING
            
            # Update rate limiting
            self.alert_timestamps.append(datetime.utcnow())
            self.hourly_alert_count += 1
            
            # Deliver alert
            await self._deliver_alert(alert)
            
            # Trigger callback
            if self.on_alert_created:
                await self.on_alert_created(alert)
            
            self.alerts_created += 1
            self.logger.info(f"Created alert: {alert_type} - {message}")
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
            return None
    
    def _check_rate_limits(self) -> bool:
        """Check if alert creation is within rate limits"""
        current_time = datetime.utcnow()
        
        # Reset hourly counter if needed
        if (current_time - self.last_hour_reset).total_seconds() >= 3600:
            self.hourly_alert_count = 0
            self.last_hour_reset = current_time
        
        # Check hourly limit
        if self.hourly_alert_count >= self.config.max_alerts_per_hour:
            return False
        
        # Check minute limit
        minute_ago = current_time - timedelta(minutes=1)
        recent_alerts = [t for t in self.alert_timestamps if t > minute_ago]
        
        if len(recent_alerts) >= self.config.max_alerts_per_minute:
            return False
        
        return True
    
    async def _deliver_alert(self, alert: RiskAlert):
        """Deliver alert through configured channels"""
        try:
            # Check cooldown for similar alerts
            if self._is_in_cooldown(alert):
                self.logger.debug(f"Alert {alert.alert_id} in cooldown, skipping delivery")
                return
            
            # Deliver through each enabled channel
            for channel in self.config.enabled_channels:
                try:
                    await self._deliver_to_channel(alert, channel)
                    self.alerts_delivered += 1
                except Exception as e:
                    self.logger.error(f"Failed to deliver alert to {channel.value}: {e}")
                    self.alerts_failed += 1
                    self.failed_deliveries.append(alert.alert_id)
            
            # Update delivery status
            self.delivery_status[alert.alert_id] = AlertStatus.SENT
            
        except Exception as e:
            self.logger.error(f"Error delivering alert: {e}")
            self.delivery_status[alert.alert_id] = AlertStatus.FAILED
    
    def _is_in_cooldown(self, alert: RiskAlert) -> bool:
        """Check if alert is in cooldown period"""
        cooldown_cutoff = datetime.utcnow() - timedelta(minutes=self.config.cooldown_minutes)
        
        # Check for similar alerts in cooldown period
        for existing_alert in self.alert_history:
            if (existing_alert.alert_type == alert.alert_type and
                existing_alert.severity == alert.severity and
                existing_alert.timestamp > cooldown_cutoff):
                return True
        
        return False
    
    async def _deliver_to_channel(self, alert: RiskAlert, channel: AlertChannel):
        """Deliver alert to specific channel"""
        if channel == AlertChannel.LOG:
            await self._deliver_to_log(alert)
        elif channel == AlertChannel.EMAIL:
            await self._deliver_to_email(alert)
        elif channel == AlertChannel.SMS:
            await self._deliver_to_sms(alert)
        elif channel == AlertChannel.WEBHOOK:
            await self._deliver_to_webhook(alert)
        elif channel == AlertChannel.DASHBOARD:
            await self._deliver_to_dashboard(alert)
    
    async def _deliver_to_log(self, alert: RiskAlert):
        """Deliver alert to log"""
        log_level = {
            RiskLevel.LOW: logging.INFO,
            RiskLevel.MEDIUM: logging.WARNING,
            RiskLevel.HIGH: logging.ERROR,
            RiskLevel.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        self.logger.log(log_level, f"RISK ALERT [{alert.alert_type}]: {alert.message}")
    
    async def _deliver_to_email(self, alert: RiskAlert):
        """Deliver alert via email"""
        if not self.config.email_recipients:
            return
        
        # This would integrate with an email service
        # For now, just log the email content
        subject = f"Risk Alert: {alert.alert_type}"
        body = self._format_alert_message(alert)
        
        self.logger.info(f"EMAIL ALERT to {self.config.email_recipients}: {subject}")
        self.logger.info(f"Email body: {body}")
    
    async def _deliver_to_sms(self, alert: RiskAlert):
        """Deliver alert via SMS"""
        if not self.config.sms_recipients:
            return
        
        # This would integrate with an SMS service
        # For now, just log the SMS content
        message = f"Risk Alert: {alert.message}"
        
        self.logger.info(f"SMS ALERT to {self.config.sms_recipients}: {message}")
    
    async def _deliver_to_webhook(self, alert: RiskAlert):
        """Deliver alert via webhook"""
        if not self.config.webhook_urls:
            return
        
        # This would make HTTP requests to webhook URLs
        # For now, just log the webhook content
        payload = {
            "alert_id": alert.alert_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "data": alert.data
        }
        
        self.logger.info(f"WEBHOOK ALERT to {self.config.webhook_urls}: {payload}")
    
    async def _deliver_to_dashboard(self, alert: RiskAlert):
        """Deliver alert to dashboard"""
        # This would update the dashboard in real-time
        # For now, just log the dashboard update
        self.logger.info(f"DASHBOARD ALERT: {alert.alert_type} - {alert.message}")
    
    def _format_alert_message(self, alert: RiskAlert) -> str:
        """Format alert message using templates"""
        template = self.config.alert_templates.get(alert.alert_type, "{message}")
        
        return template.format(
            message=alert.message,
            severity=alert.severity.value,
            alert_type=alert.alert_type,
            timestamp=alert.timestamp.isoformat(),
            **alert.data
        )
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                
                # Update delivery status
                self.delivery_status[alert_id] = AlertStatus.ACKNOWLEDGED
                
                # Trigger callback
                if self.on_alert_acknowledged:
                    await self.on_alert_acknowledged(alert)
                
                self.alerts_acknowledged += 1
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
    
    async def get_active_alerts(self) -> List[RiskAlert]:
        """Get all active (unacknowledged) alerts"""
        return [
            alert for alert in self.alerts.values()
            if not alert.acknowledged
        ]
    
    async def get_alerts_by_severity(self, severity: RiskLevel) -> List[RiskAlert]:
        """Get alerts by severity level"""
        return [
            alert for alert in self.alerts.values()
            if alert.severity == severity
        ]
    
    async def get_alerts_by_type(self, alert_type: str) -> List[RiskAlert]:
        """Get alerts by type"""
        return [
            alert for alert in self.alerts.values()
            if alert.alert_type == alert_type
        ]
    
    async def get_recent_alerts(self, hours_back: int = 24) -> List[RiskAlert]:
        """Get recent alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        return [
            alert for alert in self.alerts.values()
            if alert.timestamp > cutoff_time
        ]
    
    async def escalate_alert(self, alert_id: str, escalation_reason: str):
        """Escalate an alert"""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                
                # Create escalation alert
                escalation_alert = RiskAlert(
                    alert_id=f"escalation_{alert_id}",
                    alert_type="escalation",
                    severity=RiskLevel.CRITICAL,
                    message=f"Alert escalated: {alert.message} - Reason: {escalation_reason}",
                    timestamp=datetime.utcnow(),
                    data={
                        "original_alert_id": alert_id,
                        "escalation_reason": escalation_reason
                    }
                )
                
                # Store escalation alert
                self.alerts[escalation_alert.alert_id] = escalation_alert
                self.alert_history.append(escalation_alert)
                
                # Deliver escalation alert
                await self._deliver_alert(escalation_alert)
                
                # Trigger callback
                if self.on_alert_escalated:
                    await self.on_alert_escalated(alert, escalation_alert)
                
                self.logger.warning(f"Alert {alert_id} escalated: {escalation_reason}")
            
        except Exception as e:
            self.logger.error(f"Error escalating alert: {e}")
    
    async def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Clean up old alerts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            # Remove old alerts from active alerts
            old_alert_ids = [
                alert_id for alert_id, alert in self.alerts.items()
                if alert.timestamp < cutoff_time
            ]
            
            for alert_id in old_alert_ids:
                del self.alerts[alert_id]
                if alert_id in self.delivery_status:
                    del self.delivery_status[alert_id]
            
            # Clean up alert history
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
            
            # Clean up rate limiting data
            self.alert_timestamps = [
                timestamp for timestamp in self.alert_timestamps
                if timestamp > cutoff_time
            ]
            
            self.logger.info(f"Cleaned up {len(old_alert_ids)} old alerts")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up alerts: {e}")
    
    def get_alerting_summary(self) -> Dict[str, Any]:
        """Get alerting system summary"""
        active_alerts = [alert for alert in self.alerts.values() if not alert.acknowledged]
        
        return {
            "enabled": self.config.enable_alerts,
            "active_alerts_count": len(active_alerts),
            "total_alerts_created": self.alerts_created,
            "alerts_delivered": self.alerts_delivered,
            "alerts_acknowledged": self.alerts_acknowledged,
            "alerts_failed": self.alerts_failed,
            "enabled_channels": [channel.value for channel in self.config.enabled_channels],
            "rate_limiting": {
                "alerts_last_hour": self.hourly_alert_count,
                "alerts_last_minute": len([t for t in self.alert_timestamps 
                                         if t > datetime.utcnow() - timedelta(minutes=1)])
            },
            "failed_deliveries": len(self.failed_deliveries),
            "last_cleanup": datetime.utcnow().isoformat()
        }
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        if not self.alert_history:
            return {"error": "No alert history available"}
        
        # Calculate statistics
        total_alerts = len(self.alert_history)
        acknowledged_alerts = sum(1 for alert in self.alert_history if alert.acknowledged)
        
        # Severity distribution
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Type distribution
        type_counts = {}
        for alert in self.alert_history:
            alert_type = alert.alert_type
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "acknowledged_alerts": acknowledged_alerts,
            "acknowledgment_rate": acknowledged_alerts / total_alerts if total_alerts > 0 else 0,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "average_alerts_per_hour": total_alerts / max(1, (datetime.utcnow() - self.alert_history[0].timestamp).total_seconds() / 3600)
        }

