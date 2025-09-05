"""
Test Trading Suspension System

Tests for the automated trading suspension functionality.
"""

import unittest
import asyncio
from datetime import datetime, timedelta

from src.risk_management.controls.trading_suspension import (
    TradingSuspensionManager, SuspensionConfig, SuspensionRule,
    SuspensionReason, SuspensionLevel, SuspensionEvent
)
from src.risk_management.models import (
    PortfolioData, RiskMetrics, RiskLevel, RiskEvent
)


class TestTradingSuspensionManager(unittest.TestCase):
    """Test trading suspension manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SuspensionConfig()
        self.suspension_manager = TradingSuspensionManager(self.config)
        
        # Create test portfolio data
        self.portfolio_data = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-2000,
            unrealized_pnl=-2000,
            realized_pnl=0,
            max_drawdown=0.10,
            current_drawdown=0.05,
            positions=[]
        )
        
        # Create test risk metrics
        self.risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.08,
            risk_per_trade=0.02,
            max_drawdown=0.10,
            current_drawdown=0.05,
            var_95=0.04,
            var_99=0.06,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=0.8,
            max_consecutive_losses=2,
            win_rate=0.6,
            profit_factor=1.4,
            recovery_factor=0.7
        )
    
    def test_suspension_manager_initialization(self):
        """Test suspension manager initialization"""
        self.assertIsNotNone(self.suspension_manager.config)
        self.assertIsNone(self.suspension_manager.current_suspension)
        self.assertEqual(len(self.suspension_manager.suspension_history), 0)
        self.assertIsInstance(self.suspension_manager.suspension_history, list)
    
    def test_no_suspension_conditions(self):
        """Test when no suspension conditions are met"""
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, self.risk_metrics
        )
        
        self.assertIsNone(result)
        self.assertIsNone(self.suspension_manager.current_suspension)
    
    def test_risk_threshold_suspension(self):
        """Test suspension due to risk threshold breach"""
        # Create high risk metrics
        high_risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.20,  # Above threshold
            risk_per_trade=0.05,
            max_drawdown=0.15,
            current_drawdown=0.08,
            var_95=0.08,
            var_99=0.12,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            calmar_ratio=0.4,
            max_consecutive_losses=3,
            win_rate=0.5,
            profit_factor=1.0,
            recovery_factor=0.5
        )
        
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, high_risk_metrics
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, SuspensionReason.RISK_THRESHOLD_BREACH)
        self.assertEqual(result.level, SuspensionLevel.PARTIAL)
        self.assertFalse(result.resolved)
    
    def test_critical_risk_suspension(self):
        """Test suspension due to critical risk threshold"""
        # Create critical risk metrics
        critical_risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.30,  # Above critical threshold
            risk_per_trade=0.08,
            max_drawdown=0.25,
            current_drawdown=0.20,
            var_95=0.12,
            var_99=0.18,
            sharpe_ratio=0.5,
            sortino_ratio=0.7,
            calmar_ratio=0.2,
            max_consecutive_losses=6,
            win_rate=0.3,
            profit_factor=0.6,
            recovery_factor=0.2
        )
        
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, critical_risk_metrics
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, SuspensionReason.RISK_THRESHOLD_BREACH)
        self.assertEqual(result.level, SuspensionLevel.FULL)
    
    def test_drawdown_suspension(self):
        """Test suspension due to drawdown limit"""
        # Create high drawdown metrics
        high_drawdown_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.12,
            risk_per_trade=0.03,
            max_drawdown=0.15,
            current_drawdown=0.12,  # Above threshold
            var_95=0.06,
            var_99=0.09,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=0.6,
            max_consecutive_losses=3,
            win_rate=0.55,
            profit_factor=1.2,
            recovery_factor=0.6
        )
        
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, high_drawdown_metrics
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, SuspensionReason.DRAWDOWN_LIMIT)
        self.assertEqual(result.level, SuspensionLevel.PARTIAL)
    
    def test_consecutive_losses_suspension(self):
        """Test suspension due to consecutive losses"""
        # Create metrics with high consecutive losses
        high_losses_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.10,
            risk_per_trade=0.03,
            max_drawdown=0.12,
            current_drawdown=0.08,
            var_95=0.05,
            var_99=0.08,
            sharpe_ratio=0.9,
            sortino_ratio=1.1,
            calmar_ratio=0.5,
            max_consecutive_losses=6,  # Above threshold
            win_rate=0.45,
            profit_factor=0.9,
            recovery_factor=0.4
        )
        
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, high_losses_metrics
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, SuspensionReason.CONSECUTIVE_LOSSES)
        self.assertEqual(result.level, SuspensionLevel.PARTIAL)
    
    def test_volatility_spike_suspension(self):
        """Test suspension due to volatility spike"""
        market_data = {
            'volatility': 0.40  # Above threshold
        }
        
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, self.risk_metrics, market_data
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, SuspensionReason.VOLATILITY_SPIKE)
        self.assertEqual(result.level, SuspensionLevel.PARTIAL)
    
    def test_correlation_spike_suspension(self):
        """Test suspension due to correlation spike"""
        market_data = {
            'max_correlation': 0.90  # Above threshold
        }
        
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, self.risk_metrics, market_data
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, SuspensionReason.CORRELATION_SPIKE)
        self.assertEqual(result.level, SuspensionLevel.PARTIAL)
    
    def test_high_impact_event_suspension(self):
        """Test suspension due to high impact events"""
        # Create high impact events
        high_impact_events = [
            RiskEvent(
                event_id="event_1",
                event_type="economic_event",
                risk_level=RiskLevel.HIGH,
                description="High impact event 1",
                timestamp=datetime.utcnow(),
                source="test"
            ),
            RiskEvent(
                event_id="event_2",
                event_type="economic_event",
                risk_level=RiskLevel.HIGH,
                description="High impact event 2",
                timestamp=datetime.utcnow(),
                source="test"
            )
        ]
        
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, self.risk_metrics, active_events=high_impact_events
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, SuspensionReason.HIGH_IMPACT_EVENT)
        self.assertEqual(result.level, SuspensionLevel.PARTIAL)
    
    def test_system_error_suspension(self):
        """Test suspension due to system errors"""
        market_data = {
            'errors_per_hour': 15  # Above threshold
        }
        
        result = self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, self.risk_metrics, market_data
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, SuspensionReason.SYSTEM_ERROR)
        self.assertEqual(result.level, SuspensionLevel.FULL)
    
    def test_suspension_status(self):
        """Test suspension status retrieval"""
        # No suspension
        status = self.suspension_manager.get_suspension_status()
        self.assertFalse(status['suspended'])
        self.assertEqual(status['level'], SuspensionLevel.NONE.value)
        
        # Create suspension
        high_risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.20,
            risk_per_trade=0.05,
            max_drawdown=0.15,
            current_drawdown=0.08,
            var_95=0.08,
            var_99=0.12,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            calmar_ratio=0.4,
            max_consecutive_losses=3,
            win_rate=0.5,
            profit_factor=1.0,
            recovery_factor=0.5
        )
        
        self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, high_risk_metrics
        )
        
        status = self.suspension_manager.get_suspension_status()
        self.assertTrue(status['suspended'])
        self.assertEqual(status['level'], SuspensionLevel.PARTIAL.value)
        self.assertEqual(status['reason'], SuspensionReason.RISK_THRESHOLD_BREACH.value)
    
    def test_suspension_history(self):
        """Test suspension history retrieval"""
        # Create multiple suspensions
        high_risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.20,
            risk_per_trade=0.05,
            max_drawdown=0.15,
            current_drawdown=0.08,
            var_95=0.08,
            var_99=0.12,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            calmar_ratio=0.4,
            max_consecutive_losses=3,
            win_rate=0.5,
            profit_factor=1.0,
            recovery_factor=0.5
        )
        
        # First suspension
        self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, high_risk_metrics
        )
        
        # Manual recovery
        self.suspension_manager.manual_recover()
        
        # Second suspension
        self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, high_risk_metrics
        )
        
        history = self.suspension_manager.get_suspension_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['reason'], SuspensionReason.RISK_THRESHOLD_BREACH.value)
        self.assertEqual(history[1]['reason'], SuspensionReason.RISK_THRESHOLD_BREACH.value)
    
    def test_manual_suspension(self):
        """Test manual suspension"""
        suspension = self.suspension_manager.manual_suspend(
            "Test manual suspension",
            SuspensionLevel.FULL,
            120
        )
        
        self.assertIsNotNone(suspension)
        self.assertEqual(suspension.reason, SuspensionReason.MANUAL_OVERRIDE)
        self.assertEqual(suspension.level, SuspensionLevel.FULL)
        self.assertEqual(suspension.duration_minutes, 120)
        self.assertIn("Test manual suspension", suspension.description)
    
    def test_manual_recovery(self):
        """Test manual recovery"""
        # Create suspension first
        high_risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.20,
            risk_per_trade=0.05,
            max_drawdown=0.15,
            current_drawdown=0.08,
            var_95=0.08,
            var_99=0.12,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            calmar_ratio=0.4,
            max_consecutive_losses=3,
            win_rate=0.5,
            profit_factor=1.0,
            recovery_factor=0.5
        )
        
        self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, high_risk_metrics
        )
        
        # Verify suspension exists
        self.assertIsNotNone(self.suspension_manager.current_suspension)
        
        # Manual recovery
        recovered = self.suspension_manager.manual_recover()
        self.assertTrue(recovered)
        self.assertIsNone(self.suspension_manager.current_suspension)
    
    def test_manual_recovery_no_suspension(self):
        """Test manual recovery when no suspension exists"""
        recovered = self.suspension_manager.manual_recover()
        self.assertFalse(recovered)
    
    def test_system_health(self):
        """Test system health monitoring"""
        market_data = {
            'latency_ms': 500,
            'errors_per_hour': 5,
            'memory_usage': 0.7,
            'cpu_usage': 0.6
        }
        
        self.suspension_manager.check_suspension_conditions(
            self.portfolio_data, self.risk_metrics, market_data
        )
        
        health = self.suspension_manager.get_system_health()
        self.assertIn('last_check', health)
        self.assertIn('health_metrics', health)
        self.assertIn('recovery_monitoring', health)
        self.assertEqual(health['health_metrics']['latency_ms'], 500)
        self.assertEqual(health['health_metrics']['errors_per_hour'], 5)
    
    def test_config_update(self):
        """Test configuration update"""
        new_config = SuspensionConfig(
            max_risk_threshold=0.20,
            critical_risk_threshold=0.30
        )
        
        self.suspension_manager.update_config(new_config)
        
        self.assertEqual(self.suspension_manager.config.max_risk_threshold, 0.20)
        self.assertEqual(self.suspension_manager.config.critical_risk_threshold, 0.30)


class TestSuspensionConfig(unittest.TestCase):
    """Test suspension configuration"""
    
    def test_config_creation(self):
        """Test configuration creation"""
        config = SuspensionConfig()
        
        self.assertEqual(config.max_risk_threshold, 0.15)
        self.assertEqual(config.critical_risk_threshold, 0.25)
        self.assertEqual(config.max_drawdown_threshold, 0.10)
        self.assertEqual(config.critical_drawdown_threshold, 0.20)
        self.assertTrue(config.auto_recovery_enabled)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = SuspensionConfig(
            max_risk_threshold=0.20,
            critical_risk_threshold=0.30,
            max_drawdown_threshold=0.15,
            auto_recovery_enabled=False
        )
        
        self.assertEqual(config.max_risk_threshold, 0.20)
        self.assertEqual(config.critical_risk_threshold, 0.30)
        self.assertEqual(config.max_drawdown_threshold, 0.15)
        self.assertFalse(config.auto_recovery_enabled)


class TestSuspensionRule(unittest.TestCase):
    """Test suspension rule"""
    
    def test_rule_creation(self):
        """Test rule creation"""
        def test_condition(context):
            return True
        
        rule = SuspensionRule(
            reason=SuspensionReason.RISK_THRESHOLD_BREACH,
            condition=test_condition,
            suspension_level=SuspensionLevel.PARTIAL,
            duration_minutes=60,
            priority=3
        )
        
        self.assertEqual(rule.reason, SuspensionReason.RISK_THRESHOLD_BREACH)
        self.assertEqual(rule.suspension_level, SuspensionLevel.PARTIAL)
        self.assertEqual(rule.duration_minutes, 60)
        self.assertEqual(rule.priority, 3)
        self.assertTrue(rule.enabled)


if __name__ == '__main__':
    unittest.main()
