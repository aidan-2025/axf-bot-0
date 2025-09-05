#!/usr/bin/env python3
"""
Test Risk Management System

Comprehensive test suite for the risk management and event avoidance system.
"""

import unittest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import risk management components
from src.risk_management.models import (
    RiskConfig, RiskLevel, RiskEvent, RiskAction, TradingState,
    PortfolioData, PositionData, EconomicEventData, SentimentData,
    EventImpact, SentimentLevel
)
from src.risk_management.core.risk_engine import RiskEngine, RiskEngineConfig
from src.risk_management.core.risk_manager import RiskManager, RiskManagerConfig
from src.risk_management.core.risk_metrics import RiskMetricsCalculator, RiskMetricsConfig
from src.risk_management.event_integration.event_monitor import EventMonitor, EventMonitorConfig
from src.risk_management.event_integration.sentiment_monitor import SentimentMonitor, SentimentMonitorConfig
from src.risk_management.controls.circuit_breakers import CircuitBreaker, CircuitBreakerConfig
from src.risk_management.monitoring.risk_dashboard import RiskDashboard, RiskDashboardConfig
from src.risk_management.monitoring.alerting import RiskAlerting, AlertingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRiskModels(unittest.TestCase):
    """Test risk management models"""
    
    def test_risk_level_enum(self):
        """Test risk level enum values"""
        self.assertEqual(RiskLevel.LOW.value, "low")
        self.assertEqual(RiskLevel.MEDIUM.value, "medium")
        self.assertEqual(RiskLevel.HIGH.value, "high")
        self.assertEqual(RiskLevel.CRITICAL.value, "critical")
    
    def test_risk_config_creation(self):
        """Test risk configuration creation"""
        config = RiskConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.max_drawdown_threshold, 0.15)
        self.assertEqual(config.max_risk_per_trade, 0.02)
    
    def test_portfolio_data_creation(self):
        """Test portfolio data creation"""
        positions = [
            PositionData(
                currency_pair="EUR/USD",
                size=10000,
                entry_price=1.0850,
                current_price=1.0820,
                unrealized_pnl=-300,
                realized_pnl=0,
                timestamp=datetime.utcnow()
            )
        ]
        
        portfolio = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-300,
            unrealized_pnl=-300,
            realized_pnl=0,
            max_drawdown=0.05,
            current_drawdown=0.003,
            positions=positions
        )
        
        self.assertEqual(portfolio.total_value, 100000)
        self.assertEqual(len(portfolio.positions), 1)
        self.assertEqual(portfolio.positions[0].currency_pair, "EUR/USD")


class TestRiskEngine(unittest.TestCase):
    """Test risk engine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = RiskEngineConfig()
        self.risk_engine = RiskEngine(self.config)
    
    def test_risk_engine_initialization(self):
        """Test risk engine initialization"""
        self.assertIsNotNone(self.risk_engine)
        self.assertEqual(len(self.risk_engine.risk_thresholds), 5)
        self.assertEqual(self.risk_engine.current_state.current_risk_level, RiskLevel.LOW)
    
    def test_risk_threshold_creation(self):
        """Test risk threshold creation"""
        threshold = self.risk_engine.risk_thresholds[0]
        self.assertEqual(threshold.name, "max_drawdown")
        self.assertEqual(threshold.threshold_value, 0.15)
        self.assertEqual(threshold.risk_level, RiskLevel.HIGH)
    
    async def test_risk_assessment(self):
        """Test comprehensive risk assessment"""
        # Create mock portfolio data
        portfolio_data = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-5000,
            unrealized_pnl=-5000,
            realized_pnl=0,
            max_drawdown=0.05,
            current_drawdown=0.05,
            positions=[]
        )
        
        # Perform risk assessment
        risk_state = await self.risk_engine.assess_risk(portfolio_data)
        
        self.assertIsNotNone(risk_state)
        self.assertIsNotNone(risk_state.current_risk_level)
        self.assertIsNotNone(risk_state.trading_state)
    
    def test_risk_calculation_methods(self):
        """Test risk calculation methods"""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        
        # Test VaR calculation
        var_95 = self.risk_engine._calculate_var(returns, 0.95)
        self.assertGreaterEqual(var_95, 0)
        
        # Test drawdown calculation
        max_dd, current_dd = self.risk_engine._calculate_drawdowns(returns)
        self.assertGreaterEqual(max_dd, 0)
        self.assertGreaterEqual(current_dd, 0)
        
        # Test Sharpe ratio calculation
        sharpe = self.risk_engine._calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)


class TestEventMonitor(unittest.TestCase):
    """Test event monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = EventMonitorConfig()
        self.event_monitor = EventMonitor(self.config)
    
    def test_event_monitor_initialization(self):
        """Test event monitor initialization"""
        self.assertIsNotNone(self.event_monitor)
        self.assertEqual(len(self.event_monitor.events_cache), 0)
        self.assertEqual(len(self.event_monitor.active_events), 0)
    
    async def test_mock_events_generation(self):
        """Test mock events generation"""
        events = await self.event_monitor._get_mock_events()
        
        self.assertIsInstance(events, list)
        self.assertGreater(len(events), 0)
        
        # Check event structure
        event = events[0]
        self.assertIsInstance(event, EconomicEventData)
        self.assertIsNotNone(event.event_id)
        self.assertIsNotNone(event.title)
        self.assertIsNotNone(event.impact)
    
    def test_event_relevance_check(self):
        """Test event relevance checking"""
        # Create test event
        event = EconomicEventData(
            event_id="test_1",
            title="Test Event",
            event_time=datetime.utcnow() + timedelta(hours=1),
            impact=EventImpact.HIGH,
            currency="USD",
            currency_pairs=["EUR/USD"],
            relevance_score=0.8
        )
        
        # Test relevance
        is_relevant = self.event_monitor._is_relevant_event(event)
        self.assertTrue(is_relevant, f"Event should be relevant. Currency: {event.currency}, Impact: {event.impact.value}, Relevance: {event.relevance_score}")
        
        # Test with low impact event
        event.impact = EventImpact.LOW
        is_relevant = self.event_monitor._is_relevant_event(event)
        self.assertFalse(is_relevant)
    
    async def test_event_processing(self):
        """Test event processing"""
        events = await self.event_monitor._get_mock_events()
        await self.event_monitor._process_events(events)
        
        # Check that events were processed
        self.assertGreater(len(self.event_monitor.events_cache), 0)
        self.assertGreater(len(self.event_monitor.active_events), 0)


class TestSentimentMonitor(unittest.TestCase):
    """Test sentiment monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SentimentMonitorConfig()
        self.sentiment_monitor = SentimentMonitor(self.config)
    
    def test_sentiment_monitor_initialization(self):
        """Test sentiment monitor initialization"""
        self.assertIsNotNone(self.sentiment_monitor)
        self.assertEqual(len(self.sentiment_monitor.sentiment_cache), 0)
        self.assertEqual(len(self.sentiment_monitor.current_sentiment), 0)
    
    async def test_mock_sentiment_generation(self):
        """Test mock sentiment generation"""
        sentiment_data = await self.sentiment_monitor._get_mock_sentiment()
        
        self.assertIsInstance(sentiment_data, list)
        self.assertGreater(len(sentiment_data), 0)
        
        # Check sentiment structure
        sentiment = sentiment_data[0]
        self.assertIsInstance(sentiment, SentimentData)
        self.assertIsNotNone(sentiment.currency_pair)
        self.assertIsNotNone(sentiment.sentiment_level)
        self.assertIsNotNone(sentiment.sentiment_score)
    
    def test_sentiment_level_conversion(self):
        """Test sentiment score to level conversion"""
        # Test very bearish
        level = self.sentiment_monitor._score_to_sentiment_level(-0.9)
        self.assertEqual(level, SentimentLevel.VERY_BEARISH)
        
        # Test bearish
        level = self.sentiment_monitor._score_to_sentiment_level(-0.6)
        self.assertEqual(level, SentimentLevel.BEARISH)
        
        # Test neutral (should be bullish based on thresholds)
        level = self.sentiment_monitor._score_to_sentiment_level(0.0)
        self.assertEqual(level, SentimentLevel.BULLISH)
        
        # Test bullish (0.6 should be VERY_BULLISH based on thresholds)
        level = self.sentiment_monitor._score_to_sentiment_level(0.6)
        self.assertEqual(level, SentimentLevel.VERY_BULLISH)
        
        # Test very bullish
        level = self.sentiment_monitor._score_to_sentiment_level(0.9)
        self.assertEqual(level, SentimentLevel.VERY_BULLISH)
    
    async def test_sentiment_processing(self):
        """Test sentiment processing"""
        sentiment_data = await self.sentiment_monitor._get_mock_sentiment()
        await self.sentiment_monitor._process_sentiment_data(sentiment_data)
        
        # Check that sentiment was processed
        self.assertGreater(len(self.sentiment_monitor.sentiment_cache), 0)
        self.assertGreater(len(self.sentiment_monitor.current_sentiment), 0)


class TestCircuitBreakers(unittest.TestCase):
    """Test circuit breaker functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = CircuitBreakerConfig()
        self.circuit_breaker = CircuitBreaker(self.config)
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        self.assertIsNotNone(self.circuit_breaker)
        self.assertEqual(self.circuit_breaker.drawdown_breaker.value, "closed")
        self.assertEqual(self.circuit_breaker.daily_loss_breaker.value, "closed")
    
    def test_drawdown_circuit_breaker(self):
        """Test drawdown circuit breaker"""
        # Create portfolio data with high drawdown
        portfolio_data = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-20000,  # 20% loss
            unrealized_pnl=-20000,
            realized_pnl=0,
            max_drawdown=0.20,
            current_drawdown=0.20,
            positions=[]
        )
        
        # Create risk metrics
        from src.risk_management.models import RiskMetrics
        risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.0,
            risk_per_trade=0.0,
            max_drawdown=0.20,
            current_drawdown=0.20,
            var_95=0.0,
            var_99=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_consecutive_losses=0,
            win_rate=0.0,
            profit_factor=0.0,
            recovery_factor=0.0
        )
        
        # Check circuit breaker
        result = self.circuit_breaker.check_circuit_breakers(portfolio_data, risk_metrics)
        
        self.assertIsNotNone(result)
        self.assertIn("breakers", result)
        self.assertIn("drawdown", result["breakers"])
        self.assertTrue(result["breakers"]["drawdown"]["triggered"])
    
    def test_daily_loss_circuit_breaker(self):
        """Test daily loss circuit breaker"""
        # Update tracking data with high daily loss
        self.circuit_breaker.daily_pnl_history = [-10000, -15000, -20000]  # Increasing losses
        
        portfolio_data = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-20000,
            unrealized_pnl=-20000,
            realized_pnl=0,
            max_drawdown=0.20,
            current_drawdown=0.20,
            positions=[]
        )
        
        from src.risk_management.models import RiskMetrics
        risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.05,
            risk_per_trade=0.02,
            max_drawdown=0.10,
            current_drawdown=0.05,
            var_95=0.03,
            var_99=0.05,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_consecutive_losses=3,
            win_rate=0.6,
            profit_factor=1.5,
            recovery_factor=0.8
        )
        
        # Check circuit breaker
        result = self.circuit_breaker.check_circuit_breakers(portfolio_data, risk_metrics)
        
        self.assertIsNotNone(result)
        self.assertIn("breakers", result)
        self.assertIn("daily_loss", result["breakers"])
        self.assertTrue(result["breakers"]["daily_loss"]["triggered"])


class TestRiskDashboard(unittest.TestCase):
    """Test risk dashboard functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = RiskDashboardConfig()
        self.dashboard = RiskDashboard(self.config)
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        self.assertIsNotNone(self.dashboard)
        self.assertEqual(len(self.dashboard.risk_metrics_history), 0)
        self.assertEqual(len(self.dashboard.risk_events_history), 0)
    
    async def test_dashboard_update(self):
        """Test dashboard update"""
        # Create mock risk state
        from src.risk_management.models import RiskState, RiskMetrics, PortfolioData
        from src.risk_management.core.risk_metrics import RiskMetricsCalculator
        
        portfolio_data = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-5000,
            unrealized_pnl=-5000,
            realized_pnl=0,
            max_drawdown=0.05,
            current_drawdown=0.05,
            positions=[]
        )
        
        risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.0,
            risk_per_trade=0.0,
            max_drawdown=0.05,
            current_drawdown=0.05,
            var_95=0.0,
            var_99=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_consecutive_losses=0,
            win_rate=0.0,
            profit_factor=0.0,
            recovery_factor=0.0
        )
        
        risk_state = RiskState(
            trading_state=TradingState.ACTIVE,
            current_risk_level=RiskLevel.MEDIUM,
            risk_metrics=risk_metrics,
            portfolio_data=portfolio_data
        )
        
        # Update dashboard
        await self.dashboard.update_dashboard(risk_state)
        
        # Check that dashboard was updated
        self.assertIsNotNone(self.dashboard.current_state)
        self.assertEqual(self.dashboard.updates_count, 1)
    
    def test_dashboard_data_generation(self):
        """Test dashboard data generation"""
        # Create mock risk state
        from src.risk_management.models import RiskState, RiskMetrics, PortfolioData
        
        portfolio_data = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-5000,
            unrealized_pnl=-5000,
            realized_pnl=0,
            max_drawdown=0.05,
            current_drawdown=0.05,
            positions=[]
        )
        
        risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.05,
            risk_per_trade=0.02,
            max_drawdown=0.10,
            current_drawdown=0.05,
            var_95=0.03,
            var_99=0.05,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_consecutive_losses=3,
            win_rate=0.6,
            profit_factor=1.5,
            recovery_factor=0.8
        )
        
        risk_state = RiskState(
            trading_state=TradingState.ACTIVE,
            current_risk_level=RiskLevel.MEDIUM,
            risk_metrics=risk_metrics,
            portfolio_data=portfolio_data
        )
        
        # Set current state first
        self.dashboard.current_state = risk_state
        
        # Generate dashboard data
        asyncio.run(self.dashboard._generate_dashboard_data())
        
        # Check that data was generated
        self.assertIsNotNone(self.dashboard.dashboard_data)
        self.assertIn("timestamp", self.dashboard.dashboard_data)
        self.assertIn("risk_level", self.dashboard.dashboard_data)


class TestRiskAlerting(unittest.TestCase):
    """Test risk alerting functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AlertingConfig()
        self.alerting = RiskAlerting(self.config)
    
    def test_alerting_initialization(self):
        """Test alerting system initialization"""
        self.assertIsNotNone(self.alerting)
        self.assertEqual(len(self.alerting.alerts), 0)
        self.assertEqual(len(self.alerting.alert_history), 0)
    
    async def test_alert_creation(self):
        """Test alert creation"""
        alert_id = await self.alerting.create_alert(
            alert_type="test_alert",
            severity=RiskLevel.HIGH,
            message="Test alert message",
            data={"test": "data"}
        )
        
        self.assertIsNotNone(alert_id)
        self.assertIn(alert_id, self.alerting.alerts)
        self.assertEqual(self.alerting.alerts_created, 1)
    
    async def test_alert_acknowledgment(self):
        """Test alert acknowledgment"""
        # Create alert
        alert_id = await self.alerting.create_alert(
            alert_type="test_alert",
            severity=RiskLevel.MEDIUM,
            message="Test alert message"
        )
        
        # Acknowledge alert
        await self.alerting.acknowledge_alert(alert_id, "test_user")
        
        # Check acknowledgment
        alert = self.alerting.alerts[alert_id]
        self.assertTrue(alert.acknowledged)
        self.assertEqual(alert.acknowledged_by, "test_user")
        self.assertEqual(self.alerting.alerts_acknowledged, 1)
    
    def test_rate_limiting(self):
        """Test alert rate limiting"""
        # Test normal rate limiting
        self.assertTrue(self.alerting._check_rate_limits())
        
        # Simulate rate limit exceeded
        self.alerting.hourly_alert_count = 100  # Exceed hourly limit
        self.assertFalse(self.alerting._check_rate_limits())
    
    async def test_alert_delivery(self):
        """Test alert delivery"""
        # Create alert
        alert_id = await self.alerting.create_alert(
            alert_type="test_alert",
            severity=RiskLevel.HIGH,
            message="Test alert message"
        )
        
        # Check delivery status
        self.assertIn(alert_id, self.alerting.delivery_status)
        self.assertEqual(self.alerting.delivery_status[alert_id], "sent")


class TestRiskManager(unittest.TestCase):
    """Test main risk manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = RiskManagerConfig()
        self.risk_manager = RiskManager(self.config)
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        self.assertIsNotNone(self.risk_manager)
        self.assertIsNotNone(self.risk_manager.risk_engine)
        self.assertFalse(self.risk_manager._is_running)
    
    async def test_risk_assessment(self):
        """Test risk assessment"""
        # Force risk assessment
        risk_state = await self.risk_manager.force_risk_assessment()
        
        self.assertIsNotNone(risk_state)
        self.assertIsNotNone(risk_state.current_risk_level)
        self.assertIsNotNone(risk_state.trading_state)
    
    async def test_risk_summary(self):
        """Test risk summary generation"""
        summary = await self.risk_manager.get_risk_summary()
        
        self.assertIsNotNone(summary)
        self.assertIn("current_state", summary)
        self.assertIn("risk_engine_summary", summary)
        self.assertIn("performance", summary)
    
    def test_system_status(self):
        """Test system status"""
        status = self.risk_manager.get_status()
        
        self.assertIsNotNone(status)
        self.assertIn("is_running", status)
        self.assertIn("components", status)
        self.assertIn("current_risk_level", status)


class TestRiskMetricsCalculator(unittest.TestCase):
    """Test risk metrics calculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = RiskMetricsConfig()
        self.calculator = RiskMetricsCalculator(self.config)
    
    def test_calculator_initialization(self):
        """Test calculator initialization"""
        self.assertIsNotNone(self.calculator)
        self.assertEqual(self.calculator.calculations_count, 0)
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation"""
        # Create portfolio data
        portfolio_data = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-5000,
            unrealized_pnl=-5000,
            realized_pnl=0,
            max_drawdown=0.05,
            current_drawdown=0.05,
            positions=[]
        )
        
        # Calculate metrics
        metrics = self.calculator.calculate_risk_metrics(portfolio_data)
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.portfolio_value, 100000)
        self.assertGreaterEqual(metrics.max_drawdown, 0)
        self.assertGreaterEqual(metrics.current_drawdown, 0)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation"""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        max_dd, current_dd = self.calculator._calculate_drawdowns(returns)
        
        self.assertGreaterEqual(max_dd, 0)
        self.assertGreaterEqual(current_dd, 0)
    
    def test_var_calculation(self):
        """Test VaR calculation"""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.05, 0.01]
        var_95 = self.calculator._calculate_var(returns, 0.95)
        
        self.assertGreaterEqual(var_95, 0)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        sharpe = self.calculator._calculate_sharpe_ratio(returns)
        
        self.assertIsInstance(sharpe, float)


def run_async_test(coro):
    """Helper function to run async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
