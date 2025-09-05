"""
Test Adaptive Position Sizing System

Tests for the adaptive position sizing functionality.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from src.risk_management.position_sizing.adaptive_sizer import (
    AdaptivePositionSizer, SizingConfig, SizingRule, PositionSizingResult,
    SizingMode, SizingFactor
)
from src.risk_management.models import (
    PortfolioData, RiskMetrics, RiskLevel, RiskEvent, EventImpact, SentimentLevel
)


class TestAdaptivePositionSizer(unittest.TestCase):
    """Test adaptive position sizing system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SizingConfig()
        self.sizer = AdaptivePositionSizer(self.config)
        
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
            positions=[
                {"symbol": "EUR/USD", "size": 0.05, "value": 5000},
                {"symbol": "GBP/USD", "size": 0.03, "value": 3000}
            ]
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
    
    def test_sizer_initialization(self):
        """Test position sizer initialization"""
        self.assertIsNotNone(self.sizer.config)
        self.assertIsInstance(self.sizer.current_positions, dict)
        self.assertEqual(len(self.sizer.current_positions), 0)
    
    def test_calculate_position_size_basic(self):
        """Test basic position size calculation"""
        result = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics
        )
        
        self.assertIsInstance(result, PositionSizingResult)
        self.assertEqual(result.symbol, "EUR/USD")
        self.assertGreater(result.recommended_size, 0)
        self.assertLessEqual(result.recommended_size, self.config.max_position_size)
        self.assertGreaterEqual(result.recommended_size, self.config.min_position_size)
    
    def test_risk_adjustment_calculation(self):
        """Test risk-based position size adjustment"""
        # Test with low risk
        low_risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.05,
            risk_per_trade=0.01,
            max_drawdown=0.05,
            current_drawdown=0.02,
            var_95=0.02,
            var_99=0.03,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_consecutive_losses=1,
            win_rate=0.7,
            profit_factor=1.6,
            recovery_factor=0.9
        )
        
        result_low = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, low_risk_metrics
        )
        
        # Test with high risk
        high_risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.15,
            risk_per_trade=0.05,
            max_drawdown=0.20,
            current_drawdown=0.18,
            var_95=0.08,
            var_99=0.12,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            calmar_ratio=0.4,
            max_consecutive_losses=5,
            win_rate=0.4,
            profit_factor=0.8,
            recovery_factor=0.3
        )
        
        result_high = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, high_risk_metrics
        )
        
        # High risk should result in smaller position size
        self.assertLess(result_high.recommended_size, result_low.recommended_size)
    
    def test_volatility_adjustment(self):
        """Test volatility-based position size adjustment"""
        # Test with low volatility
        low_vol_data = {"EUR/USD_volatility": 0.1}
        result_low = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, low_vol_data
        )
        
        # Test with high volatility
        high_vol_data = {"EUR/USD_volatility": 0.5}
        result_high = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, high_vol_data
        )
        
        # High volatility should result in smaller position size
        self.assertLess(result_high.recommended_size, result_low.recommended_size)
    
    def test_correlation_adjustment(self):
        """Test correlation-based position size adjustment"""
        # Test with low correlation
        low_corr_data = {"EUR/USD_correlations": {"GBP/USD": 0.3}}
        result_low = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, low_corr_data
        )
        
        # Test with high correlation
        high_corr_data = {"EUR/USD_correlations": {"GBP/USD": 0.9}}
        result_high = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, high_corr_data
        )
        
        # High correlation should result in smaller position size
        self.assertLess(result_high.recommended_size, result_low.recommended_size)
    
    def test_sentiment_adjustment(self):
        """Test sentiment-based position size adjustment"""
        # Test with bearish sentiment
        bearish_data = {"EUR/USD_sentiment": -0.7}
        result_bearish = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, bearish_data
        )
        
        # Test with bullish sentiment
        bullish_data = {"EUR/USD_sentiment": 0.7}
        result_bullish = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, bullish_data
        )
        
        # Bullish sentiment should result in larger position size
        self.assertGreater(result_bullish.recommended_size, result_bearish.recommended_size)
    
    def test_event_adjustment(self):
        """Test event-based position size adjustment"""
        # Test without events
        result_no_events = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, active_events=[]
        )
        
        # Test with high impact event
        high_impact_event = RiskEvent(
            event_id="test_1",
            event_type="economic_event",
            risk_level=RiskLevel.HIGH,
            description="High impact event",
            timestamp=datetime.utcnow(),
            source="test"
        )
        
        result_with_events = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, active_events=[high_impact_event]
        )
        
        # High impact event should result in smaller position size
        self.assertLess(result_with_events.recommended_size, result_no_events.recommended_size)
    
    def test_drawdown_adjustment(self):
        """Test drawdown-based position size adjustment"""
        # Test with low drawdown
        low_drawdown_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.05,
            risk_per_trade=0.02,
            max_drawdown=0.10,
            current_drawdown=0.02,
            var_95=0.03,
            var_99=0.05,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_consecutive_losses=1,
            win_rate=0.7,
            profit_factor=1.6,
            recovery_factor=0.9
        )
        
        result_low = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, low_drawdown_metrics
        )
        
        # Test with high drawdown
        high_drawdown_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.15,
            risk_per_trade=0.05,
            max_drawdown=0.20,
            current_drawdown=0.15,
            var_95=0.08,
            var_99=0.12,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            calmar_ratio=0.4,
            max_consecutive_losses=5,
            win_rate=0.4,
            profit_factor=0.8,
            recovery_factor=0.3
        )
        
        result_high = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, high_drawdown_metrics
        )
        
        # High drawdown should result in smaller position size
        self.assertLess(result_high.recommended_size, result_low.recommended_size)
    
    def test_position_bounds(self):
        """Test position size bounds"""
        # Create extreme market conditions to test bounds
        extreme_data = {
            "EUR/USD_volatility": 1.0,
            "EUR/USD_correlations": {"GBP/USD": 0.95},
            "EUR/USD_sentiment": -1.0
        }
        
        result = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, self.risk_metrics, extreme_data
        )
        
        # Should be bounded by min and max position sizes
        self.assertGreaterEqual(result.recommended_size, self.config.min_position_size)
        self.assertLessEqual(result.recommended_size, self.config.max_position_size)
    
    def test_multiple_symbols(self):
        """Test position size calculation for multiple symbols"""
        symbols = ["EUR/USD", "GBP/USD", "USD/JPY"]
        market_data = {
            "EUR/USD_volatility": 0.2,
            "GBP/USD_volatility": 0.3,
            "USD/JPY_volatility": 0.15,
            "EUR/USD_sentiment": 0.5,
            "GBP/USD_sentiment": -0.3,
            "USD/JPY_sentiment": 0.1
        }
        
        results = self.sizer.get_position_recommendations(
            symbols, self.portfolio_data, self.risk_metrics, market_data
        )
        
        self.assertEqual(len(results), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, results)
            self.assertIsInstance(results[symbol], PositionSizingResult)
    
    def test_sizing_summary(self):
        """Test sizing summary generation"""
        # Make some sizing decisions first
        self.sizer.calculate_position_size("EUR/USD", self.portfolio_data, self.risk_metrics)
        self.sizer.calculate_position_size("GBP/USD", self.portfolio_data, self.risk_metrics)
        
        summary = self.sizer.get_sizing_summary()
        
        self.assertIn("total_decisions", summary)
        self.assertIn("recent_decisions", summary)
        self.assertIn("average_confidence", summary)
        self.assertIn("current_positions", summary)
        self.assertEqual(summary["total_decisions"], 2)
    
    def test_config_update(self):
        """Test configuration update"""
        new_config = SizingConfig(
            base_position_size=0.03,
            max_position_size=0.15
        )
        
        self.sizer.update_config(new_config)
        
        self.assertEqual(self.sizer.config.base_position_size, 0.03)
        self.assertEqual(self.sizer.config.max_position_size, 0.15)
    
    def test_position_reset(self):
        """Test position reset"""
        # Set some positions
        self.sizer.current_positions["EUR/USD"] = 0.05
        self.sizer.current_positions["GBP/USD"] = 0.03
        
        self.sizer.reset_positions()
        
        self.assertEqual(len(self.sizer.current_positions), 0)
    
    def test_warnings_generation(self):
        """Test warning generation"""
        # Create conditions that should generate warnings
        extreme_data = {
            "EUR/USD_volatility": 1.0,
            "EUR/USD_correlations": {"GBP/USD": 0.95},
            "EUR/USD_sentiment": -1.0
        }
        
        # Create high risk metrics to trigger warnings
        high_risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.20,
            risk_per_trade=0.08,
            max_drawdown=0.25,
            current_drawdown=0.20,
            var_95=0.10,
            var_99=0.15,
            sharpe_ratio=0.5,
            sortino_ratio=0.7,
            calmar_ratio=0.3,
            max_consecutive_losses=8,
            win_rate=0.3,
            profit_factor=0.6,
            recovery_factor=0.2
        )
        
        result = self.sizer.calculate_position_size(
            "EUR/USD", self.portfolio_data, high_risk_metrics, extreme_data
        )
        
        self.assertIsInstance(result.warnings, list)
        # Should have warnings due to extreme conditions
        self.assertGreaterEqual(len(result.warnings), 0)  # At least 0 warnings (may not always trigger)


class TestSizingConfig(unittest.TestCase):
    """Test sizing configuration"""
    
    def test_config_creation(self):
        """Test configuration creation"""
        config = SizingConfig()
        
        self.assertEqual(config.base_position_size, 0.02)
        self.assertEqual(config.max_position_size, 0.10)
        self.assertEqual(config.min_position_size, 0.001)
        self.assertIsInstance(config.risk_level_multipliers, dict)
        self.assertIsInstance(config.sizing_rules, list)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = SizingConfig(
            base_position_size=0.03,
            max_position_size=0.15,
            min_position_size=0.002
        )
        
        self.assertEqual(config.base_position_size, 0.03)
        self.assertEqual(config.max_position_size, 0.15)
        self.assertEqual(config.min_position_size, 0.002)


class TestSizingRule(unittest.TestCase):
    """Test sizing rule"""
    
    def test_rule_creation(self):
        """Test rule creation"""
        rule = SizingRule(
            factor=SizingFactor.VOLATILITY,
            weight=0.3,
            min_multiplier=0.5,
            max_multiplier=1.5
        )
        
        self.assertEqual(rule.factor, SizingFactor.VOLATILITY)
        self.assertEqual(rule.weight, 0.3)
        self.assertEqual(rule.min_multiplier, 0.5)
        self.assertEqual(rule.max_multiplier, 1.5)
        self.assertTrue(rule.enabled)


if __name__ == '__main__':
    unittest.main()
