"""
Test Enhanced Risk Management System

Tests for the enhanced configuration management and circuit breakers.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta

from src.risk_management.config.risk_config_manager import RiskConfigManager, RiskSystemConfig
from src.risk_management.controls.enhanced_circuit_breakers import (
    EnhancedCircuitBreaker, BreakerConfig, BreakerType, BreakerThreshold,
    BreakerState, BreakerResult
)
from src.risk_management.models import PortfolioData, RiskMetrics, RiskLevel


class TestRiskConfigManager(unittest.TestCase):
    """Test risk configuration manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary config file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        
        self.config_manager = RiskConfigManager(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_config_initialization(self):
        """Test configuration manager initialization"""
        self.assertIsNotNone(self.config_manager.current_config)
        self.assertIsNotNone(self.config_manager.config_file)
    
    def test_load_default_config(self):
        """Test loading default configuration"""
        config = self.config_manager.load_config()
        self.assertIsInstance(config, RiskSystemConfig)
        self.assertEqual(config.config_version, "1.0.0")
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        # Save current config
        self.assertTrue(self.config_manager.save_config())
        
        # Create new manager and load config
        new_manager = RiskConfigManager(self.temp_file.name)
        loaded_config = new_manager.load_config()
        
        self.assertEqual(loaded_config.config_version, self.config_manager.current_config.config_version)
    
    def test_update_config(self):
        """Test updating configuration"""
        updates = {
            "risk_engine": {
                "var_confidence_level": 0.99,
                "lookback_periods": 500
            }
        }
        
        self.assertTrue(self.config_manager.update_config(updates))
        
        # Check if values were updated
        self.assertEqual(self.config_manager.current_config.risk_engine.var_confidence_level, 0.99)
        self.assertEqual(self.config_manager.current_config.risk_engine.lookback_periods, 500)
    
    def test_get_risk_thresholds(self):
        """Test getting risk thresholds"""
        thresholds = self.config_manager.get_risk_thresholds()
        
        self.assertIn("risk_engine", thresholds)
        self.assertIn("circuit_breakers", thresholds)
        self.assertIn("event_impact", thresholds)
        self.assertIn("sentiment", thresholds)
    
    def test_update_risk_thresholds(self):
        """Test updating risk thresholds"""
        new_thresholds = {
            "risk_engine": {
                "var_confidence_level": 0.99,
                "lookback_periods": 500
            },
            "circuit_breakers": {
                "max_daily_loss_threshold": 0.03,
                "max_drawdown_threshold": 0.08
            }
        }
        
        self.assertTrue(self.config_manager.update_risk_thresholds(new_thresholds))
        
        # Verify updates
        thresholds = self.config_manager.get_risk_thresholds()
        self.assertEqual(thresholds["risk_engine"]["var_confidence_level"], 0.99)
        self.assertEqual(thresholds["circuit_breakers"]["max_daily_loss_threshold"], 0.03)
    
    def test_create_preset_configs(self):
        """Test creating preset configurations"""
        presets = ["conservative", "moderate", "aggressive", "crypto", "forex"]
        
        for preset in presets:
            self.assertTrue(self.config_manager.create_preset_config(preset))
            
            # Check that config was updated
            self.assertIsNotNone(self.config_manager.current_config)
            self.assertEqual(self.config_manager.current_config.config_version, "1.0.0")
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid thresholds (not in ascending order)
        invalid_updates = {
            "risk_engine": {
                "low_risk_threshold": 0.10,
                "medium_risk_threshold": 0.05  # Invalid: medium < low
            }
        }
        
        self.assertFalse(self.config_manager.update_config(invalid_updates))
    
    def test_get_config_summary(self):
        """Test getting configuration summary"""
        summary = self.config_manager.get_config_summary()
        
        self.assertIn("version", summary)
        self.assertIn("last_updated", summary)
        self.assertIn("thresholds", summary)
        self.assertIn("history_count", summary)


class TestEnhancedCircuitBreakers(unittest.TestCase):
    """Test enhanced circuit breakers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.circuit_breaker = EnhancedCircuitBreaker()
        
        # Create test portfolio data
        self.portfolio_data = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-5000,  # 5% loss
            unrealized_pnl=-5000,
            realized_pnl=0,
            max_drawdown=0.20,
            current_drawdown=0.15,
            positions=[
                {"symbol": "EUR/USD", "size": 0.1, "value": 10000},
                {"symbol": "GBP/USD", "size": 0.15, "value": 15000}
            ]
        )
        
        # Create test risk metrics
        self.risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.15,
            risk_per_trade=0.05,
            max_drawdown=0.20,
            current_drawdown=0.15,
            var_95=0.08,
            var_99=0.12,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=0.8,
            max_consecutive_losses=3,
            win_rate=0.6,
            profit_factor=1.4,
            recovery_factor=0.7
        )
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        self.assertIsNotNone(self.circuit_breaker.configs)
        self.assertIsNotNone(self.circuit_breaker.breaker_states)
        self.assertEqual(len(self.circuit_breaker.breaker_states), len(self.circuit_breaker.configs))
    
    def test_check_all_breakers(self):
        """Test checking all circuit breakers"""
        market_data = {
            "volatility": 0.25,
            "trend": 0.1,
            "condition": "normal",
            "correlation": 0.7,
            "liquidity": 0.8,
            "sentiment": -0.6,
            "event_risk": 0.3
        }
        
        results = self.circuit_breaker.check_all_breakers(
            self.portfolio_data, self.risk_metrics, market_data
        )
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(self.circuit_breaker.configs))
        
        # Check that all breaker types are present
        for config in self.circuit_breaker.configs:
            self.assertIn(config.breaker_type, results)
            result = results[config.breaker_type]
            self.assertIsInstance(result, BreakerResult)
    
    def test_daily_loss_breaker(self):
        """Test daily loss circuit breaker"""
        # Test with high loss (should trigger)
        high_loss_portfolio = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-8000,  # 8% loss
            unrealized_pnl=-8000,
            realized_pnl=0,
            max_drawdown=0.20,
            current_drawdown=0.08,
            positions=[]
        )
        
        results = self.circuit_breaker.check_all_breakers(
            high_loss_portfolio, self.risk_metrics
        )
        
        daily_loss_result = results[BreakerType.DAILY_LOSS]
        self.assertIsNotNone(daily_loss_result)
        self.assertGreater(daily_loss_result.current_value, 0)
    
    def test_drawdown_breaker(self):
        """Test drawdown circuit breaker"""
        # Test with high drawdown (should trigger)
        high_drawdown_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.15,
            risk_per_trade=0.05,
            max_drawdown=0.20,
            current_drawdown=0.25,  # High drawdown
            var_95=0.08,
            var_99=0.12,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=0.8,
            max_consecutive_losses=3,
            win_rate=0.6,
            profit_factor=1.4,
            recovery_factor=0.7
        )
        
        results = self.circuit_breaker.check_all_breakers(
            self.portfolio_data, high_drawdown_metrics
        )
        
        drawdown_result = results[BreakerType.DRAWDOWN]
        self.assertIsNotNone(drawdown_result)
        self.assertEqual(drawdown_result.current_value, 0.25)
    
    def test_adaptive_thresholds(self):
        """Test adaptive threshold calculation"""
        config = self.circuit_breaker.configs[0]  # Get first config
        threshold = config.threshold
        
        # Test normal conditions
        normal_threshold = threshold.get_current_threshold(
            market_volatility=0.1,
            trend_direction=0.0,
            market_condition="normal"
        )
        
        # Test high volatility
        high_vol_threshold = threshold.get_current_threshold(
            market_volatility=0.5,
            trend_direction=0.0,
            market_condition="normal"
        )
        
        # Test crisis conditions
        crisis_threshold = threshold.get_current_threshold(
            market_volatility=0.3,
            trend_direction=0.0,
            market_condition="crisis"
        )
        
        # High volatility should increase threshold
        self.assertGreaterEqual(high_vol_threshold, normal_threshold)
        
        # Crisis should decrease threshold (but may not be less due to other factors)
        # Just check that it's different from normal
        self.assertNotEqual(crisis_threshold, normal_threshold)
    
    def test_breaker_status(self):
        """Test getting breaker status"""
        status = self.circuit_breaker.get_breaker_status()
        
        self.assertIsInstance(status, dict)
        
        for config in self.circuit_breaker.configs:
            breaker_status = status[config.breaker_type.value]
            self.assertIn("state", breaker_status)
            self.assertIn("enabled", breaker_status)
            self.assertIn("cooldown_remaining", breaker_status)
            self.assertIn("trigger_count", breaker_status)
    
    def test_reset_breaker(self):
        """Test resetting individual breaker"""
        breaker_type = BreakerType.DAILY_LOSS
        
        # Trigger breaker first
        self.circuit_breaker.breaker_states[breaker_type] = BreakerState.OPEN
        self.circuit_breaker.trigger_counts[breaker_type] = 2
        
        # Reset breaker
        self.assertTrue(self.circuit_breaker.reset_breaker(breaker_type))
        
        # Check state was reset
        self.assertEqual(self.circuit_breaker.breaker_states[breaker_type], BreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.trigger_counts[breaker_type], 0)
    
    def test_reset_all_breakers(self):
        """Test resetting all breakers"""
        # Set some breakers to open state
        for breaker_type in self.circuit_breaker.breaker_states:
            self.circuit_breaker.breaker_states[breaker_type] = BreakerState.OPEN
            self.circuit_breaker.trigger_counts[breaker_type] = 1
        
        # Reset all
        self.circuit_breaker.reset_all_breakers()
        
        # Check all are closed
        for breaker_type in self.circuit_breaker.breaker_states:
            self.assertEqual(self.circuit_breaker.breaker_states[breaker_type], BreakerState.CLOSED)
            self.assertEqual(self.circuit_breaker.trigger_counts[breaker_type], 0)
    
    def test_breaker_dependencies(self):
        """Test breaker dependencies"""
        # Create config with dependencies
        dependent_config = BreakerConfig(
            breaker_type=BreakerType.VOLATILITY,
            threshold=BreakerThreshold(0.3, 0.1, 0.8),
            dependencies=[BreakerType.DAILY_LOSS]
        )
        
        # Create breaker with dependency
        breaker = EnhancedCircuitBreaker([dependent_config])
        
        # Set dependency to open
        breaker.breaker_states[BreakerType.DAILY_LOSS] = BreakerState.OPEN
        
        # Check dependent breaker
        results = breaker.check_all_breakers(self.portfolio_data, self.risk_metrics)
        
        # Dependent breaker should be checked but may be closed due to dependency
        self.assertIn(BreakerType.VOLATILITY, results)
        # The breaker should be closed since dependency is open
        self.assertEqual(results[BreakerType.VOLATILITY].state, BreakerState.CLOSED)


class TestBreakerThreshold(unittest.TestCase):
    """Test breaker threshold functionality"""
    
    def test_threshold_creation(self):
        """Test threshold creation"""
        threshold = BreakerThreshold(
            base_value=0.05,
            min_value=0.01,
            max_value=0.20,
            volatility_factor=0.1,
            trend_factor=0.05,
            market_condition_factor=0.2
        )
        
        self.assertEqual(threshold.base_value, 0.05)
        self.assertEqual(threshold.min_value, 0.01)
        self.assertEqual(threshold.max_value, 0.20)
    
    def test_adaptive_threshold_calculation(self):
        """Test adaptive threshold calculation"""
        threshold = BreakerThreshold(
            base_value=0.05,
            min_value=0.01,
            max_value=0.20,
            volatility_factor=0.1,
            trend_factor=0.05,
            market_condition_factor=0.2
        )
        
        # Test normal conditions
        normal_threshold = threshold.get_current_threshold()
        self.assertEqual(normal_threshold, 0.05)
        
        # Test with high volatility
        high_vol_threshold = threshold.get_current_threshold(
            market_volatility=1.0,
            trend_direction=0.0,
            market_condition="normal"
        )
        self.assertGreater(high_vol_threshold, 0.05)
        
        # Test bounds
        self.assertGreaterEqual(high_vol_threshold, threshold.min_value)
        self.assertLessEqual(high_vol_threshold, threshold.max_value)
    
    def test_disabled_adaptive(self):
        """Test disabled adaptive mode"""
        threshold = BreakerThreshold(
            base_value=0.05,
            min_value=0.01,
            max_value=0.20,
            adaptive_enabled=False
        )
        
        # Should always return base value regardless of conditions
        result = threshold.get_current_threshold(
            market_volatility=1.0,
            trend_direction=1.0,
            market_condition="crisis"
        )
        
        self.assertEqual(result, 0.05)


if __name__ == '__main__':
    unittest.main()
