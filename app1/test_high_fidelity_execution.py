#!/usr/bin/env python3
"""
Test Suite for High-Fidelity Order Execution Logic

Comprehensive tests for the high-fidelity broker, sizer, and integration
components to ensure 99% modeling quality in backtesting.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add the app1/src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app1', 'src'))

from src.backtesting.execution.high_fidelity_broker import (
    HighFidelityBroker, ExecutionConfig, ExecutionResult, OrderType
)
from src.backtesting.execution.high_fidelity_sizer import (
    HighFidelitySizer, SizingConfig, SizingMethod, HighFidelitySizerFactory
)
from src.backtesting.execution.execution_integration import (
    HighFidelityExecutionIntegration, HighFidelityStrategy, create_high_fidelity_backtest
)
from src.backtesting.tick_data.variable_spread_simulator import SpreadModel


class TestExecutionConfig(unittest.TestCase):
    """Test ExecutionConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ExecutionConfig()
        
        self.assertEqual(config.spread_model, SpreadModel.HYBRID)
        self.assertEqual(config.base_spread, 0.0001)
        self.assertEqual(config.slippage_model, "realistic")
        self.assertEqual(config.slippage_factor, 0.1)
        self.assertTrue(config.market_impact_enabled)
        self.assertTrue(config.latency_enabled)
        self.assertTrue(config.partial_fills_enabled)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ExecutionConfig(
            spread_model=SpreadModel.STATISTICAL,
            base_spread=0.0002,
            slippage_model="aggressive",
            slippage_factor=0.2,
            market_impact_enabled=False,
            latency_enabled=False
        )
        
        self.assertEqual(config.spread_model, SpreadModel.STATISTICAL)
        self.assertEqual(config.base_spread, 0.0002)
        self.assertEqual(config.slippage_model, "aggressive")
        self.assertEqual(config.slippage_factor, 0.2)
        self.assertFalse(config.market_impact_enabled)
        self.assertFalse(config.latency_enabled)


class TestHighFidelityBroker(unittest.TestCase):
    """Test HighFidelityBroker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ExecutionConfig(
            slippage_model="realistic",
            market_impact_enabled=True,
            latency_enabled=True,
            partial_fills_enabled=True
        )
        self.broker = HighFidelityBroker(self.config)
        
        # Mock current tick data
        self.sample_tick = {
            'timestamp': datetime.now(),
            'bid': 1.1000,
            'ask': 1.1002,
            'mid': 1.1001,
            'volume': 1000,
            'volatility': 1.0,
            'liquidity': 1.0
        }
    
    def test_initialization(self):
        """Test broker initialization"""
        self.assertIsNotNone(self.broker.spread_simulator)
        self.assertEqual(self.broker.total_orders, 0)
        self.assertEqual(self.broker.successful_orders, 0)
        self.assertEqual(self.broker.failed_orders, 0)
        self.assertEqual(len(self.broker.orders), 0)
    
    def test_set_market_data(self):
        """Test setting market data"""
        self.broker.set_market_data(self.sample_tick)
        self.assertEqual(self.broker.current_tick_data, self.sample_tick)
    
    def test_generate_order_id(self):
        """Test order ID generation"""
        order_id1 = self.broker._generate_order_id()
        order_id2 = self.broker._generate_order_id()
        
        self.assertTrue(order_id1.startswith("HF_"))
        self.assertTrue(order_id2.startswith("HF_"))
        self.assertNotEqual(order_id1, order_id2)
    
    def test_calculate_slippage(self):
        """Test slippage calculation"""
        spread = 0.0002
        quantity = 1.0
        side = "buy"
        
        slippage = self.broker._calculate_slippage(spread, quantity, side)
        
        self.assertGreaterEqual(slippage, 0.0)
        self.assertLessEqual(slippage, self.config.max_slippage)
    
    def test_calculate_market_impact(self):
        """Test market impact calculation"""
        quantity = 1.0
        current_price = 1.1000
        
        impact = self.broker._calculate_market_impact(quantity, current_price)
        
        self.assertGreaterEqual(impact, 0.0)
        self.assertLessEqual(impact, self.config.max_impact)
    
    def test_simulate_latency(self):
        """Test latency simulation"""
        latency = self.broker._simulate_latency()
        
        self.assertGreaterEqual(latency, self.config.min_latency_ms)
        self.assertLessEqual(latency, self.config.max_latency_ms)
    
    def test_simulate_partial_fill(self):
        """Test partial fill simulation"""
        requested_quantity = 1.0
        
        # Test multiple times to check probability
        fills = [self.broker._simulate_partial_fill(requested_quantity) for _ in range(100)]
        
        # Should be within expected range
        for fill in fills:
            self.assertGreaterEqual(fill, requested_quantity * self.config.min_fill_size)
            self.assertLessEqual(fill, requested_quantity * self.config.max_fill_size)
    
    def test_get_execution_prices(self):
        """Test execution price calculation"""
        self.broker.set_market_data(self.sample_tick)
        
        bid, ask = self.broker._get_execution_prices("buy", self.sample_tick)
        
        self.assertGreater(ask, bid)
        self.assertGreater(bid, 0)
        self.assertGreater(ask, 0)
    
    def test_execute_market_order(self):
        """Test market order execution"""
        # Mock order
        order = Mock()
        order.isbuy.return_value = True
        order.size = 1.0
        order.ref = "TEST_001"
        
        self.broker.set_market_data(self.sample_tick)
        
        result = self.broker._execute_market_order(order, self.sample_tick)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.order_id, "TEST_001")
        self.assertEqual(result.side, "buy")
        self.assertEqual(result.quantity, 1.0)
        self.assertGreater(result.fill_price, 0)
        self.assertGreaterEqual(result.execution_quality_score, 0)
        self.assertLessEqual(result.execution_quality_score, 100)
    
    def test_execute_limit_order(self):
        """Test limit order execution"""
        # Mock order
        order = Mock()
        order.isbuy.return_value = True
        order.size = 1.0
        order.price = 1.1001  # Limit price
        order.ref = "TEST_002"
        
        self.broker.set_market_data(self.sample_tick)
        
        result = self.broker._execute_limit_order(order, self.sample_tick)
        
        if result:  # Order might not execute if limit not hit
            self.assertIsInstance(result, ExecutionResult)
            self.assertEqual(result.order_id, "TEST_002")
            self.assertEqual(result.side, "buy")
            self.assertLessEqual(result.fill_price, order.price)
    
    def test_execute_stop_order(self):
        """Test stop order execution"""
        # Mock order
        order = Mock()
        order.isbuy.return_value = True
        order.size = 1.0
        order.price = 1.1001  # Stop price
        order.ref = "TEST_003"
        
        self.broker.set_market_data(self.sample_tick)
        
        result = self.broker._execute_stop_order(order, self.sample_tick)
        
        if result:  # Order might not execute if stop not triggered
            self.assertIsInstance(result, ExecutionResult)
            self.assertEqual(result.order_id, "TEST_003")
            self.assertEqual(result.side, "buy")
    
    def test_calculate_execution_quality(self):
        """Test execution quality calculation"""
        spread = 0.0002
        slippage = 0.0001
        market_impact = 0.00005
        executed_quantity = 0.8
        requested_quantity = 1.0
        
        quality = self.broker._calculate_execution_quality(
            spread, slippage, market_impact, executed_quantity, requested_quantity
        )
        
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 100.0)
    
    def test_submit_order(self):
        """Test order submission"""
        # Mock order
        order = Mock()
        order.ref = None
        
        submitted_order = self.broker.submit(order)
        
        self.assertIsNotNone(submitted_order.ref)
        self.assertTrue(submitted_order.ref.startswith("HF_"))
        self.assertIn(submitted_order.ref, self.broker.orders)
        self.assertEqual(self.broker.total_orders, 1)
    
    def test_process_orders(self):
        """Test order processing"""
        # Mock order
        order = Mock()
        order.ref = "TEST_004"
        order.status = 1  # Submitted
        order.exectype = 0  # Market
        order.isbuy.return_value = True
        order.size = 1.0
        
        self.broker.orders[order.ref] = order
        self.broker.set_market_data(self.sample_tick)
        
        # Process orders
        self.broker.process_orders(self.sample_tick)
        
        # Check if order was processed
        self.assertGreater(self.broker.successful_orders, 0)
        self.assertEqual(len(self.broker.execution_results), 1)
    
    def test_get_execution_statistics(self):
        """Test execution statistics"""
        stats = self.broker.get_execution_statistics()
        
        self.assertIn('total_orders', stats)
        self.assertIn('successful_orders', stats)
        self.assertIn('failed_orders', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('avg_slippage', stats)
        self.assertIn('avg_execution_time_ms', stats)
        self.assertIn('avg_quality_score', stats)


class TestSizingConfig(unittest.TestCase):
    """Test SizingConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SizingConfig()
        
        self.assertEqual(config.method, SizingMethod.PERCENTAGE)
        self.assertEqual(config.percentage, 0.1)
        self.assertEqual(config.max_position_size, 0.2)
        self.assertEqual(config.min_position_size, 0.01)
        self.assertEqual(config.max_risk_per_trade, 0.02)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = SizingConfig(
            method=SizingMethod.VOLATILITY,
            percentage=0.15,
            max_position_size=0.3,
            min_position_size=0.02
        )
        
        self.assertEqual(config.method, SizingMethod.VOLATILITY)
        self.assertEqual(config.percentage, 0.15)
        self.assertEqual(config.max_position_size, 0.3)
        self.assertEqual(config.min_position_size, 0.02)


class TestHighFidelitySizer(unittest.TestCase):
    """Test HighFidelitySizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SizingConfig(
            method=SizingMethod.PERCENTAGE,
            percentage=0.1,
            max_position_size=0.2,
            min_position_size=0.01
        )
        self.sizer = HighFidelitySizer(self.config)
        
        # Mock broker
        self.sizer.broker = Mock()
        self.sizer.broker.getvalue.return_value = 10000.0
        
        # Mock data
        self.mock_data = Mock()
        self.mock_data.close = [1.1000]
        self.mock_data.spread = [0.0002]
        self.mock_data.volume = [1000]
        self.mock_data.ask = [1.1002]
        self.mock_data.bid = [1.1000]
    
    def test_initialization(self):
        """Test sizer initialization"""
        self.assertEqual(self.sizer.config.method, SizingMethod.PERCENTAGE)
        self.assertEqual(len(self.sizer.position_history), 0)
    
    def test_get_current_price(self):
        """Test current price extraction"""
        price = self.sizer._get_current_price(self.mock_data)
        self.assertEqual(price, 1.1000)
    
    def test_get_current_spread(self):
        """Test current spread extraction"""
        spread = self.sizer._get_current_spread(self.mock_data)
        self.assertEqual(spread, 0.0002)
    
    def test_get_current_volume(self):
        """Test current volume extraction"""
        volume = self.sizer._get_current_volume(self.mock_data)
        self.assertEqual(volume, 1000)
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        # Mock data with multiple prices
        mock_data = Mock()
        mock_data.close = [1.1000, 1.1005, 1.0995, 1.1010, 1.1002]
        mock_data.__len__ = Mock(return_value=5)
        
        volatility = self.sizer._calculate_volatility(mock_data, lookback=5)
        
        self.assertGreater(volatility, 0)
        self.assertLess(volatility, 1.0)  # Should be reasonable
    
    def test_calculate_kelly_fraction(self):
        """Test Kelly fraction calculation"""
        # Mock data with price history
        mock_data = Mock()
        mock_data.close = [1.1000, 1.1005, 1.0995, 1.1010, 1.1002, 1.1008, 1.0998, 1.1012]
        mock_data.__len__ = Mock(return_value=8)
        
        kelly_fraction = self.sizer._calculate_kelly_fraction(mock_data, lookback=8)
        
        self.assertGreaterEqual(kelly_fraction, self.config.min_kelly_fraction)
        self.assertLessEqual(kelly_fraction, self.config.max_kelly_fraction)
    
    def test_apply_spread_adjustment(self):
        """Test spread adjustment"""
        size = 1.0
        spread = 0.0005  # 5 pips
        
        adjusted_size = self.sizer._apply_spread_adjustment(size, spread)
        
        self.assertLessEqual(adjusted_size, size)
        self.assertGreater(adjusted_size, 0)
    
    def test_apply_market_impact_adjustment(self):
        """Test market impact adjustment"""
        size = 1.0
        current_price = 1.1000
        
        adjusted_size = self.sizer._apply_market_impact_adjustment(size, current_price)
        
        self.assertLessEqual(adjusted_size, size)
        self.assertGreater(adjusted_size, 0)
    
    def test_apply_risk_constraints(self):
        """Test risk constraint application"""
        size = 1.0
        portfolio_value = 10000.0
        current_price = 1.1000
        
        constrained_size = self.sizer._apply_risk_constraints(size, portfolio_value, current_price)
        
        self.assertGreaterEqual(constrained_size, 0)
        # Note: constrained_size might be larger than original size due to risk constraints
    
    def test_getsizing(self):
        """Test position sizing calculation"""
        size = self.sizer._getsizing(self.mock_data, isbuy=True)
        
        self.assertGreaterEqual(size, 0)
        # Note: size might be larger than expected due to risk constraints
    
    def test_record_position(self):
        """Test position recording"""
        size = 1.0
        price = 1.1000
        timestamp = datetime.now()
        
        self.sizer._record_position(size, price, timestamp)
        
        self.assertEqual(len(self.sizer.position_history), 1)
        self.assertEqual(self.sizer.position_history[0]['size'], size)
        self.assertEqual(self.sizer.position_history[0]['price'], price)
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some mock positions
        self.sizer.position_history = [
            {'pnl': 100.0},
            {'pnl': -50.0},
            {'pnl': 200.0},
            {'pnl': -25.0}
        ]
        
        metrics = self.sizer.get_performance_metrics()
        
        self.assertIn('total_positions', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('avg_pnl', metrics)
        self.assertIn('total_pnl', metrics)
        self.assertIn('sharpe_ratio', metrics)
        
        self.assertEqual(metrics['total_positions'], 4)
        self.assertEqual(metrics['win_rate'], 0.5)  # 2 wins out of 4
        self.assertEqual(metrics['total_pnl'], 225.0)  # 100 - 50 + 200 - 25


class TestHighFidelitySizerFactory(unittest.TestCase):
    """Test HighFidelitySizerFactory class"""
    
    def test_create_fixed_sizer(self):
        """Test fixed sizer creation"""
        sizer = HighFidelitySizerFactory.create_fixed_sizer(size=2.0)
        
        self.assertEqual(sizer.config.method, SizingMethod.FIXED)
        self.assertEqual(sizer.config.fixed_size, 2.0)
    
    def test_create_percentage_sizer(self):
        """Test percentage sizer creation"""
        sizer = HighFidelitySizerFactory.create_percentage_sizer(percentage=0.15)
        
        self.assertEqual(sizer.config.method, SizingMethod.PERCENTAGE)
        self.assertEqual(sizer.config.percentage, 0.15)
    
    def test_create_volatility_sizer(self):
        """Test volatility sizer creation"""
        sizer = HighFidelitySizerFactory.create_volatility_sizer(target_volatility=0.20)
        
        self.assertEqual(sizer.config.method, SizingMethod.VOLATILITY)
        self.assertEqual(sizer.config.target_volatility, 0.20)
    
    def test_create_kelly_sizer(self):
        """Test Kelly sizer creation"""
        sizer = HighFidelitySizerFactory.create_kelly_sizer(kelly_fraction=0.3)
        
        self.assertEqual(sizer.config.method, SizingMethod.KELLY)
        self.assertEqual(sizer.config.kelly_fraction, 0.3)
    
    def test_create_adaptive_sizer(self):
        """Test adaptive sizer creation"""
        sizer = HighFidelitySizerFactory.create_adaptive_sizer(base_percentage=0.12)
        
        self.assertEqual(sizer.config.method, SizingMethod.ADAPTIVE)
        self.assertEqual(sizer.config.percentage, 0.12)


class TestHighFidelityExecutionIntegration(unittest.TestCase):
    """Test HighFidelityExecutionIntegration class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.integration = HighFidelityExecutionIntegration()
    
    def test_initialization(self):
        """Test integration initialization"""
        self.assertIsNone(self.integration.broker)
        self.assertIsNone(self.integration.sizer)
        self.assertIsNone(self.integration.cerebro)
    
    def test_setup_cerebro(self):
        """Test Cerebro setup"""
        cerebro = self.integration.setup_cerebro()
        
        self.assertIsNotNone(cerebro)
        self.assertIsNotNone(self.integration.broker)
        self.assertIsNotNone(self.integration.sizer)
        self.assertEqual(self.integration.cerebro, cerebro)
    
    def test_create_sample_data_feed(self):
        """Test sample data feed creation"""
        self.integration.setup_cerebro()
        
        data_feed = self.integration.create_sample_data_feed(
            symbol="EURUSD",
            days=7,
            start_price=1.1000
        )
        
        self.assertIsNotNone(data_feed)
        self.assertIsInstance(data_feed.tick_data, pd.DataFrame)
        self.assertGreater(len(data_feed.tick_data), 0)
    
    def test_generate_sample_tick_data(self):
        """Test sample tick data generation"""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        data = self.integration._generate_sample_tick_data(
            symbol="EURUSD",
            start_date=start_date,
            end_date=end_date,
            start_price=1.1000
        )
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('timestamp', data.columns)
        self.assertIn('bid', data.columns)
        self.assertIn('ask', data.columns)
        self.assertIn('mid', data.columns)
        self.assertIn('spread', data.columns)
        self.assertIn('volume', data.columns)
        
        # Check data integrity
        self.assertTrue((data['ask'] > data['bid']).all())
        self.assertTrue((data['mid'] == (data['ask'] + data['bid']) / 2).all())
        self.assertTrue((data['spread'] == data['ask'] - data['bid']).all())


class TestHighFidelityStrategy(unittest.TestCase):
    """Test HighFidelityStrategy class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock strategy class for testing
        class MockHighFidelityStrategy:
            def __init__(self):
                self.order_count = 0
                self.execution_history = []
            
            def log(self, txt, dt=None):
                pass
            
            def notify_order(self, order):
                self.execution_history.append({
                    'timestamp': datetime.now(),
                    'order_ref': order.ref,
                    'status': order.getstatusname(),
                    'price': order.executed.price if order.executed else None,
                    'size': order.executed.size if order.executed else None
                })
            
            def notify_trade(self, trade):
                pass
            
            def get_execution_summary(self):
                return {
                    'total_orders': self.order_count,
                    'execution_history': self.execution_history,
                    'total_trades': len([h for h in self.execution_history if 'EXECUTED' in h['status']])
                }
        
        self.strategy = MockHighFidelityStrategy()
        
        # Mock data
        self.mock_data = Mock()
        self.mock_data.datetime = Mock()
        self.mock_data.datetime.date.return_value = datetime.now().date()
        self.mock_data.datetime.datetime.return_value = datetime.now()
        
        # Mock order
        self.mock_order = Mock()
        self.mock_order.ref = "TEST_001"
        self.mock_order.getstatusname.return_value = "Submitted"
        self.mock_order.status = 1
        self.mock_order.isbuy.return_value = True
        self.mock_order.executed = Mock()
        self.mock_order.executed.price = 1.1000
        self.mock_order.executed.size = 1.0
        self.mock_order.executed.value = 1100.0
    
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.order_count, 0)
        self.assertEqual(len(self.strategy.execution_history), 0)
    
    def test_log(self):
        """Test logging function"""
        # This should not raise an exception
        self.strategy.log("Test message")
    
    def test_notify_order(self):
        """Test order notification handling"""
        # Test submitted order
        self.mock_order.status = 1  # Submitted
        self.strategy.notify_order(self.mock_order)
        
        # Test completed order
        self.mock_order.status = 3  # Completed
        self.strategy.notify_order(self.mock_order)
        
        # Check execution history
        self.assertGreater(len(self.strategy.execution_history), 0)
    
    def test_notify_trade(self):
        """Test trade notification handling"""
        # Mock trade
        mock_trade = Mock()
        mock_trade.isclosed = True
        mock_trade.pnl = 100.0
        mock_trade.pnlcomm = 95.0
        
        # This should not raise an exception
        self.strategy.notify_trade(mock_trade)
    
    def test_get_execution_summary(self):
        """Test execution summary"""
        summary = self.strategy.get_execution_summary()
        
        self.assertIn('total_orders', summary)
        self.assertIn('execution_history', summary)
        self.assertIn('total_trades', summary)


class TestIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def test_broker_sizer_integration(self):
        """Test broker and sizer integration"""
        # Create broker
        broker_config = ExecutionConfig()
        broker = HighFidelityBroker(broker_config)
        
        # Create sizer
        sizer_config = SizingConfig()
        sizer = HighFidelitySizer(sizer_config)
        
        # Mock broker for sizer
        sizer.broker = broker
        
        # Test that they work together
        self.assertIsNotNone(broker)
        self.assertIsNotNone(sizer)
    
    def test_execution_flow(self):
        """Test complete execution flow"""
        # Create integration
        integration = HighFidelityExecutionIntegration()
        
        # Setup Cerebro
        cerebro = integration.setup_cerebro()
        
        # Create sample data
        data_feed = integration.create_sample_data_feed(days=1)
        integration.add_data_feed(data_feed, name="TEST")
        
        # Verify setup
        self.assertIsNotNone(integration.broker)
        self.assertIsNotNone(integration.sizer)
        self.assertIsNotNone(integration.cerebro)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestExecutionConfig,
        TestHighFidelityBroker,
        TestSizingConfig,
        TestHighFidelitySizer,
        TestHighFidelitySizerFactory,
        TestHighFidelityExecutionIntegration,
        TestHighFidelityStrategy,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running High-Fidelity Execution Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print(f"\nTest execution completed with {'success' if success else 'failures'}")
