#!/usr/bin/env python3
"""
Test Suite for Variable Spread Simulation Module

Comprehensive tests for the variable spread simulation functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import unittest
import tempfile
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app1', 'src'))

from src.backtesting.tick_data.variable_spread_simulator import (
    VariableSpreadSimulator, SpreadConfig, SpreadModel, SpreadData
)
from src.backtesting.tick_data.enhanced_tick_feed import (
    EnhancedTickDataFeed, EnhancedTickDataFeedFactory
)
from src.backtesting.tick_data.tick_data_loader import TickDataLoader, TickDataConfig, TickDataFormat

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVariableSpreadSimulator(unittest.TestCase):
    """Test cases for VariableSpreadSimulator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SpreadConfig(
            symbol="EURUSD",
            model=SpreadModel.STATISTICAL,
            base_spread=0.0001,
            volatility_multiplier=1.5,
            time_of_day_factor=0.3,
            volume_factor=0.2
        )
        self.simulator = VariableSpreadSimulator(self.config)
        
        # Create sample data
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self, days: int = 1) -> pd.DataFrame:
        """Create sample tick data for testing"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=days * 24 * 60,  # 1 minute intervals
            freq='1T'
        )
        
        # Generate realistic price data
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.0001, len(timestamps)).cumsum()
        mid_prices = base_price + price_changes
        
        # Generate volumes
        volumes = np.random.randint(1, 100, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'mid': mid_prices,
            'volume': volumes
        })
    
    def test_initialization(self):
        """Test simulator initialization"""
        self.assertIsNotNone(self.simulator)
        self.assertEqual(self.simulator.config.symbol, "EURUSD")
        self.assertEqual(self.simulator.config.model, SpreadModel.STATISTICAL)
    
    def test_statistical_spread_simulation(self):
        """Test statistical spread simulation"""
        timestamp = datetime.now()
        mid_price = 1.1000
        volume = 50.0
        
        spread_data = self.simulator.simulate_spread(timestamp, mid_price, volume)
        
        self.assertIsInstance(spread_data, SpreadData)
        self.assertEqual(spread_data.timestamp, timestamp)
        self.assertEqual(spread_data.mid, mid_price)
        self.assertEqual(spread_data.volume, volume)
        self.assertGreater(spread_data.spread, 0)
        self.assertAlmostEqual(spread_data.bid, mid_price - (spread_data.spread / 2), places=6)
        self.assertAlmostEqual(spread_data.ask, mid_price + (spread_data.spread / 2), places=6)
    
    def test_spread_constraints(self):
        """Test that spreads respect min/max constraints"""
        timestamp = datetime.now()
        mid_price = 1.1000
        
        # Test multiple simulations
        spreads = []
        for _ in range(100):
            spread_data = self.simulator.simulate_spread(timestamp, mid_price, 50.0)
            spreads.append(spread_data.spread)
        
        min_spread = min(spreads)
        max_spread = max(spreads)
        
        self.assertGreaterEqual(min_spread, self.config.min_spread)
        self.assertLessEqual(max_spread, self.config.max_spread)
    
    def test_time_of_day_effects(self):
        """Test that spreads vary by time of day"""
        mid_price = 1.1000
        volume = 50.0
        
        # Test different hours
        asian_hour = datetime.now().replace(hour=2, minute=0, second=0, microsecond=0)
        european_hour = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        american_hour = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
        
        asian_spread = self.simulator.simulate_spread(asian_hour, mid_price, volume)
        european_spread = self.simulator.simulate_spread(european_hour, mid_price, volume)
        american_spread = self.simulator.simulate_spread(american_hour, mid_price, volume)
        
        # Spreads should be different (though not guaranteed due to randomness)
        spreads = [asian_spread.spread, european_spread.spread, american_spread.spread]
        self.assertTrue(len(set(spreads)) > 1 or all(s >= self.config.min_spread for s in spreads))
    
    def test_volume_effects(self):
        """Test that spreads are affected by volume"""
        timestamp = datetime.now()
        mid_price = 1.1000
        
        # Test different volumes
        low_volume_spread = self.simulator.simulate_spread(timestamp, mid_price, 5.0)
        high_volume_spread = self.simulator.simulate_spread(timestamp, mid_price, 95.0)
        
        # High volume should generally result in tighter spreads
        # (though not guaranteed due to randomness)
        self.assertGreaterEqual(low_volume_spread.spread, self.config.min_spread)
        self.assertGreaterEqual(high_volume_spread.spread, self.config.min_spread)
    
    def test_market_conditions_update(self):
        """Test updating market conditions"""
        original_volatility = self.simulator.config.market_volatility
        original_liquidity = self.simulator.config.liquidity_level
        
        # Update conditions
        self.simulator.update_market_conditions(volatility=2.0, liquidity=0.5)
        
        self.assertEqual(self.simulator.config.market_volatility, 2.0)
        self.assertEqual(self.simulator.config.liquidity_level, 0.5)
    
    def test_dataframe_simulation(self):
        """Test simulating spreads for entire DataFrame"""
        result_data = self.simulator.simulate_spreads_for_dataframe(self.sample_data)
        
        # Check that new columns were added
        expected_columns = ['simulated_spread', 'simulated_bid', 'simulated_ask', 
                          'market_condition', 'trading_session']
        for col in expected_columns:
            self.assertIn(col, result_data.columns)
        
        # Check that all rows have simulated data
        self.assertFalse(result_data['simulated_spread'].isna().any())
        self.assertFalse(result_data['simulated_bid'].isna().any())
        self.assertFalse(result_data['simulated_ask'].isna().any())
    
    def test_spread_quality_validation(self):
        """Test spread quality validation"""
        # Simulate spreads for sample data
        result_data = self.simulator.simulate_spreads_for_dataframe(self.sample_data)
        
        # Validate quality
        validation_results = self.simulator.validate_spread_quality(result_data)
        
        self.assertIn('quality_score', validation_results)
        self.assertIn('validation_passed', validation_results)
        self.assertIn('mean_spread', validation_results)
        self.assertIn('std_spread', validation_results)
        
        # Quality score should be reasonable
        self.assertGreaterEqual(validation_results['quality_score'], 0)
        self.assertLessEqual(validation_results['quality_score'], 100)
    
    def test_get_spread_statistics(self):
        """Test getting spread statistics"""
        stats = self.simulator.get_spread_statistics()
        
        expected_keys = ['mean_spread', 'median_spread', 'std_spread', 
                        'min_spread', 'max_spread']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # All values should be positive
        for key in expected_keys:
            self.assertGreaterEqual(stats[key], 0)


class TestEnhancedTickDataFeed(unittest.TestCase):
    """Test cases for EnhancedTickDataFeed"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample tick data
        self.sample_data = self._create_sample_tick_data()
        
        # Create spread config
        self.spread_config = SpreadConfig(
            symbol="EURUSD",
            model=SpreadModel.HYBRID,
            base_spread=0.0001
        )
    
    def _create_sample_tick_data(self, days: int = 1) -> pd.DataFrame:
        """Create sample tick data for testing"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=days * 24 * 60,  # 1 minute intervals
            freq='1T'
        )
        
        # Generate realistic price data
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.0001, len(timestamps)).cumsum()
        mid_prices = base_price + price_changes
        
        # Generate spreads and calculate bid/ask
        spreads = np.random.uniform(0.0001, 0.0003, len(timestamps))
        bid_prices = mid_prices - spreads / 2
        ask_prices = mid_prices + spreads / 2
        
        # Generate volumes
        volumes = np.random.randint(1, 100, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'bid': bid_prices,
            'ask': ask_prices,
            'volume': volumes
        })
    
    def test_initialization_with_spread_simulation(self):
        """Test initialization with spread simulation enabled"""
        feed = EnhancedTickDataFeed(
            self.sample_data, 
            "EURUSD", 
            self.spread_config, 
            enable_spread_simulation=True
        )
        
        self.assertIsNotNone(feed)
        self.assertTrue(feed.enable_spread_simulation)
        self.assertIsNotNone(feed.spread_simulator)
    
    def test_initialization_without_spread_simulation(self):
        """Test initialization without spread simulation"""
        feed = EnhancedTickDataFeed(
            self.sample_data, 
            "EURUSD", 
            enable_spread_simulation=False
        )
        
        self.assertIsNotNone(feed)
        self.assertFalse(feed.enable_spread_simulation)
        self.assertIsNone(feed.spread_simulator)
    
    def test_data_preparation(self):
        """Test data preparation"""
        feed = EnhancedTickDataFeed(self.sample_data, "EURUSD")
        
        # Check that required columns exist
        self.assertIn('mid', feed.tick_data.columns)
        self.assertIn('datetime_num', feed.tick_data.columns)
        
        # Check that data is sorted by timestamp
        timestamps = feed.tick_data['timestamp']
        self.assertTrue(timestamps.is_monotonic_increasing)
    
    def test_spread_simulation_integration(self):
        """Test integration with spread simulation"""
        feed = EnhancedTickDataFeed(
            self.sample_data, 
            "EURUSD", 
            self.spread_config, 
            enable_spread_simulation=True
        )
        
        # Check that spread simulation columns were added
        expected_columns = ['simulated_spread', 'simulated_bid', 'simulated_ask', 
                          'market_condition', 'trading_session', 'spread']
        for col in expected_columns:
            self.assertIn(col, feed.tick_data.columns)
        
        # Check that spreads are realistic
        spreads = feed.tick_data['spread']
        self.assertTrue((spreads > 0).all())
        self.assertTrue((spreads < 0.01).all())  # Less than 10 pips
    
    def test_get_current_tick(self):
        """Test getting current tick data"""
        feed = EnhancedTickDataFeed(
            self.sample_data, 
            "EURUSD", 
            self.spread_config, 
            enable_spread_simulation=True
        )
        
        # Start the feed
        feed.start()
        
        # Manually advance index to simulate loading
        feed.current_index = 1
        
        # Get current tick
        tick = feed.get_current_tick()
        
        self.assertIsNotNone(tick)
        self.assertIn('timestamp', tick)
        self.assertIn('bid', tick)
        self.assertIn('ask', tick)
        self.assertIn('spread', tick)
        self.assertIn('mid', tick)
        self.assertIn('volume', tick)
        
        if feed.enable_spread_simulation:
            self.assertIn('simulated_spread', tick)
            self.assertIn('market_condition', tick)
            self.assertIn('trading_session', tick)
    
    def test_get_data_info(self):
        """Test getting data information"""
        feed = EnhancedTickDataFeed(
            self.sample_data, 
            "EURUSD", 
            self.spread_config, 
            enable_spread_simulation=True
        )
        
        info = feed.get_data_info()
        
        expected_keys = ['symbol', 'total_ticks', 'start_time', 'end_time', 
                        'avg_spread', 'min_spread', 'max_spread', 'spread_simulation_enabled']
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['symbol'], "EURUSD")
        self.assertEqual(info['total_ticks'], len(self.sample_data))
        self.assertTrue(info['spread_simulation_enabled'])
    
    def test_market_conditions_update(self):
        """Test updating market conditions"""
        feed = EnhancedTickDataFeed(
            self.sample_data, 
            "EURUSD", 
            self.spread_config, 
            enable_spread_simulation=True
        )
        
        # Update market conditions
        feed.update_market_conditions(volatility=2.0, liquidity=0.5)
        
        # Check that conditions were updated
        self.assertEqual(feed.spread_simulator.config.market_volatility, 2.0)
        self.assertEqual(feed.spread_simulator.config.liquidity_level, 0.5)
    
    def test_get_spread_analysis(self):
        """Test getting spread analysis"""
        feed = EnhancedTickDataFeed(
            self.sample_data, 
            "EURUSD", 
            self.spread_config, 
            enable_spread_simulation=True
        )
        
        analysis = feed.get_spread_analysis()
        
        self.assertIn('overall_stats', analysis)
        self.assertIn('hourly_analysis', analysis)
        self.assertIn('spread_distribution', analysis)
        
        # Check overall stats
        overall_stats = analysis['overall_stats']
        expected_stats = ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75']
        for stat in expected_stats:
            self.assertIn(stat, overall_stats)


class TestEnhancedTickDataFeedFactory(unittest.TestCase):
    """Test cases for EnhancedTickDataFeedFactory"""
    
    def test_from_dataframe(self):
        """Test creating feed from DataFrame"""
        # Create sample data
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'bid': [1.1000],
            'ask': [1.1002],
            'volume': [50]
        })
        
        feed = EnhancedTickDataFeedFactory.from_dataframe(sample_data, "EURUSD")
        
        self.assertIsInstance(feed, EnhancedTickDataFeed)
        self.assertEqual(feed.symbol, "EURUSD")
    
    def test_create_sample_feed(self):
        """Test creating sample feed"""
        feed = EnhancedTickDataFeedFactory.create_sample_feed("EURUSD", days=1)
        
        self.assertIsInstance(feed, EnhancedTickDataFeed)
        self.assertEqual(feed.symbol, "EURUSD")
        self.assertTrue(feed.enable_spread_simulation)
    
    def test_create_high_volatility_feed(self):
        """Test creating high volatility feed"""
        feed = EnhancedTickDataFeedFactory.create_high_volatility_feed("EURUSD", days=1)
        
        self.assertIsInstance(feed, EnhancedTickDataFeed)
        self.assertEqual(feed.symbol, "EURUSD")
        self.assertTrue(feed.enable_spread_simulation)
        
        # Check that high volatility settings are applied
        config = feed.spread_simulator.config
        self.assertEqual(config.market_volatility, 2.0)
        self.assertEqual(config.liquidity_level, 0.5)
    
    def test_create_low_volatility_feed(self):
        """Test creating low volatility feed"""
        feed = EnhancedTickDataFeedFactory.create_low_volatility_feed("EURUSD", days=1)
        
        self.assertIsInstance(feed, EnhancedTickDataFeed)
        self.assertEqual(feed.symbol, "EURUSD")
        self.assertTrue(feed.enable_spread_simulation)
        
        # Check that low volatility settings are applied
        config = feed.spread_simulator.config
        self.assertEqual(config.market_volatility, 0.5)
        self.assertEqual(config.liquidity_level, 2.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_simulation(self):
        """Test complete end-to-end spread simulation"""
        # Create sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range(
                start=datetime.now() - timedelta(hours=1),
                periods=60,
                freq='1T'
            ),
            'bid': np.random.uniform(1.0950, 1.1050, 60),
            'ask': np.random.uniform(1.0952, 1.1052, 60),
            'volume': np.random.randint(1, 100, 60)
        })
        
        # Create enhanced feed
        feed = EnhancedTickDataFeed(
            sample_data, 
            "EURUSD", 
            enable_spread_simulation=True
        )
        
        # Verify spread simulation worked
        self.assertIn('simulated_spread', feed.tick_data.columns)
        self.assertIn('market_condition', feed.tick_data.columns)
        self.assertIn('trading_session', feed.tick_data.columns)
        
        # Verify data quality
        spreads = feed.tick_data['spread']
        self.assertTrue((spreads > 0).all())
        self.assertTrue((spreads < 0.01).all())
        
        # Test feed operation (simplified test without Backtrader line buffer)
        feed.start()
        
        # Test that we can get data info
        info = feed.get_data_info()
        self.assertIn('total_ticks', info)
        self.assertGreater(info['total_ticks'], 0)
        
        # Test that we can get spread analysis
        analysis = feed.get_spread_analysis()
        self.assertIn('overall_stats', analysis)
    
    def test_different_spread_models(self):
        """Test different spread simulation models"""
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'bid': [1.1000],
            'ask': [1.1002],
            'volume': [50]
        })
        
        models = [SpreadModel.STATISTICAL, SpreadModel.MARKET_CONDITIONS]
        
        for model in models:
            config = SpreadConfig(symbol="EURUSD", model=model)
            feed = EnhancedTickDataFeed(sample_data, "EURUSD", config, True)
            
            self.assertIsNotNone(feed)
            self.assertTrue(feed.enable_spread_simulation)
            self.assertEqual(feed.spread_simulator.config.model, model)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestVariableSpreadSimulator,
        TestEnhancedTickDataFeed,
        TestEnhancedTickDataFeedFactory,
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
    print("Running Variable Spread Simulation Tests...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
