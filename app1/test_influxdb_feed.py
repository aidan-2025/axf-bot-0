#!/usr/bin/env python3
"""
Test script for InfluxDB data feed integration with Backtrader
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta

# Add the app1 directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app1'))

import backtrader as bt
from src.strategy_validation.backtesting.influxdb_feed import create_data_feed, FeedConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStrategy(bt.Strategy):
    """Simple test strategy to verify data feed"""
    
    def __init__(self):
        self.data_count = 0
        self.last_price = None
        
    def next(self):
        """Called for each data point"""
        self.data_count += 1
        current_price = self.datas[0].close[0]
        
        if self.data_count <= 10:  # Log first 10 data points
            logger.info(f"Data point {self.data_count}: Close = {current_price:.5f}")
        
        self.last_price = current_price
    
    def stop(self):
        """Called when backtest ends"""
        logger.info(f"Strategy completed. Processed {self.data_count} data points")
        logger.info(f"Last price: {self.last_price:.5f}")


def test_influxdb_feed():
    """Test InfluxDB data feed with Backtrader"""
    logger.info("Testing InfluxDB data feed integration...")
    
    try:
        # Create a simple test strategy
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(TestStrategy)
        
        # Create data feed
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        logger.info(f"Loading data for EURUSD from {start_date} to {end_date}")
        
        # Try to create data feed
        try:
            data_feed = create_data_feed(
                symbol='EURUSD',
                timeframe='1h',  # Use 1-hour data for testing
                start_date=start_date,
                end_date=end_date,
                use_tick_data=False
            )
            
            # Add data to cerebro
            cerebro.adddata(data_feed)
            
            # Set initial capital
            cerebro.broker.setcash(10000.0)
            
            # Set commission
            cerebro.broker.setcommission(commission=0.001)
            
            # Print starting conditions
            logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
            
            # Run backtest
            logger.info("Running backtest...")
            cerebro.run()
            
            # Print final conditions
            logger.info(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
            
            logger.info("✅ InfluxDB data feed test completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create data feed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_mock_data_feed():
    """Test with mock data when InfluxDB is not available"""
    logger.info("Testing with mock data...")
    
    try:
        # Create a simple test strategy
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(TestStrategy)
        
        # Create mock data
        import pandas as pd
        import numpy as np
        
        # Generate mock OHLCV data
        dates = pd.date_range(start='2025-01-01', end='2025-01-07', freq='1h')
        n_points = len(dates)
        
        # Generate realistic price data
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.0001, n_points)
        prices = base_price + np.cumsum(price_changes)
        
        # Create OHLCV data
        mock_data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 0.0005, n_points),
            'low': prices - np.random.uniform(0, 0.0005, n_points),
            'close': prices + np.random.normal(0, 0.0002, n_points),
            'volume': np.random.randint(100, 1000, n_points)
        })
        
        # Create data feed from DataFrame
        data_feed = bt.feeds.PandasData(
            dataname=mock_data,
            datetime='datetime',
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        # Add data to cerebro
        cerebro.adddata(data_feed)
        
        # Set initial capital
        cerebro.broker.setcash(10000.0)
        
        # Set commission
        cerebro.broker.setcommission(commission=0.001)
        
        # Print starting conditions
        logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
        
        # Run backtest
        logger.info("Running backtest with mock data...")
        cerebro.run()
        
        # Print final conditions
        logger.info(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
        
        logger.info("✅ Mock data feed test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock data test failed: {e}")
        return False


async def main():
    """Main test function"""
    logger.info("Starting InfluxDB data feed tests...")
    
    # Test 1: Try InfluxDB data feed
    logger.info("\n" + "="*50)
    logger.info("TEST 1: InfluxDB Data Feed")
    logger.info("="*50)
    
    influxdb_success = test_influxdb_feed()
    
    # Test 2: Mock data feed (fallback)
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Mock Data Feed (Fallback)")
    logger.info("="*50)
    
    mock_success = test_mock_data_feed()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"InfluxDB Data Feed: {'✅ PASSED' if influxdb_success else '❌ FAILED'}")
    logger.info(f"Mock Data Feed: {'✅ PASSED' if mock_success else '❌ FAILED'}")
    
    if influxdb_success or mock_success:
        logger.info("🎉 At least one data feed is working!")
        return True
    else:
        logger.error("💥 All data feed tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
