#!/usr/bin/env python3
"""
Test script for Spread-Aware Broker
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add the app1 directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app1'))

import backtrader as bt
from src.strategy_validation.backtesting.spread_aware_broker import (
    SpreadAwareBroker, create_spread_aware_broker
)
from src.strategy_validation.backtesting.spread_simulator import SpreadModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStrategy(bt.Strategy):
    """Test strategy for spread-aware broker"""
    
    def __init__(self):
        self.order = None
        self.trade_count = 0
        self.buy_price = None
        self.sell_price = None
        
    def next(self):
        """Called for each data point"""
        if not self.position:
            # Buy on first data point
            if self.trade_count == 0:
                self.order = self.buy()
                self.trade_count += 1
        else:
            # Sell after holding for a few periods
            if self.trade_count == 1 and len(self.data) > 5:
                self.order = self.sell()
                self.trade_count += 1
    
    def notify_order(self, order):
        """Called when order status changes"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed_price
                logger.info(f"Buy executed at {order.executed_price:.5f}, commission: ${order.commission:.2f}")
            else:
                self.sell_price = order.executed_price
                logger.info(f"Sell executed at {order.executed_price:.5f}, commission: ${order.commission:.2f}")
    
    def stop(self):
        """Called when backtest ends"""
        logger.info(f"Strategy completed. Trades: {self.trade_count}")
        if self.buy_price and self.sell_price:
            profit = (self.sell_price - self.buy_price) / self.buy_price * 100
            logger.info(f"Trade profit: {profit:.2f}%")


def test_spread_aware_broker():
    """Test spread-aware broker with Backtrader"""
    logger.info("Testing Spread-Aware Broker...")
    
    try:
        # Create cerebro
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(TestStrategy)
        
        # Create mock data
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
        
        # Create data feed
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
        
        # Create spread-aware broker
        broker = create_spread_aware_broker(
            cash=10000.0,
            commission=0.001,
            spread_model=SpreadModel.STATISTICAL,
            base_spread=0.0001
        )
        
        # Set broker
        cerebro.broker = broker
        
        # Print starting conditions
        logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
        
        # Run backtest
        logger.info("Running backtest with spread-aware broker...")
        cerebro.run()
        
        # Print final conditions
        logger.info(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
        
        # Test broker methods
        logger.info(f"Cash: ${cerebro.broker.get_cash():.2f}")
        logger.info(f"Portfolio Value: ${cerebro.broker.get_value():.2f}")
        
        # Get spread statistics
        stats = broker.spread_simulator.get_spread_statistics('EURUSD')
        logger.info(f"Spread Statistics: {stats}")
        
        logger.info("✅ Spread-aware broker test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Spread-aware broker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_spread_models():
    """Test different spread models"""
    logger.info("Testing Different Spread Models...")
    
    models = [
        SpreadModel.FIXED,
        SpreadModel.TIME_BASED,
        SpreadModel.VOLATILITY_BASED,
        SpreadModel.STATISTICAL
    ]
    
    results = {}
    
    for model in models:
        logger.info(f"Testing {model.value} model...")
        
        try:
            # Create broker with specific model
            broker = create_spread_aware_broker(
                cash=10000.0,
                commission=0.001,
                spread_model=model,
                base_spread=0.0001
            )
            
            # Generate some spread data
            base_bid, base_ask = 1.1000, 1.1001
            spreads = []
            
            for i in range(20):
                timestamp = datetime.now() + timedelta(minutes=i)
                adjusted_bid, adjusted_ask = broker.spread_simulator.get_spread(
                    'EURUSD', timestamp, base_bid, base_ask
                )
                spread = adjusted_ask - adjusted_bid
                spreads.append(spread)
            
            # Calculate statistics
            mean_spread = np.mean(spreads)
            std_spread = np.std(spreads)
            min_spread = np.min(spreads)
            max_spread = np.max(spreads)
            
            results[model.value] = {
                'mean': mean_spread,
                'std': std_spread,
                'min': min_spread,
                'max': max_spread
            }
            
            logger.info(f"{model.value}: Mean={mean_spread:.5f}, Std={std_spread:.5f}, Range=[{min_spread:.5f}, {max_spread:.5f}]")
            
        except Exception as e:
            logger.error(f"Error testing {model.value}: {e}")
            results[model.value] = None
    
    logger.info("✅ Different spread models test completed!")
    return results


def test_order_execution():
    """Test order execution with spread-aware broker"""
    logger.info("Testing Order Execution...")
    
    try:
        # Create broker
        broker = create_spread_aware_broker(
            cash=10000.0,
            commission=0.001,
            spread_model=SpreadModel.STATISTICAL
        )
        
        # Test broker methods
        logger.info(f"Initial cash: ${broker.get_cash():.2f}")
        logger.info(f"Initial value: ${broker.get_value():.2f}")
        
        # Test spread calculation
        base_bid, base_ask = 1.1000, 1.1001
        adjusted_bid, adjusted_ask = broker.spread_simulator.get_spread(
            'EURUSD', datetime.now(), base_bid, base_ask
        )
        spread = adjusted_ask - adjusted_bid
        
        logger.info(f"Spread calculation: {base_bid:.4f}/{base_ask:.4f} -> {adjusted_bid:.4f}/{adjusted_ask:.4f}")
        logger.info(f"Spread: {spread:.5f}")
        
        logger.info("✅ Order execution test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Order execution test failed: {e}")
        return False


def main():
    """Run all spread-aware broker tests"""
    logger.info("Starting Spread-Aware Broker tests...")
    
    tests = [
        test_spread_aware_broker,
        test_different_spread_models,
        test_order_execution
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}")
            failed += 1
    
    logger.info("\n" + "="*50)
    logger.info("SPREAD-AWARE BROKER TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All spread-aware broker tests passed!")
        return True
    else:
        logger.error("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
