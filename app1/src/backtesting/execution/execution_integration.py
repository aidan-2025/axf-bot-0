#!/usr/bin/env python3
"""
High-Fidelity Execution Integration

Integrates the high-fidelity broker and sizer with Backtrader for
realistic order execution and position sizing in backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

try:
    import backtrader as bt
    from backtrader import Cerebro, Strategy
    from backtrader.broker import BrokerBase
    from backtrader.sizer import SizerBase
except ImportError:
    bt = None
    # Create mock classes for testing when backtrader is not available
    class MockCerebro:
        def __init__(self):
            pass
        def setbroker(self, broker):
            pass
        def addsizer(self, sizer):
            pass
        def adddata(self, data, name=None):
            pass
        def addstrategy(self, strategy, **kwargs):
            pass
        def run(self):
            return []
    
    class MockStrategy:
        def __init__(self):
            pass
        def next(self):
            pass
        def notify_order(self, order):
            pass
        def notify_trade(self, trade):
            pass
        def buy(self):
            pass
        def sell(self):
            pass
        def log(self, txt, dt=None):
            pass
    
    Cerebro = MockCerebro
    Strategy = MockStrategy
    BrokerBase = None
    SizerBase = None

from .high_fidelity_broker import HighFidelityBroker, ExecutionConfig, ExecutionResult
from .high_fidelity_sizer import HighFidelitySizer, SizingConfig, SizingMethod, HighFidelitySizerFactory
from ..tick_data.enhanced_tick_feed import EnhancedTickDataFeed, EnhancedTickDataFeedFactory
from ..tick_data.variable_spread_simulator import SpreadConfig, SpreadModel

logger = logging.getLogger(__name__)


@dataclass
class ExecutionIntegrationConfig:
    """Configuration for execution integration"""
    
    # Broker configuration
    execution_config: ExecutionConfig = None
    
    # Sizer configuration
    sizing_config: SizingConfig = None
    
    # Data feed configuration
    spread_model: SpreadModel = SpreadModel.HYBRID
    base_spread: float = 0.0001
    
    # Integration settings
    enable_realistic_execution: bool = True
    enable_variable_spreads: bool = True
    enable_market_impact: bool = True
    enable_slippage: bool = True
    enable_latency: bool = True
    
    # Performance settings
    log_execution_details: bool = True
    track_performance_metrics: bool = True
    generate_execution_reports: bool = True


class HighFidelityExecutionIntegration:
    """
    Integrates high-fidelity execution components with Backtrader
    for realistic backtesting with variable spreads and order execution.
    """
    
    def __init__(self, config: ExecutionIntegrationConfig = None):
        self.config = config or ExecutionIntegrationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.broker: Optional[HighFidelityBroker] = None
        self.sizer: Optional[HighFidelitySizer] = None
        self.cerebro: Optional[Cerebro] = None
        
        # Performance tracking
        self.execution_statistics: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        logger.info("HighFidelityExecutionIntegration initialized")
    
    def setup_cerebro(self, cerebro: Cerebro = None) -> Cerebro:
        """Setup Cerebro with high-fidelity execution components"""
        
        if cerebro is None:
            cerebro = Cerebro()
        
        self.cerebro = cerebro
        
        # Setup broker
        self._setup_broker()
        
        # Setup sizer
        self._setup_sizer()
        
        # Configure Cerebro
        cerebro.setbroker(self.broker)
        cerebro.addsizer(self.sizer)
        
        logger.info("Cerebro configured with high-fidelity execution")
        
        return cerebro
    
    def _setup_broker(self):
        """Setup high-fidelity broker"""
        
        # Create execution config
        if self.config.execution_config is None:
            self.config.execution_config = ExecutionConfig(
                spread_model=self.config.spread_model,
                base_spread=self.config.base_spread,
                slippage_model="realistic" if self.config.enable_slippage else "none",
                market_impact_enabled=self.config.enable_market_impact,
                latency_enabled=self.config.enable_latency,
                log_execution_details=self.config.log_execution_details
            )
        
        # Create broker
        self.broker = HighFidelityBroker(self.config.execution_config)
        
        logger.info("High-fidelity broker configured")
    
    def _setup_sizer(self):
        """Setup high-fidelity sizer"""
        
        # Create sizing config
        if self.config.sizing_config is None:
            self.config.sizing_config = SizingConfig(
                method=SizingMethod.PERCENTAGE,
                percentage=0.1,
                max_position_size=0.2,
                min_position_size=0.01
            )
        
        # Create sizer
        self.sizer = HighFidelitySizer(self.config.sizing_config)
        
        logger.info("High-fidelity sizer configured")
    
    def add_data_feed(self, data_feed: EnhancedTickDataFeed, name: str = None):
        """Add enhanced tick data feed to Cerebro"""
        
        if self.cerebro is None:
            raise ValueError("Cerebro not initialized. Call setup_cerebro() first.")
        
        # Add data feed
        self.cerebro.adddata(data_feed, name=name)
        
        logger.info(f"Added data feed: {name or 'unnamed'}")
    
    def create_sample_data_feed(self, symbol: str = "EURUSD", 
                              days: int = 30, 
                              start_price: float = 1.1000) -> EnhancedTickDataFeed:
        """Create sample data feed for testing"""
        
        # Create sample data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate sample tick data
        sample_data = self._generate_sample_tick_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            start_price=start_price
        )
        
        # Create spread config
        spread_config = SpreadConfig(
            model=self.config.spread_model,
            base_spread=self.config.base_spread
        )
        
        # Create enhanced tick data feed
        data_feed = EnhancedTickDataFeed(
            tick_data=sample_data,
            spread_config=spread_config
        )
        
        return data_feed
    
    def _generate_sample_tick_data(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, start_price: float) -> pd.DataFrame:
        """Generate sample tick data for testing"""
        
        # Generate timestamps (1-minute intervals)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # Generate price data (random walk with trend)
        np.random.seed(42)  # For reproducible results
        n_ticks = len(timestamps)
        
        # Generate returns
        returns = np.random.normal(0.0001, 0.001, n_ticks)  # Small positive drift
        
        # Calculate prices
        prices = [start_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate bid/ask spreads
        base_spread = self.config.base_spread
        spreads = np.random.uniform(base_spread * 0.5, base_spread * 2.0, n_ticks)
        
        # Calculate bid and ask prices
        mid_prices = np.array(prices)
        bid_prices = mid_prices - spreads / 2
        ask_prices = mid_prices + spreads / 2
        
        # Ensure ask > bid
        ask_prices = np.maximum(ask_prices, bid_prices + 0.00001)
        
        # Generate volume data
        volumes = np.random.uniform(100, 1000, n_ticks)
        
        # Recalculate spreads to ensure accuracy
        calculated_spreads = ask_prices - bid_prices
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'bid': bid_prices,
            'ask': ask_prices,
            'mid': mid_prices,
            'spread': calculated_spreads,
            'volume': volumes
        })
        
        return data
    
    def run_backtest(self, strategy_class, **strategy_params) -> Dict[str, Any]:
        """Run backtest with high-fidelity execution"""
        
        if self.cerebro is None:
            raise ValueError("Cerebro not initialized. Call setup_cerebro() first.")
        
        # Add strategy
        self.cerebro.addstrategy(strategy_class, **strategy_params)
        
        # Run backtest
        logger.info("Starting high-fidelity backtest...")
        results = self.cerebro.run()
        
        # Collect execution statistics
        self._collect_execution_statistics()
        
        # Generate performance report
        if self.config.generate_execution_reports:
            self._generate_performance_report()
        
        return {
            'results': results,
            'execution_statistics': self.execution_statistics,
            'performance_metrics': self.performance_metrics
        }
    
    def _collect_execution_statistics(self):
        """Collect execution statistics from broker"""
        
        if self.broker:
            self.execution_statistics = self.broker.get_execution_statistics()
        
        if self.sizer:
            self.performance_metrics = self.sizer.get_performance_metrics()
    
    def _generate_performance_report(self):
        """Generate performance report"""
        
        report = {
            'execution_summary': self.execution_statistics,
            'sizing_performance': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log summary
        logger.info("=== EXECUTION PERFORMANCE REPORT ===")
        logger.info(f"Total Orders: {self.execution_statistics.get('total_orders', 0)}")
        logger.info(f"Success Rate: {self.execution_statistics.get('success_rate', 0):.2f}%")
        logger.info(f"Avg Slippage: {self.execution_statistics.get('avg_slippage', 0):.6f}")
        logger.info(f"Avg Quality Score: {self.execution_statistics.get('avg_quality_score', 0):.2f}")
        logger.info(f"Avg Execution Time: {self.execution_statistics.get('avg_execution_time_ms', 0):.2f}ms")
        
        return report
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get detailed execution history"""
        
        if self.broker:
            return self.broker.get_execution_history()
        return []
    
    def reset_statistics(self):
        """Reset all statistics"""
        
        if self.broker:
            self.broker.reset_statistics()
        
        if self.sizer:
            self.sizer.position_history.clear()
        
        self.execution_statistics.clear()
        self.performance_metrics.clear()
        
        logger.info("Statistics reset")


class HighFidelityStrategy(Strategy):
    """
    Base strategy class with high-fidelity execution support
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Execution tracking
        self.order_count = 0
        self.execution_history = []
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        self.logger.info(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Handle order notifications"""
        
        if order.status in [order.Submitted, order.Accepted]:
            self.log(f'Order {order.ref} {order.getstatusname()}')
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED - Price: {order.executed.price:.5f}, '
                        f'Size: {order.executed.size:.2f}, Cost: {order.executed.value:.2f}')
            else:
                self.log(f'SELL EXECUTED - Price: {order.executed.price:.5f}, '
                        f'Size: {order.executed.size:.2f}, Cost: {order.executed.value:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.ref} {order.getstatusname()}')
        
        # Track execution
        self.execution_history.append({
            'timestamp': self.datas[0].datetime.datetime(0),
            'order_ref': order.ref,
            'status': order.getstatusname(),
            'price': order.executed.price if order.executed else None,
            'size': order.executed.size if order.executed else None
        })
    
    def notify_trade(self, trade):
        """Handle trade notifications"""
        
        if not trade.isclosed:
            return
        
        self.log(f'TRADE PROFIT - Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary for this strategy"""
        
        return {
            'total_orders': self.order_count,
            'execution_history': self.execution_history,
            'total_trades': len([h for h in self.execution_history if 'EXECUTED' in h['status']])
        }


def create_high_fidelity_backtest(symbol: str = "EURUSD", 
                                days: int = 30,
                                strategy_class: Strategy = None,
                                **strategy_params) -> Dict[str, Any]:
    """
    Create and run a high-fidelity backtest
    
    Args:
        symbol: Symbol to backtest
        days: Number of days of data
        strategy_class: Strategy class to use
        **strategy_params: Strategy parameters
    
    Returns:
        Backtest results with execution statistics
    """
    
    # Create integration
    integration = HighFidelityExecutionIntegration()
    
    # Setup Cerebro
    cerebro = integration.setup_cerebro()
    
    # Add sample data
    data_feed = integration.create_sample_data_feed(symbol=symbol, days=days)
    integration.add_data_feed(data_feed, name=symbol)
    
    # Use default strategy if none provided
    if strategy_class is None:
        strategy_class = HighFidelityStrategy
    
    # Run backtest
    results = integration.run_backtest(strategy_class, **strategy_params)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run sample backtest
    results = create_high_fidelity_backtest(
        symbol="EURUSD",
        days=7,  # 1 week of data
        strategy_class=HighFidelityStrategy
    )
    
    print("Backtest completed!")
    print(f"Execution Statistics: {results['execution_statistics']}")
    print(f"Performance Metrics: {results['performance_metrics']}")
