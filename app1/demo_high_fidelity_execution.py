#!/usr/bin/env python3
"""
High-Fidelity Order Execution Demonstration

Demonstrates the high-fidelity order execution system with realistic
spread simulation, slippage, market impact, and latency modeling.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add the app1/src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app1', 'src'))

from src.backtesting.execution.high_fidelity_broker import (
    HighFidelityBroker, ExecutionConfig, ExecutionResult
)
from src.backtesting.execution.high_fidelity_sizer import (
    HighFidelitySizer, SizingConfig, SizingMethod, HighFidelitySizerFactory
)
from src.backtesting.execution.execution_integration import (
    HighFidelityExecutionIntegration, HighFidelityStrategy, create_high_fidelity_backtest
)
from src.backtesting.tick_data.variable_spread_simulator import SpreadModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_broker_execution():
    """Demonstrate high-fidelity broker execution"""
    
    print("\n" + "="*60)
    print("HIGH-FIDELITY BROKER EXECUTION DEMONSTRATION")
    print("="*60)
    
    # Create execution config
    config = ExecutionConfig(
        spread_model=SpreadModel.HYBRID,
        base_spread=0.0001,
        slippage_model="realistic",
        slippage_factor=0.15,
        market_impact_enabled=True,
        latency_enabled=True,
        partial_fills_enabled=True,
        log_execution_details=True
    )
    
    # Create broker
    broker = HighFidelityBroker(config)
    
    # Create sample market data
    sample_ticks = []
    base_price = 1.1000
    
    for i in range(100):
        # Simulate price movement
        price_change = np.random.normal(0, 0.0001)
        base_price += price_change
        
        # Create tick data
        spread = np.random.uniform(0.0001, 0.0003)
        tick = {
            'timestamp': datetime.now() + timedelta(minutes=i),
            'bid': base_price - spread/2,
            'ask': base_price + spread/2,
            'mid': base_price,
            'volume': np.random.uniform(100, 1000),
            'volatility': np.random.uniform(0.8, 1.2),
            'liquidity': np.random.uniform(0.9, 1.1)
        }
        sample_ticks.append(tick)
    
    print(f"Generated {len(sample_ticks)} sample ticks")
    
    # Simulate order execution
    execution_results = []
    
    for i, tick in enumerate(sample_ticks):
        # Set market data
        broker.set_market_data(tick)
        
        # Create mock orders
        if i % 10 == 0:  # Every 10th tick
            # Market buy order
            mock_order = type('MockOrder', (), {
                'ref': f'BUY_{i:03d}',
                'isbuy': lambda self: True,
                'size': 1.0,
                'exectype': 0,  # Market
                'data': type('MockData', (), {'_name': 'EURUSD'})()
            })()
            
            result = broker._execute_market_order(mock_order, tick)
            if result:
                execution_results.append(result)
        
        elif i % 15 == 0:  # Every 15th tick
            # Market sell order
            mock_order = type('MockOrder', (), {
                'ref': f'SELL_{i:03d}',
                'isbuy': lambda self: False,
                'size': 0.5,
                'exectype': 0,  # Market
                'data': type('MockData', (), {'_name': 'EURUSD'})()
            })()
            
            result = broker._execute_market_order(mock_order, tick)
            if result:
                execution_results.append(result)
    
    print(f"Executed {len(execution_results)} orders")
    
    # Analyze execution results
    if execution_results:
        avg_slippage = np.mean([r.slippage for r in execution_results])
        avg_latency = np.mean([r.latency_ms for r in execution_results])
        avg_quality = np.mean([r.execution_quality_score for r in execution_results])
        avg_spread = np.mean([r.spread_at_execution for r in execution_results])
        
        print(f"\nExecution Statistics:")
        print(f"  Average Slippage: {avg_slippage:.6f} ({avg_slippage*10000:.2f} pips)")
        print(f"  Average Latency: {avg_latency:.1f} ms")
        print(f"  Average Quality Score: {avg_quality:.1f}/100")
        print(f"  Average Spread: {avg_spread:.6f} ({avg_spread*10000:.2f} pips)")
        
        # Show sample executions
        print(f"\nSample Executions:")
        for i, result in enumerate(execution_results[:5]):
            print(f"  {i+1}. {result.side.upper()} {result.executed_quantity:.2f} @ {result.fill_price:.5f} "
                  f"(Slippage: {result.slippage:.6f}, Quality: {result.execution_quality_score:.1f})")
    
    return execution_results


def demonstrate_sizer_functionality():
    """Demonstrate high-fidelity sizer functionality"""
    
    print("\n" + "="*60)
    print("HIGH-FIDELITY SIZER FUNCTIONALITY DEMONSTRATION")
    print("="*60)
    
    # Test different sizing methods
    methods = [
        (SizingMethod.FIXED, "Fixed Size"),
        (SizingMethod.PERCENTAGE, "Percentage Based"),
        (SizingMethod.VOLATILITY, "Volatility Based"),
        (SizingMethod.KELLY, "Kelly Criterion"),
        (SizingMethod.ADAPTIVE, "Adaptive Sizing")
    ]
    
    # Create sample data
    sample_data = []
    base_price = 1.1000
    
    for i in range(50):
        price_change = np.random.normal(0, 0.0001)
        base_price += price_change
        
        data_point = {
            'close': [base_price],
            'spread': [np.random.uniform(0.0001, 0.0003)],
            'volume': [np.random.uniform(100, 1000)],
            'ask': [base_price + 0.0001],
            'bid': [base_price - 0.0001]
        }
        sample_data.append(data_point)
    
    print(f"Generated {len(sample_data)} data points for sizing tests")
    
    # Test each sizing method
    for method, name in methods:
        print(f"\n{name} Sizing:")
        
        # Create sizer
        if method == SizingMethod.FIXED:
            sizer = HighFidelitySizerFactory.create_fixed_sizer(size=1.0)
        elif method == SizingMethod.PERCENTAGE:
            sizer = HighFidelitySizerFactory.create_percentage_sizer(percentage=0.1)
        elif method == SizingMethod.VOLATILITY:
            sizer = HighFidelitySizerFactory.create_volatility_sizer(target_volatility=0.15)
        elif method == SizingMethod.KELLY:
            sizer = HighFidelitySizerFactory.create_kelly_sizer(kelly_fraction=0.25)
        else:  # ADAPTIVE
            sizer = HighFidelitySizerFactory.create_adaptive_sizer(base_percentage=0.1)
        
        # Mock broker
        sizer.broker = type('MockBroker', (), {
            'getvalue': lambda: 10000.0
        })()
        
        # Test sizing
        sizes = []
        for data_point in sample_data:
            mock_data = type('MockData', (), data_point)()
            size = sizer._getsizing(mock_data, isbuy=True)
            sizes.append(size)
        
        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        min_size = np.min(sizes)
        max_size = np.max(sizes)
        
        print(f"  Average Size: {avg_size:.4f}")
        print(f"  Size Std Dev: {std_size:.4f}")
        print(f"  Min Size: {min_size:.4f}")
        print(f"  Max Size: {max_size:.4f}")
        
        # Show performance metrics
        metrics = sizer.get_performance_metrics()
        print(f"  Total Positions: {metrics['total_positions']}")


def demonstrate_integration():
    """Demonstrate complete integration"""
    
    print("\n" + "="*60)
    print("COMPLETE INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Create integration
    integration = HighFidelityExecutionIntegration()
    
    # Setup Cerebro
    cerebro = integration.setup_cerebro()
    
    # Create sample data feed
    data_feed = integration.create_sample_data_feed(
        symbol="EURUSD",
        days=7,  # 1 week of data
        start_price=1.1000
    )
    
    # Add data feed
    integration.add_data_feed(data_feed, name="EURUSD")
    
            print(f"Created data feed with {len(data_feed.tick_data)} data points")
    
    # Create a simple strategy
    class SimpleStrategy(HighFidelityStrategy):
        def __init__(self):
            super().__init__()
            self.order_count = 0
        
        def next(self):
            # Simple buy and hold strategy
            if not self.position and self.order_count == 0:
                self.buy()
                self.order_count += 1
                self.log("BUY ORDER CREATED")
        
        def notify_order(self, order):
            super().notify_order(order)
            if order.status in [order.Completed]:
                self.log(f"Order {order.ref} completed at {order.executed.price:.5f}")
    
    # Run backtest
    print("Running backtest...")
    results = integration.run_backtest(SimpleStrategy)
    
    # Display results
    print(f"\nBacktest Results:")
    print(f"  Strategy executed successfully")
    
    # Get execution statistics
    stats = integration.broker.get_execution_statistics()
    print(f"\nExecution Statistics:")
    print(f"  Total Orders: {stats['total_orders']}")
    print(f"  Successful Orders: {stats['successful_orders']}")
    print(f"  Success Rate: {stats['success_rate']:.2f}%")
    print(f"  Average Slippage: {stats['avg_slippage']:.6f}")
    print(f"  Average Quality Score: {stats['avg_quality_score']:.2f}")
    print(f"  Average Execution Time: {stats['avg_execution_time_ms']:.2f}ms")
    
    return results


def create_execution_visualization(execution_results):
    """Create visualization of execution results"""
    
    if not execution_results:
        print("No execution results to visualize")
        return
    
    print("\nCreating execution visualization...")
    
    # Extract data for plotting
    timestamps = [r.execution_time for r in execution_results]
    prices = [r.fill_price for r in execution_results]
    spreads = [r.spread_at_execution for r in execution_results]
    slippages = [r.slippage for r in execution_results]
    quality_scores = [r.execution_quality_score for r in execution_results]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('High-Fidelity Execution Analysis', fontsize=16)
    
    # Plot 1: Execution prices over time
    ax1.plot(timestamps, prices, 'b-', alpha=0.7, linewidth=1)
    ax1.scatter(timestamps, prices, c='red', s=20, alpha=0.8)
    ax1.set_title('Execution Prices Over Time')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spread distribution
    ax2.hist(spreads, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Spread Distribution')
    ax2.set_xlabel('Spread')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Slippage analysis
    ax3.scatter(spreads, slippages, alpha=0.7, color='orange')
    ax3.set_title('Slippage vs Spread')
    ax3.set_xlabel('Spread')
    ax3.set_ylabel('Slippage')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Quality scores
    ax4.hist(quality_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_title('Execution Quality Score Distribution')
    ax4.set_xlabel('Quality Score')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'high_fidelity_execution_demo.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")
    
    # Show plot
    plt.show()


def main():
    """Main demonstration function"""
    
    print("HIGH-FIDELITY ORDER EXECUTION SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases realistic order execution with:")
    print("  • Variable spread simulation")
    print("  • Realistic slippage modeling")
    print("  • Market impact calculation")
    print("  • Latency simulation")
    print("  • Partial fill simulation")
    print("  • Quality scoring")
    print("  • Multiple position sizing methods")
    
    try:
        # Demonstrate broker execution
        execution_results = demonstrate_broker_execution()
        
        # Demonstrate sizer functionality
        demonstrate_sizer_functionality()
        
        # Demonstrate complete integration
        integration_results = demonstrate_integration()
        
        # Create visualization
        if execution_results:
            create_execution_visualization(execution_results)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Key Features Demonstrated:")
        print("  ✅ Variable spread simulation")
        print("  ✅ Realistic order execution")
        print("  ✅ Multiple position sizing methods")
        print("  ✅ Market impact modeling")
        print("  ✅ Slippage calculation")
        print("  ✅ Latency simulation")
        print("  ✅ Quality scoring")
        print("  ✅ Complete Backtrader integration")
        print("\nThe system is ready for high-fidelity backtesting!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
