#!/usr/bin/env python3
"""
Variable Spread Simulation Demo

Demonstrates the variable spread simulation functionality with different
market conditions and spread models.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.backtesting.tick_data.variable_spread_simulator import (
    VariableSpreadSimulator, SpreadConfig, SpreadModel
)
from src.backtesting.tick_data.enhanced_tick_feed import (
    EnhancedTickDataFeed, EnhancedTickDataFeedFactory
)

def create_demo_data():
    """Create sample data for demonstration"""
    print("Creating sample tick data...")
    
    # Create 1 hour of minute-by-minute data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=1),
        periods=60,
        freq='1T'
    )
    
    # Generate realistic price movement
    base_price = 1.1000
    price_changes = np.random.normal(0, 0.0001, len(timestamps)).cumsum()
    mid_prices = base_price + price_changes
    
    # Generate varying volumes (higher during market hours)
    volumes = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        if 8 <= hour < 16:  # European session
            base_vol = 80
        elif 16 <= hour < 24:  # American session
            base_vol = 60
        else:  # Asian session
            base_vol = 20
        volumes.append(np.random.randint(base_vol, base_vol + 40))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'mid': mid_prices,
        'volume': volumes
    })

def demonstrate_spread_models():
    """Demonstrate different spread simulation models"""
    print("\n" + "="*60)
    print("VARIABLE SPREAD SIMULATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data = create_demo_data()
    
    # Test different spread models
    models = [
        (SpreadModel.STATISTICAL, "Statistical Model"),
        (SpreadModel.MARKET_CONDITIONS, "Market Conditions Model"),
        (SpreadModel.HYBRID, "Hybrid Model")
    ]
    
    results = {}
    
    for model, name in models:
        print(f"\n--- {name} ---")
        
        # Create spread config
        config = SpreadConfig(
            symbol="EURUSD",
            model=model,
            base_spread=0.0001,  # 1 pip
            volatility_multiplier=1.5,
            time_of_day_factor=0.3,
            volume_factor=0.2
        )
        
        # Create simulator
        simulator = VariableSpreadSimulator(config)
        
        # Simulate spreads
        result_data = simulator.simulate_spreads_for_dataframe(data)
        
        # Calculate statistics
        spreads = result_data['simulated_spread']
        stats = {
            'mean': spreads.mean(),
            'std': spreads.std(),
            'min': spreads.min(),
            'max': spreads.max(),
            'quality_score': simulator.validate_spread_quality(result_data)['quality_score']
        }
        
        print(f"Mean Spread: {stats['mean']:.6f} ({stats['mean']*10000:.1f} pips)")
        print(f"Std Dev: {stats['std']:.6f} ({stats['std']*10000:.1f} pips)")
        print(f"Min Spread: {stats['min']:.6f} ({stats['min']*10000:.1f} pips)")
        print(f"Max Spread: {stats['max']:.6f} ({stats['max']*10000:.1f} pips)")
        print(f"Quality Score: {stats['quality_score']:.1f}%")
        
        results[name] = {
            'data': result_data,
            'stats': stats
        }
    
    return results

def demonstrate_market_conditions():
    """Demonstrate spread simulation under different market conditions"""
    print("\n" + "="*60)
    print("MARKET CONDITIONS IMPACT ON SPREADS")
    print("="*60)
    
    # Create sample data
    data = create_demo_data()
    
    # Test different market conditions
    conditions = [
        (1.0, 1.0, "Normal Market"),
        (2.0, 0.5, "High Volatility, Low Liquidity"),
        (0.5, 2.0, "Low Volatility, High Liquidity"),
        (3.0, 0.3, "Crisis Conditions")
    ]
    
    for volatility, liquidity, name in conditions:
        print(f"\n--- {name} (Vol: {volatility}, Liq: {liquidity}) ---")
        
        # Create spread config
        config = SpreadConfig(
            symbol="EURUSD",
            model=SpreadModel.MARKET_CONDITIONS,
            market_volatility=volatility,
            liquidity_level=liquidity,
            base_spread=0.0001
        )
        
        # Create simulator
        simulator = VariableSpreadSimulator(config)
        
        # Simulate spreads
        result_data = simulator.simulate_spreads_for_dataframe(data)
        
        # Calculate statistics
        spreads = result_data['simulated_spread']
        mean_spread = spreads.mean()
        std_spread = spreads.std()
        
        print(f"Mean Spread: {mean_spread:.6f} ({mean_spread*10000:.1f} pips)")
        print(f"Std Dev: {std_spread:.6f} ({std_spread*10000:.1f} pips)")
        print(f"Spread Range: {spreads.min()*10000:.1f} - {spreads.max()*10000:.1f} pips")

def demonstrate_enhanced_feed():
    """Demonstrate the enhanced tick data feed"""
    print("\n" + "="*60)
    print("ENHANCED TICK DATA FEED DEMONSTRATION")
    print("="*60)
    
    # Create sample tick data
    data = create_demo_data()
    
    # Add bid/ask columns (required for enhanced feed)
    data['bid'] = data['mid'] - 0.0001
    data['ask'] = data['mid'] + 0.0001
    
    # Create enhanced feed with spread simulation
    feed = EnhancedTickDataFeed(
        data, 
        "EURUSD", 
        enable_spread_simulation=True
    )
    
    print(f"Feed initialized with {feed.total_ticks} ticks")
    print(f"Spread simulation enabled: {feed.enable_spread_simulation}")
    
    # Get data info
    info = feed.get_data_info()
    print(f"\nData Information:")
    print(f"Symbol: {info['symbol']}")
    print(f"Start Time: {info['start_time']}")
    print(f"End Time: {info['end_time']}")
    print(f"Average Spread: {info['avg_spread']:.6f} ({info['avg_spread']*10000:.1f} pips)")
    print(f"Min Spread: {info['min_spread']:.6f} ({info['min_spread']*10000:.1f} pips)")
    print(f"Max Spread: {info['max_spread']:.6f} ({info['max_spread']*10000:.1f} pips)")
    
    # Get spread analysis
    analysis = feed.get_spread_analysis()
    print(f"\nSpread Analysis:")
    print(f"Overall Mean: {analysis['overall_stats']['mean']:.6f}")
    print(f"Overall Std: {analysis['overall_stats']['std']:.6f}")
    
    # Show market conditions distribution
    if 'market_conditions' in analysis:
        print(f"\nMarket Conditions Distribution:")
        for condition, count in analysis['market_conditions'].items():
            print(f"  {condition}: {count} ticks")
    
    # Show trading sessions distribution
    if 'trading_sessions' in analysis:
        print(f"\nTrading Sessions Distribution:")
        for session, count in analysis['trading_sessions'].items():
            print(f"  {session}: {count} ticks")

def create_visualization(results):
    """Create a visualization of spread patterns"""
    print("\n" + "="*60)
    print("CREATING SPREAD VISUALIZATION")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Variable Spread Simulation Comparison', fontsize=16)
        
        # Plot 1: Spread over time
        ax1 = axes[0, 0]
        for name, result in results.items():
            data = result['data']
            ax1.plot(data['timestamp'], data['simulated_spread'] * 10000, 
                    label=name, alpha=0.7, linewidth=1)
        ax1.set_title('Spread Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Spread (pips)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spread distribution
        ax2 = axes[0, 1]
        for name, result in results.items():
            spreads = result['data']['simulated_spread'] * 10000
            ax2.hist(spreads, bins=20, alpha=0.6, label=name, density=True)
        ax2.set_title('Spread Distribution')
        ax2.set_xlabel('Spread (pips)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Spread vs Volume
        ax3 = axes[1, 0]
        for name, result in results.items():
            data = result['data']
            ax3.scatter(data['volume'], data['simulated_spread'] * 10000, 
                       label=name, alpha=0.6, s=20)
        ax3.set_title('Spread vs Volume')
        ax3.set_xlabel('Volume')
        ax3.set_ylabel('Spread (pips)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Quality scores
        ax4 = axes[1, 1]
        names = list(results.keys())
        quality_scores = [result['stats']['quality_score'] for result in results.values()]
        bars = ax4.bar(names, quality_scores, color=['blue', 'green', 'orange'])
        ax4.set_title('Quality Scores')
        ax4.set_ylabel('Quality Score (%)')
        ax4.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, quality_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'variable_spread_simulation_demo.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: {output_file}")
        
        # Show the plot
        plt.show()
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    """Main demonstration function"""
    print("Variable Spread Simulation Demo")
    print("This demo shows the variable spread simulation functionality")
    print("with different models and market conditions.")
    
    # Demonstrate different spread models
    results = demonstrate_spread_models()
    
    # Demonstrate market conditions impact
    demonstrate_market_conditions()
    
    # Demonstrate enhanced feed
    demonstrate_enhanced_feed()
    
    # Create visualization
    create_visualization(results)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("✅ Multiple spread simulation models")
    print("✅ Market condition adjustments")
    print("✅ Time-of-day effects")
    print("✅ Volume impact on spreads")
    print("✅ Quality validation and scoring")
    print("✅ Enhanced tick data feed integration")
    print("✅ Comprehensive analysis and statistics")
    
    print("\nThe variable spread simulation module provides realistic")
    print("spread modeling for accurate backtesting with proper")
    print("market condition representation.")

if __name__ == "__main__":
    main()

