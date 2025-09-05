#!/usr/bin/env python3
"""
Test script for technical analysis engine
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from technical_analysis import (
    Timeframe, TechnicalIndicatorCalculator, TimeframeAggregator,
    AnalysisProcessor, AnalysisStorage, INDICATOR_PRESETS, TIMEFRAME_CONFIGS
)


def generate_sample_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    print(f"Generating {days} days of sample data for {symbol}...")
    
    # Generate timestamps (1-minute intervals)
    start_time = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, periods=days*24*60, freq='1T')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible results
    base_price = 1.1000 if 'USD' in symbol else 100.0
    
    # Generate price movements
    returns = np.random.normal(0, 0.0001, len(timestamps))  # Small random returns
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.0001, 0.0005)
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = close * (1 + np.random.uniform(-volatility/2, volatility/2))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


async def test_technical_analysis():
    """Test the complete technical analysis pipeline"""
    print("🧪 Testing Technical Analysis Engine")
    print("=" * 50)
    
    # Generate sample data
    sample_data = generate_sample_data("EURUSD", days=7)
    print(f"✅ Generated {len(sample_data)} data points")
    
    # Test 1: Timeframe Aggregation
    print("\n📊 Testing Timeframe Aggregation...")
    aggregator = TimeframeAggregator()
    
    timeframes_to_test = [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]
    aggregated_data = {}
    
    for timeframe in timeframes_to_test:
        aggregated = aggregator.aggregate_to_timeframe(
            sample_data, "EURUSD", timeframe
        )
        
        if aggregated:
            aggregated_data[timeframe] = aggregated
            print(f"✅ {timeframe.value}: {len(aggregated.data)} data points")
        else:
            print(f"❌ {timeframe.value}: Failed to aggregate")
    
    # Test 2: Technical Indicators
    print("\n📈 Testing Technical Indicators...")
    calculator = TechnicalIndicatorCalculator()
    
    # Test indicators on H1 data
    if Timeframe.H1 in aggregated_data:
        h1_data = aggregated_data[Timeframe.H1].data
        
        # Test individual indicators
        test_indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands']
        indicator_results = {}
        
        for indicator_name in test_indicators:
            indicator = calculator.calculate_indicator(
                indicator_name, h1_data, "EURUSD", Timeframe.H1,
                INDICATOR_PRESETS.get(indicator_name, {})
            )
            
            if indicator:
                indicator_results[indicator_name] = indicator
                print(f"✅ {indicator_name}: {len(indicator.values)} values")
            else:
                print(f"❌ {indicator_name}: Failed to calculate")
        
        # Test multiple indicators at once
        print("\n🔄 Testing Multiple Indicators...")
        multiple_results = calculator.calculate_multiple_indicators(
            h1_data, "EURUSD", Timeframe.H1,
            [{'name': name, 'parameters': INDICATOR_PRESETS.get(name, {})} 
             for name in test_indicators]
        )
        
        print(f"✅ Multiple indicators: {len(multiple_results)} calculated")
    
    # Test 3: Analysis Processor
    print("\n⚙️ Testing Analysis Processor...")
    processor = AnalysisProcessor(max_workers=2)
    
    # Test single symbol analysis
    analysis_result = await processor.analyze_symbol(
        "EURUSD", aggregated_data, timeframes_to_test
    )
    
    if analysis_result.success:
        print(f"✅ Analysis successful: {len(analysis_result.analyses)} timeframes")
        print(f"   Processing time: {analysis_result.processing_time_ms:.2f}ms")
        
        # Show summary
        summary = processor.get_analysis_summary(analysis_result)
        print(f"   Total indicators: {summary['total_indicators']}")
        
        for tf, details in summary['timeframe_details'].items():
            print(f"   {tf}: {details['indicators_count']} indicators, {details['data_points']} data points")
    else:
        print(f"❌ Analysis failed: {analysis_result.error_message}")
    
    # Test 4: Analysis Storage (if InfluxDB is available)
    print("\n💾 Testing Analysis Storage...")
    
    # Mock InfluxDB config for testing
    influxdb_config = {
        'url': 'http://localhost:8086',
        'token': 'test-token',
        'org': 'test-org',
        'bucket': 'technical_analysis'
    }
    
    storage = AnalysisStorage(influxdb_config)
    
    # Test connection (will fail in test environment, but we can test the logic)
    connected = storage.connect()
    if connected:
        print("✅ Connected to InfluxDB")
        
        # Test storing analysis result
        if analysis_result.success:
            stored = storage.store_analysis_result(analysis_result)
            if stored:
                print("✅ Analysis result stored successfully")
            else:
                print("❌ Failed to store analysis result")
    else:
        print("⚠️ InfluxDB not available (expected in test environment)")
    
    # Test 5: Performance Test
    print("\n🚀 Performance Test...")
    
    # Generate more data for performance testing
    large_data = generate_sample_data("GBPUSD", days=14)
    large_aggregated = {}
    
    for timeframe in [Timeframe.M15, Timeframe.H1, Timeframe.H4]:
        aggregated = aggregator.aggregate_to_timeframe(
            large_data, "GBPUSD", timeframe
        )
        if aggregated:
            large_aggregated[timeframe] = aggregated
    
    if large_aggregated:
        start_time = datetime.now()
        
        performance_result = await processor.analyze_symbol(
            "GBPUSD", large_aggregated, list(large_aggregated.keys())
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        print(f"✅ Performance test completed in {processing_time:.2f}ms")
        print(f"   Data points processed: {sum(len(data.data) for data in large_aggregated.values())}")
        print(f"   Indicators calculated: {performance_result.analyses.get(Timeframe.H1, {}).indicators if Timeframe.H1 in performance_result.analyses else 0}")
    
    print("\n🎉 Technical Analysis Engine Test Complete!")
    return True


def test_indicator_calculations():
    """Test specific indicator calculations"""
    print("\n🔍 Testing Individual Indicator Calculations...")
    
    # Generate simple test data
    test_data = pd.DataFrame({
        'open': [1.1000, 1.1010, 1.1020, 1.1015, 1.1025],
        'high': [1.1015, 1.1025, 1.1030, 1.1025, 1.1030],
        'low': [1.0995, 1.1005, 1.1015, 1.1010, 1.1020],
        'close': [1.1010, 1.1020, 1.1015, 1.1025, 1.1028],
        'volume': [1000, 1200, 1100, 1300, 1400]
    })
    
    calculator = TechnicalIndicatorCalculator()
    
    # Test SMA
    sma = calculator.calculate_indicator('sma', test_data, 'TEST', Timeframe.M1, {'period': 3})
    if sma and len(sma.values) > 0:
        print(f"✅ SMA: {sma.values[-1]:.4f}")
    
    # Test RSI
    rsi = calculator.calculate_indicator('rsi', test_data, 'TEST', Timeframe.M1, {'period': 3})
    if rsi and len(rsi.values) > 0:
        print(f"✅ RSI: {rsi.values[-1]:.2f}")
    
    # Test MACD
    macd = calculator.calculate_indicator('macd', test_data, 'TEST', Timeframe.M1, {})
    if macd and len(macd.values) > 0:
        print(f"✅ MACD: {macd.values[-1]:.6f}")


if __name__ == "__main__":
    print("🚀 Starting Technical Analysis Engine Tests...")
    
    # Test individual indicators first
    test_indicator_calculations()
    
    # Run main async test
    try:
        asyncio.run(test_technical_analysis())
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

