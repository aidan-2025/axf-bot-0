"""
Comprehensive test script for the Strategy Generation Engine
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import strategy generation components
from src.strategy_generation.core.strategy_generation_engine import (
    StrategyGenerationEngine, GenerationConfig, GenerationStatus, StrategyCategory
)
from src.strategy_generation.core.strategy_template import StrategyParameters, StrategyType
from src.strategy_generation.core.parameter_space import ParameterSpace, ParameterDefinition, ParameterType
from src.strategy_generation.optimization.advanced_genetic_optimizer import OptimizationConfig
from src.strategy_generation.optimization.advanced_monte_carlo import MonteCarloConfig, SimulationType
from src.strategy_generation.optimization.advanced_walk_forward import WalkForwardConfig, WalkForwardMode


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_market_data(periods: int = 1000) -> Dict[str, Any]:
    """Create mock market data for testing"""
    np.random.seed(42)  # For reproducibility
    
    # Generate price data using random walk
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, periods)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    ohlcv = {
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': [np.random.randint(1000, 5000) for _ in range(periods)],
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(periods, 0, -1)]
    }
    
    # Add technical indicators
    closes = np.array(ohlcv['close'])
    
    # Simple Moving Averages
    sma_20 = np.convolve(closes, np.ones(20)/20, mode='valid')
    sma_50 = np.convolve(closes, np.ones(50)/50, mode='valid')
    
    # RSI
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.convolve(gains, np.ones(14)/14, mode='valid')
    avg_losses = np.convolve(losses, np.ones(14)/14, mode='valid')
    
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    indicators = {
        'sma_20': sma_20.tolist(),
        'sma_50': sma_50.tolist(),
        'rsi': rsi.tolist()
    }
    
    return {
        'ohlcv': ohlcv,
        'indicators': indicators,
        'sentiment': {'overall_sentiment': 0.5, 'confidence': 0.7},
        'economic_events': [],
        'news': []
    }


def create_generation_config() -> GenerationConfig:
    """Create generation configuration for testing"""
    # Create optimization config
    opt_config = OptimizationConfig(
        population_size=20,
        generations=10,
        use_multi_objective=True
    )
    
    # Create Monte Carlo config
    mc_config = MonteCarloConfig(
        iterations=50,
        simulation_type=SimulationType.BOOTSTRAP
    )
    
    # Create walk-forward config
    wf_config = WalkForwardConfig(
        mode=WalkForwardMode.ROLLING_WINDOW,
        in_sample_ratio=0.7,
        out_sample_ratio=0.3,
        min_in_sample_periods=100,
        min_out_sample_periods=50
    )
    
    return GenerationConfig(
        max_strategies=10,  # Reduced for testing
        generation_timeout=300,  # 5 minutes
        parallel_generation=False,  # Use sequential for testing
        optimization_config=opt_config,
        monte_carlo_config=mc_config,
        walk_forward_config=wf_config,
        min_performance_score=0.3,  # Lower threshold for testing
        max_drawdown_threshold=0.3,
        min_sharpe_ratio=0.1,
        min_win_rate=0.3,
        min_trades=5
    )


async def test_strategy_generation_engine():
    """Test the complete strategy generation engine"""
    logger.info("Testing Strategy Generation Engine...")
    
    try:
        # Create market data
        market_data = create_mock_market_data(500)
        
        # Create generation config
        config = create_generation_config()
        
        # Create strategy generation engine
        engine = StrategyGenerationEngine(config)
        
        # Generate strategies
        logger.info("Starting strategy generation...")
        result = await engine.generate_strategies_async(market_data)
        
        # Validate results
        assert result.total_strategies_generated > 0, "No strategies generated"
        assert result.successful_strategies >= 0, "Invalid successful strategies count"
        assert result.failed_strategies >= 0, "Invalid failed strategies count"
        assert result.validated_strategies >= 0, "Invalid validated strategies count"
        assert result.generation_time > 0, "Invalid generation time"
        
        # Check strategy details
        for strategy in result.strategies:
            assert strategy.strategy_id, "Missing strategy ID"
            assert strategy.strategy_name, "Missing strategy name"
            assert strategy.strategy_type, "Missing strategy type"
            assert strategy.category, "Missing strategy category"
            assert strategy.template_class, "Missing template class"
            assert strategy.parameters, "Missing parameters"
            assert strategy.status, "Missing status"
            assert strategy.created_at, "Missing created_at"
            assert strategy.updated_at, "Missing updated_at"
        
        logger.info(f"✓ Strategy generation completed successfully")
        logger.info(f"  Total Strategies: {result.total_strategies_generated}")
        logger.info(f"  Successful: {result.successful_strategies}")
        logger.info(f"  Failed: {result.failed_strategies}")
        logger.info(f"  Validated: {result.validated_strategies}")
        logger.info(f"  Generation Time: {result.generation_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Strategy generation engine test failed: {e}")
        return False


async def test_strategy_diversity():
    """Test strategy diversity in generation"""
    logger.info("Testing Strategy Diversity...")
    
    try:
        # Create market data
        market_data = create_mock_market_data(300)
        
        # Create generation config
        config = create_generation_config()
        config.max_strategies = 20  # Generate more strategies for diversity test
        
        # Create strategy generation engine
        engine = StrategyGenerationEngine(config)
        
        # Generate strategies
        result = await engine.generate_strategies_async(market_data)
        
        # Check strategy type diversity
        strategy_types = [s.strategy_type for s in result.strategies]
        unique_types = set(strategy_types)
        
        assert len(unique_types) > 1, "Insufficient strategy type diversity"
        
        # Check category diversity
        categories = [s.category for s in result.strategies]
        unique_categories = set(categories)
        
        assert len(unique_categories) > 1, "Insufficient category diversity"
        
        # Check parameter diversity
        parameter_diversity = 0
        for strategy in result.strategies:
            if strategy.parameters:
                parameter_diversity += len(strategy.parameters)
        
        assert parameter_diversity > 0, "No parameter diversity"
        
        logger.info(f"✓ Strategy diversity test passed")
        logger.info(f"  Strategy Types: {len(unique_types)}")
        logger.info(f"  Categories: {len(unique_categories)}")
        logger.info(f"  Total Parameters: {parameter_diversity}")
        
        return True
        
    except Exception as e:
        logger.error(f"Strategy diversity test failed: {e}")
        return False


async def test_validation_pipeline():
    """Test the validation pipeline"""
    logger.info("Testing Validation Pipeline...")
    
    try:
        # Create market data
        market_data = create_mock_market_data(400)
        
        # Create generation config with strict validation
        config = create_generation_config()
        config.min_performance_score = 0.5
        config.max_drawdown_threshold = 0.2
        config.min_sharpe_ratio = 0.3
        
        # Create strategy generation engine
        engine = StrategyGenerationEngine(config)
        
        # Generate strategies
        result = await engine.generate_strategies_async(market_data)
        
        # Check validation results
        validated_count = 0
        for strategy in result.strategies:
            if strategy.status == GenerationStatus.VALIDATED:
                validated_count += 1
                
                # Check validation results exist
                assert strategy.validation_results, "Missing validation results"
                assert strategy.performance_metrics, "Missing performance metrics"
                
                # Check specific validation components
                validation_results = strategy.validation_results
                assert 'monte_carlo' in validation_results, "Missing Monte Carlo results"
                assert 'walk_forward' in validation_results, "Missing walk-forward results"
                assert 'strategy_validation' in validation_results, "Missing strategy validation"
        
        logger.info(f"✓ Validation pipeline test passed")
        logger.info(f"  Validated Strategies: {validated_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation pipeline test failed: {e}")
        return False


async def test_performance_metrics():
    """Test performance metrics extraction"""
    logger.info("Testing Performance Metrics...")
    
    try:
        # Create market data
        market_data = create_mock_market_data(300)
        
        # Create generation config
        config = create_generation_config()
        
        # Create strategy generation engine
        engine = StrategyGenerationEngine(config)
        
        # Generate strategies
        result = await engine.generate_strategies_async(market_data)
        
        # Check performance metrics
        for strategy in result.strategies:
            if strategy.performance_metrics:
                metrics = strategy.performance_metrics
                
                # Check that metrics are numeric where expected
                for key, value in metrics.items():
                    if 'score' in key or 'return' in key:
                        assert isinstance(value, (int, float)), f"Invalid metric type for {key}"
                        assert not np.isnan(value), f"NaN value in metric {key}"
        
        logger.info(f"✓ Performance metrics test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance metrics test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling and resilience"""
    logger.info("Testing Error Handling...")
    
    try:
        # Create invalid market data
        invalid_market_data = {
            'ohlcv': {},  # Empty OHLCV data
            'indicators': {},
            'sentiment': {},
            'economic_events': [],
            'news': []
        }
        
        # Create generation config
        config = create_generation_config()
        config.max_strategies = 5  # Small number for error testing
        
        # Create strategy generation engine
        engine = StrategyGenerationEngine(config)
        
        # Generate strategies with invalid data
        result = await engine.generate_strategies_async(invalid_market_data)
        
        # Should handle errors gracefully
        assert result.total_strategies_generated >= 0, "Invalid total strategies count"
        assert result.failed_strategies >= 0, "Invalid failed strategies count"
        
        # Check that engine doesn't crash
        assert engine.get_generation_summary(), "Engine summary not available"
        
        logger.info(f"✓ Error handling test passed")
        logger.info(f"  Total Strategies: {result.total_strategies_generated}")
        logger.info(f"  Failed Strategies: {result.failed_strategies}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False


async def test_result_persistence():
    """Test result saving and loading"""
    logger.info("Testing Result Persistence...")
    
    try:
        # Create market data
        market_data = create_mock_market_data(200)
        
        # Create generation config
        config = create_generation_config()
        config.max_strategies = 3  # Small number for persistence testing
        
        # Create strategy generation engine
        engine = StrategyGenerationEngine(config)
        
        # Generate strategies
        result = await engine.generate_strategies_async(market_data)
        
        # Save results
        test_file = "test_strategy_results.json"
        engine.save_results(test_file)
        
        # Check file exists
        import os
        assert os.path.exists(test_file), "Results file not created"
        
        # Load results
        load_success = engine.load_results(test_file)
        assert load_success, "Failed to load results"
        
        # Clean up
        os.remove(test_file)
        
        logger.info(f"✓ Result persistence test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"Result persistence test failed: {e}")
        return False


async def test_engine_summary():
    """Test engine summary and statistics"""
    logger.info("Testing Engine Summary...")
    
    try:
        # Create market data
        market_data = create_mock_market_data(300)
        
        # Create generation config
        config = create_generation_config()
        
        # Create strategy generation engine
        engine = StrategyGenerationEngine(config)
        
        # Generate strategies
        result = await engine.generate_strategies_async(market_data)
        
        # Get summary
        summary = engine.get_generation_summary()
        
        # Check summary structure
        assert 'total_generations' in summary, "Missing total_generations"
        assert 'total_strategies' in summary, "Missing total_strategies"
        assert 'successful_strategies' in summary, "Missing successful_strategies"
        assert 'validated_strategies' in summary, "Missing validated_strategies"
        assert 'failed_strategies' in summary, "Missing failed_strategies"
        
        # Check summary values
        assert summary['total_generations'] > 0, "Invalid total_generations"
        assert summary['total_strategies'] >= 0, "Invalid total_strategies"
        assert summary['successful_strategies'] >= 0, "Invalid successful_strategies"
        assert summary['validated_strategies'] >= 0, "Invalid validated_strategies"
        assert summary['failed_strategies'] >= 0, "Invalid failed_strategies"
        
        logger.info(f"✓ Engine summary test passed")
        logger.info(f"  Summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Engine summary test failed: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("Starting Strategy Generation Engine Testing...")
    
    tests = [
        ("Strategy Generation Engine", test_strategy_generation_engine),
        ("Strategy Diversity", test_strategy_diversity),
        ("Validation Pipeline", test_validation_pipeline),
        ("Performance Metrics", test_performance_metrics),
        ("Error Handling", test_error_handling),
        ("Result Persistence", test_result_persistence),
        ("Engine Summary", test_engine_summary)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed successfully!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    exit(0 if success else 1)

