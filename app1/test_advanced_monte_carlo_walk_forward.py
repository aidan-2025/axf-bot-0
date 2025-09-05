"""
Comprehensive test script for advanced Monte Carlo simulation and walk-forward testing
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import strategy generation components
from src.strategy_generation.core.strategy_template import StrategyTemplate, StrategyParameters, Signal
from src.strategy_generation.core.parameter_space import ParameterSpace, ParameterType
from src.strategy_generation.optimization.advanced_monte_carlo import (
    AdvancedMonteCarloSimulator, MonteCarloConfig, SimulationType
)
from src.strategy_generation.optimization.advanced_walk_forward import (
    AdvancedWalkForwardTester, WalkForwardConfig, WalkForwardMode
)
from src.strategy_generation.optimization.advanced_genetic_optimizer import OptimizationConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockStrategy(StrategyTemplate):
    """Mock strategy for testing"""
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        self.initialized = False
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self.parameters.name
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization"""
        return ParameterSpace({
            'sma_period': ParameterType.INTEGER,
            'rsi_threshold': ParameterType.FLOAT,
            'volume_threshold': ParameterType.FLOAT,
            'trend_direction': ParameterType.CATEGORICAL
        })
    
    def initialize(self):
        """Initialize strategy"""
        self.initialized = True
        logger.info(f"Initialized strategy: {self.parameters.name}")
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters"""
        self.parameters.parameters.update(new_parameters)
        logger.info(f"Updated parameters: {new_parameters}")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameter values"""
        try:
            # Basic validation
            if 'sma_period' in parameters:
                if not isinstance(parameters['sma_period'], int) or parameters['sma_period'] < 1:
                    return False
            
            if 'rsi_threshold' in parameters:
                if not isinstance(parameters['rsi_threshold'], (int, float)) or not 0 <= parameters['rsi_threshold'] <= 100:
                    return False
            
            if 'volume_threshold' in parameters:
                if not isinstance(parameters['volume_threshold'], (int, float)) or parameters['volume_threshold'] < 0:
                    return False
            
            if 'trend_direction' in parameters:
                if parameters['trend_direction'] not in ['bullish', 'bearish', 'neutral']:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on market data"""
        try:
            if not self.initialized:
                self.initialize()
            
            signals = []
            ohlcv = market_data.get('ohlcv', {})
            
            if not ohlcv or 'close' not in ohlcv:
                return signals
            
            closes = ohlcv['close']
            if len(closes) < 20:  # Need minimum data
                return signals
            
            # Get parameters
            sma_period = self.parameters.parameters.get('sma_period', 20)
            rsi_threshold = self.parameters.parameters.get('rsi_threshold', 70)
            volume_threshold = self.parameters.parameters.get('volume_threshold', 1000)
            trend_direction = self.parameters.parameters.get('trend_direction', 'neutral')
            
            # Calculate simple moving average
            if len(closes) >= sma_period:
                sma = np.mean(closes[-sma_period:])
                current_price = closes[-1]
                
                # Calculate RSI (simplified)
                deltas = np.diff(closes[-14:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                rs = avg_gain / (avg_loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                
                # Get volume
                volume = ohlcv.get('volume', [0])[-1] if ohlcv.get('volume') else 0
                
                # Generate signals based on strategy logic
                signal_strength = 0.5
                signal_confidence = 0.7
                
                # Trend following signal
                if trend_direction == 'bullish' and current_price > sma and rsi < rsi_threshold and volume > volume_threshold:
                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol='EURUSD',
                        signal_type='buy',
                        strength=signal_strength,
                        confidence=signal_confidence,
                        price=current_price,
                        metadata={'strategy': 'trend_following', 'rsi': rsi, 'sma': sma}
                    )
                    signals.append(signal)
                
                elif trend_direction == 'bearish' and current_price < sma and rsi > (100 - rsi_threshold) and volume > volume_threshold:
                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol='EURUSD',
                        signal_type='sell',
                        strength=signal_strength,
                        confidence=signal_confidence,
                        price=current_price,
                        metadata={'strategy': 'trend_following', 'rsi': rsi, 'sma': sma}
                    )
                    signals.append(signal)
                
                # Mean reversion signal
                elif trend_direction == 'neutral':
                    if rsi < 30:  # Oversold
                        signal = Signal(
                            timestamp=datetime.now(),
                            symbol='EURUSD',
                            signal_type='buy',
                            strength=signal_strength * 0.8,
                            confidence=signal_confidence * 0.9,
                            price=current_price,
                            metadata={'strategy': 'mean_reversion', 'rsi': rsi}
                        )
                        signals.append(signal)
                    elif rsi > 70:  # Overbought
                        signal = Signal(
                            timestamp=datetime.now(),
                            symbol='EURUSD',
                            signal_type='sell',
                            strength=signal_strength * 0.8,
                            confidence=signal_confidence * 0.9,
                            price=current_price,
                            metadata={'strategy': 'mean_reversion', 'rsi': rsi}
                        )
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []


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
        'economic_events': []
    }


def create_parameter_space() -> ParameterSpace:
    """Create parameter space for testing"""
    from src.strategy_generation.core.parameter_space import ParameterDefinition
    
    space = ParameterSpace()
    
    # Add parameter definitions
    space.add_parameter(ParameterDefinition(
        name='sma_period',
        param_type=ParameterType.INTEGER,
        min_value=5,
        max_value=50,
        default_value=20,
        description='Simple Moving Average period'
    ))
    
    space.add_parameter(ParameterDefinition(
        name='rsi_threshold',
        param_type=ParameterType.FLOAT,
        min_value=50.0,
        max_value=90.0,
        default_value=70.0,
        description='RSI threshold for overbought/oversold'
    ))
    
    space.add_parameter(ParameterDefinition(
        name='volume_threshold',
        param_type=ParameterType.FLOAT,
        min_value=500.0,
        max_value=2000.0,
        default_value=1000.0,
        description='Minimum volume threshold'
    ))
    
    space.add_parameter(ParameterDefinition(
        name='trend_direction',
        param_type=ParameterType.CATEGORICAL,
        categories=['bullish', 'bearish', 'neutral'],
        default_value='bullish',
        description='Trend direction bias'
    ))
    
    return space


async def test_advanced_monte_carlo():
    """Test advanced Monte Carlo simulation"""
    logger.info("Testing Advanced Monte Carlo Simulation...")
    
    try:
        # Create mock strategy
        parameters = StrategyParameters(
            strategy_id="test_strategy_001",
            strategy_type="trend_following",
            name="Test Trend Strategy",
            description="Test strategy for Monte Carlo simulation",
            parameters={
                'sma_period': 20,
                'rsi_threshold': 70,
                'volume_threshold': 1000,
                'trend_direction': 'bullish'
            }
        )
        
        strategy = MockStrategy(parameters)
        
        # Create market data
        market_data = create_mock_market_data(500)
        
        # Test different simulation types
        simulation_types = [
            SimulationType.BOOTSTRAP,
            SimulationType.RANDOM_WALK,
            SimulationType.HISTORICAL_SIMULATION,
            SimulationType.STRESS_TEST
        ]
        
        for sim_type in simulation_types:
            logger.info(f"Testing {sim_type.value} simulation...")
            
            # Create Monte Carlo config
            config = MonteCarloConfig(
                iterations=100,  # Reduced for testing
                simulation_type=sim_type,
                confidence_level=0.95,
                noise_level=0.01
            )
            
            # Create simulator
            simulator = AdvancedMonteCarloSimulator(config)
            
            # Run simulation
            result = await simulator.simulate_async(strategy, market_data)
            
            # Validate results
            assert result['success'], f"Simulation failed for {sim_type.value}"
            assert result['iterations'] == config.iterations, "Incorrect iteration count"
            assert len(result['results']) == config.iterations, "Incorrect result count"
            
            # Check analysis
            analysis = result['analysis']
            assert 'mean_return' in analysis, "Missing mean_return in analysis"
            assert 'robustness_score' in analysis, "Missing robustness_score in analysis"
            assert 'stability_score' in analysis, "Missing stability_score in analysis"
            
            logger.info(f"✓ {sim_type.value} simulation completed successfully")
            logger.info(f"  Mean Return: {analysis['mean_return']:.4f}")
            logger.info(f"  Robustness Score: {analysis['robustness_score']:.4f}")
            logger.info(f"  Stability Score: {analysis['stability_score']:.4f}")
        
        logger.info("✓ Advanced Monte Carlo simulation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Advanced Monte Carlo test failed: {e}")
        return False


async def test_advanced_walk_forward():
    """Test advanced walk-forward testing"""
    logger.info("Testing Advanced Walk-Forward Testing...")
    
    try:
        # Create mock strategy
        parameters = StrategyParameters(
            strategy_id="test_strategy_002",
            strategy_type="trend_following",
            name="Test Walk-Forward Strategy",
            description="Test strategy for walk-forward testing",
            parameters={
                'sma_period': 20,
                'rsi_threshold': 70,
                'volume_threshold': 1000,
                'trend_direction': 'bullish'
            }
        )
        
        strategy = MockStrategy(parameters)
        
        # Create market data with more periods for walk-forward
        market_data = create_mock_market_data(1000)
        
        # Create parameter space
        parameter_space = create_parameter_space()
        
        # Test different walk-forward modes
        walk_forward_modes = [
            WalkForwardMode.FIXED_WINDOW,
            WalkForwardMode.EXPANDING_WINDOW,
            WalkForwardMode.ROLLING_WINDOW
        ]
        
        for mode in walk_forward_modes:
            logger.info(f"Testing {mode.value} walk-forward...")
            
            # Create walk-forward config
            config = WalkForwardConfig(
                mode=mode,
                in_sample_ratio=0.7,
                out_sample_ratio=0.3,
                min_in_sample_periods=100,
                min_out_sample_periods=50,
                reoptimize_frequency=1,
                use_monte_carlo=True,
                monte_carlo_iterations=50  # Reduced for testing
            )
            
            # Create walk-forward tester
            tester = AdvancedWalkForwardTester(config)
            
            # Run walk-forward test
            result = await tester.test_async(strategy, market_data, parameter_space)
            
            # Validate results
            assert result.total_periods > 0, "No walk-forward periods created"
            assert result.successful_periods >= 0, "Invalid successful periods count"
            assert result.failed_periods >= 0, "Invalid failed periods count"
            assert result.robustness_score >= 0, "Invalid robustness score"
            assert result.stability_score >= 0, "Invalid stability score"
            assert result.consistency_score >= 0, "Invalid consistency score"
            assert result.overall_grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D'], "Invalid overall grade"
            
            logger.info(f"✓ {mode.value} walk-forward completed successfully")
            logger.info(f"  Total Periods: {result.total_periods}")
            logger.info(f"  Successful Periods: {result.successful_periods}")
            logger.info(f"  Robustness Score: {result.robustness_score:.4f}")
            logger.info(f"  Stability Score: {result.stability_score:.4f}")
            logger.info(f"  Overall Grade: {result.overall_grade}")
        
        logger.info("✓ Advanced walk-forward testing passed")
        return True
        
    except Exception as e:
        logger.error(f"Advanced walk-forward test failed: {e}")
        return False


async def test_integration():
    """Test integration between Monte Carlo and walk-forward testing"""
    logger.info("Testing Integration...")
    
    try:
        # Create mock strategy
        parameters = StrategyParameters(
            strategy_id="test_strategy_003",
            strategy_type="trend_following",
            name="Test Integration Strategy",
            description="Test strategy for integration testing",
            parameters={
                'sma_period': 20,
                'rsi_threshold': 70,
                'volume_threshold': 1000,
                'trend_direction': 'bullish'
            }
        )
        
        strategy = MockStrategy(parameters)
        
        # Create market data
        market_data = create_mock_market_data(800)
        
        # Create parameter space
        parameter_space = create_parameter_space()
        
        # Create walk-forward config with Monte Carlo
        wf_config = WalkForwardConfig(
            mode=WalkForwardMode.ROLLING_WINDOW,
            in_sample_ratio=0.6,
            out_sample_ratio=0.4,
            min_in_sample_periods=80,
            min_out_sample_periods=40,
            use_monte_carlo=True,
            monte_carlo_iterations=50
        )
        
        # Run walk-forward test
        tester = AdvancedWalkForwardTester(wf_config)
        wf_result = await tester.test_async(strategy, market_data, parameter_space)
        
        # Validate integration
        assert wf_result.total_periods > 0, "No walk-forward periods created"
        
        # Check that Monte Carlo results are included
        monte_carlo_periods = 0
        for period in wf_result.periods:
            if period.monte_carlo_results and period.monte_carlo_results.get('success'):
                monte_carlo_periods += 1
        
        logger.info(f"✓ Integration test completed successfully")
        logger.info(f"  Walk-Forward Periods: {wf_result.total_periods}")
        logger.info(f"  Monte Carlo Periods: {monte_carlo_periods}")
        logger.info(f"  Overall Grade: {wf_result.overall_grade}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


async def test_performance():
    """Test performance and scalability"""
    logger.info("Testing Performance...")
    
    try:
        # Create mock strategy
        parameters = StrategyParameters(
            strategy_id="test_strategy_004",
            strategy_type="trend_following",
            name="Test Performance Strategy",
            description="Test strategy for performance testing",
            parameters={
                'sma_period': 20,
                'rsi_threshold': 70,
                'volume_threshold': 1000,
                'trend_direction': 'bullish'
            }
        )
        
        strategy = MockStrategy(parameters)
        
        # Create larger market data
        market_data = create_mock_market_data(2000)
        
        # Test Monte Carlo performance
        start_time = datetime.now()
        
        mc_config = MonteCarloConfig(
            iterations=200,  # Moderate size
            simulation_type=SimulationType.BOOTSTRAP,
            use_parallel=False  # Test sequential performance
        )
        
        simulator = AdvancedMonteCarloSimulator(mc_config)
        mc_result = await simulator.simulate_async(strategy, market_data)
        
        mc_time = (datetime.now() - start_time).total_seconds()
        
        # Test walk-forward performance
        start_time = datetime.now()
        
        wf_config = WalkForwardConfig(
            mode=WalkForwardMode.ROLLING_WINDOW,
            in_sample_ratio=0.7,
            out_sample_ratio=0.3,
            min_in_sample_periods=200,
            min_out_sample_periods=100,
            use_monte_carlo=False  # Test without Monte Carlo for speed
        )
        
        parameter_space = create_parameter_space()
        tester = AdvancedWalkForwardTester(wf_config)
        wf_result = await tester.test_async(strategy, market_data, parameter_space)
        
        wf_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ Performance test completed")
        logger.info(f"  Monte Carlo Time: {mc_time:.2f}s")
        logger.info(f"  Walk-Forward Time: {wf_time:.2f}s")
        logger.info(f"  Monte Carlo Iterations: {mc_result['iterations']}")
        logger.info(f"  Walk-Forward Periods: {wf_result.total_periods}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("Starting Advanced Monte Carlo and Walk-Forward Testing...")
    
    tests = [
        ("Advanced Monte Carlo Simulation", test_advanced_monte_carlo),
        ("Advanced Walk-Forward Testing", test_advanced_walk_forward),
        ("Integration Testing", test_integration),
        ("Performance Testing", test_performance)
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
