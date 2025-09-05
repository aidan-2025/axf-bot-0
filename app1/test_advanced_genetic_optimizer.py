#!/usr/bin/env python3
"""
Test script for advanced genetic algorithm optimizer
"""

import unittest
import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List

# Suppress logging for cleaner test output
logging.getLogger('root').setLevel(logging.CRITICAL)

from src.strategy_generation.optimization.advanced_genetic_optimizer import (
    AdvancedGeneticOptimizer, OptimizationConfig, OptimizationObjective, FitnessMetrics
)
from src.strategy_generation.core.strategy_template import StrategyTemplate, StrategyType, StrategyParameters, Signal
from src.strategy_generation.core.parameter_space import ParameterSpace, ParameterDefinition, ParameterType
from src.strategy_generation.templates.trend_strategy import TrendStrategy


class MockStrategy(StrategyTemplate):
    """Mock strategy for testing"""
    
    def __init__(self, strategy_id: str, name: str, parameters: StrategyParameters):
        super().__init__(parameters)
        self._name = name
    
    @property
    def name(self) -> str:
        """Get strategy name"""
        return self._name
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate mock signals"""
        signals = []
        ohlcv = market_data.get('ohlcv', {})
        closes = ohlcv.get('close', [])
        
        if not closes:
            return signals
        
        # Generate mock signals based on price movement
        for i in range(1, len(closes)):
            price_change = (closes[i] - closes[i-1]) / closes[i-1]
            
            if abs(price_change) > 0.001:  # 0.1% threshold
                signal_type = 'buy' if price_change > 0 else 'sell'
                strength = min(1.0, abs(price_change) * 100)  # Scale to 0-1
                confidence = min(1.0, strength * 1.2)
                
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=market_data.get('symbol', 'EURUSD'),
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    price=closes[i]
                )
                signals.append(signal)
        
        return signals
    
    def evaluate_performance(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate mock performance"""
        return {
            'profit_factor': 1.5,
            'drawdown': 0.05,
            'win_rate': 0.6,
            'total_trades': 100
        }
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for this strategy"""
        space = ParameterSpace()
        space.add_parameter(ParameterDefinition(
            "rsi_period", ParameterType.INTEGER, 10, 30, default_value=14
        ))
        space.add_parameter(ParameterDefinition(
            "ma_period", ParameterType.INTEGER, 20, 100, default_value=50
        ))
        return space
    
    def initialize(self) -> bool:
        """Initialize the strategy"""
        return True
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """Update strategy parameters"""
        try:
            self.parameters.parameters.update(new_parameters)
            return True
        except Exception:
            return False
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate strategy parameters"""
        errors = []
        
        if 'rsi_period' in parameters:
            if not (10 <= parameters['rsi_period'] <= 30):
                errors.append("RSI period must be between 10 and 30")
        
        if 'ma_period' in parameters:
            if not (20 <= parameters['ma_period'] <= 100):
                errors.append("MA period must be between 20 and 100")
        
        return len(errors) == 0, errors


class TestAdvancedGeneticOptimizer(unittest.TestCase):
    """Test cases for advanced genetic optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create optimization config
        self.config = OptimizationConfig(
            population_size=20,
            generations=10,
            use_multi_objective=True,
            objectives=[
                OptimizationObjective.PROFIT,
                OptimizationObjective.SHARPE_RATIO,
                OptimizationObjective.MAX_DRAWDOWN
            ]
        )
        
        # Create optimizer
        self.optimizer = AdvancedGeneticOptimizer(self.config)
        
        # Create parameter space
        self.parameter_space = ParameterSpace()
        self.parameter_space.add_parameter(ParameterDefinition(
            "rsi_period", ParameterType.INTEGER, 10, 30, default_value=14
        ))
        self.parameter_space.add_parameter(ParameterDefinition(
            "ma_period", ParameterType.INTEGER, 20, 100, default_value=50
        ))
        self.parameter_space.add_parameter(ParameterDefinition(
            "risk_per_trade", ParameterType.FLOAT, 0.005, 0.05, default_value=0.01
        ))
        self.parameter_space.add_parameter(ParameterDefinition(
            "take_profit_multiplier", ParameterType.FLOAT, 1.0, 3.0, default_value=1.5
        ))
        self.parameter_space.add_parameter(ParameterDefinition(
            "stop_loss_multiplier", ParameterType.FLOAT, 0.5, 2.0, default_value=1.0
        ))
        
        # Create test strategy
        self.test_parameters = StrategyParameters(
            strategy_id="test_strategy",
            strategy_type=StrategyType.TREND,
            name="Test Strategy",
            description="Test strategy for optimization",
            parameters={
                "symbol": "EURUSD",
                "timeframe": "H1",
                "risk_per_trade": 0.01
            }
        )
        self.test_strategy = MockStrategy("test_strategy", "Test Strategy", self.test_parameters)
        
        # Create test market data
        self.test_market_data = {
            'ohlcv': {
                'open': [1.1000, 1.1010, 1.1020, 1.1015, 1.1025, 1.1030, 1.1025, 1.1035, 1.1040, 1.1035],
                'high': [1.1015, 1.1025, 1.1030, 1.1020, 1.1035, 1.1040, 1.1035, 1.1045, 1.1050, 1.1045],
                'low': [1.0995, 1.1005, 1.1015, 1.1010, 1.1020, 1.1025, 1.1020, 1.1030, 1.1035, 1.1030],
                'close': [1.1010, 1.1020, 1.1015, 1.1025, 1.1030, 1.1035, 1.1030, 1.1040, 1.1045, 1.1040],
                'volume': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
                'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10, 0, -1)]
            },
            'sentiment': {
                'news': [{'sentiment_score': 0.6, 'relevance': 0.8}],
                'social': [{'sentiment_score': 0.4, 'engagement': 0.7}]
            },
            'economic_events': [
                {
                    'title': 'NFP Release',
                    'event_time': datetime.now() + timedelta(hours=2),
                    'market_impact_score': 0.9
                }
            ]
        }
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.config.population_size, 20)
        self.assertEqual(self.optimizer.config.generations, 10)
        self.assertTrue(self.optimizer.config.use_multi_objective)
        self.assertEqual(len(self.optimizer.config.objectives), 3)
    
    def test_fitness_metrics_creation(self):
        """Test fitness metrics creation"""
        metrics = FitnessMetrics(
            net_profit=1000.0,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.6
        )
        
        self.assertEqual(metrics.net_profit, 1000.0)
        self.assertEqual(metrics.sharpe_ratio, 1.5)
        self.assertEqual(metrics.max_drawdown, 0.1)
        self.assertEqual(metrics.win_rate, 0.6)
        
        # Test to_dict conversion
        metrics_dict = metrics.to_dict()
        self.assertIn('net_profit', metrics_dict)
        self.assertIn('sharpe_ratio', metrics_dict)
        self.assertEqual(metrics_dict['net_profit'], 1000.0)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters
        valid_params = {
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'risk_per_trade': 0.01,
            'take_profit_multiplier': 1.5,
            'stop_loss_multiplier': 1.0
        }
        self.assertTrue(self.optimizer._validate_parameters(valid_params))
        
        # Invalid parameters
        invalid_params = {
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'risk_per_trade': 0.1,  # Too high
            'take_profit_multiplier': 5.0,  # Too high
            'stop_loss_multiplier': 1.0
        }
        self.assertFalse(self.optimizer._validate_parameters(invalid_params))
    
    def test_strategy_instance_creation(self):
        """Test strategy instance creation"""
        parameters = {
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'risk_per_trade': 0.01,
            'take_profit_multiplier': 1.5,
            'stop_loss_multiplier': 1.0
        }
        
        strategy_instance = self.optimizer._create_strategy_instance(
            self.test_strategy, parameters
        )
        
        self.assertIsNotNone(strategy_instance)
        self.assertEqual(strategy_instance.parameters.parameters['symbol'], 'EURUSD')
        self.assertEqual(strategy_instance.parameters.parameters['timeframe'], 'H1')
        self.assertEqual(strategy_instance.parameters.parameters['risk_per_trade'], 0.01)
    
    def test_fitness_calculation(self):
        """Test fitness calculation"""
        # Create mock signals
        signals = [
            {'signal_type': 'buy', 'strength': 0.8, 'confidence': 0.7, 'price': 1.1000},
            {'signal_type': 'sell', 'strength': 0.6, 'confidence': 0.8, 'price': 1.1010},
            {'signal_type': 'buy', 'strength': 0.9, 'confidence': 0.9, 'price': 1.1020}
        ]
        
        # Mock strategy with signals
        class MockStrategyWithSignals(MockStrategy):
            def generate_signals(self, market_data):
                return {"signals": signals}
        
        strategy = MockStrategyWithSignals("test", "Test", self.test_parameters)
        
        # Calculate fitness
        metrics = self.optimizer._calculate_comprehensive_fitness(strategy, self.test_market_data)
        
        self.assertIsInstance(metrics, FitnessMetrics)
        self.assertGreaterEqual(metrics.total_trades, 0)
    
    def test_combined_fitness_calculation(self):
        """Test combined fitness calculation"""
        metrics = FitnessMetrics(
            net_profit=500.0,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            win_rate=0.65,
            stability_score=0.8
        )
        
        combined_score = self.optimizer._calculate_combined_fitness(metrics)
        
        self.assertIsInstance(combined_score, float)
        self.assertGreaterEqual(combined_score, 0.0)
        self.assertLessEqual(combined_score, 1.0)
    
    def test_convergence_check(self):
        """Test convergence checking"""
        # Test with insufficient generations
        self.assertFalse(self.optimizer._check_convergence(5))
        
        # Test with no history
        self.optimizer.optimization_history = []
        self.assertFalse(self.optimizer._check_convergence(100))
        
        # Test with stagnant fitness
        self.optimizer.optimization_history = [
            {"max": 0.5} for _ in range(60)
        ]
        self.assertTrue(self.optimizer._check_convergence(100))
    
    async def test_async_optimization(self):
        """Test asynchronous optimization"""
        results = await self.optimizer.optimize_async(
            self.test_strategy,
            self.parameter_space,
            self.test_market_data
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('success', results)
        self.assertIn('best_individual', results)
        self.assertIn('optimization_history', results)
        
        if results['success']:
            self.assertIsNotNone(results['best_individual'])
            self.assertGreater(len(results['optimization_history']), 0)
    
    def test_sync_optimization(self):
        """Test synchronous optimization"""
        results = self.optimizer.optimize(
            self.test_strategy,
            self.parameter_space,
            self.test_market_data
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('success', results)
        self.assertIn('best_individual', results)
    
    def test_optimization_summary(self):
        """Test optimization summary"""
        summary = self.optimizer.get_optimization_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('evaluation_count', summary)
        self.assertIn('generations_completed', summary)
        self.assertIn('total_time', summary)
    
    def test_single_objective_config(self):
        """Test single-objective optimization configuration"""
        single_obj_config = OptimizationConfig(
            population_size=10,
            generations=5,
            use_multi_objective=False,
            objectives=[OptimizationObjective.PROFIT]
        )
        
        single_obj_optimizer = AdvancedGeneticOptimizer(single_obj_config)
        
        self.assertFalse(single_obj_optimizer.config.use_multi_objective)
        self.assertEqual(len(single_obj_optimizer.config.objectives), 1)
    
    def test_statistics_setup(self):
        """Test statistics setup"""
        stats = self.optimizer._setup_statistics()
        
        self.assertIsNotNone(stats)
        self.assertIn('avg', stats.functions)
        self.assertIn('std', stats.functions)
        self.assertIn('min', stats.functions)
        self.assertIn('max', stats.functions)


class TestFitnessMetrics(unittest.TestCase):
    """Test cases for fitness metrics"""
    
    def test_fitness_metrics_initialization(self):
        """Test fitness metrics initialization"""
        metrics = FitnessMetrics()
        
        self.assertEqual(metrics.net_profit, 0.0)
        self.assertEqual(metrics.sharpe_ratio, 0.0)
        self.assertEqual(metrics.max_drawdown, 0.0)
        self.assertEqual(metrics.win_rate, 0.0)
    
    def test_fitness_metrics_custom_values(self):
        """Test fitness metrics with custom values"""
        metrics = FitnessMetrics(
            net_profit=1500.0,
            sharpe_ratio=2.1,
            max_drawdown=0.05,
            win_rate=0.75,
            total_trades=200
        )
        
        self.assertEqual(metrics.net_profit, 1500.0)
        self.assertEqual(metrics.sharpe_ratio, 2.1)
        self.assertEqual(metrics.max_drawdown, 0.05)
        self.assertEqual(metrics.win_rate, 0.75)
        self.assertEqual(metrics.total_trades, 200)
    
    def test_fitness_metrics_to_dict(self):
        """Test fitness metrics to dictionary conversion"""
        metrics = FitnessMetrics(
            net_profit=1000.0,
            sharpe_ratio=1.5,
            max_drawdown=0.1
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['net_profit'], 1000.0)
        self.assertEqual(metrics_dict['sharpe_ratio'], 1.5)
        self.assertEqual(metrics_dict['max_drawdown'], 0.1)
        self.assertIn('win_rate', metrics_dict)
        self.assertIn('profit_factor', metrics_dict)


class TestOptimizationConfig(unittest.TestCase):
    """Test cases for optimization configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = OptimizationConfig()
        
        self.assertEqual(config.population_size, 100)
        self.assertEqual(config.generations, 200)
        self.assertTrue(config.use_multi_objective)
        self.assertEqual(config.crossover_prob, 0.8)
        self.assertEqual(config.mutation_prob, 0.2)
        self.assertEqual(config.tournament_size, 5)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = OptimizationConfig(
            population_size=50,
            generations=100,
            use_multi_objective=False,
            crossover_prob=0.7,
            mutation_prob=0.3
        )
        
        self.assertEqual(config.population_size, 50)
        self.assertEqual(config.generations, 100)
        self.assertFalse(config.use_multi_objective)
        self.assertEqual(config.crossover_prob, 0.7)
        self.assertEqual(config.mutation_prob, 0.3)
    
    def test_objectives_configuration(self):
        """Test objectives configuration"""
        config = OptimizationConfig(
            objectives=[
                OptimizationObjective.PROFIT,
                OptimizationObjective.SHARPE_RATIO
            ]
        )
        
        self.assertEqual(len(config.objectives), 2)
        self.assertIn(OptimizationObjective.PROFIT, config.objectives)
        self.assertIn(OptimizationObjective.SHARPE_RATIO, config.objectives)


def run_async_tests():
    """Run asynchronous tests"""
    async def run_async_test_suite():
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add async test cases
        test_suite.addTest(TestAdvancedGeneticOptimizer('test_async_optimization'))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        return result.wasSuccessful()
    
    return asyncio.run(run_async_test_suite())


if __name__ == '__main__':
    print("Testing Advanced Genetic Algorithm Optimizer")
    print("=" * 50)
    
    # Run synchronous tests
    print("\nRunning synchronous tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run asynchronous tests
    print("\nRunning asynchronous tests...")
    success = run_async_tests()
    
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed!")
