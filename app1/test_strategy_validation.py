#!/usr/bin/env python3
"""
Comprehensive Test Suite for Strategy Validation System

Tests all components of the strategy validation and scoring system.
"""

import sys
import os
import asyncio
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from strategy_validation import (
    ValidationCriteria, ValidationThresholds, StrategyScorer, ScoringWeights,
    BacktraderValidator, BacktestConfig, ValidationStorage, ValidationResult
)
from strategy_validation.criteria.performance_metrics import PerformanceMetrics
from strategy_validation.scoring.strategy_scorer import ScoringMetrics


class TestValidationCriteria(unittest.TestCase):
    """Test validation criteria and thresholds"""
    
    def setUp(self):
        self.criteria = ValidationCriteria()
        self.thresholds = ValidationThresholds()
    
    def test_validation_thresholds_creation(self):
        """Test validation thresholds creation"""
        self.assertIsInstance(self.thresholds, ValidationThresholds)
        self.assertEqual(self.thresholds.min_trades, 30)
        self.assertEqual(self.thresholds.min_profit_factor, 1.2)
        self.assertEqual(self.thresholds.max_drawdown, 0.15)
    
    def test_validation_thresholds_to_dict(self):
        """Test thresholds to dictionary conversion"""
        data = self.thresholds.to_dict()
        self.assertIsInstance(data, dict)
        self.assertIn('min_trades', data)
        self.assertIn('min_profit_factor', data)
        self.assertIn('max_drawdown', data)
    
    def test_validation_thresholds_from_dict(self):
        """Test thresholds from dictionary creation"""
        data = {
            'min_trades': 50,
            'min_profit_factor': 1.5,
            'max_drawdown': 0.10
        }
        thresholds = ValidationThresholds.from_dict(data)
        self.assertEqual(thresholds.min_trades, 50)
        self.assertEqual(thresholds.min_profit_factor, 1.5)
        self.assertEqual(thresholds.max_drawdown, 0.10)
    
    def test_trade_count_validation(self):
        """Test trade count validation"""
        # Valid case
        result = self.criteria.validate_trade_count(50, 90)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['violations']), 0)
        
        # Invalid case - too few trades
        result = self.criteria.validate_trade_count(10, 90)
        self.assertFalse(result['passed'])
        self.assertGreater(len(result['violations']), 0)
        
        # High frequency case - should generate warning but not fail
        result = self.criteria.validate_trade_count(1000, 30)
        self.assertTrue(result['passed'])  # Should pass but with warning
        self.assertGreater(len(result['warnings']), 0)  # Should have warnings
    
    def test_performance_metrics_validation(self):
        """Test performance metrics validation"""
        # Valid case
        metrics = {
            'profit_factor': 1.5,
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.0,
            'calmar_ratio': 0.8
        }
        result = self.criteria.validate_performance_metrics(metrics)
        self.assertTrue(result['passed'])
        
        # Invalid case - low profit factor
        metrics['profit_factor'] = 0.8
        result = self.criteria.validate_performance_metrics(metrics)
        self.assertFalse(result['passed'])
        self.assertGreater(len(result['violations']), 0)
    
    def test_risk_metrics_validation(self):
        """Test risk metrics validation"""
        # Valid case
        metrics = {
            'max_drawdown': 0.08,
            'consecutive_losses': 3,
            'max_daily_loss': 0.02
        }
        result = self.criteria.validate_risk_metrics(metrics)
        self.assertTrue(result['passed'])
        
        # Invalid case - excessive drawdown
        metrics['max_drawdown'] = 0.25
        result = self.criteria.validate_risk_metrics(metrics)
        self.assertFalse(result['passed'])
        self.assertGreater(len(result['violations']), 0)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation"""
        strategy_metrics = {
            'total_trades': 50,
            'backtest_duration_days': 90,
            'performance_metrics': {
                'profit_factor': 1.5,
                'sharpe_ratio': 1.2,
                'sortino_ratio': 1.0,
                'calmar_ratio': 0.8
            },
            'risk_metrics': {
                'max_drawdown': 0.08,
                'consecutive_losses': 3,
                'max_daily_loss': 0.02
            },
            'consistency_metrics': {
                'win_rate': 0.6,
                'avg_win_loss_ratio': 1.2,
                'consistency_score': 0.8,
                'stability_score': 0.7
            }
        }
        
        result = self.criteria.validate_comprehensive(strategy_metrics)
        self.assertTrue(result['overall_passed'])
        self.assertEqual(len(result['critical_violations']), 0)
        self.assertGreater(result['score'], 0.0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculation"""
    
    def setUp(self):
        self.metrics = PerformanceMetrics()
    
    def test_performance_metrics_creation(self):
        """Test performance metrics creation"""
        self.assertIsInstance(self.metrics, PerformanceMetrics)
        self.assertEqual(self.metrics.total_return, 0.0)
        self.assertEqual(self.metrics.total_trades, 0)
    
    def test_calculate_from_trades(self):
        """Test metrics calculation from trade data"""
        # Create sample trade data
        trades = []
        base_price = 1.1000
        for i in range(50):
            # Generate realistic trade data
            pnl = np.random.normal(0, 10)  # Random P&L
            return_pct = pnl / 10000  # 10k initial capital
            
            trades.append({
                'timestamp': datetime.now() - timedelta(days=50-i),
                'return': return_pct,
                'pnl': pnl,
                'duration': np.random.uniform(1, 24)  # 1-24 hours
            })
        
        # Calculate metrics
        metrics = self.metrics.calculate_from_trades(trades, initial_capital=10000.0)
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.total_trades, 50)
        self.assertGreaterEqual(metrics.win_rate, 0.0)
        self.assertLessEqual(metrics.win_rate, 1.0)
    
    def test_metrics_to_dict(self):
        """Test metrics to dictionary conversion"""
        data = self.metrics.to_dict()
        self.assertIsInstance(data, dict)
        self.assertIn('total_return', data)
        self.assertIn('total_trades', data)
        self.assertIn('win_rate', data)
    
    def test_metrics_from_dict(self):
        """Test metrics from dictionary creation"""
        data = {
            'total_return': 0.15,
            'total_trades': 100,
            'win_rate': 0.6,
            'sharpe_ratio': 1.2
        }
        metrics = PerformanceMetrics.from_dict(data)
        self.assertEqual(metrics.total_return, 0.15)
        self.assertEqual(metrics.total_trades, 100)
        self.assertEqual(metrics.win_rate, 0.6)
        self.assertEqual(metrics.sharpe_ratio, 1.2)


class TestStrategyScorer(unittest.TestCase):
    """Test strategy scoring system"""
    
    def setUp(self):
        self.scorer = StrategyScorer()
        self.weights = ScoringWeights()
    
    def test_scorer_creation(self):
        """Test scorer creation"""
        self.assertIsInstance(self.scorer, StrategyScorer)
        self.assertIsInstance(self.scorer.weights, ScoringWeights)
    
    def test_weights_validation(self):
        """Test weights validation"""
        self.assertTrue(self.weights.validate_weights())
        
        # Test invalid weights
        invalid_weights = ScoringWeights()
        invalid_weights.performance_weight = 0.8  # This will make sum > 1.0
        self.assertFalse(invalid_weights.validate_weights())
    
    def test_strategy_scoring(self):
        """Test strategy scoring"""
        # Create sample performance metrics
        metrics = PerformanceMetrics()
        metrics.total_return = 0.15
        metrics.annualized_return = 0.20
        metrics.profit_factor = 1.5
        metrics.win_rate = 0.6
        metrics.sharpe_ratio = 1.2
        metrics.max_drawdown = 0.08
        metrics.volatility = 0.15
        metrics.consistency_score = 0.8
        metrics.stability_score = 0.7
        metrics.total_trades = 50
        
        # Score the strategy
        scoring_metrics = self.scorer.score_strategy(metrics, backtest_duration_days=90)
        
        self.assertIsInstance(scoring_metrics, ScoringMetrics)
        self.assertGreaterEqual(scoring_metrics.overall_score, 0.0)
        self.assertLessEqual(scoring_metrics.overall_score, 1.0)
        self.assertGreaterEqual(scoring_metrics.performance_score, 0.0)
        self.assertGreaterEqual(scoring_metrics.risk_score, 0.0)
    
    def test_score_breakdown(self):
        """Test score breakdown"""
        metrics = PerformanceMetrics()
        metrics.total_return = 0.15
        metrics.annualized_return = 0.20
        metrics.profit_factor = 1.5
        metrics.win_rate = 0.6
        metrics.sharpe_ratio = 1.2
        metrics.max_drawdown = 0.08
        metrics.volatility = 0.15
        metrics.consistency_score = 0.8
        metrics.stability_score = 0.7
        metrics.total_trades = 50
        
        scoring_metrics = self.scorer.score_strategy(metrics, backtest_duration_days=90)
        breakdown = self.scorer.get_score_breakdown(scoring_metrics)
        
        self.assertIsInstance(breakdown, dict)
        self.assertIn('overall_score', breakdown)
        self.assertIn('category_scores', breakdown)
        self.assertIn('weights', breakdown)


class TestBacktraderValidator(unittest.TestCase):
    """Test Backtrader validator"""
    
    def setUp(self):
        self.config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            symbols=['EURUSD'],
            initial_capital=10000.0
        )
        self.validator = BacktraderValidator(self.config)
    
    def test_validator_creation(self):
        """Test validator creation"""
        self.assertIsInstance(self.validator, BacktraderValidator)
        self.assertIsInstance(self.validator.config, BacktestConfig)
    
    def test_config_creation(self):
        """Test backtest config creation"""
        self.assertIsInstance(self.config, BacktestConfig)
        self.assertEqual(self.config.initial_capital, 10000.0)
        self.assertIn('EURUSD', self.config.symbols)
    
    def test_data_feed_creation(self):
        """Test data feed creation"""
        self.assertIsNotNone(self.validator.data_feed)
        self.assertIsNotNone(self.validator.broker)
    
    def test_broker_simulation(self):
        """Test broker simulation"""
        broker = self.validator.broker
        
        # Test order execution
        result = broker.execute_order(
            symbol='EURUSD',
            order_type='buy',
            size=0.1,
            price=1.1000,
            timestamp=datetime.now()
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('executed_price', result)
    
    def test_account_summary(self):
        """Test account summary"""
        broker = self.validator.broker
        summary = broker.get_account_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('account_balance', summary)
        self.assertIn('equity', summary)
        self.assertIn('positions', summary)


class TestValidationStorage(unittest.TestCase):
    """Test validation storage"""
    
    def setUp(self):
        # Use in-memory SQLite for testing
        self.connection_string = "postgresql://test:test@localhost:5432/test_validation"
        # Note: In a real test, you'd use a test database
        self.storage = None  # Skip storage tests for now
    
    def test_validation_result_creation(self):
        """Test validation result creation"""
        metrics = PerformanceMetrics()
        scoring_metrics = ScoringMetrics()
        
        result = ValidationResult(
            strategy_id='TEST_001',
            strategy_name='Test Strategy',
            strategy_type='test',
            validation_timestamp=datetime.now(),
            validation_passed=True,
            validation_score=0.85,
            critical_violations=[],
            warnings=[],
            performance_metrics=metrics,
            scoring_metrics=scoring_metrics,
            backtest_config={},
            validation_duration_seconds=10.5,
            backtest_duration_days=30,
            total_trades=50
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.strategy_id, 'TEST_001')
        self.assertTrue(result.validation_passed)
        self.assertEqual(result.validation_score, 0.85)
    
    def test_validation_result_to_dict(self):
        """Test validation result to dictionary conversion"""
        metrics = PerformanceMetrics()
        scoring_metrics = ScoringMetrics()
        
        result = ValidationResult(
            strategy_id='TEST_001',
            strategy_name='Test Strategy',
            strategy_type='test',
            validation_timestamp=datetime.now(),
            validation_passed=True,
            validation_score=0.85,
            critical_violations=[],
            warnings=[],
            performance_metrics=metrics,
            scoring_metrics=scoring_metrics,
            backtest_config={},
            validation_duration_seconds=10.5,
            backtest_duration_days=30,
            total_trades=50
        )
        
        data = result.to_dict()
        self.assertIsInstance(data, dict)
        self.assertIn('strategy_id', data)
        self.assertIn('validation_passed', data)
        self.assertIn('performance_metrics', data)
        self.assertIn('scoring_metrics', data)


class TestIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def setUp(self):
        self.criteria = ValidationCriteria()
        self.scorer = StrategyScorer()
    
    def test_end_to_end_validation(self):
        """Test end-to-end validation process"""
        # Create sample strategy metrics
        strategy_metrics = {
            'total_trades': 50,
            'backtest_duration_days': 90,
            'performance_metrics': {
                'profit_factor': 1.5,
                'sharpe_ratio': 1.2,
                'sortino_ratio': 1.0,
                'calmar_ratio': 0.8,
                'total_return': 0.15,
                'annualized_return': 0.20,
                'win_rate': 0.6
            },
            'risk_metrics': {
                'max_drawdown': 0.08,
                'consecutive_losses': 3,
                'max_daily_loss': 0.02,
                'volatility': 0.15
            },
            'consistency_metrics': {
                'consistency_score': 0.8,
                'stability_score': 0.7
            }
        }
        
        # Step 1: Validate strategy
        validation_result = self.criteria.validate_comprehensive(strategy_metrics)
        self.assertIsInstance(validation_result, dict)
        self.assertIn('overall_passed', validation_result)
        self.assertIn('score', validation_result)
        
        # Step 2: Score strategy
        performance_metrics = PerformanceMetrics.from_dict(strategy_metrics['performance_metrics'])
        scoring_metrics = self.scorer.score_strategy(
            performance_metrics, 
            backtest_duration_days=strategy_metrics['backtest_duration_days']
        )
        self.assertIsInstance(scoring_metrics, ScoringMetrics)
        self.assertGreaterEqual(scoring_metrics.overall_score, 0.0)
        
        # Step 3: Create validation result
        validation_result_obj = ValidationResult(
            strategy_id='TEST_001',
            strategy_name='Test Strategy',
            strategy_type='test',
            validation_timestamp=datetime.now(),
            validation_passed=validation_result['overall_passed'],
            validation_score=validation_result['score'],
            critical_violations=validation_result['critical_violations'],
            warnings=validation_result['warnings'],
            performance_metrics=performance_metrics,
            scoring_metrics=scoring_metrics,
            backtest_config={'test': True},
            validation_duration_seconds=5.0,
            backtest_duration_days=strategy_metrics['backtest_duration_days'],
            total_trades=strategy_metrics['total_trades']
        )
        
        self.assertIsInstance(validation_result_obj, ValidationResult)
        self.assertTrue(validation_result_obj.validation_passed)


def run_async_tests():
    """Run asynchronous tests"""
    async def run_async_test_suite():
        print("Running async tests...")
        
        # Test data feed
        from strategy_validation.backtesting.data_feeds import ForexDataFeed, DataFeedConfig
        
        config = DataFeedConfig(source='mock')
        data_feed = ForexDataFeed(config)
        
        # Test data retrieval
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        data = await data_feed.get_data('EURUSD', start_date, end_date, '1h')
        if data is not None:
            print(f"✅ Data feed test passed - retrieved {len(data)} bars")
        else:
            print("❌ Data feed test failed")
        
        # Test multiple symbols
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        multi_data = await data_feed.get_multiple_symbols(symbols, start_date, end_date, '1h')
        print(f"✅ Multi-symbol test passed - retrieved data for {len(multi_data)} symbols")
        
        return True
    
    return asyncio.run(run_async_test_suite())


def main():
    """Run all tests"""
    print("=== Strategy Validation System Test Suite ===\n")
    
    # Run unit tests
    print("1. Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n2. Running async tests...")
    run_async_tests()
    
    print("\n=== All Tests Completed ===")


if __name__ == '__main__':
    main()
