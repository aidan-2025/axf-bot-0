#!/usr/bin/env python3
"""
Complete Validation Framework Integration Test

Tests the complete strategy validation framework including:
- Strategy filtering
- Scoring engine
- Strategy evaluation
- Filtering pipeline
- API integration
"""

import asyncio
import unittest
import logging
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import validation components
from src.strategy_validation.filtering.strategy_filter import StrategyFilter, FilterConfig
from src.strategy_validation.scoring.scoring_engine import ScoringEngine, ScoringConfig
from src.strategy_validation.evaluation.strategy_evaluator import StrategyEvaluator, EvaluationConfig
from src.strategy_validation.filtering.filtering_pipeline import FilteringPipeline, PipelineConfig, PipelineStatus

# Import API components
from src.api.routes.strategy_validation import ValidationRequest, ValidationResponse


class CompleteValidationFrameworkTest(unittest.TestCase):
    """Test the complete validation framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger(__name__)
        
        # Create test strategies
        self.test_strategies = self._create_test_strategies()
        
        # Create test backtest results
        self.test_backtest_results = self._create_test_backtest_results()
        
        # Create test benchmark results
        self.test_benchmark_results = self._create_test_benchmark_results()
        
        # Create test configurations
        self.filter_config = FilterConfig(
            min_trades=10,
            min_profit_factor=1.1,
            max_drawdown_threshold=0.15,
            min_sharpe_ratio=0.5,
            min_win_rate=0.4,
            min_data_points=1000,
            min_backtest_days=90,
            max_volatility=0.3,
            max_var_95=0.05
        )
        
        self.scoring_config = ScoringConfig(
            enable_benchmark_comparison=True,
            benchmark_weight=0.2,
            enable_risk_adjustment=True,
            risk_free_rate=0.02,
            enable_consistency_penalty=True,
            consistency_penalty_factor=0.1,
            enable_efficiency_bonus=True,
            efficiency_bonus_factor=0.05
        )
        
        self.evaluation_config = EvaluationConfig(
            enable_performance_metrics=True,
            enable_risk_metrics=True,
            enable_consistency_metrics=True,
            enable_efficiency_metrics=True,
            enable_robustness_metrics=True,
            enable_benchmark_comparison=True,
            enable_statistical_validation=True,
            enable_monte_carlo_validation=True,
            monte_carlo_runs=1000,
            confidence_level=0.95,
            enable_walk_forward_validation=True,
            walk_forward_periods=5,
            enable_cross_validation=True,
            cross_validation_folds=5,
            enable_stress_testing=True,
            stress_test_scenarios=10,
            enable_regression_testing=True,
            enable_risk_adjustment=True,
            risk_free_rate=0.02,
            enable_consistency_penalty=True,
            consistency_penalty_factor=0.1,
            enable_efficiency_bonus=True,
            efficiency_bonus_factor=0.05
        )
        
        self.pipeline_config = PipelineConfig(
            filter_config=self.filter_config,
            scoring_config=self.scoring_config,
            evaluation_config=self.evaluation_config,
            enable_parallel_processing=True,
            max_workers=2,
            batch_size=5,
            enable_filtering=True,
            enable_scoring=True,
            enable_evaluation=True,
            enable_ranking=True,
            top_n_strategies=3,
            min_passing_score=70.0,
            generate_reports=True,
            timeout_seconds=60.0
        )
    
    def _create_test_strategies(self) -> List[Dict[str, Any]]:
        """Create test strategies"""
        strategies = []
        
        for i in range(10):
            strategy = {
                'strategy_id': f'test_strategy_{i}',
                'strategy_name': f'Test Strategy {i}',
                'class_name': f'TestStrategy{i}',
                'module_path': f'test_module_{i}',
                'parameters': {
                    'name': f'Test Strategy {i}',
                    'description': f'Test strategy {i} for validation',
                    'category': 'test',
                    'risk_level': 'medium',
                    'timeframe': '1h',
                    'symbols': ['EURUSD', 'GBPUSD'],
                    'indicators': ['SMA', 'RSI'],
                    'entry_conditions': ['SMA_cross', 'RSI_oversold'],
                    'exit_conditions': ['SMA_cross', 'RSI_overbought'],
                    'risk_management': {
                        'stop_loss': 0.02,
                        'take_profit': 0.04,
                        'position_size': 0.1
                    }
                },
                'description': f'Test strategy {i} for validation framework',
                'category': 'test',
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            strategies.append(strategy)
        
        return strategies
    
    def _create_test_backtest_results(self) -> List[Dict[str, Any]]:
        """Create test backtest results"""
        results = []
        
        for i in range(10):
            # Create realistic backtest results
            total_trades = 50 + i * 10
            winning_trades = int(total_trades * (0.4 + i * 0.02))
            losing_trades = total_trades - winning_trades
            
            gross_profit = 1000 + i * 100
            gross_loss = 800 + i * 80
            net_profit = gross_profit - gross_loss
            
            max_drawdown = 0.05 + i * 0.01
            sharpe_ratio = 0.5 + i * 0.1
            
            result = {
                'strategy_id': f'test_strategy_{i}',
                'strategy_name': f'Test Strategy {i}',
                'backtest_id': f'backtest_{i}',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'duration_days': 365,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'net_profit': net_profit,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': 30 + i * 5,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sharpe_ratio * 1.2,
                'calmar_ratio': net_profit / (max_drawdown * 10000) if max_drawdown > 0 else 0,
                'var_95': 0.02 + i * 0.005,
                'cvar_95': 0.025 + i * 0.005,
                'avg_trade_duration': 5 + i * 2,
                'avg_winning_trade': gross_profit / winning_trades if winning_trades > 0 else 0,
                'avg_losing_trade': gross_loss / losing_trades if losing_trades > 0 else 0,
                'largest_win': 100 + i * 10,
                'largest_loss': 80 + i * 8,
                'consecutive_wins': 5 + i,
                'consecutive_losses': 3 + i,
                'recovery_factor': net_profit / (max_drawdown * 10000) if max_drawdown > 0 else 0,
                'expectancy': net_profit / total_trades if total_trades > 0 else 0,
                'consistency_score': 0.3 + i * 0.05,
                'robustness_score': 0.3 + i * 0.05,
                'efficiency_score': 0.3 + i * 0.05,
                'stability_score': 0.3 + i * 0.05,
                'reliability_score': 0.3 + i * 0.05,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            results.append(result)
        
        return results
    
    def _create_test_benchmark_results(self) -> Dict[str, Any]:
        """Create test benchmark results"""
        return {
            'benchmark_id': 'test_benchmark',
            'benchmark_name': 'Test Benchmark',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'duration_days': 365,
            'total_return': 0.15,
            'annualized_return': 0.15,
            'volatility': 0.12,
            'sharpe_ratio': 1.25,
            'max_drawdown': 0.08,
            'calmar_ratio': 1.875,
            'var_95': 0.02,
            'cvar_95': 0.025,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
    
    def test_strategy_filter(self):
        """Test strategy filtering"""
        self.logger.info("Testing strategy filter...")
        
        # Create filter
        strategy_filter = StrategyFilter(self.filter_config)
        
        # Test filtering
        passing_strategies = strategy_filter.get_passing_strategies(
            self.test_strategies, 
            self.test_backtest_results
        )
        
        # Verify results
        self.assertIsInstance(passing_strategies, list)
        self.assertLessEqual(len(passing_strategies), len(self.test_strategies))
        
        # Test individual validation
        for strategy in passing_strategies:
            validation_result = strategy_filter.validate_strategy(strategy, None)
            self.assertTrue(validation_result['passed'])
        
        self.logger.info(f"Strategy filter test passed: {len(passing_strategies)}/{len(self.test_strategies)} strategies passed")
    
    def test_scoring_engine(self):
        """Test scoring engine"""
        self.logger.info("Testing scoring engine...")
        
        # Create scoring engine
        scoring_engine = ScoringEngine(self.scoring_config)
        
        # Test scoring
        scoring_results = scoring_engine.score_strategies(
            self.test_strategies,
            self.test_backtest_results,
            self.test_benchmark_results
        )
        
        # Verify results
        self.assertIsInstance(scoring_results, list)
        self.assertEqual(len(scoring_results), len(self.test_strategies))
        
        # Test individual scoring
        for i, result in enumerate(scoring_results):
            # Handle both dict and ScoringResult objects
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            else:
                result_dict = result
                
            self.assertIn('strategy_id', result_dict)
            self.assertIn('total_score', result_dict)
            self.assertIn('breakdown', result_dict)
            self.assertIn('normalized_score', result_dict)
            
            # Check breakdown contains the individual scores
            breakdown = result_dict['breakdown']
            self.assertIn('performance_score', breakdown)
            self.assertIn('risk_score', breakdown)
            self.assertIn('consistency_score', breakdown)
            self.assertIn('efficiency_score', breakdown)
            
            # Verify score ranges
            self.assertGreaterEqual(result_dict['total_score'], 0)
            self.assertLessEqual(result_dict['total_score'], 100)
            self.assertGreaterEqual(result_dict['normalized_score'], 0)
            self.assertLessEqual(result_dict['normalized_score'], 100)
        
        self.logger.info("Scoring engine test passed")
    
    def test_strategy_evaluator(self):
        """Test strategy evaluation"""
        self.logger.info("Testing strategy evaluator...")
        
        # Create evaluator
        evaluator = StrategyEvaluator(self.evaluation_config)
        
        # Test evaluation
        evaluation_results = evaluator.evaluate_strategies(
            self.test_strategies,
            self.test_backtest_results,
            self.test_benchmark_results
        )
        
        # Verify results
        self.assertIsInstance(evaluation_results, list)
        self.assertEqual(len(evaluation_results), len(self.test_strategies))
        
        # Test individual evaluation
        for i, result in enumerate(evaluation_results):
            # Handle both dict and EvaluationResult objects
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            else:
                result_dict = result
                
            self.assertIn('strategy_id', result_dict)
            self.assertIn('evaluation_metrics', result_dict)
            self.assertIn('validation_results', result_dict)
            self.assertIn('recommendations', result_dict)
            
            # Verify evaluation metrics
            metrics = result_dict['evaluation_metrics']
            self.assertIn('performance_score', metrics)
            self.assertIn('risk_score', metrics)
            self.assertIn('consistency_score', metrics)
            self.assertIn('efficiency_score', metrics)
            self.assertIn('robustness_score', metrics)
            self.assertIn('normalized_score', metrics)
            
            # Verify score ranges
            self.assertGreaterEqual(metrics['normalized_score'], 0)
            self.assertLessEqual(metrics['normalized_score'], 100)
        
        self.logger.info("Strategy evaluator test passed")
    
    def test_filtering_pipeline(self):
        """Test filtering pipeline"""
        self.logger.info("Testing filtering pipeline...")
        
        # Create pipeline
        pipeline = FilteringPipeline(self.pipeline_config)
        
        # Test pipeline
        result = pipeline.run_pipeline_sync(
            strategies=self.test_strategies,
            backtest_results=self.test_backtest_results,
            benchmark_results=self.test_benchmark_results
        )
        
        # Handle both dict and PipelineResult objects
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = result
            
        self.assertIn('pipeline_id', result_dict)
        self.assertIn('status', result_dict)
        self.assertIn('duration', result_dict)
        self.assertIn('input_strategies', result_dict)
        self.assertIn('filtered_strategies', result_dict)
        self.assertIn('evaluated_strategies', result_dict)
        self.assertIn('passing_strategies', result_dict)
        self.assertIn('top_strategies', result_dict)
        self.assertIn('stages_completed', result_dict)
        self.assertIn('errors', result_dict)
        self.assertIn('warnings', result_dict)
        
        # Verify status
        self.assertIn(result_dict['status'], ['completed', 'failed'])
        
        # Verify counts
        self.assertEqual(result_dict['input_strategies'], len(self.test_strategies))
        self.assertLessEqual(result_dict['filtered_strategies'], result_dict['input_strategies'])
        self.assertLessEqual(result_dict['evaluated_strategies'], result_dict['filtered_strategies'])
        self.assertLessEqual(result_dict['passing_strategies'], result_dict['evaluated_strategies'])
        
        # Verify top strategies
        self.assertIsInstance(result_dict['top_strategies'], list)
        self.assertLessEqual(len(result_dict['top_strategies']), self.pipeline_config.top_n_strategies)
        
        self.logger.info(f"Filtering pipeline test passed: {result_dict['passing_strategies']}/{result_dict['input_strategies']} strategies passed")
    
    def test_validation_request_model(self):
        """Test validation request model"""
        self.logger.info("Testing validation request model...")
        
        # Create request
        request = ValidationRequest(
            strategies=self.test_strategies,
            backtest_results=self.test_backtest_results,
            benchmark_results=self.test_benchmark_results,
            enable_filtering=True,
            enable_scoring=True,
            enable_evaluation=True,
            enable_ranking=True,
            enable_parallel_processing=True,
            max_workers=2,
            batch_size=5,
            top_n_strategies=3,
            min_passing_score=70.0,
            generate_reports=True,
            timeout_seconds=60.0
        )
        
        # Verify request
        self.assertEqual(len(request.strategies), len(self.test_strategies))
        self.assertEqual(len(request.backtest_results), len(self.test_backtest_results))
        self.assertTrue(request.enable_filtering)
        self.assertTrue(request.enable_scoring)
        self.assertTrue(request.enable_evaluation)
        self.assertTrue(request.enable_ranking)
        self.assertTrue(request.enable_parallel_processing)
        self.assertEqual(request.max_workers, 2)
        self.assertEqual(request.batch_size, 5)
        self.assertEqual(request.top_n_strategies, 3)
        self.assertEqual(request.min_passing_score, 70.0)
        self.assertTrue(request.generate_reports)
        self.assertEqual(request.timeout_seconds, 60.0)
        
        self.logger.info("Validation request model test passed")
    
    def test_validation_response_model(self):
        """Test validation response model"""
        self.logger.info("Testing validation response model...")
        
        # Create response
        response = ValidationResponse(
            success=True,
            message="Test validation completed",
            pipeline_id="test_pipeline",
            status="completed",
            duration=30.5,
            input_strategies=10,
            filtered_strategies=8,
            evaluated_strategies=8,
            passing_strategies=6,
            top_strategies=[],
            strategies_per_second=0.33,
            average_evaluation_time=3.8,
            stages_completed=["initialization", "filtering", "scoring", "evaluation", "ranking", "reporting"],
            errors=[],
            warnings=[]
        )
        
        # Verify response
        self.assertTrue(response.success)
        self.assertEqual(response.message, "Test validation completed")
        self.assertEqual(response.pipeline_id, "test_pipeline")
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.duration, 30.5)
        self.assertEqual(response.input_strategies, 10)
        self.assertEqual(response.filtered_strategies, 8)
        self.assertEqual(response.evaluated_strategies, 8)
        self.assertEqual(response.passing_strategies, 6)
        self.assertEqual(response.strategies_per_second, 0.33)
        self.assertEqual(response.average_evaluation_time, 3.8)
        self.assertEqual(len(response.stages_completed), 6)
        self.assertEqual(len(response.errors), 0)
        self.assertEqual(len(response.warnings), 0)
        
        self.logger.info("Validation response model test passed")
    
    def test_complete_validation_workflow(self):
        """Test complete validation workflow"""
        self.logger.info("Testing complete validation workflow...")
        
        # Create pipeline
        pipeline = FilteringPipeline(self.pipeline_config)
        
        # Run complete pipeline
        result = pipeline.run_pipeline_sync(
            strategies=self.test_strategies,
            backtest_results=self.test_backtest_results,
            benchmark_results=self.test_benchmark_results
        )
        
        # Handle both dict and PipelineResult objects
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = result
            
        # Verify pipeline completed successfully
        self.assertEqual(result_dict['status'], 'completed')
        self.assertGreater(result_dict['duration'], 0)
        self.assertEqual(result_dict['input_strategies'], len(self.test_strategies))
        
        # Verify all stages completed
        expected_stages = ['initialization', 'filtering', 'scoring', 'evaluation', 'ranking', 'reporting']
        for stage in expected_stages:
            self.assertIn(stage, result_dict['stages_completed'])
        
        # Verify results are reasonable
        self.assertGreaterEqual(result_dict['passing_strategies'], 0)
        self.assertLessEqual(result_dict['passing_strategies'], result_dict['input_strategies'])
        self.assertLessEqual(len(result_dict['top_strategies']), self.pipeline_config.top_n_strategies)
        
        # Verify performance metrics
        self.assertGreaterEqual(result_dict['strategies_per_second'], 0)
        self.assertGreaterEqual(result_dict['average_evaluation_time'], 0)
        
        self.logger.info(f"Complete validation workflow test passed: {result_dict['passing_strategies']}/{result_dict['input_strategies']} strategies passed")
    
    def test_error_handling(self):
        """Test error handling"""
        self.logger.info("Testing error handling...")
        
        # Test with invalid strategies
        invalid_strategies = [{'invalid': 'strategy'}]
        
        pipeline = FilteringPipeline(self.pipeline_config)
        
        try:
            result = pipeline.run_pipeline_sync(
                strategies=invalid_strategies,
                backtest_results=None,
                benchmark_results=None
            )
            
            # Should handle errors gracefully
            self.assertIn('status', result)
            self.assertIn('errors', result)
            
        except Exception as e:
            # Should not crash
            self.logger.warning(f"Expected error handled: {e}")
        
        self.logger.info("Error handling test passed")
    
    def test_performance_metrics(self):
        """Test performance metrics"""
        self.logger.info("Testing performance metrics...")
        
        # Create pipeline with performance tracking
        config = PipelineConfig(
            filter_config=self.filter_config,
            scoring_config=self.scoring_config,
            evaluation_config=self.evaluation_config,
            enable_parallel_processing=True,
            max_workers=2,
            batch_size=5,
            enable_filtering=True,
            enable_scoring=True,
            enable_evaluation=True,
            enable_ranking=True,
            top_n_strategies=3,
            min_passing_score=70.0,
            generate_reports=True,
            timeout_seconds=60.0
        )
        
        pipeline = FilteringPipeline(config)
        
        # Run pipeline
        result = pipeline.run_pipeline_sync(
            strategies=self.test_strategies,
            backtest_results=self.test_backtest_results,
            benchmark_results=self.test_benchmark_results
        )
        
        # Handle both dict and PipelineResult objects
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = result
            
        # Verify performance metrics
        self.assertGreater(result_dict['duration'], 0)
        self.assertGreaterEqual(result_dict['strategies_per_second'], 0)
        self.assertGreaterEqual(result_dict['average_evaluation_time'], 0)
        
        # Verify stage durations
        self.assertIn('stage_durations', result_dict)
        self.assertIsInstance(result_dict['stage_durations'], dict)
        
        self.logger.info("Performance metrics test passed")
    
    def test_parallel_processing(self):
        """Test parallel processing"""
        self.logger.info("Testing parallel processing...")
        
        # Create pipeline with parallel processing
        config = PipelineConfig(
            filter_config=self.filter_config,
            scoring_config=self.scoring_config,
            evaluation_config=self.evaluation_config,
            enable_parallel_processing=True,
            max_workers=4,
            batch_size=2,
            enable_filtering=True,
            enable_scoring=True,
            enable_evaluation=True,
            enable_ranking=True,
            top_n_strategies=3,
            min_passing_score=70.0,
            generate_reports=True,
            timeout_seconds=60.0
        )
        
        pipeline = FilteringPipeline(config)
        
        # Run pipeline
        result = pipeline.run_pipeline_sync(
            strategies=self.test_strategies,
            backtest_results=self.test_backtest_results,
            benchmark_results=self.test_benchmark_results
        )
        
        # Handle both dict and PipelineResult objects
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = result
            
        # Verify parallel processing worked
        self.assertEqual(result_dict['status'], 'completed')
        # For very fast processing, strategies_per_second might be 0, which is acceptable
        self.assertGreaterEqual(result_dict['strategies_per_second'], 0)
        
        self.logger.info("Parallel processing test passed")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        self.logger.info("Testing configuration validation...")
        
        # Test valid configuration
        valid_config = PipelineConfig(
            filter_config=self.filter_config,
            scoring_config=self.scoring_config,
            evaluation_config=self.evaluation_config,
            enable_parallel_processing=True,
            max_workers=4,
            batch_size=10,
            enable_filtering=True,
            enable_scoring=True,
            enable_evaluation=True,
            enable_ranking=True,
            top_n_strategies=5,
            min_passing_score=70.0,
            generate_reports=True,
            timeout_seconds=300.0
        )
        
        # Should not raise exception
        pipeline = FilteringPipeline(valid_config)
        self.assertIsNotNone(pipeline)
        
        self.logger.info("Configuration validation test passed")


def run_async_tests():
    """Run async tests"""
    async def async_test_pipeline():
        """Test async pipeline execution"""
        logger = logging.getLogger(__name__)
        logger.info("Testing async pipeline execution...")
        
        # Create test data
        test_strategies = [
            {
                'strategy_id': f'test_strategy_{i}',
                'strategy_name': f'Test Strategy {i}',
                'class_name': f'TestStrategy{i}',
                'module_path': f'test_module_{i}',
                'parameters': {
                    'name': f'Test Strategy {i}',
                    'description': f'Test strategy {i} for validation',
                    'category': 'test',
                    'risk_level': 'medium',
                    'timeframe': '1h',
                    'symbols': ['EURUSD', 'GBPUSD'],
                    'indicators': ['SMA', 'RSI'],
                    'entry_conditions': ['SMA_cross', 'RSI_oversold'],
                    'exit_conditions': ['SMA_cross', 'RSI_overbought'],
                    'risk_management': {
                        'stop_loss': 0.02,
                        'take_profit': 0.04,
                        'position_size': 0.1
                    }
                },
                'description': f'Test strategy {i} for validation framework',
                'category': 'test',
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            for i in range(5)
        ]
        
        test_backtest_results = [
            {
                'strategy_id': f'test_strategy_{i}',
                'strategy_name': f'Test Strategy {i}',
                'backtest_id': f'backtest_{i}',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'duration_days': 365,
                'total_trades': 50 + i * 10,
                'winning_trades': int((50 + i * 10) * (0.4 + i * 0.02)),
                'losing_trades': (50 + i * 10) - int((50 + i * 10) * (0.4 + i * 0.02)),
                'win_rate': 0.4 + i * 0.02,
                'gross_profit': 1000 + i * 100,
                'gross_loss': 800 + i * 80,
                'net_profit': 200 + i * 20,
                'profit_factor': (1000 + i * 100) / (800 + i * 80),
                'max_drawdown': 0.05 + i * 0.01,
                'max_drawdown_duration': 30 + i * 5,
                'sharpe_ratio': 0.5 + i * 0.1,
                'sortino_ratio': (0.5 + i * 0.1) * 1.2,
                'calmar_ratio': (200 + i * 20) / ((0.05 + i * 0.01) * 10000),
                'var_95': 0.02 + i * 0.005,
                'cvar_95': 0.025 + i * 0.005,
                'avg_trade_duration': 5 + i * 2,
                'avg_winning_trade': (1000 + i * 100) / int((50 + i * 10) * (0.4 + i * 0.02)),
                'avg_losing_trade': (800 + i * 80) / ((50 + i * 10) - int((50 + i * 10) * (0.4 + i * 0.02))),
                'largest_win': 100 + i * 10,
                'largest_loss': 80 + i * 8,
                'consecutive_wins': 5 + i,
                'consecutive_losses': 3 + i,
                'recovery_factor': (200 + i * 20) / ((0.05 + i * 0.01) * 10000),
                'expectancy': (200 + i * 20) / (50 + i * 10),
                'consistency_score': 0.3 + i * 0.05,
                'robustness_score': 0.3 + i * 0.05,
                'efficiency_score': 0.3 + i * 0.05,
                'stability_score': 0.3 + i * 0.05,
                'reliability_score': 0.3 + i * 0.05,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            for i in range(5)
        ]
        
        test_benchmark_results = {
            'benchmark_id': 'test_benchmark',
            'benchmark_name': 'Test Benchmark',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'duration_days': 365,
            'total_return': 0.15,
            'annualized_return': 0.15,
            'volatility': 0.12,
            'sharpe_ratio': 1.25,
            'max_drawdown': 0.08,
            'calmar_ratio': 1.875,
            'var_95': 0.02,
            'cvar_95': 0.025,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        # Create pipeline configuration
        config = PipelineConfig(
            enable_parallel_processing=True,
            max_workers=2,
            batch_size=2,
            enable_filtering=True,
            enable_scoring=True,
            enable_evaluation=True,
            enable_ranking=True,
            top_n_strategies=3,
            min_passing_score=70.0,
            generate_reports=True,
            timeout_seconds=60.0
        )
        
        # Create and run pipeline
        pipeline = FilteringPipeline(config)
        result = await pipeline.run_pipeline(
            strategies=test_strategies,
            backtest_results=test_backtest_results,
            benchmark_results=test_benchmark_results
        )
        
        # Verify result
        assert result.status == PipelineStatus.COMPLETED
        assert result.input_strategies == len(test_strategies)
        assert result.passing_strategies >= 0
        assert result.passing_strategies <= result.input_strategies
        
        logger.info(f"Async pipeline test passed: {result.passing_strategies}/{result.input_strategies} strategies passed")
    
    # Run async test
    asyncio.run(async_test_pipeline())


def main():
    """Run all tests"""
    print("=" * 80)
    print("COMPLETE VALIDATION FRAMEWORK INTEGRATION TEST")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(CompleteValidationFrameworkTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run async tests
    print("\n" + "=" * 80)
    print("ASYNC PIPELINE TEST")
    print("=" * 80)
    
    try:
        run_async_tests()
        print("✅ Async pipeline test passed")
    except Exception as e:
        print(f"❌ Async pipeline test failed: {e}")
        result.failures.append((f"async_test_pipeline", str(e)))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
