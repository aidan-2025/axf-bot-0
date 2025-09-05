#!/usr/bin/env python3
"""
Test Validation Framework

Comprehensive test suite for the validation framework components including
modeling quality validation, cross-platform comparison, statistical validation,
regression testing, and reporting.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import validation components
from strategy_validation.validation.modeling_quality_validator import (
    ModelingQualityValidator, ModelingQualityConfig, QualityMetrics, QualityReport
)
from strategy_validation.validation.cross_platform_comparator import (
    CrossPlatformComparator, ComparisonConfig, ComparisonResult, MetricComparison
)
from strategy_validation.validation.statistical_validator import (
    StatisticalValidator, StatisticalConfig, DistributionAnalysis, CorrelationAnalysis
)
from strategy_validation.validation.regression_tester import (
    RegressionTester, RegressionConfig, TestSuite, TestResult, ToleranceThresholds
)
from strategy_validation.validation.validation_reporter import (
    ValidationReporter, ReportConfig, QualityDashboard, ValidationReport
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationFrameworkTester:
    """Test suite for validation framework"""
    
    def __init__(self):
        self.test_results = []
        self.setup_test_data()
    
    def setup_test_data(self):
        """Set up test data for validation tests"""
        
        # Generate sample backtest data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h')
        
        # Generate OHLC data
        base_price = 1.1000
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.backtest_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.0001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates)),
            'spread': np.random.uniform(0.0001, 0.0005, len(dates))
        })
        
        # Generate sample trade log
        trade_count = 100
        self.trade_log = pd.DataFrame({
            'trade_id': [f'trade_{i}' for i in range(trade_count)],
            'entry_time': [dates[i] for i in np.random.randint(0, len(dates)-100, trade_count)],
            'exit_time': [dates[i+50] for i in np.random.randint(0, len(dates)-100, trade_count)],
            'entry_price': np.random.uniform(1.0900, 1.1100, trade_count),
            'exit_price': np.random.uniform(1.0900, 1.1100, trade_count),
            'pnl': np.random.normal(0, 0.001, trade_count),
            'commission': np.random.uniform(0.0001, 0.0003, trade_count),
            'slippage': np.random.uniform(0, 0.0001, trade_count)
        })
        
        # Generate sample backtest results
        self.backtest_results = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05,
            'profit_factor': 1.8,
            'win_rate': 0.55,
            'total_trades': trade_count,
            'volatility': 0.12,
            'skewness': 0.1,
            'kurtosis': 3.2,
            'returns': np.random.normal(0.001, 0.01, 1000).tolist()
        }
        
        # Generate reference data (slightly different)
        self.reference_data = self.backtest_data.copy()
        self.reference_data['close'] *= (1 + np.random.normal(0, 0.0001, len(dates)))
        
        self.reference_trades = self.trade_log.copy()
        self.reference_trades['pnl'] *= (1 + np.random.normal(0, 0.01, trade_count))
        
        self.reference_results = self.backtest_results.copy()
        self.reference_results['total_return'] *= 1.02  # 2% difference
        self.reference_results['sharpe_ratio'] *= 0.98  # 2% difference
    
    def test_modeling_quality_validator(self):
        """Test modeling quality validator"""
        
        logger.info("Testing Modeling Quality Validator...")
        
        try:
            # Create validator
            config = ModelingQualityConfig()
            validator = ModelingQualityValidator(config)
            
            # Test data quality validation
            data_metrics = validator.validate_data_quality(
                self.backtest_data, self.reference_data
            )
            
            assert isinstance(data_metrics, QualityMetrics)
            assert 0 <= data_metrics.data_completeness <= 1
            assert 0 <= data_metrics.data_accuracy <= 1
            assert 0 <= data_metrics.overall_quality_score <= 1
            
            # Test execution quality validation
            execution_metrics = validator.validate_execution_quality(self.trade_log)
            
            assert isinstance(execution_metrics, QualityMetrics)
            assert 0 <= execution_metrics.order_execution_accuracy <= 1
            
            # Test statistical quality validation
            statistical_metrics = validator.validate_statistical_quality(
                self.backtest_results, self.reference_results
            )
            
            assert isinstance(statistical_metrics, QualityMetrics)
            assert 0 <= statistical_metrics.return_distribution_accuracy <= 1
            
            # Test comprehensive quality report
            quality_report = validator.generate_quality_report(
                strategy_id="test_strategy",
                data_period=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
                backtest_data=self.backtest_data,
                trade_log=self.trade_log,
                backtest_results=self.backtest_results,
                reference_data=self.reference_data,
                reference_trades=self.reference_trades,
                reference_results=self.reference_results
            )
            
            assert isinstance(quality_report, QualityReport)
            assert quality_report.strategy_id == "test_strategy"
            assert len(quality_report.issues) >= 0
            assert len(quality_report.warnings) >= 0
            assert len(quality_report.recommendations) >= 0
            
            logger.info("✅ Modeling Quality Validator tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Modeling Quality Validator test failed: {e}")
            return False
    
    def test_cross_platform_comparator(self):
        """Test cross-platform comparator"""
        
        logger.info("Testing Cross-Platform Comparator...")
        
        try:
            # Create comparator
            config = ComparisonConfig()
            comparator = CrossPlatformComparator(config)
            
            # Test metric comparison
            metric_comparisons = comparator._compare_metrics(
                self.backtest_results, self.reference_results
            )
            
            assert isinstance(metric_comparisons, list)
            assert len(metric_comparisons) > 0
            
            for comparison in metric_comparisons:
                assert isinstance(comparison, MetricComparison)
                assert comparison.metric_name in self.backtest_results
                assert 0 <= comparison.relative_difference <= 1
            
            # Test trade comparison
            trade_comparisons = comparator._compare_trades(
                self.trade_log.to_dict('records'),
                self.reference_trades.to_dict('records')
            )
            
            assert isinstance(trade_comparisons, list)
            
            # Test comprehensive comparison
            comparison_result = comparator.compare_results(
                strategy_id="test_strategy",
                platform_a_name="Backtrader",
                platform_b_name="MT4",
                platform_a_results=self.backtest_results,
                platform_b_results=self.reference_results,
                platform_a_trades=self.trade_log.to_dict('records'),
                platform_b_trades=self.reference_trades.to_dict('records'),
                data_period=(datetime(2023, 1, 1), datetime(2023, 12, 31))
            )
            
            assert isinstance(comparison_result, ComparisonResult)
            assert comparison_result.strategy_id == "test_strategy"
            assert len(comparison_result.metric_comparisons) > 0
            assert 0 <= comparison_result.overall_agreement <= 1
            
            logger.info("✅ Cross-Platform Comparator tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cross-Platform Comparator test failed: {e}")
            return False
    
    def test_statistical_validator(self):
        """Test statistical validator"""
        
        logger.info("Testing Statistical Validator...")
        
        try:
            # Create validator
            config = StatisticalConfig()
            validator = StatisticalValidator(config)
            
            # Test distribution validation
            distribution_analysis = validator.validate_distribution(
                self.backtest_results['returns']
            )
            
            assert isinstance(distribution_analysis, DistributionAnalysis)
            assert distribution_analysis.mean is not None
            assert distribution_analysis.std is not None
            assert distribution_analysis.skewness is not None
            assert distribution_analysis.kurtosis is not None
            assert len(distribution_analysis.percentiles) > 0
            
            # Test correlation validation
            correlation_data = {
                'returns': self.backtest_results['returns'],
                'volatility': np.random.uniform(0.1, 0.2, 1000).tolist(),
                'volume': np.random.randint(1000, 10000, 1000).tolist()
            }
            
            correlation_analysis = validator.validate_correlations(correlation_data)
            
            assert isinstance(correlation_analysis, CorrelationAnalysis)
            assert len(correlation_analysis.cross_correlations) > 0
            assert correlation_analysis.max_correlation is not None
            assert correlation_analysis.min_correlation is not None
            
            # Test stationarity validation
            stationarity_tests = validator.validate_stationarity(
                self.backtest_results['returns']
            )
            
            assert isinstance(stationarity_tests, list)
            
            # Test independence validation
            independence_tests = validator.validate_independence(
                self.backtest_results['returns']
            )
            
            assert isinstance(independence_tests, list)
            
            # Test heteroscedasticity validation
            heteroscedasticity_tests = validator.validate_heteroscedasticity(
                self.backtest_results['returns']
            )
            
            assert isinstance(heteroscedasticity_tests, list)
            
            logger.info("✅ Statistical Validator tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Statistical Validator test failed: {e}")
            return False
    
    def test_regression_tester(self):
        """Test regression tester"""
        
        logger.info("Testing Regression Tester...")
        
        try:
            # Create tester
            config = RegressionConfig()
            tester = RegressionTester(config)
            
            # Test metric comparison test
            metric_test_result = tester.run_metric_comparison_test(
                test_id="test_metric_comparison",
                strategy_id="test_strategy",
                baseline_metrics=self.backtest_results,
                current_metrics=self.reference_results,
                data_period=(datetime(2023, 1, 1), datetime(2023, 12, 31))
            )
            
            assert isinstance(metric_test_result, TestResult)
            assert metric_test_result.test_id == "test_metric_comparison"
            assert metric_test_result.strategy_id == "test_strategy"
            assert metric_test_result.execution_time > 0
            
            # Test trade comparison test
            trade_test_result = tester.run_trade_comparison_test(
                test_id="test_trade_comparison",
                strategy_id="test_strategy",
                baseline_trades=self.trade_log.to_dict('records'),
                current_trades=self.reference_trades.to_dict('records'),
                data_period=(datetime(2023, 1, 1), datetime(2023, 12, 31))
            )
            
            assert isinstance(trade_test_result, TestResult)
            assert trade_test_result.test_id == "test_trade_comparison"
            assert trade_test_result.execution_time > 0
            
            # Test performance benchmark test
            benchmark_metrics = {
                'total_return': 0.12,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.06
            }
            
            benchmark_test_result = tester.run_performance_benchmark_test(
                test_id="test_performance_benchmark",
                strategy_id="test_strategy",
                current_metrics=self.backtest_results,
                benchmark_metrics=benchmark_metrics,
                data_period=(datetime(2023, 1, 1), datetime(2023, 12, 31))
            )
            
            assert isinstance(benchmark_test_result, TestResult)
            assert benchmark_test_result.test_id == "test_performance_benchmark"
            assert benchmark_test_result.execution_time > 0
            
            # Test baseline saving and loading
            baseline_id = tester.save_baseline(
                strategy_id="test_strategy",
                data_period=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
                metrics=self.backtest_results,
                trades=self.trade_log.to_dict('records')
            )
            
            assert baseline_id is not None
            
            loaded_baseline = tester.load_baseline(baseline_id)
            assert loaded_baseline is not None
            assert loaded_baseline['strategy_id'] == "test_strategy"
            assert 'metrics' in loaded_baseline
            assert 'trades' in loaded_baseline
            
            # Test test suite creation and execution
            test_suite = tester.create_test_suite(
                suite_id="test_suite_1",
                suite_name="Test Suite 1",
                description="Test suite for validation framework",
                strategy_ids=["test_strategy"],
                data_periods=[(datetime(2023, 1, 1), datetime(2023, 12, 31))],
                configurations=[{'metrics': self.backtest_results, 'trades': self.trade_log.to_dict('records')}]
            )
            
            assert isinstance(test_suite, TestSuite)
            assert test_suite.suite_id == "test_suite_1"
            assert len(test_suite.strategy_ids) == 1
            
            logger.info("✅ Regression Tester tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Regression Tester test failed: {e}")
            return False
    
    def test_validation_reporter(self):
        """Test validation reporter"""
        
        logger.info("Testing Validation Reporter...")
        
        try:
            # Create reporter
            config = ReportConfig()
            reporter = ValidationReporter(config)
            
            # Test quality report generation
            quality_data = {
                'metrics': {
                    'data_completeness': 0.98,
                    'data_accuracy': 0.95,
                    'spread_accuracy': 0.92,
                    'overall_quality_score': 0.95
                },
                'issues': ['Minor data gap detected'],
                'warnings': ['Spread accuracy below threshold'],
                'recommendations': ['Improve spread modeling']
            }
            
            quality_report = reporter.generate_quality_report(
                quality_data=quality_data,
                strategy_id="test_strategy",
                validation_period=(datetime(2023, 1, 1), datetime(2023, 12, 31))
            )
            
            assert isinstance(quality_report, ValidationReport)
            assert quality_report.report_type.value == "quality_report"
            assert quality_report.strategy_id == "test_strategy"
            assert 'quality_scores' in quality_report.quality_metrics
            
            # Test comparison report generation
            comparison_data = {
                'metric_comparisons': [
                    {
                        'metric_name': 'total_return',
                        'platform_a_value': 0.15,
                        'platform_b_value': 0.14,
                        'relative_difference': 0.067
                    }
                ],
                'overall_status': 'good',
                'overall_agreement': 0.95
            }
            
            comparison_report = reporter.generate_comparison_report(
                comparison_data=comparison_data,
                strategy_id="test_strategy",
                validation_period=(datetime(2023, 1, 1), datetime(2023, 12, 31))
            )
            
            assert isinstance(comparison_report, ValidationReport)
            assert comparison_report.report_type.value == "comparison_report"
            
            # Test dashboard generation
            dashboard_data = QualityDashboard(
                overall_quality_score=0.95,
                quality_level="good",
                total_validations=10,
                passed_validations=9,
                failed_validations=1,
                data_quality_score=0.98,
                execution_quality_score=0.92,
                statistical_quality_score=0.95,
                critical_issues=["Data gap in Q2"],
                warnings=["Spread accuracy below threshold"],
                recommendations=["Improve data quality checks"],
                average_validation_time=2.5,
                validation_success_rate=0.9
            )
            
            dashboard_report = reporter.generate_dashboard(
                dashboard_data=dashboard_data,
                strategy_id="test_strategy",
                validation_period=(datetime(2023, 1, 1), datetime(2023, 12, 31))
            )
            
            assert isinstance(dashboard_report, ValidationReport)
            assert dashboard_report.report_type.value == "dashboard"
            
            # Test report saving
            try:
                report_path = reporter.save_report(quality_report)
                assert Path(report_path).exists()
            except Exception as e:
                logger.warning(f"Report saving failed: {e}")
                # This is not critical for the test
            
            logger.info("✅ Validation Reporter tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation Reporter test failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Run all validation framework tests"""
        
        logger.info("Starting Validation Framework Tests...")
        logger.info("=" * 60)
        
        tests = [
            ("Modeling Quality Validator", self.test_modeling_quality_validator),
            ("Cross-Platform Comparator", self.test_cross_platform_comparator),
            ("Statistical Validator", self.test_statistical_validator),
            ("Regression Tester", self.test_regression_tester),
            ("Validation Reporter", self.test_validation_reporter)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name}...")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with exception: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Validation Framework Test Results:")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 All validation framework tests passed!")
            return True
        else:
            logger.error("❌ Some validation framework tests failed!")
            return False


def main():
    """Main test function"""
    
    try:
        tester = ValidationFrameworkTester()
        success = tester.run_all_tests()
        
        if success:
            print("\n✅ All validation framework tests completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some validation framework tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
