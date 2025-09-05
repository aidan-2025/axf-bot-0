"""
Regression Tester

Implements automated regression testing for backtesting validation,
comparing results across different runs and configurations.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from pathlib import Path
import pickle
from scipy import stats

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of regression test"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class TestType(Enum):
    """Type of regression test"""
    METRIC_COMPARISON = "metric_comparison"
    TRADE_COMPARISON = "trade_comparison"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    DATA_CONSISTENCY = "data_consistency"
    STATISTICAL_VALIDATION = "statistical_validation"
    CROSS_PLATFORM = "cross_platform"


@dataclass
class ToleranceThresholds:
    """Tolerance thresholds for regression testing"""
    
    # Metric tolerances (relative differences)
    total_return_tolerance: float = 0.02      # 2%
    sharpe_ratio_tolerance: float = 0.05      # 5%
    max_drawdown_tolerance: float = 0.03      # 3%
    profit_factor_tolerance: float = 0.05     # 5%
    win_rate_tolerance: float = 0.02          # 2%
    total_trades_tolerance: float = 0.01      # 1%
    volatility_tolerance: float = 0.03        # 3%
    
    # Trade-level tolerances
    entry_price_tolerance: float = 0.0001     # 0.01 pips
    exit_price_tolerance: float = 0.0001      # 0.01 pips
    pnl_tolerance: float = 0.001              # 0.1 pips
    commission_tolerance: float = 0.0001      # 0.01 pips
    slippage_tolerance: float = 0.0001        # 0.01 pips
    
    # Time tolerances (seconds)
    entry_time_tolerance: int = 60            # 1 minute
    exit_time_tolerance: int = 60             # 1 minute
    
    # Statistical tolerances
    correlation_threshold: float = 0.95       # 95% correlation
    distribution_similarity_threshold: float = 0.90  # 90% similarity
    significance_level: float = 0.05          # 5% significance level
    
    def get_tolerance(self, metric_name: str) -> float:
        """Get tolerance for specific metric"""
        tolerance_map = {
            'total_return': self.total_return_tolerance,
            'sharpe_ratio': self.sharpe_ratio_tolerance,
            'max_drawdown': self.max_drawdown_tolerance,
            'profit_factor': self.profit_factor_tolerance,
            'win_rate': self.win_rate_tolerance,
            'total_trades': self.total_trades_tolerance,
            'volatility': self.volatility_tolerance,
            'entry_price': self.entry_price_tolerance,
            'exit_price': self.exit_price_tolerance,
            'pnl': self.pnl_tolerance,
            'commission': self.commission_tolerance,
            'slippage': self.slippage_tolerance
        }
        return tolerance_map.get(metric_name, 0.05)  # Default 5%


@dataclass
class TestResult:
    """Result of a regression test"""
    
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    timestamp: datetime
    
    # Test details
    baseline_value: Optional[float] = None
    current_value: Optional[float] = None
    difference: Optional[float] = None
    relative_difference: Optional[float] = None
    tolerance_threshold: Optional[float] = None
    
    # Test metadata
    strategy_id: str = ""
    data_period: Optional[Tuple[datetime, datetime]] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Results and messages
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Performance metrics
    execution_time: float = 0.0
    memory_usage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'difference': self.difference,
            'relative_difference': self.relative_difference,
            'tolerance_threshold': self.tolerance_threshold,
            'strategy_id': self.strategy_id,
            'data_period': {
                'start': self.data_period[0].isoformat() if self.data_period else None,
                'end': self.data_period[1].isoformat() if self.data_period else None
            } if self.data_period else None,
            'configuration': self.configuration,
            'message': self.message,
            'details': self.details,
            'warnings': self.warnings,
            'errors': self.errors,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage
        }


@dataclass
class TestSuite:
    """Collection of regression tests"""
    
    suite_id: str
    suite_name: str
    description: str
    created_at: datetime
    updated_at: datetime
    
    # Test configuration
    strategy_ids: List[str] = field(default_factory=list)
    data_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)
    configurations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Test results
    test_results: List[TestResult] = field(default_factory=list)
    
    # Suite statistics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    warning_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    
    # Suite status
    overall_status: TestStatus = TestStatus.PASS
    success_rate: float = 0.0
    
    def __post_init__(self):
        """Calculate suite statistics"""
        self.total_tests = len(self.test_results)
        
        if self.total_tests > 0:
            self.passed_tests = sum(1 for result in self.test_results if result.status == TestStatus.PASS)
            self.failed_tests = sum(1 for result in self.test_results if result.status == TestStatus.FAIL)
            self.warning_tests = sum(1 for result in self.test_results if result.status == TestStatus.WARNING)
            self.error_tests = sum(1 for result in self.test_results if result.status == TestStatus.ERROR)
            self.skipped_tests = sum(1 for result in self.test_results if result.status == TestStatus.SKIPPED)
            
            self.success_rate = self.passed_tests / self.total_tests
            
            # Determine overall status
            if self.failed_tests > 0 or self.error_tests > 0:
                self.overall_status = TestStatus.FAIL
            elif self.warning_tests > 0:
                self.overall_status = TestStatus.WARNING
            else:
                self.overall_status = TestStatus.PASS


@dataclass
class RegressionConfig:
    """Configuration for regression testing"""
    
    # Test settings
    enable_metric_comparison: bool = True
    enable_trade_comparison: bool = True
    enable_performance_benchmark: bool = True
    enable_data_consistency: bool = True
    enable_statistical_validation: bool = True
    enable_cross_platform: bool = True
    
    # Tolerance settings
    tolerance_thresholds: ToleranceThresholds = field(default_factory=ToleranceThresholds)
    
    # Baseline settings
    baseline_storage_path: str = "baselines"
    baseline_retention_days: int = 30
    auto_update_baseline: bool = False
    
    # Test execution settings
    max_parallel_tests: int = 4
    test_timeout_seconds: int = 300
    enable_test_parallelization: bool = True
    
    # Reporting settings
    generate_detailed_reports: bool = True
    report_format: str = "json"  # json, html, csv
    report_storage_path: str = "reports"
    
    # Data validation settings
    validate_data_integrity: bool = True
    check_data_consistency: bool = True
    verify_timestamp_alignment: bool = True


class RegressionTester:
    """Implements automated regression testing for backtesting validation"""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.baseline_storage = Path(config.baseline_storage_path)
        self.report_storage = Path(config.report_storage_path)
        
        # Create storage directories
        self.baseline_storage.mkdir(parents=True, exist_ok=True)
        self.report_storage.mkdir(parents=True, exist_ok=True)
    
    def create_test_suite(
        self,
        suite_id: str,
        suite_name: str,
        description: str,
        strategy_ids: List[str],
        data_periods: List[Tuple[datetime, datetime]],
        configurations: Optional[List[Dict[str, Any]]] = None
    ) -> TestSuite:
        """Create a new test suite"""
        
        if configurations is None:
            configurations = [{}] * len(strategy_ids)
        
        return TestSuite(
            suite_id=suite_id,
            suite_name=suite_name,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            strategy_ids=strategy_ids,
            data_periods=data_periods,
            configurations=configurations
        )
    
    def run_test_suite(self, test_suite: TestSuite) -> TestSuite:
        """Run all tests in a test suite"""
        
        self.logger.info(f"Running test suite: {test_suite.suite_name}")
        
        test_suite.test_results = []
        
        for i, strategy_id in enumerate(test_suite.strategy_ids):
            data_period = test_suite.data_periods[i] if i < len(test_suite.data_periods) else None
            configuration = test_suite.configurations[i] if i < len(test_suite.configurations) else {}
            
            # Run tests for this strategy
            strategy_tests = self._run_strategy_tests(strategy_id, data_period, configuration)
            test_suite.test_results.extend(strategy_tests)
        
        # Update suite statistics
        test_suite.updated_at = datetime.now()
        test_suite.__post_init__()  # Recalculate statistics
        
        # Generate report
        if self.config.generate_detailed_reports:
            self._generate_test_report(test_suite)
        
        return test_suite
    
    def run_metric_comparison_test(
        self,
        test_id: str,
        strategy_id: str,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        data_period: Optional[Tuple[datetime, datetime]] = None
    ) -> TestResult:
        """Run metric comparison test"""
        
        start_time = datetime.now()
        
        try:
            # Compare metrics
            comparison_results = []
            overall_status = TestStatus.PASS
            warnings = []
            errors = []
            
            for metric_name, current_value in current_metrics.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    tolerance = self.config.tolerance_thresholds.get_tolerance(metric_name)
                    
                    # Calculate differences
                    if isinstance(current_value, (list, tuple)) and isinstance(baseline_value, (list, tuple)):
                        # Handle list/tuple comparisons
                        if len(current_value) == len(baseline_value):
                            differences = [abs(c - b) for c, b in zip(current_value, baseline_value)]
                            difference = np.mean(differences)
                            relative_difference = difference / (np.mean(np.abs(baseline_value)) + 1e-8)
                        else:
                            difference = float('inf')
                            relative_difference = float('inf')
                    else:
                        # Handle scalar comparisons
                        difference = abs(current_value - baseline_value)
                        relative_difference = difference / abs(baseline_value) if baseline_value != 0 else float('inf')
                    
                    # Determine status
                    if relative_difference <= tolerance:
                        status = TestStatus.PASS
                    elif relative_difference <= tolerance * 2:
                        status = TestStatus.WARNING
                        warnings.append(f"Metric {metric_name} exceeds tolerance: {relative_difference:.3f} > {tolerance:.3f}")
                    else:
                        status = TestStatus.FAIL
                        errors.append(f"Metric {metric_name} significantly exceeds tolerance: {relative_difference:.3f} > {tolerance:.3f}")
                    
                    comparison_results.append({
                        'metric_name': metric_name,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'difference': difference,
                        'relative_difference': relative_difference,
                        'tolerance': tolerance,
                        'status': status.value
                    })
                    
                    if status == TestStatus.FAIL:
                        overall_status = TestStatus.FAIL
                    elif status == TestStatus.WARNING and overall_status == TestStatus.PASS:
                        overall_status = TestStatus.WARNING
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_id=test_id,
                test_name="Metric Comparison Test",
                test_type=TestType.METRIC_COMPARISON,
                status=overall_status,
                timestamp=start_time,
                strategy_id=strategy_id,
                data_period=data_period,
                message=f"Compared {len(comparison_results)} metrics",
                details={'comparison_results': comparison_results},
                warnings=warnings,
                errors=errors,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Metric comparison test failed: {e}")
            return TestResult(
                test_id=test_id,
                test_name="Metric Comparison Test",
                test_type=TestType.METRIC_COMPARISON,
                status=TestStatus.ERROR,
                timestamp=start_time,
                strategy_id=strategy_id,
                data_period=data_period,
                message=f"Test failed with error: {str(e)}",
                errors=[str(e)],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def run_trade_comparison_test(
        self,
        test_id: str,
        strategy_id: str,
        baseline_trades: List[Dict[str, Any]],
        current_trades: List[Dict[str, Any]],
        data_period: Optional[Tuple[datetime, datetime]] = None
    ) -> TestResult:
        """Run trade comparison test"""
        
        start_time = datetime.now()
        
        try:
            # Align trades
            aligned_trades = self._align_trades(baseline_trades, current_trades)
            
            comparison_results = []
            overall_status = TestStatus.PASS
            warnings = []
            errors = []
            
            for trade_id, (baseline_trade, current_trade) in aligned_trades.items():
                trade_comparison = self._compare_trade(baseline_trade, current_trade)
                comparison_results.append(trade_comparison)
                
                if trade_comparison['status'] == 'fail':
                    overall_status = TestStatus.FAIL
                    errors.append(f"Trade {trade_id} failed comparison")
                elif trade_comparison['status'] == 'warning':
                    if overall_status == TestStatus.PASS:
                        overall_status = TestStatus.WARNING
                    warnings.append(f"Trade {trade_id} has warnings")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_id=test_id,
                test_name="Trade Comparison Test",
                test_type=TestType.TRADE_COMPARISON,
                status=overall_status,
                timestamp=start_time,
                strategy_id=strategy_id,
                data_period=data_period,
                message=f"Compared {len(comparison_results)} trades",
                details={'trade_comparisons': comparison_results},
                warnings=warnings,
                errors=errors,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Trade comparison test failed: {e}")
            return TestResult(
                test_id=test_id,
                test_name="Trade Comparison Test",
                test_type=TestType.TRADE_COMPARISON,
                status=TestStatus.ERROR,
                timestamp=start_time,
                strategy_id=strategy_id,
                data_period=data_period,
                message=f"Test failed with error: {str(e)}",
                errors=[str(e)],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def run_performance_benchmark_test(
        self,
        test_id: str,
        strategy_id: str,
        current_metrics: Dict[str, float],
        benchmark_metrics: Dict[str, float],
        data_period: Optional[Tuple[datetime, datetime]] = None
    ) -> TestResult:
        """Run performance benchmark test"""
        
        start_time = datetime.now()
        
        try:
            # Compare against benchmark
            benchmark_results = []
            overall_status = TestStatus.PASS
            warnings = []
            errors = []
            
            for metric_name, current_value in current_metrics.items():
                if metric_name in benchmark_metrics:
                    benchmark_value = benchmark_metrics[metric_name]
                    
                    # Calculate performance relative to benchmark
                    if benchmark_value != 0:
                        performance_ratio = current_value / benchmark_value
                    else:
                        performance_ratio = float('inf') if current_value > 0 else 0.0
                    
                    # Determine if performance meets benchmark
                    if performance_ratio >= 0.95:  # Within 5% of benchmark
                        status = TestStatus.PASS
                    elif performance_ratio >= 0.90:  # Within 10% of benchmark
                        status = TestStatus.WARNING
                        warnings.append(f"Metric {metric_name} below benchmark: {performance_ratio:.3f}")
                    else:
                        status = TestStatus.FAIL
                        errors.append(f"Metric {metric_name} significantly below benchmark: {performance_ratio:.3f}")
                    
                    benchmark_results.append({
                        'metric_name': metric_name,
                        'current_value': current_value,
                        'benchmark_value': benchmark_value,
                        'performance_ratio': performance_ratio,
                        'status': status.value
                    })
                    
                    if status == TestStatus.FAIL:
                        overall_status = TestStatus.FAIL
                    elif status == TestStatus.WARNING and overall_status == TestStatus.PASS:
                        overall_status = TestStatus.WARNING
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_id=test_id,
                test_name="Performance Benchmark Test",
                test_type=TestType.PERFORMANCE_BENCHMARK,
                status=overall_status,
                timestamp=start_time,
                strategy_id=strategy_id,
                data_period=data_period,
                message=f"Benchmarked {len(benchmark_results)} metrics",
                details={'benchmark_results': benchmark_results},
                warnings=warnings,
                errors=errors,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Performance benchmark test failed: {e}")
            return TestResult(
                test_id=test_id,
                test_name="Performance Benchmark Test",
                test_type=TestType.PERFORMANCE_BENCHMARK,
                status=TestStatus.ERROR,
                timestamp=start_time,
                strategy_id=strategy_id,
                data_period=data_period,
                message=f"Test failed with error: {str(e)}",
                errors=[str(e)],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def save_baseline(self, strategy_id: str, data_period: Tuple[datetime, datetime], 
                     metrics: Dict[str, float], trades: List[Dict[str, Any]]) -> str:
        """Save baseline data for regression testing"""
        
        baseline_id = self._generate_baseline_id(strategy_id, data_period)
        baseline_data = {
            'strategy_id': strategy_id,
            'data_period': data_period,
            'metrics': metrics,
            'trades': trades,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        baseline_path = self.baseline_storage / f"{baseline_id}.json"
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved baseline for strategy {strategy_id}: {baseline_path}")
        return baseline_id
    
    def load_baseline(self, baseline_id: str) -> Optional[Dict[str, Any]]:
        """Load baseline data for regression testing"""
        
        baseline_path = self.baseline_storage / f"{baseline_id}.json"
        
        if not baseline_path.exists():
            self.logger.warning(f"Baseline not found: {baseline_path}")
            return None
        
        try:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            
            # Convert data_period back to datetime tuples
            if 'data_period' in baseline_data and isinstance(baseline_data['data_period'], list):
                baseline_data['data_period'] = (
                    datetime.fromisoformat(baseline_data['data_period'][0]),
                    datetime.fromisoformat(baseline_data['data_period'][1])
                )
            
            return baseline_data
            
        except Exception as e:
            self.logger.error(f"Failed to load baseline {baseline_id}: {e}")
            return None
    
    def _run_strategy_tests(self, strategy_id: str, data_period: Optional[Tuple[datetime, datetime]], 
                          configuration: Dict[str, Any]) -> List[TestResult]:
        """Run all tests for a specific strategy"""
        
        tests = []
        
        # Generate test ID
        test_id_base = f"{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load baseline if available
        baseline_id = self._generate_baseline_id(strategy_id, data_period)
        baseline_data = self.load_baseline(baseline_id)
        
        if baseline_data and self.config.enable_metric_comparison:
            # Run metric comparison test
            test_id = f"{test_id_base}_metric_comparison"
            test_result = self.run_metric_comparison_test(
                test_id, strategy_id, baseline_data['metrics'], 
                configuration.get('metrics', {}), data_period
            )
            tests.append(test_result)
        
        if baseline_data and self.config.enable_trade_comparison:
            # Run trade comparison test
            test_id = f"{test_id_base}_trade_comparison"
            test_result = self.run_trade_comparison_test(
                test_id, strategy_id, baseline_data['trades'], 
                configuration.get('trades', []), data_period
            )
            tests.append(test_result)
        
        if self.config.enable_performance_benchmark:
            # Run performance benchmark test
            test_id = f"{test_id_base}_performance_benchmark"
            benchmark_metrics = self._get_benchmark_metrics(strategy_id)
            test_result = self.run_performance_benchmark_test(
                test_id, strategy_id, configuration.get('metrics', {}), 
                benchmark_metrics, data_period
            )
            tests.append(test_result)
        
        return tests
    
    def _align_trades(self, trades_a: List[Dict[str, Any]], 
                     trades_b: List[Dict[str, Any]]) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Align trades between two sets"""
        
        aligned = {}
        
        # Try to align by trade ID first
        trades_a_by_id = {trade.get('trade_id', f"trade_{i}"): trade for i, trade in enumerate(trades_a)}
        trades_b_by_id = {trade.get('trade_id', f"trade_{i}"): trade for i, trade in enumerate(trades_b)}
        
        common_ids = set(trades_a_by_id.keys()) & set(trades_b_by_id.keys())
        
        for trade_id in common_ids:
            aligned[trade_id] = (trades_a_by_id[trade_id], trades_b_by_id[trade_id])
        
        # If no common IDs, align by index
        if not aligned and len(trades_a) == len(trades_b):
            for i, (trade_a, trade_b) in enumerate(zip(trades_a, trades_b)):
                aligned[f"trade_{i}"] = (trade_a, trade_b)
        
        return aligned
    
    def _compare_trade(self, baseline_trade: Dict[str, Any], 
                      current_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Compare individual trade"""
        
        comparison = {
            'trade_id': baseline_trade.get('trade_id', 'unknown'),
            'status': 'pass',
            'differences': {},
            'warnings': []
        }
        
        # Compare common fields
        common_fields = set(baseline_trade.keys()) & set(current_trade.keys())
        
        for field in common_fields:
            baseline_value = baseline_trade[field]
            current_value = current_trade[field]
            
            if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                # Numeric comparison
                tolerance = self.config.tolerance_thresholds.get_tolerance(field)
                difference = abs(current_value - baseline_value)
                relative_difference = difference / abs(baseline_value) if baseline_value != 0 else float('inf')
                
                comparison['differences'][field] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'difference': difference,
                    'relative_difference': relative_difference,
                    'tolerance': tolerance
                }
                
                if relative_difference > tolerance * 2:
                    comparison['status'] = 'fail'
                elif relative_difference > tolerance:
                    comparison['status'] = 'warning'
                    comparison['warnings'].append(f"Field {field} exceeds tolerance")
            
            elif field in ['entry_time', 'exit_time']:
                # Time comparison
                try:
                    time_a = pd.to_datetime(baseline_value)
                    time_b = pd.to_datetime(current_value)
                    time_diff = abs((time_a - time_b).total_seconds())
                    
                    comparison['differences'][field] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'difference_seconds': time_diff
                    }
                    
                    if time_diff > self.config.tolerance_thresholds.entry_time_tolerance:
                        comparison['status'] = 'warning'
                        comparison['warnings'].append(f"Time field {field} exceeds tolerance")
                except:
                    comparison['status'] = 'fail'
                    comparison['warnings'].append(f"Invalid time format in field {field}")
        
        return comparison
    
    def _generate_baseline_id(self, strategy_id: str, data_period: Optional[Tuple[datetime, datetime]]) -> str:
        """Generate unique baseline ID"""
        
        if data_period:
            period_str = f"{data_period[0].strftime('%Y%m%d')}_{data_period[1].strftime('%Y%m%d')}"
        else:
            period_str = "unknown_period"
        
        content = f"{strategy_id}_{period_str}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_benchmark_metrics(self, strategy_id: str) -> Dict[str, float]:
        """Get benchmark metrics for strategy"""
        
        # This would typically load from a benchmark database or configuration
        # For now, return default benchmarks
        return {
            'total_return': 0.10,      # 10% annual return
            'sharpe_ratio': 1.0,       # Sharpe ratio of 1.0
            'max_drawdown': 0.05,      # 5% max drawdown
            'profit_factor': 1.5,      # Profit factor of 1.5
            'win_rate': 0.55,          # 55% win rate
            'volatility': 0.15         # 15% volatility
        }
    
    def _generate_test_report(self, test_suite: TestSuite):
        """Generate detailed test report"""
        
        report_data = {
            'suite_info': {
                'suite_id': test_suite.suite_id,
                'suite_name': test_suite.suite_name,
                'description': test_suite.description,
                'created_at': test_suite.created_at.isoformat(),
                'updated_at': test_suite.updated_at.isoformat()
            },
            'test_summary': {
                'total_tests': test_suite.total_tests,
                'passed_tests': test_suite.passed_tests,
                'failed_tests': test_suite.failed_tests,
                'warning_tests': test_suite.warning_tests,
                'error_tests': test_suite.error_tests,
                'skipped_tests': test_suite.skipped_tests,
                'success_rate': test_suite.success_rate,
                'overall_status': test_suite.overall_status.value
            },
            'test_results': [result.to_dict() for result in test_suite.test_results]
        }
        
        # Save report
        report_filename = f"test_report_{test_suite.suite_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.report_storage / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Generated test report: {report_path}")
