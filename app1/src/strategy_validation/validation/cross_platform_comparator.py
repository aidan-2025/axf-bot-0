"""
Cross-Platform Comparator

Compares backtest results between different platforms (e.g., Backtrader vs MT4)
to ensure consistency and identify discrepancies.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json
from scipy import stats
from scipy.stats import ks_2samp, pearsonr

logger = logging.getLogger(__name__)


class ComparisonStatus(Enum):
    """Status of comparison results"""
    EXCELLENT = "excellent"  # <1% difference
    GOOD = "good"           # 1-3% difference
    ACCEPTABLE = "acceptable"  # 3-5% difference
    POOR = "poor"           # >5% difference
    FAILED = "failed"       # Critical differences


@dataclass
class MetricComparison:
    """Comparison result for a specific metric"""
    
    metric_name: str
    platform_a_value: float
    platform_b_value: float
    absolute_difference: float
    relative_difference: float
    tolerance_threshold: float
    status: ComparisonStatus
    significance_level: float = 0.05
    is_significant: bool = False
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.absolute_difference = abs(self.platform_a_value - self.platform_b_value)
        
        if self.platform_b_value != 0:
            self.relative_difference = self.absolute_difference / abs(self.platform_b_value)
        else:
            self.relative_difference = float('inf') if self.absolute_difference > 0 else 0.0
        
        # Determine status based on relative difference
        if self.relative_difference <= 0.01:
            self.status = ComparisonStatus.EXCELLENT
        elif self.relative_difference <= 0.03:
            self.status = ComparisonStatus.GOOD
        elif self.relative_difference <= 0.05:
            self.status = ComparisonStatus.ACCEPTABLE
        elif self.relative_difference <= 0.10:
            self.status = ComparisonStatus.POOR
        else:
            self.status = ComparisonStatus.FAILED


@dataclass
class TradeComparison:
    """Comparison result for trade-level analysis"""
    
    trade_id: str
    platform_a_trade: Dict[str, Any]
    platform_b_trade: Dict[str, Any]
    differences: Dict[str, float]
    status: ComparisonStatus
    issues: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate trade differences"""
        self.differences = {}
        
        # Compare common fields
        common_fields = set(self.platform_a_trade.keys()) & set(self.platform_b_trade.keys())
        
        for field in common_fields:
            if field in ['entry_time', 'exit_time']:
                # Handle datetime fields
                try:
                    time_a = pd.to_datetime(self.platform_a_trade[field])
                    time_b = pd.to_datetime(self.platform_b_trade[field])
                    diff_seconds = abs((time_a - time_b).total_seconds())
                    self.differences[field] = diff_seconds
                except:
                    self.differences[field] = float('inf')
            else:
                # Handle numeric fields
                try:
                    val_a = float(self.platform_a_trade[field])
                    val_b = float(self.platform_b_trade[field])
                    self.differences[field] = abs(val_a - val_b)
                except:
                    self.differences[field] = float('inf')
        
        # Determine overall status
        self._determine_status()
    
    def _determine_status(self):
        """Determine comparison status based on differences"""
        max_diff = max(self.differences.values()) if self.differences else 0
        
        if max_diff == 0:
            self.status = ComparisonStatus.EXCELLENT
        elif max_diff <= 0.001:  # 0.1%
            self.status = ComparisonStatus.GOOD
        elif max_diff <= 0.005:  # 0.5%
            self.status = ComparisonStatus.ACCEPTABLE
        elif max_diff <= 0.01:   # 1%
            self.status = ComparisonStatus.POOR
        else:
            self.status = ComparisonStatus.FAILED


@dataclass
class ComparisonResult:
    """Comprehensive comparison result"""
    
    comparison_timestamp: datetime
    strategy_id: str
    platform_a_name: str
    platform_b_name: str
    data_period: Tuple[datetime, datetime]
    
    # Metric comparisons
    metric_comparisons: List[MetricComparison] = field(default_factory=list)
    
    # Trade comparisons
    trade_comparisons: List[TradeComparison] = field(default_factory=list)
    
    # Overall statistics
    overall_status: ComparisonStatus = ComparisonStatus.FAILED
    overall_agreement: float = 0.0
    critical_differences: List[str] = field(default_factory=list)
    
    # Statistical tests
    correlation_analysis: Dict[str, float] = field(default_factory=dict)
    distribution_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'comparison_timestamp': self.comparison_timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'platform_a_name': self.platform_a_name,
            'platform_b_name': self.platform_b_name,
            'data_period': {
                'start': self.data_period[0].isoformat(),
                'end': self.data_period[1].isoformat()
            },
            'metric_comparisons': [
                {
                    'metric_name': mc.metric_name,
                    'platform_a_value': mc.platform_a_value,
                    'platform_b_value': mc.platform_b_value,
                    'absolute_difference': mc.absolute_difference,
                    'relative_difference': mc.relative_difference,
                    'tolerance_threshold': mc.tolerance_threshold,
                    'status': mc.status.value,
                    'is_significant': mc.is_significant
                }
                for mc in self.metric_comparisons
            ],
            'trade_comparisons': [
                {
                    'trade_id': tc.trade_id,
                    'differences': tc.differences,
                    'status': tc.status.value,
                    'issues': tc.issues
                }
                for tc in self.trade_comparisons
            ],
            'overall_status': self.overall_status.value,
            'overall_agreement': self.overall_agreement,
            'critical_differences': self.critical_differences,
            'correlation_analysis': self.correlation_analysis,
            'distribution_tests': self.distribution_tests,
            'summary': self.summary
        }


@dataclass
class ComparisonConfig:
    """Configuration for cross-platform comparison"""
    
    # Tolerance thresholds
    excellent_threshold: float = 0.01    # 1%
    good_threshold: float = 0.03         # 3%
    acceptable_threshold: float = 0.05   # 5%
    poor_threshold: float = 0.10         # 10%
    
    # Metric-specific tolerances
    metric_tolerances: Dict[str, float] = field(default_factory=lambda: {
        'total_return': 0.02,      # 2%
        'sharpe_ratio': 0.05,      # 5%
        'max_drawdown': 0.03,      # 3%
        'profit_factor': 0.05,     # 5%
        'win_rate': 0.02,          # 2%
        'total_trades': 0.01,      # 1%
        'avg_trade_duration': 0.05, # 5%
        'volatility': 0.03,        # 3%
        'skewness': 0.10,          # 10%
        'kurtosis': 0.20           # 20%
    })
    
    # Trade-level tolerances
    trade_tolerances: Dict[str, float] = field(default_factory=lambda: {
        'entry_price': 0.0001,     # 0.01 pips
        'exit_price': 0.0001,      # 0.01 pips
        'pnl': 0.001,              # 0.1 pips
        'commission': 0.0001,      # 0.01 pips
        'slippage': 0.0001,        # 0.01 pips
        'entry_time': 60,          # 60 seconds
        'exit_time': 60            # 60 seconds
    })
    
    # Statistical test settings
    significance_level: float = 0.05
    enable_correlation_tests: bool = True
    enable_distribution_tests: bool = True
    enable_trade_level_analysis: bool = True
    
    # Data alignment settings
    max_time_difference_seconds: int = 300  # 5 minutes
    price_tolerance_pips: float = 0.1      # 0.1 pips
    enable_data_alignment: bool = True


class CrossPlatformComparator:
    """Compares backtest results between different platforms"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compare_results(
        self,
        strategy_id: str,
        platform_a_name: str,
        platform_b_name: str,
        platform_a_results: Dict[str, Any],
        platform_b_results: Dict[str, Any],
        platform_a_trades: Optional[List[Dict[str, Any]]] = None,
        platform_b_trades: Optional[List[Dict[str, Any]]] = None,
        data_period: Optional[Tuple[datetime, datetime]] = None
    ) -> ComparisonResult:
        """Compare results between two platforms"""
        
        self.logger.info(f"Comparing results for strategy {strategy_id} between {platform_a_name} and {platform_b_name}")
        
        # Compare metrics
        metric_comparisons = self._compare_metrics(platform_a_results, platform_b_results)
        
        # Compare trades if available
        trade_comparisons = []
        if platform_a_trades and platform_b_trades:
            trade_comparisons = self._compare_trades(platform_a_trades, platform_b_trades)
        
        # Perform statistical analysis
        correlation_analysis = self._analyze_correlations(platform_a_results, platform_b_results)
        distribution_tests = self._perform_distribution_tests(platform_a_results, platform_b_results)
        
        # Calculate overall agreement
        overall_agreement = self._calculate_overall_agreement(metric_comparisons, trade_comparisons)
        
        # Determine overall status
        overall_status = self._determine_overall_status(metric_comparisons, trade_comparisons)
        
        # Identify critical differences
        critical_differences = self._identify_critical_differences(metric_comparisons, trade_comparisons)
        
        # Generate summary
        summary = self._generate_summary(
            metric_comparisons, trade_comparisons, correlation_analysis, 
            distribution_tests, overall_agreement, overall_status
        )
        
        return ComparisonResult(
            comparison_timestamp=datetime.now(),
            strategy_id=strategy_id,
            platform_a_name=platform_a_name,
            platform_b_name=platform_b_name,
            data_period=data_period or (datetime.now() - timedelta(days=30), datetime.now()),
            metric_comparisons=metric_comparisons,
            trade_comparisons=trade_comparisons,
            overall_status=overall_status,
            overall_agreement=overall_agreement,
            critical_differences=critical_differences,
            correlation_analysis=correlation_analysis,
            distribution_tests=distribution_tests,
            summary=summary
        )
    
    def _compare_metrics(self, results_a: Dict[str, Any], results_b: Dict[str, Any]) -> List[MetricComparison]:
        """Compare metrics between two platforms"""
        
        comparisons = []
        
        # Define metrics to compare
        metrics_to_compare = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor',
            'win_rate', 'total_trades', 'avg_trade_duration', 'volatility',
            'skewness', 'kurtosis', 'calmar_ratio', 'sortino_ratio'
        ]
        
        for metric in metrics_to_compare:
            if metric in results_a and metric in results_b:
                value_a = float(results_a[metric])
                value_b = float(results_b[metric])
                tolerance = self.config.metric_tolerances.get(metric, 0.05)
                
                comparison = MetricComparison(
                    metric_name=metric,
                    platform_a_value=value_a,
                    platform_b_value=value_b,
                    absolute_difference=0.0,  # Will be calculated in __post_init__
                    relative_difference=0.0,  # Will be calculated in __post_init__
                    tolerance_threshold=tolerance,
                    status=ComparisonStatus.FAILED  # Will be determined in __post_init__
                )
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _compare_trades(self, trades_a: List[Dict[str, Any]], 
                       trades_b: List[Dict[str, Any]]) -> List[TradeComparison]:
        """Compare individual trades between platforms"""
        
        comparisons = []
        
        # Align trades by ID or timestamp
        aligned_trades = self._align_trades(trades_a, trades_b)
        
        for trade_id, (trade_a, trade_b) in aligned_trades.items():
            comparison = TradeComparison(
                trade_id=trade_id,
                platform_a_trade=trade_a,
                platform_b_trade=trade_b,
                differences={},  # Will be calculated in __post_init__
                status=ComparisonStatus.FAILED  # Will be determined in __post_init__
            )
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _align_trades(self, trades_a: List[Dict[str, Any]], 
                     trades_b: List[Dict[str, Any]]) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Align trades between platforms by ID or timestamp"""
        
        aligned = {}
        
        # First try to align by trade ID
        trades_a_by_id = {trade.get('trade_id', f"trade_{i}"): trade for i, trade in enumerate(trades_a)}
        trades_b_by_id = {trade.get('trade_id', f"trade_{i}"): trade for i, trade in enumerate(trades_b)}
        
        common_ids = set(trades_a_by_id.keys()) & set(trades_b_by_id.keys())
        
        for trade_id in common_ids:
            aligned[trade_id] = (trades_a_by_id[trade_id], trades_b_by_id[trade_id])
        
        # If no common IDs, try to align by timestamp
        if not aligned and len(trades_a) == len(trades_b):
            for i, (trade_a, trade_b) in enumerate(zip(trades_a, trades_b)):
                aligned[f"trade_{i}"] = (trade_a, trade_b)
        
        return aligned
    
    def _analyze_correlations(self, results_a: Dict[str, Any], 
                            results_b: Dict[str, Any]) -> Dict[str, float]:
        """Analyze correlations between platform results"""
        
        correlations = {}
        
        # Extract numeric values for correlation analysis
        numeric_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
        
        values_a = []
        values_b = []
        
        for metric in numeric_metrics:
            if metric in results_a and metric in results_b:
                try:
                    values_a.append(float(results_a[metric]))
                    values_b.append(float(results_b[metric]))
                except (ValueError, TypeError):
                    continue
        
        if len(values_a) >= 2 and len(values_b) >= 2:
            try:
                correlation, p_value = pearsonr(values_a, values_b)
                correlations['pearson_correlation'] = correlation
                correlations['p_value'] = p_value
                correlations['is_significant'] = p_value < self.config.significance_level
            except:
                correlations['pearson_correlation'] = 0.0
                correlations['p_value'] = 1.0
                correlations['is_significant'] = False
        
        return correlations
    
    def _perform_distribution_tests(self, results_a: Dict[str, Any], 
                                  results_b: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Perform statistical distribution tests"""
        
        tests = {}
        
        # Extract return data if available
        returns_a = results_a.get('returns', [])
        returns_b = results_b.get('returns', [])
        
        if returns_a and returns_b and len(returns_a) > 10 and len(returns_b) > 10:
            try:
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = ks_2samp(returns_a, returns_b)
                tests['kolmogorov_smirnov'] = {
                    'statistic': ks_statistic,
                    'p_value': ks_p_value,
                    'is_significant': ks_p_value < self.config.significance_level
                }
                
                # T-test for means
                t_statistic, t_p_value = stats.ttest_ind(returns_a, returns_b)
                tests['t_test'] = {
                    'statistic': t_statistic,
                    'p_value': t_p_value,
                    'is_significant': t_p_value < self.config.significance_level
                }
                
                # Mann-Whitney U test (non-parametric)
                u_statistic, u_p_value = stats.mannwhitneyu(returns_a, returns_b, alternative='two-sided')
                tests['mann_whitney_u'] = {
                    'statistic': u_statistic,
                    'p_value': u_p_value,
                    'is_significant': u_p_value < self.config.significance_level
                }
                
            except Exception as e:
                self.logger.warning(f"Distribution tests failed: {e}")
                tests['error'] = str(e)
        
        return tests
    
    def _calculate_overall_agreement(self, metric_comparisons: List[MetricComparison], 
                                   trade_comparisons: List[TradeComparison]) -> float:
        """Calculate overall agreement score"""
        
        if not metric_comparisons and not trade_comparisons:
            return 0.0
        
        # Weight metric comparisons more heavily
        metric_weight = 0.7
        trade_weight = 0.3
        
        # Calculate metric agreement
        metric_agreement = 0.0
        if metric_comparisons:
            status_scores = {
                ComparisonStatus.EXCELLENT: 1.0,
                ComparisonStatus.GOOD: 0.8,
                ComparisonStatus.ACCEPTABLE: 0.6,
                ComparisonStatus.POOR: 0.3,
                ComparisonStatus.FAILED: 0.0
            }
            
            metric_scores = [status_scores[mc.status] for mc in metric_comparisons]
            metric_agreement = np.mean(metric_scores)
        
        # Calculate trade agreement
        trade_agreement = 0.0
        if trade_comparisons:
            status_scores = {
                ComparisonStatus.EXCELLENT: 1.0,
                ComparisonStatus.GOOD: 0.8,
                ComparisonStatus.ACCEPTABLE: 0.6,
                ComparisonStatus.POOR: 0.3,
                ComparisonStatus.FAILED: 0.0
            }
            
            trade_scores = [status_scores[tc.status] for tc in trade_comparisons]
            trade_agreement = np.mean(trade_scores)
        
        # Calculate weighted overall agreement
        if metric_comparisons and trade_comparisons:
            overall_agreement = (metric_weight * metric_agreement + 
                               trade_weight * trade_agreement)
        elif metric_comparisons:
            overall_agreement = metric_agreement
        elif trade_comparisons:
            overall_agreement = trade_agreement
        else:
            overall_agreement = 0.0
        
        return overall_agreement
    
    def _determine_overall_status(self, metric_comparisons: List[MetricComparison], 
                                trade_comparisons: List[TradeComparison]) -> ComparisonStatus:
        """Determine overall comparison status"""
        
        # Collect all statuses
        all_statuses = []
        all_statuses.extend([mc.status for mc in metric_comparisons])
        all_statuses.extend([tc.status for tc in trade_comparisons])
        
        if not all_statuses:
            return ComparisonStatus.FAILED
        
        # Count status occurrences
        status_counts = {}
        for status in all_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status based on worst status present
        if ComparisonStatus.FAILED in status_counts:
            return ComparisonStatus.FAILED
        elif ComparisonStatus.POOR in status_counts:
            return ComparisonStatus.POOR
        elif ComparisonStatus.ACCEPTABLE in status_counts:
            return ComparisonStatus.ACCEPTABLE
        elif ComparisonStatus.GOOD in status_counts:
            return ComparisonStatus.GOOD
        else:
            return ComparisonStatus.EXCELLENT
    
    def _identify_critical_differences(self, metric_comparisons: List[MetricComparison], 
                                     trade_comparisons: List[TradeComparison]) -> List[str]:
        """Identify critical differences that need attention"""
        
        critical_differences = []
        
        # Check metric comparisons
        for mc in metric_comparisons:
            if mc.status in [ComparisonStatus.FAILED, ComparisonStatus.POOR]:
                critical_differences.append(
                    f"Critical difference in {mc.metric_name}: "
                    f"{mc.relative_difference:.3f} relative difference "
                    f"(threshold: {mc.tolerance_threshold:.3f})"
                )
        
        # Check trade comparisons
        for tc in trade_comparisons:
            if tc.status in [ComparisonStatus.FAILED, ComparisonStatus.POOR]:
                critical_differences.append(
                    f"Critical difference in trade {tc.trade_id}: "
                    f"Status {tc.status.value}"
                )
        
        return critical_differences
    
    def _generate_summary(self, metric_comparisons: List[MetricComparison], 
                         trade_comparisons: List[TradeComparison],
                         correlation_analysis: Dict[str, float],
                         distribution_tests: Dict[str, Dict[str, Any]],
                         overall_agreement: float,
                         overall_status: ComparisonStatus) -> Dict[str, Any]:
        """Generate comparison summary"""
        
        # Count statuses
        metric_status_counts = {}
        for mc in metric_comparisons:
            metric_status_counts[mc.status] = metric_status_counts.get(mc.status, 0) + 1
        
        trade_status_counts = {}
        for tc in trade_comparisons:
            trade_status_counts[tc.status] = trade_status_counts.get(tc.status, 0) + 1
        
        # Calculate average differences
        avg_relative_difference = 0.0
        if metric_comparisons:
            avg_relative_difference = np.mean([mc.relative_difference for mc in metric_comparisons])
        
        # Determine if results are statistically significant
        is_statistically_significant = False
        if 'is_significant' in correlation_analysis:
            is_statistically_significant = correlation_analysis['is_significant']
        
        return {
            'total_metrics_compared': len(metric_comparisons),
            'total_trades_compared': len(trade_comparisons),
            'metric_status_counts': {status.value: count for status, count in metric_status_counts.items()},
            'trade_status_counts': {status.value: count for status, count in trade_status_counts.items()},
            'average_relative_difference': avg_relative_difference,
            'overall_agreement': overall_agreement,
            'overall_status': overall_status.value,
            'is_statistically_significant': is_statistically_significant,
            'correlation_strength': correlation_analysis.get('pearson_correlation', 0.0),
            'distribution_test_results': len(distribution_tests) > 0
        }

