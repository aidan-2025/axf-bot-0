"""
Modeling Quality Validator

Validates the quality and accuracy of backtesting models by analyzing data integrity,
execution logic, and statistical consistency.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality levels for modeling validation"""
    EXCELLENT = "excellent"  # 99%+ quality
    GOOD = "good"           # 95-99% quality
    ACCEPTABLE = "acceptable"  # 90-95% quality
    POOR = "poor"           # <90% quality


@dataclass
class QualityMetrics:
    """Metrics for assessing modeling quality"""
    
    # Data Quality Metrics
    data_completeness: float = 0.0  # Percentage of expected data points present
    data_accuracy: float = 0.0      # Accuracy of price data vs reference
    spread_accuracy: float = 0.0    # Accuracy of spread simulation
    timestamp_accuracy: float = 0.0 # Accuracy of timestamp alignment
    
    # Execution Quality Metrics
    order_execution_accuracy: float = 0.0  # Accuracy of order execution logic
    slippage_modeling_accuracy: float = 0.0  # Accuracy of slippage modeling
    commission_calculation_accuracy: float = 0.0  # Accuracy of commission calculations
    
    # Statistical Quality Metrics
    return_distribution_accuracy: float = 0.0  # Accuracy of return distributions
    volatility_modeling_accuracy: float = 0.0  # Accuracy of volatility modeling
    correlation_accuracy: float = 0.0  # Accuracy of price correlations
    
    # Overall Quality Score
    overall_quality_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.POOR
    
    def calculate_overall_score(self) -> float:
        """Calculate overall quality score as weighted average"""
        weights = {
            'data_completeness': 0.20,
            'data_accuracy': 0.15,
            'spread_accuracy': 0.15,
            'timestamp_accuracy': 0.10,
            'order_execution_accuracy': 0.15,
            'slippage_modeling_accuracy': 0.10,
            'commission_calculation_accuracy': 0.05,
            'return_distribution_accuracy': 0.05,
            'volatility_modeling_accuracy': 0.03,
            'correlation_accuracy': 0.02
        }
        
        score = sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
        )
        
        self.overall_quality_score = score
        
        # Determine quality level
        if score >= 0.99:
            self.quality_level = QualityLevel.EXCELLENT
        elif score >= 0.95:
            self.quality_level = QualityLevel.GOOD
        elif score >= 0.90:
            self.quality_level = QualityLevel.ACCEPTABLE
        else:
            self.quality_level = QualityLevel.POOR
            
        return score


@dataclass
class QualityReport:
    """Comprehensive quality validation report"""
    
    validation_timestamp: datetime
    strategy_id: str
    data_period: Tuple[datetime, datetime]
    
    # Quality metrics
    metrics: QualityMetrics
    
    # Data integrity checks
    data_integrity_checks: Dict[str, bool] = field(default_factory=dict)
    
    # Issues and warnings
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Validation details
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'data_period': {
                'start': self.data_period[0].isoformat(),
                'end': self.data_period[1].isoformat()
            },
            'metrics': {
                'data_completeness': self.metrics.data_completeness,
                'data_accuracy': self.metrics.data_accuracy,
                'spread_accuracy': self.metrics.spread_accuracy,
                'timestamp_accuracy': self.metrics.timestamp_accuracy,
                'order_execution_accuracy': self.metrics.order_execution_accuracy,
                'slippage_modeling_accuracy': self.metrics.slippage_modeling_accuracy,
                'commission_calculation_accuracy': self.metrics.commission_calculation_accuracy,
                'return_distribution_accuracy': self.metrics.return_distribution_accuracy,
                'volatility_modeling_accuracy': self.metrics.volatility_modeling_accuracy,
                'correlation_accuracy': self.metrics.correlation_accuracy,
                'overall_quality_score': self.metrics.overall_quality_score,
                'quality_level': self.metrics.quality_level.value
            },
            'data_integrity_checks': self.data_integrity_checks,
            'issues': self.issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'validation_details': self.validation_details
        }


@dataclass
class ModelingQualityConfig:
    """Configuration for modeling quality validation"""
    
    # Data validation settings
    max_data_gap_seconds: int = 300  # Maximum allowed gap in seconds
    price_tolerance_pips: float = 0.1  # Price tolerance in pips
    spread_tolerance_pips: float = 0.05  # Spread tolerance in pips
    timestamp_tolerance_seconds: int = 1  # Timestamp tolerance in seconds
    
    # Execution validation settings
    order_execution_tolerance: float = 0.01  # Order execution tolerance
    slippage_tolerance: float = 0.001  # Slippage modeling tolerance
    commission_tolerance: float = 0.0001  # Commission calculation tolerance
    
    # Statistical validation settings
    distribution_test_alpha: float = 0.05  # Significance level for distribution tests
    correlation_threshold: float = 0.95  # Minimum correlation threshold
    volatility_tolerance: float = 0.05  # Volatility modeling tolerance
    
    # Quality thresholds
    excellent_threshold: float = 0.99
    good_threshold: float = 0.95
    acceptable_threshold: float = 0.90
    
    # Reference data settings
    reference_data_source: str = "mt4"  # Reference data source for comparison
    enable_checksum_validation: bool = True
    enable_statistical_tests: bool = True


class ModelingQualityValidator:
    """Validates modeling quality and consistency"""
    
    def __init__(self, config: ModelingQualityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def validate_data_quality(
        self, 
        backtest_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None
    ) -> QualityMetrics:
        """Validate data quality metrics"""
        
        metrics = QualityMetrics()
        
        # Data completeness check
        metrics.data_completeness = self._check_data_completeness(backtest_data)
        
        # Data accuracy check (if reference data provided)
        if reference_data is not None:
            metrics.data_accuracy = self._check_data_accuracy(backtest_data, reference_data)
            metrics.spread_accuracy = self._check_spread_accuracy(backtest_data, reference_data)
            metrics.timestamp_accuracy = self._check_timestamp_accuracy(backtest_data, reference_data)
        else:
            # Use internal consistency checks
            metrics.data_accuracy = self._check_internal_data_consistency(backtest_data)
            metrics.spread_accuracy = self._check_spread_consistency(backtest_data)
            metrics.timestamp_accuracy = self._check_timestamp_consistency(backtest_data)
        
        # Calculate overall score
        metrics.calculate_overall_score()
        
        return metrics
    
    def validate_execution_quality(
        self,
        trade_log: pd.DataFrame,
        reference_trades: Optional[pd.DataFrame] = None
    ) -> QualityMetrics:
        """Validate execution quality metrics"""
        
        metrics = QualityMetrics()
        
        # Order execution accuracy
        metrics.order_execution_accuracy = self._check_order_execution_accuracy(trade_log)
        
        # Slippage modeling accuracy
        metrics.slippage_modeling_accuracy = self._check_slippage_modeling(trade_log)
        
        # Commission calculation accuracy
        metrics.commission_calculation_accuracy = self._check_commission_calculations(trade_log)
        
        # Calculate overall score
        metrics.calculate_overall_score()
        
        return metrics
    
    def validate_statistical_quality(
        self,
        backtest_results: Dict[str, Any],
        reference_results: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate statistical quality metrics"""
        
        metrics = QualityMetrics()
        
        # Return distribution accuracy
        metrics.return_distribution_accuracy = self._check_return_distributions(
            backtest_results, reference_results
        )
        
        # Volatility modeling accuracy
        metrics.volatility_modeling_accuracy = self._check_volatility_modeling(
            backtest_results, reference_results
        )
        
        # Correlation accuracy
        metrics.correlation_accuracy = self._check_correlation_accuracy(
            backtest_results, reference_results
        )
        
        # Calculate overall score
        metrics.calculate_overall_score()
        
        return metrics
    
    def generate_quality_report(
        self,
        strategy_id: str,
        data_period: Tuple[datetime, datetime],
        backtest_data: pd.DataFrame,
        trade_log: pd.DataFrame,
        backtest_results: Dict[str, Any],
        reference_data: Optional[pd.DataFrame] = None,
        reference_trades: Optional[pd.DataFrame] = None,
        reference_results: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """Generate comprehensive quality validation report"""
        
        # Validate data quality
        data_metrics = self.validate_data_quality(backtest_data, reference_data)
        
        # Validate execution quality
        execution_metrics = self.validate_execution_quality(trade_log, reference_trades)
        
        # Validate statistical quality
        statistical_metrics = self.validate_statistical_quality(backtest_results, reference_results)
        
        # Combine metrics
        combined_metrics = self._combine_metrics(data_metrics, execution_metrics, statistical_metrics)
        
        # Perform data integrity checks
        integrity_checks = self._perform_data_integrity_checks(backtest_data, trade_log)
        
        # Generate issues and warnings
        issues, warnings = self._identify_issues_and_warnings(combined_metrics, integrity_checks)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(combined_metrics, issues, warnings)
        
        # Create validation details
        validation_details = self._create_validation_details(
            backtest_data, trade_log, backtest_results
        )
        
        return QualityReport(
            validation_timestamp=datetime.now(),
            strategy_id=strategy_id,
            data_period=data_period,
            metrics=combined_metrics,
            data_integrity_checks=integrity_checks,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            validation_details=validation_details
        )
    
    def _check_data_completeness(self, data: pd.DataFrame) -> float:
        """Check data completeness percentage"""
        if data.empty:
            return 0.0
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        total_values = data.size
        
        completeness = 1.0 - (missing_values / total_values)
        
        # Check for data gaps
        if 'timestamp' in data.columns:
            time_diffs = data['timestamp'].diff()
            expected_interval = time_diffs.median()
            large_gaps = (time_diffs > expected_interval * 2).sum()
            gap_penalty = large_gaps / len(data) * 0.1
            completeness = max(0.0, completeness - gap_penalty)
        
        return min(1.0, completeness)
    
    def _check_data_accuracy(self, data: pd.DataFrame, reference: pd.DataFrame) -> float:
        """Check data accuracy against reference"""
        if reference.empty:
            return 0.0
        
        # Align data by timestamp
        common_timestamps = set(data['timestamp']) & set(reference['timestamp'])
        if not common_timestamps:
            return 0.0
        
        # Compare price data
        price_columns = ['open', 'high', 'low', 'close']
        accuracy_scores = []
        
        for col in price_columns:
            if col in data.columns and col in reference.columns:
                data_subset = data[data['timestamp'].isin(common_timestamps)][col]
                ref_subset = reference[reference['timestamp'].isin(common_timestamps)][col]
                
                if len(data_subset) > 0 and len(ref_subset) > 0:
                    # Calculate accuracy as 1 - (mean absolute error / mean reference value)
                    mae = np.mean(np.abs(data_subset.values - ref_subset.values))
                    mean_ref = np.mean(ref_subset.values)
                    accuracy = max(0.0, 1.0 - (mae / mean_ref))
                    accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.0
    
    def _check_spread_accuracy(self, data: pd.DataFrame, reference: pd.DataFrame) -> float:
        """Check spread accuracy against reference"""
        if 'spread' not in data.columns or 'spread' not in reference.columns:
            return 1.0  # No spread data to validate
        
        # Align data by timestamp
        common_timestamps = set(data['timestamp']) & set(reference['timestamp'])
        if not common_timestamps:
            return 0.0
        
        data_spreads = data[data['timestamp'].isin(common_timestamps)]['spread']
        ref_spreads = reference[reference['timestamp'].isin(common_timestamps)]['spread']
        
        if len(data_spreads) == 0 or len(ref_spreads) == 0:
            return 0.0
        
        # Calculate spread accuracy
        mae = np.mean(np.abs(data_spreads.values - ref_spreads.values))
        mean_ref = np.mean(ref_spreads.values)
        accuracy = max(0.0, 1.0 - (mae / mean_ref))
        
        return accuracy
    
    def _check_timestamp_accuracy(self, data: pd.DataFrame, reference: pd.DataFrame) -> float:
        """Check timestamp accuracy against reference"""
        if 'timestamp' not in data.columns or 'timestamp' not in reference.columns:
            return 1.0
        
        # Convert to datetime if needed
        data_timestamps = pd.to_datetime(data['timestamp'])
        ref_timestamps = pd.to_datetime(reference['timestamp'])
        
        # Find common timestamps
        data_set = set(data_timestamps)
        ref_set = set(ref_timestamps)
        common_timestamps = data_set & ref_set
        
        if not common_timestamps:
            return 0.0
        
        # Calculate timestamp accuracy
        accuracy = len(common_timestamps) / max(len(data_set), len(ref_set))
        
        return accuracy
    
    def _check_internal_data_consistency(self, data: pd.DataFrame) -> float:
        """Check internal data consistency"""
        if data.empty:
            return 0.0
        
        consistency_score = 1.0
        
        # Check OHLC consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            ).sum()
            
            ohlc_consistency = 1.0 - (invalid_ohlc / len(data))
            consistency_score = min(consistency_score, ohlc_consistency)
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                negative_prices = (data[col] <= 0).sum()
                price_consistency = 1.0 - (negative_prices / len(data))
                consistency_score = min(consistency_score, price_consistency)
                
                # Check for finite values
                finite_prices = data[col].notna().sum()
                finite_consistency = finite_prices / len(data)
                consistency_score = min(consistency_score, finite_consistency)
        
        return max(0.0, consistency_score)
    
    def _check_spread_consistency(self, data: pd.DataFrame) -> float:
        """Check spread consistency"""
        if 'spread' not in data.columns:
            return 1.0
        
        # Check for negative spreads
        negative_spreads = (data['spread'] < 0).sum()
        spread_consistency = 1.0 - (negative_spreads / len(data))
        
        # Check for unrealistic spreads (e.g., > 100 pips)
        unrealistic_spreads = (data['spread'] > 0.01).sum()  # 100 pips
        spread_realism = 1.0 - (unrealistic_spreads / len(data))
        
        return max(0.0, min(spread_consistency, spread_realism))
    
    def _check_timestamp_consistency(self, data: pd.DataFrame) -> float:
        """Check timestamp consistency"""
        if 'timestamp' not in data.columns:
            return 1.0
        
        # Convert to datetime
        timestamps = pd.to_datetime(data['timestamp'])
        
        # Check for duplicate timestamps
        duplicate_timestamps = timestamps.duplicated().sum()
        duplicate_consistency = 1.0 - (duplicate_timestamps / len(timestamps))
        
        # Check for chronological order
        is_sorted = timestamps.is_monotonic_increasing
        order_consistency = 1.0 if is_sorted else 0.5
        
        return max(0.0, min(duplicate_consistency, order_consistency))
    
    def _check_order_execution_accuracy(self, trade_log: pd.DataFrame) -> float:
        """Check order execution accuracy"""
        if trade_log.empty:
            return 1.0
        
        # Check for required columns
        required_columns = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl']
        if not all(col in trade_log.columns for col in required_columns):
            return 0.0
        
        # Check for logical consistency
        logical_issues = 0
        
        # Entry time should be before exit time
        if 'entry_time' in trade_log.columns and 'exit_time' in trade_log.columns:
            invalid_timing = (pd.to_datetime(trade_log['entry_time']) >= 
                            pd.to_datetime(trade_log['exit_time'])).sum()
            logical_issues += invalid_timing
        
        # PnL should be calculable from entry/exit prices
        if all(col in trade_log.columns for col in ['entry_price', 'exit_price', 'pnl']):
            calculated_pnl = trade_log['exit_price'] - trade_log['entry_price']
            pnl_accuracy = 1.0 - np.mean(np.abs(calculated_pnl - trade_log['pnl']) / 
                                        np.abs(trade_log['pnl'] + 1e-8))
            logical_issues += (1.0 - pnl_accuracy) * len(trade_log)
        
        accuracy = 1.0 - (logical_issues / len(trade_log))
        return max(0.0, accuracy)
    
    def _check_slippage_modeling(self, trade_log: pd.DataFrame) -> float:
        """Check slippage modeling accuracy"""
        if 'slippage' not in trade_log.columns:
            return 1.0  # No slippage data to validate
        
        # Check for realistic slippage values
        realistic_slippage = ((trade_log['slippage'] >= 0) & 
                            (trade_log['slippage'] <= 0.01)).sum()  # Max 100 pips
        accuracy = realistic_slippage / len(trade_log)
        
        return accuracy
    
    def _check_commission_calculations(self, trade_log: pd.DataFrame) -> float:
        """Check commission calculation accuracy"""
        if 'commission' not in trade_log.columns:
            return 1.0  # No commission data to validate
        
        # Check for realistic commission values
        realistic_commission = ((trade_log['commission'] >= 0) & 
                              (trade_log['commission'] <= 0.01)).sum()  # Max 100 pips
        accuracy = realistic_commission / len(trade_log)
        
        return accuracy
    
    def _check_return_distributions(self, backtest_results: Dict[str, Any], 
                                  reference_results: Optional[Dict[str, Any]] = None) -> float:
        """Check return distribution accuracy"""
        if reference_results is None:
            return 1.0  # No reference to compare against
        
        # Extract return data
        backtest_returns = backtest_results.get('returns', [])
        reference_returns = reference_results.get('returns', [])
        
        if not backtest_returns or not reference_returns:
            return 0.0
        
        # Calculate distribution similarity (simplified)
        backtest_mean = np.mean(backtest_returns)
        reference_mean = np.mean(reference_returns)
        
        backtest_std = np.std(backtest_returns)
        reference_std = np.std(reference_returns)
        
        mean_accuracy = 1.0 - abs(backtest_mean - reference_mean) / abs(reference_mean + 1e-8)
        std_accuracy = 1.0 - abs(backtest_std - reference_std) / abs(reference_std + 1e-8)
        
        return max(0.0, (mean_accuracy + std_accuracy) / 2)
    
    def _check_volatility_modeling(self, backtest_results: Dict[str, Any], 
                                 reference_results: Optional[Dict[str, Any]] = None) -> float:
        """Check volatility modeling accuracy"""
        if reference_results is None:
            return 1.0
        
        # Extract volatility data
        backtest_vol = backtest_results.get('volatility', 0)
        reference_vol = reference_results.get('volatility', 0)
        
        if reference_vol == 0:
            return 1.0
        
        accuracy = 1.0 - abs(backtest_vol - reference_vol) / reference_vol
        return max(0.0, accuracy)
    
    def _check_correlation_accuracy(self, backtest_results: Dict[str, Any], 
                                  reference_results: Optional[Dict[str, Any]] = None) -> float:
        """Check correlation accuracy"""
        if reference_results is None:
            return 1.0
        
        # Extract correlation data
        backtest_corr = backtest_results.get('correlation', 0)
        reference_corr = reference_results.get('correlation', 0)
        
        accuracy = 1.0 - abs(backtest_corr - reference_corr)
        return max(0.0, accuracy)
    
    def _combine_metrics(self, data_metrics: QualityMetrics, 
                        execution_metrics: QualityMetrics, 
                        statistical_metrics: QualityMetrics) -> QualityMetrics:
        """Combine metrics from different validation areas"""
        
        combined = QualityMetrics()
        
        # Data quality metrics
        combined.data_completeness = data_metrics.data_completeness
        combined.data_accuracy = data_metrics.data_accuracy
        combined.spread_accuracy = data_metrics.spread_accuracy
        combined.timestamp_accuracy = data_metrics.timestamp_accuracy
        
        # Execution quality metrics
        combined.order_execution_accuracy = execution_metrics.order_execution_accuracy
        combined.slippage_modeling_accuracy = execution_metrics.slippage_modeling_accuracy
        combined.commission_calculation_accuracy = execution_metrics.commission_calculation_accuracy
        
        # Statistical quality metrics
        combined.return_distribution_accuracy = statistical_metrics.return_distribution_accuracy
        combined.volatility_modeling_accuracy = statistical_metrics.volatility_modeling_accuracy
        combined.correlation_accuracy = statistical_metrics.correlation_accuracy
        
        # Calculate overall score
        combined.calculate_overall_score()
        
        return combined
    
    def _perform_data_integrity_checks(self, data: pd.DataFrame, 
                                     trade_log: pd.DataFrame) -> Dict[str, bool]:
        """Perform data integrity checks"""
        
        checks = {}
        
        # Data integrity checks
        checks['data_not_empty'] = not data.empty
        checks['trade_log_not_empty'] = not trade_log.empty
        checks['data_has_required_columns'] = all(
            col in data.columns for col in ['timestamp', 'open', 'high', 'low', 'close']
        )
        checks['trade_log_has_required_columns'] = all(
            col in trade_log.columns for col in ['entry_time', 'exit_time', 'entry_price', 'exit_price']
        )
        
        # Timestamp checks
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            checks['timestamps_chronological'] = timestamps.is_monotonic_increasing
            checks['no_duplicate_timestamps'] = not timestamps.duplicated().any()
        
        # Price checks
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                checks[f'{col}_positive'] = (data[col] > 0).all()
                checks[f'{col}_finite'] = data[col].notna().all()
        
        # OHLC consistency
        if all(col in data.columns for col in price_columns):
            ohlc_consistent = (
                (data['high'] >= data['low']) &
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close'])
            ).all()
            checks['ohlc_consistent'] = ohlc_consistent
        
        return checks
    
    def _identify_issues_and_warnings(self, metrics: QualityMetrics, 
                                    integrity_checks: Dict[str, bool]) -> Tuple[List[str], List[str]]:
        """Identify issues and warnings based on metrics and checks"""
        
        issues = []
        warnings = []
        
        # Check overall quality
        if metrics.overall_quality_score < self.config.acceptable_threshold:
            issues.append(f"Overall quality score {metrics.overall_quality_score:.3f} below acceptable threshold")
        
        # Check individual metrics
        if metrics.data_completeness < 0.95:
            warnings.append(f"Data completeness {metrics.data_completeness:.3f} below 95%")
        
        if metrics.data_accuracy < 0.90:
            issues.append(f"Data accuracy {metrics.data_accuracy:.3f} below 90%")
        
        if metrics.spread_accuracy < 0.85:
            warnings.append(f"Spread accuracy {metrics.spread_accuracy:.3f} below 85%")
        
        if metrics.order_execution_accuracy < 0.95:
            issues.append(f"Order execution accuracy {metrics.order_execution_accuracy:.3f} below 95%")
        
        # Check integrity issues
        for check_name, passed in integrity_checks.items():
            if not passed:
                if 'required' in check_name or 'consistent' in check_name:
                    issues.append(f"Data integrity check failed: {check_name}")
                else:
                    warnings.append(f"Data integrity warning: {check_name}")
        
        return issues, warnings
    
    def _generate_recommendations(self, metrics: QualityMetrics, 
                                issues: List[str], warnings: List[str]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Data quality recommendations
        if metrics.data_completeness < 0.95:
            recommendations.append("Improve data completeness by filling gaps or using higher quality data sources")
        
        if metrics.data_accuracy < 0.90:
            recommendations.append("Verify data source accuracy and implement data validation checks")
        
        if metrics.spread_accuracy < 0.85:
            recommendations.append("Improve spread modeling accuracy by using historical spread data")
        
        # Execution quality recommendations
        if metrics.order_execution_accuracy < 0.95:
            recommendations.append("Review and improve order execution logic")
        
        if metrics.slippage_modeling_accuracy < 0.90:
            recommendations.append("Implement more realistic slippage modeling")
        
        # Statistical quality recommendations
        if metrics.return_distribution_accuracy < 0.90:
            recommendations.append("Improve return distribution modeling")
        
        if metrics.volatility_modeling_accuracy < 0.85:
            recommendations.append("Enhance volatility modeling accuracy")
        
        # General recommendations
        if len(issues) > 0:
            recommendations.append("Address critical issues before using results for trading decisions")
        
        if len(warnings) > 3:
            recommendations.append("Review and address multiple warnings to improve overall quality")
        
        return recommendations
    
    def _create_validation_details(self, data: pd.DataFrame, trade_log: pd.DataFrame, 
                                 backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed validation information"""
        
        details = {
            'data_summary': {
                'total_records': len(data),
                'date_range': {
                    'start': data['timestamp'].min() if 'timestamp' in data.columns else None,
                    'end': data['timestamp'].max() if 'timestamp' in data.columns else None
                },
                'columns': list(data.columns)
            },
            'trade_summary': {
                'total_trades': len(trade_log),
                'columns': list(trade_log.columns)
            },
            'backtest_summary': {
                'metrics_available': list(backtest_results.keys()),
                'total_return': backtest_results.get('total_return', 0),
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 0)
            }
        }
        
        return details
