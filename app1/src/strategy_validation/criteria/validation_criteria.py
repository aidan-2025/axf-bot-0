#!/usr/bin/env python3
"""
Validation Criteria and Thresholds

Defines comprehensive validation criteria for trading strategies based on industry standards
and risk management requirements.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"  # Must pass - strategy rejected if failed
    HIGH = "high"         # Strongly recommended - affects scoring significantly
    MEDIUM = "medium"     # Recommended - affects scoring moderately
    LOW = "low"          # Optional - affects scoring minimally


@dataclass
class ValidationThresholds:
    """Validation thresholds for trading strategies"""
    
    # Trade Count Requirements
    min_trades: int = 30
    min_trades_per_year: int = 10
    max_trades_per_year: int = 1000
    
    # Performance Requirements
    min_profit_factor: float = 1.2
    min_sharpe_ratio: float = 0.5
    min_sortino_ratio: float = 0.3
    min_calmar_ratio: float = 0.2
    
    # Risk Management
    max_drawdown: float = 0.15  # 15% maximum drawdown
    max_consecutive_losses: int = 5
    max_daily_loss: float = 0.05  # 5% maximum daily loss
    
    # Win Rate Requirements
    min_win_rate: float = 0.35  # 35% minimum win rate
    min_avg_win_loss_ratio: float = 1.0
    
    # Consistency Requirements
    min_consistency_score: float = 0.6
    min_stability_score: float = 0.7
    max_volatility: float = 0.3  # 30% maximum annual volatility
    
    # Time-based Requirements
    min_backtest_duration_days: int = 90  # Minimum 3 months
    min_live_trading_days: int = 30  # Minimum 1 month live
    
    # Economic Event Requirements
    min_event_avoidance_score: float = 0.8
    max_event_impact_drawdown: float = 0.05  # 5% max drawdown during events
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert thresholds to dictionary"""
        return {
            'min_trades': self.min_trades,
            'min_trades_per_year': self.min_trades_per_year,
            'max_trades_per_year': self.max_trades_per_year,
            'min_profit_factor': self.min_profit_factor,
            'min_sharpe_ratio': self.min_sharpe_ratio,
            'min_sortino_ratio': self.min_sortino_ratio,
            'min_calmar_ratio': self.min_calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_consecutive_losses': self.max_consecutive_losses,
            'max_daily_loss': self.max_daily_loss,
            'min_win_rate': self.min_win_rate,
            'min_avg_win_loss_ratio': self.min_avg_win_loss_ratio,
            'min_consistency_score': self.min_consistency_score,
            'min_stability_score': self.min_stability_score,
            'max_volatility': self.max_volatility,
            'min_backtest_duration_days': self.min_backtest_duration_days,
            'min_live_trading_days': self.min_live_trading_days,
            'min_event_avoidance_score': self.min_event_avoidance_score,
            'max_event_impact_drawdown': self.max_event_impact_drawdown
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationThresholds':
        """Create thresholds from dictionary"""
        return cls(**data)


@dataclass
class ValidationCriteria:
    """Comprehensive validation criteria for trading strategies"""
    
    thresholds: ValidationThresholds
    validation_levels: Dict[str, ValidationLevel]
    
    def __init__(self, thresholds: Optional[ValidationThresholds] = None):
        self.thresholds = thresholds or ValidationThresholds()
        self.validation_levels = self._initialize_validation_levels()
    
    def _initialize_validation_levels(self) -> Dict[str, ValidationLevel]:
        """Initialize validation levels for different criteria"""
        return {
            # Critical validations - must pass
            'min_trades': ValidationLevel.CRITICAL,
            'min_profit_factor': ValidationLevel.CRITICAL,
            'max_drawdown': ValidationLevel.CRITICAL,
            'min_backtest_duration_days': ValidationLevel.CRITICAL,
            
            # High priority validations
            'min_sharpe_ratio': ValidationLevel.HIGH,
            'min_win_rate': ValidationLevel.HIGH,
            'min_consistency_score': ValidationLevel.HIGH,
            'min_stability_score': ValidationLevel.HIGH,
            
            # Medium priority validations
            'min_sortino_ratio': ValidationLevel.MEDIUM,
            'min_calmar_ratio': ValidationLevel.MEDIUM,
            'max_consecutive_losses': ValidationLevel.MEDIUM,
            'min_avg_win_loss_ratio': ValidationLevel.MEDIUM,
            
            # Low priority validations
            'max_volatility': ValidationLevel.LOW,
            'min_event_avoidance_score': ValidationLevel.LOW,
            'max_event_impact_drawdown': ValidationLevel.LOW
        }
    
    def validate_trade_count(self, total_trades: int, backtest_days: int) -> Dict[str, Any]:
        """Validate trade count requirements"""
        trades_per_year = (total_trades / backtest_days) * 365 if backtest_days > 0 else 0
        
        results = {
            'passed': True,
            'violations': [],
            'warnings': [],
            'level': ValidationLevel.CRITICAL
        }
        
        # Check minimum trades
        if total_trades < self.thresholds.min_trades:
            results['passed'] = False
            results['violations'].append(
                f"Insufficient trades: {total_trades} < {self.thresholds.min_trades}"
            )
        
        # Check trades per year
        if trades_per_year < self.thresholds.min_trades_per_year:
            results['passed'] = False
            results['violations'].append(
                f"Low trading frequency: {trades_per_year:.1f} trades/year < {self.thresholds.min_trades_per_year}"
            )
        elif trades_per_year > self.thresholds.max_trades_per_year:
            results['warnings'].append(
                f"High trading frequency: {trades_per_year:.1f} trades/year > {self.thresholds.max_trades_per_year}"
            )
        
        return results
    
    def validate_performance_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate performance metrics"""
        results = {
            'passed': True,
            'violations': [],
            'warnings': [],
            'level': ValidationLevel.CRITICAL
        }
        
        # Check profit factor
        profit_factor = metrics.get('profit_factor', 0.0)
        if profit_factor < self.thresholds.min_profit_factor:
            results['passed'] = False
            results['violations'].append(
                f"Low profit factor: {profit_factor:.3f} < {self.thresholds.min_profit_factor}"
            )
        
        # Check Sharpe ratio
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        if sharpe_ratio < self.thresholds.min_sharpe_ratio:
            level = self.validation_levels['min_sharpe_ratio']
            if level == ValidationLevel.CRITICAL:
                results['passed'] = False
                results['violations'].append(
                    f"Low Sharpe ratio: {sharpe_ratio:.3f} < {self.thresholds.min_sharpe_ratio}"
                )
            else:
                results['warnings'].append(
                    f"Low Sharpe ratio: {sharpe_ratio:.3f} < {self.thresholds.min_sharpe_ratio}"
                )
        
        # Check Sortino ratio
        sortino_ratio = metrics.get('sortino_ratio', 0.0)
        if sortino_ratio < self.thresholds.min_sortino_ratio:
            level = self.validation_levels['min_sortino_ratio']
            if level == ValidationLevel.CRITICAL:
                results['passed'] = False
                results['violations'].append(
                    f"Low Sortino ratio: {sortino_ratio:.3f} < {self.thresholds.min_sortino_ratio}"
                )
            else:
                results['warnings'].append(
                    f"Low Sortino ratio: {sortino_ratio:.3f} < {self.thresholds.min_sortino_ratio}"
                )
        
        # Check Calmar ratio
        calmar_ratio = metrics.get('calmar_ratio', 0.0)
        if calmar_ratio < self.thresholds.min_calmar_ratio:
            level = self.validation_levels['min_calmar_ratio']
            if level == ValidationLevel.CRITICAL:
                results['passed'] = False
                results['violations'].append(
                    f"Low Calmar ratio: {calmar_ratio:.3f} < {self.thresholds.min_calmar_ratio}"
                )
            else:
                results['warnings'].append(
                    f"Low Calmar ratio: {calmar_ratio:.3f} < {self.thresholds.min_calmar_ratio}"
                )
        
        return results
    
    def validate_risk_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate risk management metrics"""
        results = {
            'passed': True,
            'violations': [],
            'warnings': [],
            'level': ValidationLevel.CRITICAL
        }
        
        # Check maximum drawdown
        max_drawdown = abs(metrics.get('max_drawdown', 0.0))
        if max_drawdown > self.thresholds.max_drawdown:
            results['passed'] = False
            results['violations'].append(
                f"Excessive drawdown: {max_drawdown:.3f} > {self.thresholds.max_drawdown}"
            )
        
        # Check consecutive losses
        consecutive_losses = metrics.get('consecutive_losses', 0)
        if consecutive_losses > self.thresholds.max_consecutive_losses:
            level = self.validation_levels['max_consecutive_losses']
            if level == ValidationLevel.CRITICAL:
                results['passed'] = False
                results['violations'].append(
                    f"Too many consecutive losses: {consecutive_losses} > {self.thresholds.max_consecutive_losses}"
                )
            else:
                results['warnings'].append(
                    f"Many consecutive losses: {consecutive_losses} > {self.thresholds.max_consecutive_losses}"
                )
        
        # Check daily loss
        max_daily_loss = abs(metrics.get('max_daily_loss', 0.0))
        if max_daily_loss > self.thresholds.max_daily_loss:
            results['warnings'].append(
                f"High daily loss: {max_daily_loss:.3f} > {self.thresholds.max_daily_loss}"
            )
        
        return results
    
    def validate_consistency_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate consistency and stability metrics"""
        results = {
            'passed': True,
            'violations': [],
            'warnings': [],
            'level': ValidationLevel.HIGH
        }
        
        # Check win rate
        win_rate = metrics.get('win_rate', 0.0)
        if win_rate < self.thresholds.min_win_rate:
            level = self.validation_levels['min_win_rate']
            if level == ValidationLevel.CRITICAL:
                results['passed'] = False
                results['violations'].append(
                    f"Low win rate: {win_rate:.3f} < {self.thresholds.min_win_rate}"
                )
            else:
                results['warnings'].append(
                    f"Low win rate: {win_rate:.3f} < {self.thresholds.min_win_rate}"
                )
        
        # Check average win/loss ratio
        avg_win_loss_ratio = metrics.get('avg_win_loss_ratio', 0.0)
        if avg_win_loss_ratio < self.thresholds.min_avg_win_loss_ratio:
            level = self.validation_levels['min_avg_win_loss_ratio']
            if level == ValidationLevel.CRITICAL:
                results['passed'] = False
                results['violations'].append(
                    f"Low win/loss ratio: {avg_win_loss_ratio:.3f} < {self.thresholds.min_avg_win_loss_ratio}"
                )
            else:
                results['warnings'].append(
                    f"Low win/loss ratio: {avg_win_loss_ratio:.3f} < {self.thresholds.min_avg_win_loss_ratio}"
                )
        
        # Check consistency score
        consistency_score = metrics.get('consistency_score', 0.0)
        if consistency_score < self.thresholds.min_consistency_score:
            level = self.validation_levels['min_consistency_score']
            if level == ValidationLevel.CRITICAL:
                results['passed'] = False
                results['violations'].append(
                    f"Low consistency score: {consistency_score:.3f} < {self.thresholds.min_consistency_score}"
                )
            else:
                results['warnings'].append(
                    f"Low consistency score: {consistency_score:.3f} < {self.thresholds.min_consistency_score}"
                )
        
        # Check stability score
        stability_score = metrics.get('stability_score', 0.0)
        if stability_score < self.thresholds.min_stability_score:
            level = self.validation_levels['min_stability_score']
            if level == ValidationLevel.CRITICAL:
                results['passed'] = False
                results['violations'].append(
                    f"Low stability score: {stability_score:.3f} < {self.thresholds.min_stability_score}"
                )
            else:
                results['warnings'].append(
                    f"Low stability score: {stability_score:.3f} < {self.thresholds.min_stability_score}"
                )
        
        return results
    
    def validate_comprehensive(self, strategy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive validation of strategy metrics"""
        logger.info("Starting comprehensive strategy validation")
        
        validation_results = {
            'overall_passed': True,
            'critical_violations': [],
            'warnings': [],
            'validation_details': {},
            'score': 0.0
        }
        
        # Extract basic metrics
        total_trades = strategy_metrics.get('total_trades', 0)
        backtest_days = strategy_metrics.get('backtest_duration_days', 0)
        performance_metrics = strategy_metrics.get('performance_metrics', {})
        risk_metrics = strategy_metrics.get('risk_metrics', {})
        consistency_metrics = strategy_metrics.get('consistency_metrics', {})
        
        # Validate trade count
        trade_count_result = self.validate_trade_count(total_trades, backtest_days)
        validation_results['validation_details']['trade_count'] = trade_count_result
        if not trade_count_result['passed']:
            validation_results['overall_passed'] = False
            validation_results['critical_violations'].extend(trade_count_result['violations'])
        validation_results['warnings'].extend(trade_count_result['warnings'])
        
        # Validate performance metrics
        performance_result = self.validate_performance_metrics(performance_metrics)
        validation_results['validation_details']['performance'] = performance_result
        if not performance_result['passed']:
            validation_results['overall_passed'] = False
            validation_results['critical_violations'].extend(performance_result['violations'])
        validation_results['warnings'].extend(performance_result['warnings'])
        
        # Validate risk metrics
        risk_result = self.validate_risk_metrics(risk_metrics)
        validation_results['validation_details']['risk'] = risk_result
        if not risk_result['passed']:
            validation_results['overall_passed'] = False
            validation_results['critical_violations'].extend(risk_result['violations'])
        validation_results['warnings'].extend(risk_result['warnings'])
        
        # Validate consistency metrics
        consistency_result = self.validate_consistency_metrics(consistency_metrics)
        validation_results['validation_details']['consistency'] = consistency_result
        if not consistency_result['passed']:
            validation_results['overall_passed'] = False
            validation_results['critical_violations'].extend(consistency_result['violations'])
        validation_results['warnings'].extend(consistency_result['warnings'])
        
        # Calculate overall score
        validation_results['score'] = self._calculate_validation_score(validation_results)
        
        logger.info(f"Validation completed. Overall passed: {validation_results['overall_passed']}")
        logger.info(f"Critical violations: {len(validation_results['critical_violations'])}")
        logger.info(f"Warnings: {len(validation_results['warnings'])}")
        
        return validation_results
    
    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score (0.0 to 1.0)"""
        base_score = 1.0
        
        # Deduct points for critical violations
        critical_penalty = len(validation_results['critical_violations']) * 0.2
        base_score -= critical_penalty
        
        # Deduct points for warnings (less severe)
        warning_penalty = len(validation_results['warnings']) * 0.05
        base_score -= warning_penalty
        
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, base_score))

