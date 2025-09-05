"""
Strategy Evaluator

Comprehensive evaluation system for trading strategies including
performance analysis, risk assessment, and validation metrics.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of strategy evaluation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationType(Enum):
    """Type of evaluation"""
    PERFORMANCE = "performance"
    RISK = "risk"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    BENCHMARK = "benchmark"
    STATISTICAL = "statistical"
    MONTE_CARLO = "monte_carlo"
    WALK_FORWARD = "walk_forward"
    CROSS_VALIDATION = "cross_validation"
    STRESS_TEST = "stress_test"
    REGRESSION = "regression"


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a strategy"""
    
    # Performance metrics
    performance_score: float = 0.0
    return_score: float = 0.0
    profit_factor_score: float = 0.0
    win_rate_score: float = 0.0
    
    # Risk metrics
    risk_score: float = 0.0
    drawdown_score: float = 0.0
    var_score: float = 0.0
    volatility_score: float = 0.0
    
    # Consistency metrics
    consistency_score: float = 0.0
    stability_score: float = 0.0
    reliability_score: float = 0.0
    predictability_score: float = 0.0
    
    # Efficiency metrics
    efficiency_score: float = 0.0
    sharpe_score: float = 0.0
    calmar_score: float = 0.0
    sortino_score: float = 0.0
    
    # Robustness metrics
    robustness_score: float = 0.0
    stress_test_score: float = 0.0
    monte_carlo_score: float = 0.0
    
    # Benchmark metrics
    benchmark_score: float = 0.0
    alpha_score: float = 0.0
    beta_score: float = 0.0
    
    # Overall metrics
    normalized_score: float = 0.0
    weighted_score: float = 0.0
    
    # Evaluation metadata
    evaluation_duration: float = 0.0
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'performance_score': self.performance_score,
            'return_score': self.return_score,
            'profit_factor_score': self.profit_factor_score,
            'win_rate_score': self.win_rate_score,
            'risk_score': self.risk_score,
            'drawdown_score': self.drawdown_score,
            'var_score': self.var_score,
            'volatility_score': self.volatility_score,
            'consistency_score': self.consistency_score,
            'stability_score': self.stability_score,
            'reliability_score': self.reliability_score,
            'predictability_score': self.predictability_score,
            'efficiency_score': self.efficiency_score,
            'sharpe_score': self.sharpe_score,
            'calmar_score': self.calmar_score,
            'sortino_score': self.sortino_score,
            'robustness_score': self.robustness_score,
            'stress_test_score': self.stress_test_score,
            'monte_carlo_score': self.monte_carlo_score,
            'benchmark_score': self.benchmark_score,
            'alpha_score': self.alpha_score,
            'beta_score': self.beta_score,
            'normalized_score': self.normalized_score,
            'weighted_score': self.weighted_score,
            'evaluation_duration': self.evaluation_duration,
            'evaluation_timestamp': self.evaluation_timestamp.isoformat()
        }


@dataclass
class EvaluationResult:
    """Result of strategy evaluation"""
    
    strategy_id: str
    strategy_name: str
    evaluation_metrics: EvaluationMetrics
    
    # Validation results
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Status
    status: EvaluationStatus = EvaluationStatus.COMPLETED
    
    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    evaluation_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'evaluation_metrics': self.evaluation_metrics.to_dict(),
            'validation_results': self.validation_results,
            'recommendations': self.recommendations,
            'status': self.status.value,
            'evaluation_timestamp': self.evaluation_timestamp.isoformat(),
            'evaluation_duration': self.evaluation_duration
        }


@dataclass
class EvaluationConfig:
    """Configuration for strategy evaluation"""
    
    # Evaluation types
    enable_performance_metrics: bool = True
    enable_risk_metrics: bool = True
    enable_consistency_metrics: bool = True
    enable_efficiency_metrics: bool = True
    enable_robustness_metrics: bool = True
    enable_benchmark_comparison: bool = True
    
    # Advanced evaluation
    enable_statistical_validation: bool = True
    enable_monte_carlo_validation: bool = True
    monte_carlo_runs: int = 1000
    confidence_level: float = 0.95
    
    # Walk-forward validation
    enable_walk_forward_validation: bool = True
    walk_forward_periods: int = 5
    
    # Cross-validation
    enable_cross_validation: bool = True
    cross_validation_folds: int = 5
    
    # Stress testing
    enable_stress_testing: bool = True
    stress_test_scenarios: int = 10
    
    # Regression testing
    enable_regression_testing: bool = True
    
    # Benchmark comparison
    enable_benchmark_comparison: bool = True
    
    # Risk adjustment
    enable_risk_adjustment: bool = True
    risk_free_rate: float = 0.02
    
    # Consistency penalty
    enable_consistency_penalty: bool = True
    consistency_penalty_factor: float = 0.1
    
    # Efficiency bonus
    enable_efficiency_bonus: bool = True
    efficiency_bonus_factor: float = 0.05
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Output settings
    include_breakdown: bool = True
    include_recommendations: bool = True
    include_validation_details: bool = True


class StrategyEvaluator:
    """Comprehensive strategy evaluator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_strategies(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: Optional[List[Dict[str, Any]]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """Evaluate multiple strategies"""
        
        self.logger.info(f"Evaluating {len(strategies)} strategies")
        
        if self.config.enable_parallel_processing and len(strategies) > 4:
            return self._evaluate_strategies_parallel(strategies, backtest_results, benchmark_results)
        else:
            return self._evaluate_strategies_sequential(strategies, backtest_results, benchmark_results)
    
    def _evaluate_strategies_sequential(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: Optional[List[Dict[str, Any]]],
        benchmark_results: Optional[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Evaluate strategies sequentially"""
        
        results = []
        
        for strategy in strategies:
            try:
                # Find corresponding backtest result
                backtest_result = self._find_backtest_result(strategy, backtest_results) if backtest_results else None
                
                # Evaluate the strategy
                evaluation_result = self._evaluate_single_strategy(
                    strategy, backtest_result, benchmark_results
                )
                results.append(evaluation_result)
            
            except Exception as e:
                self.logger.error(f"Error evaluating strategy {strategy.get('strategy_id', 'unknown')}: {e}")
                
                # Create failed result
                failed_result = EvaluationResult(
                    strategy_id=strategy.get('strategy_id', 'unknown'),
                    strategy_name=strategy.get('strategy_name', 'Unknown Strategy'),
                    evaluation_metrics=EvaluationMetrics(),
                    validation_results={'error': str(e)},
                    recommendations=['Fix evaluation errors'],
                    status=EvaluationStatus.FAILED
                )
                results.append(failed_result)
        
        return results
    
    def _evaluate_strategies_parallel(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: Optional[List[Dict[str, Any]]],
        benchmark_results: Optional[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Evaluate strategies in parallel"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit evaluation tasks
            futures = []
            for strategy in strategies:
                backtest_result = self._find_backtest_result(strategy, backtest_results) if backtest_results else None
                future = executor.submit(
                    self._evaluate_single_strategy,
                    strategy, backtest_result, benchmark_results
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel evaluation: {e}")
        
        return results
    
    def _evaluate_single_strategy(
        self,
        strategy: Dict[str, Any],
        backtest_result: Optional[Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate a single strategy"""
        
        start_time = datetime.now()
        
        # Create evaluation metrics
        metrics = EvaluationMetrics()
        
        # Evaluate different aspects
        if self.config.enable_performance_metrics and backtest_result:
            self._evaluate_performance(backtest_result, metrics)
        
        if self.config.enable_risk_metrics and backtest_result:
            self._evaluate_risk(backtest_result, metrics)
        
        if self.config.enable_consistency_metrics and backtest_result:
            self._evaluate_consistency(backtest_result, metrics)
        
        if self.config.enable_efficiency_metrics and backtest_result:
            self._evaluate_efficiency(backtest_result, metrics)
        
        if self.config.enable_robustness_metrics and backtest_result:
            self._evaluate_robustness(backtest_result, metrics)
        
        if self.config.enable_benchmark_comparison and benchmark_results and backtest_result:
            self._evaluate_benchmark(backtest_result, benchmark_results, metrics)
        
        # Calculate overall scores
        self._calculate_overall_scores(metrics)
        
        # Generate validation results
        validation_results = self._generate_validation_results(strategy, backtest_result, metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(strategy, backtest_result, metrics)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        metrics.evaluation_duration = duration
        
        # Create result
        result = EvaluationResult(
            strategy_id=strategy.get('strategy_id', 'unknown'),
            strategy_name=strategy.get('strategy_name', 'Unknown Strategy'),
            evaluation_metrics=metrics,
            validation_results=validation_results,
            recommendations=recommendations,
            status=EvaluationStatus.COMPLETED,
            evaluation_duration=duration
        )
        
        return result
    
    def _evaluate_performance(self, backtest_result: Dict[str, Any], metrics: EvaluationMetrics) -> None:
        """Evaluate performance metrics"""
        
        # Return score
        net_profit = backtest_result.get('net_profit', 0)
        initial_capital = backtest_result.get('initial_capital', 100000)
        return_rate = net_profit / initial_capital if initial_capital > 0 else 0
        metrics.return_score = min(100, max(0, return_rate * 1000))
        
        # Profit factor score
        profit_factor = backtest_result.get('profit_factor', 0)
        metrics.profit_factor_score = min(100, max(0, (profit_factor - 1) * 50))
        
        # Win rate score
        win_rate = backtest_result.get('win_rate', 0)
        metrics.win_rate_score = min(100, max(0, win_rate * 100))
        
        # Overall performance score
        metrics.performance_score = (
            metrics.return_score * 0.4 +
            metrics.profit_factor_score * 0.3 +
            metrics.win_rate_score * 0.3
        )
    
    def _evaluate_risk(self, backtest_result: Dict[str, Any], metrics: EvaluationMetrics) -> None:
        """Evaluate risk metrics"""
        
        # Drawdown score (inverted)
        max_drawdown = backtest_result.get('max_drawdown', 0)
        metrics.drawdown_score = max(0, 100 - max_drawdown * 1000)
        
        # VaR score (inverted)
        var_95 = backtest_result.get('var_95', 0)
        metrics.var_score = max(0, 100 - var_95 * 2000)
        
        # Volatility score (inverted)
        volatility = backtest_result.get('volatility', 0)
        metrics.volatility_score = max(0, 100 - volatility * 1000)
        
        # Overall risk score
        metrics.risk_score = (
            metrics.drawdown_score * 0.4 +
            metrics.var_score * 0.3 +
            metrics.volatility_score * 0.3
        )
    
    def _evaluate_consistency(self, backtest_result: Dict[str, Any], metrics: EvaluationMetrics) -> None:
        """Evaluate consistency metrics"""
        
        # Stability score
        stability_score = backtest_result.get('stability_score', 0)
        metrics.stability_score = min(100, max(0, stability_score * 100))
        
        # Reliability score
        reliability_score = backtest_result.get('reliability_score', 0)
        metrics.reliability_score = min(100, max(0, reliability_score * 100))
        
        # Predictability score
        predictability_score = backtest_result.get('predictability_score', 0)
        metrics.predictability_score = min(100, max(0, predictability_score * 100))
        
        # Overall consistency score
        metrics.consistency_score = (
            metrics.stability_score * 0.4 +
            metrics.reliability_score * 0.3 +
            metrics.predictability_score * 0.3
        )
    
    def _evaluate_efficiency(self, backtest_result: Dict[str, Any], metrics: EvaluationMetrics) -> None:
        """Evaluate efficiency metrics"""
        
        # Sharpe ratio score
        sharpe_ratio = backtest_result.get('sharpe_ratio', 0)
        metrics.sharpe_score = min(100, max(0, sharpe_ratio * 20))
        
        # Calmar ratio score
        calmar_ratio = backtest_result.get('calmar_ratio', 0)
        metrics.calmar_score = min(100, max(0, calmar_ratio * 10))
        
        # Sortino ratio score
        sortino_ratio = backtest_result.get('sortino_ratio', 0)
        metrics.sortino_score = min(100, max(0, sortino_ratio * 20))
        
        # Overall efficiency score
        metrics.efficiency_score = (
            metrics.sharpe_score * 0.4 +
            metrics.calmar_score * 0.3 +
            metrics.sortino_score * 0.3
        )
    
    def _evaluate_robustness(self, backtest_result: Dict[str, Any], metrics: EvaluationMetrics) -> None:
        """Evaluate robustness metrics"""
        
        # Robustness score
        robustness_score = backtest_result.get('robustness_score', 0)
        metrics.robustness_score = min(100, max(0, robustness_score * 100))
        
        # Stress test score (mock)
        metrics.stress_test_score = min(100, max(0, robustness_score * 100))
        
        # Monte Carlo score (mock)
        metrics.monte_carlo_score = min(100, max(0, robustness_score * 100))
    
    def _evaluate_benchmark(
        self,
        backtest_result: Dict[str, Any],
        benchmark_results: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> None:
        """Evaluate benchmark comparison metrics"""
        
        # Strategy return
        strategy_return = backtest_result.get('net_profit', 0) / backtest_result.get('initial_capital', 100000)
        benchmark_return = benchmark_results.get('total_return', 0)
        
        # Alpha score
        alpha = strategy_return - benchmark_return
        metrics.alpha_score = min(100, max(0, 50 + alpha * 1000))
        
        # Beta score (mock)
        metrics.beta_score = 1.0
        
        # Overall benchmark score
        metrics.benchmark_score = metrics.alpha_score
    
    def _calculate_overall_scores(self, metrics: EvaluationMetrics) -> None:
        """Calculate overall scores"""
        
        # Normalized score (average of all component scores)
        component_scores = [
            metrics.performance_score,
            metrics.risk_score,
            metrics.consistency_score,
            metrics.efficiency_score,
            metrics.robustness_score,
            metrics.benchmark_score
        ]
        
        metrics.normalized_score = np.mean(component_scores)
        
        # Weighted score
        metrics.weighted_score = (
            metrics.performance_score * 0.3 +
            metrics.risk_score * 0.25 +
            metrics.consistency_score * 0.2 +
            metrics.efficiency_score * 0.15 +
            metrics.robustness_score * 0.05 +
            metrics.benchmark_score * 0.05
        )
    
    def _generate_validation_results(
        self,
        strategy: Dict[str, Any],
        backtest_result: Optional[Dict[str, Any]],
        metrics: EvaluationMetrics
    ) -> Dict[str, Any]:
        """Generate validation results"""
        
        validation_results = {
            'strategy_id': strategy.get('strategy_id', 'unknown'),
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics.to_dict(),
            'validation_passed': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for warnings
        if metrics.performance_score < 50:
            validation_results['warnings'].append('Low performance score')
        
        if metrics.risk_score < 50:
            validation_results['warnings'].append('High risk score')
        
        if metrics.consistency_score < 50:
            validation_results['warnings'].append('Low consistency score')
        
        if metrics.efficiency_score < 50:
            validation_results['warnings'].append('Low efficiency score')
        
        # Check for errors
        if metrics.normalized_score < 30:
            validation_results['errors'].append('Very low overall score')
            validation_results['validation_passed'] = False
        
        return validation_results
    
    def _generate_recommendations(
        self,
        strategy: Dict[str, Any],
        backtest_result: Optional[Dict[str, Any]],
        metrics: EvaluationMetrics
    ) -> List[str]:
        """Generate recommendations for strategy improvement"""
        
        recommendations = []
        
        # Performance recommendations
        if metrics.performance_score < 60:
            recommendations.append('Improve strategy performance by optimizing entry/exit conditions')
        
        if metrics.return_score < 50:
            recommendations.append('Focus on increasing strategy returns')
        
        if metrics.profit_factor_score < 50:
            recommendations.append('Improve profit factor by reducing losses or increasing wins')
        
        if metrics.win_rate_score < 50:
            recommendations.append('Increase win rate by improving signal quality')
        
        # Risk recommendations
        if metrics.risk_score < 60:
            recommendations.append('Implement better risk management to reduce drawdowns')
        
        if metrics.drawdown_score < 50:
            recommendations.append('Add stop-loss mechanisms to limit maximum drawdown')
        
        # Consistency recommendations
        if metrics.consistency_score < 60:
            recommendations.append('Improve strategy consistency across different market conditions')
        
        if metrics.stability_score < 50:
            recommendations.append('Enhance strategy stability by reducing parameter sensitivity')
        
        # Efficiency recommendations
        if metrics.efficiency_score < 60:
            recommendations.append('Improve strategy efficiency by optimizing risk-adjusted returns')
        
        if metrics.sharpe_score < 50:
            recommendations.append('Focus on improving Sharpe ratio')
        
        # General recommendations
        if metrics.normalized_score < 70:
            recommendations.append('Consider comprehensive strategy review and optimization')
        
        if not recommendations:
            recommendations.append('Strategy appears to be performing well')
        
        return recommendations
    
    def _find_backtest_result(
        self,
        strategy: Dict[str, Any],
        backtest_results: Optional[List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Find backtest result for a strategy"""
        
        if not backtest_results:
            return None
        
        strategy_id = strategy.get('strategy_id')
        if not strategy_id:
            return None
        
        for result in backtest_results:
            if result.get('strategy_id') == strategy_id:
                return result
        
        return None

