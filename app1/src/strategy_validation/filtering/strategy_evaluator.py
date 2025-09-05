"""
Strategy Evaluator

Comprehensive evaluation system that combines filtering and scoring
to provide overall strategy assessment and ranking.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json

from .strategy_filter import StrategyFilter, FilterConfig, FilterResult
from .scoring_engine import ScoringEngine, ScoringConfig, ScoringResult

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of strategy evaluation"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 80-89%
    ACCEPTABLE = "acceptable"  # 70-79%
    POOR = "poor"           # 60-69%
    FAILED = "failed"       # <60%


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    
    # Overall scores
    total_score: float = 0.0
    normalized_score: float = 0.0
    status: EvaluationStatus = EvaluationStatus.FAILED
    
    # Filter results
    filter_passed: bool = False
    filter_score: float = 0.0
    filter_issues: List[str] = field(default_factory=list)
    
    # Scoring results
    scoring_passed: bool = False
    scoring_score: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    performance_rank: int = 0
    risk_rank: int = 0
    consistency_rank: int = 0
    
    # Evaluation metadata
    evaluated_at: datetime = field(default_factory=datetime.now)
    evaluation_duration: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics"""
        # Determine status based on normalized score
        if self.normalized_score >= 90:
            self.status = EvaluationStatus.EXCELLENT
        elif self.normalized_score >= 80:
            self.status = EvaluationStatus.GOOD
        elif self.normalized_score >= 70:
            self.status = EvaluationStatus.ACCEPTABLE
        elif self.normalized_score >= 60:
            self.status = EvaluationStatus.POOR
        else:
            self.status = EvaluationStatus.FAILED


@dataclass
class EvaluationResult:
    """Result of strategy evaluation"""
    
    strategy_id: str
    strategy_name: str
    evaluation_metrics: EvaluationMetrics
    
    # Detailed results
    filter_results: List[FilterResult] = field(default_factory=list)
    scoring_result: Optional[ScoringResult] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Evaluation details
    evaluation_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'evaluation_metrics': {
                'total_score': self.evaluation_metrics.total_score,
                'normalized_score': self.evaluation_metrics.normalized_score,
                'status': self.evaluation_metrics.status.value,
                'filter_passed': self.evaluation_metrics.filter_passed,
                'filter_score': self.evaluation_metrics.filter_score,
                'filter_issues': self.evaluation_metrics.filter_issues,
                'scoring_passed': self.evaluation_metrics.scoring_passed,
                'scoring_score': self.evaluation_metrics.scoring_score,
                'component_scores': self.evaluation_metrics.component_scores,
                'performance_rank': self.evaluation_metrics.performance_rank,
                'risk_rank': self.evaluation_metrics.risk_rank,
                'consistency_rank': self.evaluation_metrics.consistency_rank,
                'evaluated_at': self.evaluation_metrics.evaluated_at.isoformat(),
                'evaluation_duration': self.evaluation_metrics.evaluation_duration
            },
            'filter_results': [
                {
                    'criteria': result.criteria.value,
                    'status': result.status.value,
                    'message': result.message,
                    'passed': result.passed
                }
                for result in self.filter_results
            ],
            'scoring_result': self.scoring_result.to_dict() if self.scoring_result else None,
            'recommendations': self.recommendations,
            'warnings': self.warnings,
            'evaluation_details': self.evaluation_details
        }


@dataclass
class EvaluationConfig:
    """Configuration for strategy evaluation"""
    
    # Filter configuration
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    
    # Scoring configuration
    scoring_config: ScoringConfig = field(default_factory=ScoringConfig)
    
    # Evaluation thresholds
    min_total_score: float = 60.0  # Minimum total score to pass
    min_filter_score: float = 80.0  # Minimum filter score to pass
    min_scoring_score: float = 70.0  # Minimum scoring score to pass
    
    # Ranking settings
    enable_ranking: bool = True
    ranking_metrics: List[str] = field(default_factory=lambda: [
        'total_score', 'sharpe_ratio', 'max_drawdown', 'win_rate'
    ])
    
    # Recommendation settings
    generate_recommendations: bool = True
    generate_warnings: bool = True
    
    # Performance settings
    max_evaluation_time: float = 30.0  # Maximum evaluation time in seconds
    enable_parallel_evaluation: bool = True


class StrategyEvaluator:
    """Comprehensive strategy evaluation system"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.filter = StrategyFilter(config.filter_config)
        self.scorer = ScoringEngine(config.scoring_config)
    
    def evaluate_strategy(
        self, 
        strategy: Dict[str, Any], 
        backtest_results: Optional[Dict[str, Any]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate a single strategy"""
        
        start_time = datetime.now()
        strategy_id = strategy.get('strategy_id', 'unknown')
        strategy_name = strategy.get('name', 'Unknown Strategy')
        
        self.logger.info(f"Evaluating strategy: {strategy_name} ({strategy_id})")
        
        # Step 1: Filter the strategy
        filter_results = self.filter.filter_strategy(strategy, backtest_results)
        filter_passed = all(result.passed for result in filter_results)
        filter_score = self._calculate_filter_score(filter_results)
        
        # Step 2: Score the strategy (if backtest results available)
        scoring_result = None
        scoring_passed = False
        scoring_score = 0.0
        
        if backtest_results and filter_passed:
            try:
                scoring_result = self.scorer.score_strategy(strategy, backtest_results, benchmark_results)
                scoring_passed = scoring_result.normalized_score >= self.config.min_scoring_score
                scoring_score = scoring_result.normalized_score
            except Exception as e:
                self.logger.warning(f"Scoring failed for strategy {strategy_id}: {e}")
                scoring_result = None
        
        # Step 3: Calculate overall metrics
        total_score = self._calculate_total_score(filter_score, scoring_score, filter_passed, scoring_passed)
        normalized_score = total_score
        
        # Step 4: Generate recommendations and warnings
        recommendations = self._generate_recommendations(filter_results, scoring_result, total_score)
        warnings = self._generate_warnings(filter_results, scoring_result, total_score)
        
        # Step 5: Create evaluation metrics
        evaluation_metrics = EvaluationMetrics(
            total_score=total_score,
            normalized_score=normalized_score,
            filter_passed=filter_passed,
            filter_score=filter_score,
            filter_issues=[r.message for r in filter_results if not r.passed],
            scoring_passed=scoring_passed,
            scoring_score=scoring_score,
            component_scores=scoring_result.component_scores if scoring_result else {},
            evaluated_at=datetime.now(),
            evaluation_duration=(datetime.now() - start_time).total_seconds()
        )
        
        # Step 6: Create evaluation result
        result = EvaluationResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            evaluation_metrics=evaluation_metrics,
            filter_results=filter_results,
            scoring_result=scoring_result,
            recommendations=recommendations,
            warnings=warnings,
            evaluation_details={
                'filter_config': self.config.filter_config.__dict__,
                'scoring_config': self.config.scoring_config.__dict__,
                'evaluation_config': self.config.__dict__
            }
        )
        
        self.logger.info(f"Strategy {strategy_name} evaluated: {normalized_score:.1f}% ({evaluation_metrics.status.value})")
        
        return result
    
    def evaluate_strategies(
        self, 
        strategies: List[Dict[str, Any]], 
        backtest_results: Optional[List[Dict[str, Any]]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """Evaluate multiple strategies"""
        
        results = []
        
        for i, strategy in enumerate(strategies):
            bt_results = backtest_results[i] if backtest_results and i < len(backtest_results) else None
            result = self.evaluate_strategy(strategy, bt_results, benchmark_results)
            results.append(result)
        
        # Rank strategies if enabled
        if self.config.enable_ranking:
            results = self._rank_strategies(results)
        
        return results
    
    def get_passing_strategies(
        self, 
        strategies: List[Dict[str, Any]], 
        backtest_results: Optional[List[Dict[str, Any]]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """Get strategies that pass evaluation"""
        
        all_results = self.evaluate_strategies(strategies, backtest_results, benchmark_results)
        
        passing_results = [
            result for result in all_results
            if result.evaluation_metrics.normalized_score >= self.config.min_total_score
        ]
        
        return passing_results
    
    def get_top_strategies(
        self, 
        strategies: List[Dict[str, Any]], 
        backtest_results: Optional[List[Dict[str, Any]]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None,
        top_n: int = 10
    ) -> List[EvaluationResult]:
        """Get top N strategies by score"""
        
        all_results = self.evaluate_strategies(strategies, backtest_results, benchmark_results)
        
        # Sort by normalized score (descending)
        sorted_results = sorted(all_results, key=lambda x: x.evaluation_metrics.normalized_score, reverse=True)
        
        return sorted_results[:top_n]
    
    def _calculate_filter_score(self, filter_results: List[FilterResult]) -> float:
        """Calculate overall filter score"""
        
        if not filter_results:
            return 0.0
        
        # Calculate weighted average of filter scores
        total_score = 0.0
        total_weight = 0.0
        
        for result in filter_results:
            weight = 1.0  # Equal weight for all filters
            total_score += result.score * weight
            total_weight += weight
        
        return (total_score / total_weight) * 100.0 if total_weight > 0 else 0.0
    
    def _calculate_total_score(self, filter_score: float, scoring_score: float, 
                             filter_passed: bool, scoring_passed: bool) -> float:
        """Calculate total evaluation score"""
        
        if not filter_passed:
            # If filter fails, total score is limited by filter score
            return min(filter_score, self.config.min_total_score)
        
        if scoring_score == 0.0:
            # If no scoring available, use filter score
            return filter_score
        
        # Weighted combination of filter and scoring
        filter_weight = 0.3
        scoring_weight = 0.7
        
        total_score = (filter_score * filter_weight + scoring_score * scoring_weight)
        
        return total_score
    
    def _rank_strategies(self, results: List[EvaluationResult]) -> List[EvaluationResult]:
        """Rank strategies based on evaluation metrics"""
        
        if not results:
            return results
        
        # Create DataFrame for ranking
        data = []
        for i, result in enumerate(results):
            data.append({
                'index': i,
                'strategy_id': result.strategy_id,
                'total_score': result.evaluation_metrics.total_score,
                'normalized_score': result.evaluation_metrics.normalized_score,
                'filter_score': result.evaluation_metrics.filter_score,
                'scoring_score': result.evaluation_metrics.scoring_score
            })
        
        df = pd.DataFrame(data)
        
        # Calculate ranks for different metrics
        df['performance_rank'] = df['normalized_score'].rank(ascending=False, method='dense')
        df['filter_rank'] = df['filter_score'].rank(ascending=False, method='dense')
        df['scoring_rank'] = df['scoring_score'].rank(ascending=False, method='dense')
        
        # Update results with ranks
        for _, row in df.iterrows():
            idx = int(row['index'])
            results[idx].evaluation_metrics.performance_rank = int(row['performance_rank'])
            results[idx].evaluation_metrics.risk_rank = int(row['filter_rank'])
            results[idx].evaluation_metrics.consistency_rank = int(row['scoring_rank'])
        
        return results
    
    def _generate_recommendations(
        self, 
        filter_results: List[FilterResult], 
        scoring_result: Optional[ScoringResult], 
        total_score: float
    ) -> List[str]:
        """Generate recommendations for strategy improvement"""
        
        if not self.config.generate_recommendations:
            return []
        
        recommendations = []
        
        # Filter-based recommendations
        for result in filter_results:
            if not result.passed:
                if result.criteria.value == 'completeness':
                    recommendations.append("Add missing required fields to strategy definition")
                elif result.criteria.value == 'logical_consistency':
                    recommendations.append("Review and fix logical inconsistencies in strategy rules")
                elif result.criteria.value == 'feasibility':
                    recommendations.append("Simplify strategy to use only supported indicators and timeframes")
                elif result.criteria.value == 'parameter_bounds':
                    recommendations.append("Adjust parameters to fall within acceptable bounds")
                elif result.criteria.value == 'performance_threshold':
                    recommendations.append("Improve strategy performance metrics")
                elif result.criteria.value == 'risk_threshold':
                    recommendations.append("Reduce strategy risk exposure")
                elif result.criteria.value == 'trade_frequency':
                    recommendations.append("Adjust trade frequency to optimal range")
        
        # Scoring-based recommendations
        if scoring_result:
            for component, score in scoring_result.component_scores.items():
                if score < 3.0:  # Low score
                    if component.value == 'statistical_significance':
                        recommendations.append("Improve statistical significance of strategy performance")
                    elif component.value == 'risk_adjusted_return':
                        recommendations.append("Enhance risk-adjusted return metrics")
                    elif component.value == 'drawdown_profile':
                        recommendations.append("Reduce maximum drawdown and improve recovery time")
                    elif component.value == 'consistency':
                        recommendations.append("Improve strategy consistency across different market conditions")
                    elif component.value == 'robustness':
                        recommendations.append("Enhance strategy robustness through walk-forward analysis")
                    elif component.value == 'execution_feasibility':
                        recommendations.append("Optimize strategy for better execution feasibility")
                    elif component.value == 'complexity_plausibility':
                        recommendations.append("Simplify strategy complexity for better interpretability")
        
        # Overall recommendations
        if total_score < 70:
            recommendations.append("Consider significant strategy improvements or alternative approaches")
        elif total_score < 80:
            recommendations.append("Minor improvements needed to reach good performance level")
        
        return recommendations
    
    def _generate_warnings(
        self, 
        filter_results: List[FilterResult], 
        scoring_result: Optional[ScoringResult], 
        total_score: float
    ) -> List[str]:
        """Generate warnings for strategy evaluation"""
        
        if not self.config.generate_warnings:
            return []
        
        warnings = []
        
        # Filter-based warnings
        for result in filter_results:
            if result.status.value == 'warning':
                warnings.append(f"Warning: {result.message}")
        
        # Scoring-based warnings
        if scoring_result:
            for component, score in scoring_result.component_scores.items():
                if 2.0 <= score < 3.0:  # Warning range
                    warnings.append(f"Warning: {component.value} score is low ({score:.1f}/5.0)")
        
        # Overall warnings
        if 60 <= total_score < 70:
            warnings.append("Strategy performance is below acceptable threshold")
        elif total_score < 60:
            warnings.append("Strategy failed evaluation and requires significant improvements")
        
        return warnings

