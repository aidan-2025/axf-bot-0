"""
Scoring Engine

Comprehensive scoring system for trading strategies based on performance,
risk, consistency, and efficiency metrics.
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


class ScoreComponent(Enum):
    """Components of the scoring system"""
    PERFORMANCE = "performance"
    RISK = "risk"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    BENCHMARK = "benchmark"


class ScoringMethod(Enum):
    """Scoring methods"""
    WEIGHTED_SUM = "weighted_sum"
    NORMALIZED_SUM = "normalized_sum"
    RANK_BASED = "rank_based"
    PERCENTILE = "percentile"
    Z_SCORE = "z_score"


@dataclass
class ScoringWeights:
    """Weights for different scoring components"""
    
    performance_weight: float = 0.4
    risk_weight: float = 0.3
    consistency_weight: float = 0.2
    efficiency_weight: float = 0.1
    
    # Sub-weights for performance
    return_weight: float = 0.4
    profit_factor_weight: float = 0.3
    win_rate_weight: float = 0.3
    
    # Sub-weights for risk
    drawdown_weight: float = 0.4
    var_weight: float = 0.3
    volatility_weight: float = 0.3
    
    # Sub-weights for consistency
    stability_weight: float = 0.4
    reliability_weight: float = 0.3
    predictability_weight: float = 0.3
    
    # Sub-weights for efficiency
    sharpe_weight: float = 0.4
    calmar_weight: float = 0.3
    sortino_weight: float = 0.3
    
    # Benchmark comparison
    benchmark_weight: float = 0.2
    
    def normalize_weights(self) -> 'ScoringWeights':
        """Normalize weights to sum to 1.0"""
        total = (self.performance_weight + self.risk_weight + 
                self.consistency_weight + self.efficiency_weight)
        
        if total > 0:
            return ScoringWeights(
                performance_weight=self.performance_weight / total,
                risk_weight=self.risk_weight / total,
                consistency_weight=self.consistency_weight / total,
                efficiency_weight=self.efficiency_weight / total,
                return_weight=self.return_weight,
                profit_factor_weight=self.profit_factor_weight,
                win_rate_weight=self.win_rate_weight,
                drawdown_weight=self.drawdown_weight,
                var_weight=self.var_weight,
                volatility_weight=self.volatility_weight,
                stability_weight=self.stability_weight,
                reliability_weight=self.reliability_weight,
                predictability_weight=self.predictability_weight,
                sharpe_weight=self.sharpe_weight,
                calmar_weight=self.calmar_weight,
                sortino_weight=self.sortino_weight,
                benchmark_weight=self.benchmark_weight
            )
        
        return self


@dataclass
class ScoringConfig:
    """Configuration for the scoring engine"""
    
    # Scoring method
    method: ScoringMethod = ScoringMethod.WEIGHTED_SUM
    
    # Weights
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    
    # Benchmark comparison
    enable_benchmark_comparison: bool = True
    benchmark_weight: float = 0.2
    
    # Risk adjustment
    enable_risk_adjustment: bool = True
    risk_free_rate: float = 0.02
    
    # Consistency penalty
    enable_consistency_penalty: bool = True
    consistency_penalty_factor: float = 0.1
    
    # Efficiency bonus
    enable_efficiency_bonus: bool = True
    efficiency_bonus_factor: float = 0.05
    
    # Normalization
    enable_normalization: bool = True
    normalization_method: str = "min_max"  # min_max, z_score, percentile
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Output settings
    include_breakdown: bool = True
    include_rankings: bool = True
    include_percentiles: bool = True


@dataclass
class ScoreBreakdown:
    """Breakdown of scores by component"""
    
    performance_score: float = 0.0
    risk_score: float = 0.0
    consistency_score: float = 0.0
    efficiency_score: float = 0.0
    robustness_score: float = 0.0
    benchmark_score: float = 0.0
    
    # Sub-scores
    return_score: float = 0.0
    profit_factor_score: float = 0.0
    win_rate_score: float = 0.0
    drawdown_score: float = 0.0
    var_score: float = 0.0
    volatility_score: float = 0.0
    stability_score: float = 0.0
    reliability_score: float = 0.0
    predictability_score: float = 0.0
    sharpe_score: float = 0.0
    calmar_score: float = 0.0
    sortino_score: float = 0.0
    
    # Adjustments
    risk_adjustment: float = 0.0
    consistency_penalty: float = 0.0
    efficiency_bonus: float = 0.0
    
    # Final scores
    total_score: float = 0.0
    normalized_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'performance_score': self.performance_score,
            'risk_score': self.risk_score,
            'consistency_score': self.consistency_score,
            'efficiency_score': self.efficiency_score,
            'robustness_score': self.robustness_score,
            'benchmark_score': self.benchmark_score,
            'return_score': self.return_score,
            'profit_factor_score': self.profit_factor_score,
            'win_rate_score': self.win_rate_score,
            'drawdown_score': self.drawdown_score,
            'var_score': self.var_score,
            'volatility_score': self.volatility_score,
            'stability_score': self.stability_score,
            'reliability_score': self.reliability_score,
            'predictability_score': self.predictability_score,
            'sharpe_score': self.sharpe_score,
            'calmar_score': self.calmar_score,
            'sortino_score': self.sortino_score,
            'risk_adjustment': self.risk_adjustment,
            'consistency_penalty': self.consistency_penalty,
            'efficiency_bonus': self.efficiency_bonus,
            'total_score': self.total_score,
            'normalized_score': self.normalized_score
        }


@dataclass
class ScoringResult:
    """Result of scoring a strategy"""
    
    strategy_id: str
    strategy_name: str
    total_score: float
    normalized_score: float
    breakdown: ScoreBreakdown
    
    # Rankings
    performance_rank: Optional[int] = None
    risk_rank: Optional[int] = None
    consistency_rank: Optional[int] = None
    efficiency_rank: Optional[int] = None
    overall_rank: Optional[int] = None
    
    # Percentiles
    performance_percentile: Optional[float] = None
    risk_percentile: Optional[float] = None
    consistency_percentile: Optional[float] = None
    efficiency_percentile: Optional[float] = None
    overall_percentile: Optional[float] = None
    
    # Metadata
    scoring_timestamp: datetime = field(default_factory=datetime.now)
    scoring_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'total_score': self.total_score,
            'normalized_score': self.normalized_score,
            'breakdown': self.breakdown.to_dict(),
            'performance_rank': self.performance_rank,
            'risk_rank': self.risk_rank,
            'consistency_rank': self.consistency_rank,
            'efficiency_rank': self.efficiency_rank,
            'overall_rank': self.overall_rank,
            'performance_percentile': self.performance_percentile,
            'risk_percentile': self.risk_percentile,
            'consistency_percentile': self.consistency_percentile,
            'efficiency_percentile': self.efficiency_percentile,
            'overall_percentile': self.overall_percentile,
            'scoring_timestamp': self.scoring_timestamp.isoformat(),
            'scoring_duration': self.scoring_duration
        }


class ScoringEngine:
    """Comprehensive scoring engine for trading strategies"""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Normalize weights
        self.weights = config.weights.normalize_weights()
    
    def score_strategies(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: List[Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> List[ScoringResult]:
        """Score multiple strategies"""
        
        self.logger.info(f"Scoring {len(strategies)} strategies")
        
        if self.config.enable_parallel_processing and len(strategies) > 4:
            return self._score_strategies_parallel(strategies, backtest_results, benchmark_results)
        else:
            return self._score_strategies_sequential(strategies, backtest_results, benchmark_results)
    
    def _score_strategies_sequential(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: List[Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> List[ScoringResult]:
        """Score strategies sequentially"""
        
        results = []
        
        for strategy in strategies:
            try:
                # Find corresponding backtest result
                backtest_result = self._find_backtest_result(strategy, backtest_results)
                
                if backtest_result:
                    # Score the strategy
                    scoring_result = self._score_single_strategy(
                        strategy, backtest_result, benchmark_results
                    )
                    results.append(scoring_result)
                else:
                    self.logger.warning(f"No backtest result found for strategy {strategy.get('strategy_id', 'unknown')}")
            
            except Exception as e:
                self.logger.error(f"Error scoring strategy {strategy.get('strategy_id', 'unknown')}: {e}")
        
        # Add rankings and percentiles
        if self.config.include_rankings or self.config.include_percentiles:
            self._add_rankings_and_percentiles(results)
        
        return results
    
    def _score_strategies_parallel(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: List[Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> List[ScoringResult]:
        """Score strategies in parallel"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit scoring tasks
            futures = []
            for strategy in strategies:
                backtest_result = self._find_backtest_result(strategy, backtest_results)
                if backtest_result:
                    future = executor.submit(
                        self._score_single_strategy,
                        strategy, backtest_result, benchmark_results
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel scoring: {e}")
        
        # Add rankings and percentiles
        if self.config.include_rankings or self.config.include_percentiles:
            self._add_rankings_and_percentiles(results)
        
        return results
    
    def _score_single_strategy(
        self,
        strategy: Dict[str, Any],
        backtest_result: Dict[str, Any],
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        """Score a single strategy"""
        
        start_time = datetime.now()
        
        # Create breakdown
        breakdown = ScoreBreakdown()
        
        # Calculate component scores
        self._calculate_performance_score(backtest_result, breakdown)
        self._calculate_risk_score(backtest_result, breakdown)
        self._calculate_consistency_score(backtest_result, breakdown)
        self._calculate_efficiency_score(backtest_result, breakdown)
        self._calculate_robustness_score(backtest_result, breakdown)
        
        # Calculate benchmark score if available
        if benchmark_results and self.config.enable_benchmark_comparison:
            self._calculate_benchmark_score(backtest_result, benchmark_results, breakdown)
        
        # Apply adjustments
        self._apply_risk_adjustment(backtest_result, breakdown)
        self._apply_consistency_penalty(backtest_result, breakdown)
        self._apply_efficiency_bonus(backtest_result, breakdown)
        
        # Calculate total score
        self._calculate_total_score(breakdown)
        
        # Normalize score
        if self.config.enable_normalization:
            self._normalize_score(breakdown)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = ScoringResult(
            strategy_id=strategy.get('strategy_id', 'unknown'),
            strategy_name=strategy.get('strategy_name', 'Unknown Strategy'),
            total_score=breakdown.total_score,
            normalized_score=breakdown.normalized_score,
            breakdown=breakdown,
            scoring_duration=duration
        )
        
        return result
    
    def _calculate_performance_score(self, backtest_result: Dict[str, Any], breakdown: ScoreBreakdown) -> None:
        """Calculate performance score"""
        
        # Return score
        net_profit = backtest_result.get('net_profit', 0)
        initial_capital = backtest_result.get('initial_capital', 100000)
        return_rate = net_profit / initial_capital if initial_capital > 0 else 0
        breakdown.return_score = min(100, max(0, return_rate * 1000))  # Scale to 0-100
        
        # Profit factor score
        profit_factor = backtest_result.get('profit_factor', 0)
        breakdown.profit_factor_score = min(100, max(0, (profit_factor - 1) * 50))  # Scale to 0-100
        
        # Win rate score
        win_rate = backtest_result.get('win_rate', 0)
        breakdown.win_rate_score = min(100, max(0, win_rate * 100))  # Scale to 0-100
        
        # Overall performance score
        breakdown.performance_score = (
            breakdown.return_score * self.weights.return_weight +
            breakdown.profit_factor_score * self.weights.profit_factor_weight +
            breakdown.win_rate_score * self.weights.win_rate_weight
        )
    
    def _calculate_risk_score(self, backtest_result: Dict[str, Any], breakdown: ScoreBreakdown) -> None:
        """Calculate risk score"""
        
        # Drawdown score (inverted - lower drawdown is better)
        max_drawdown = backtest_result.get('max_drawdown', 0)
        breakdown.drawdown_score = max(0, 100 - max_drawdown * 1000)  # Scale to 0-100
        
        # VaR score (inverted - lower VaR is better)
        var_95 = backtest_result.get('var_95', 0)
        breakdown.var_score = max(0, 100 - var_95 * 2000)  # Scale to 0-100
        
        # Volatility score (inverted - lower volatility is better)
        volatility = backtest_result.get('volatility', 0)
        breakdown.volatility_score = max(0, 100 - volatility * 1000)  # Scale to 0-100
        
        # Overall risk score
        breakdown.risk_score = (
            breakdown.drawdown_score * self.weights.drawdown_weight +
            breakdown.var_score * self.weights.var_weight +
            breakdown.volatility_score * self.weights.volatility_weight
        )
    
    def _calculate_consistency_score(self, backtest_result: Dict[str, Any], breakdown: ScoreBreakdown) -> None:
        """Calculate consistency score"""
        
        # Stability score
        stability_score = backtest_result.get('stability_score', 0)
        breakdown.stability_score = min(100, max(0, stability_score * 100))
        
        # Reliability score
        reliability_score = backtest_result.get('reliability_score', 0)
        breakdown.reliability_score = min(100, max(0, reliability_score * 100))
        
        # Predictability score
        predictability_score = backtest_result.get('predictability_score', 0)
        breakdown.predictability_score = min(100, max(0, predictability_score * 100))
        
        # Overall consistency score
        breakdown.consistency_score = (
            breakdown.stability_score * self.weights.stability_weight +
            breakdown.reliability_score * self.weights.reliability_weight +
            breakdown.predictability_score * self.weights.predictability_weight
        )
    
    def _calculate_efficiency_score(self, backtest_result: Dict[str, Any], breakdown: ScoreBreakdown) -> None:
        """Calculate efficiency score"""
        
        # Sharpe ratio score
        sharpe_ratio = backtest_result.get('sharpe_ratio', 0)
        breakdown.sharpe_score = min(100, max(0, sharpe_ratio * 20))  # Scale to 0-100
        
        # Calmar ratio score
        calmar_ratio = backtest_result.get('calmar_ratio', 0)
        breakdown.calmar_score = min(100, max(0, calmar_ratio * 10))  # Scale to 0-100
        
        # Sortino ratio score
        sortino_ratio = backtest_result.get('sortino_ratio', 0)
        breakdown.sortino_score = min(100, max(0, sortino_ratio * 20))  # Scale to 0-100
        
        # Overall efficiency score
        breakdown.efficiency_score = (
            breakdown.sharpe_score * self.weights.sharpe_weight +
            breakdown.calmar_score * self.weights.calmar_weight +
            breakdown.sortino_score * self.weights.sortino_weight
        )
    
    def _calculate_robustness_score(self, backtest_result: Dict[str, Any], breakdown: ScoreBreakdown) -> None:
        """Calculate robustness score"""
        
        # Robustness score
        robustness_score = backtest_result.get('robustness_score', 0)
        breakdown.robustness_score = min(100, max(0, robustness_score * 100))
    
    def _calculate_benchmark_score(
        self,
        backtest_result: Dict[str, Any],
        benchmark_results: Dict[str, Any],
        breakdown: ScoreBreakdown
    ) -> None:
        """Calculate benchmark comparison score"""
        
        # Compare key metrics with benchmark
        strategy_return = backtest_result.get('net_profit', 0) / backtest_result.get('initial_capital', 100000)
        benchmark_return = benchmark_results.get('total_return', 0)
        
        # Calculate relative performance
        relative_performance = strategy_return - benchmark_return
        breakdown.benchmark_score = min(100, max(0, 50 + relative_performance * 1000))
    
    def _apply_risk_adjustment(self, backtest_result: Dict[str, Any], breakdown: ScoreBreakdown) -> None:
        """Apply risk adjustment to scores"""
        
        if not self.config.enable_risk_adjustment:
            return
        
        # Calculate risk adjustment based on volatility
        volatility = backtest_result.get('volatility', 0)
        risk_adjustment = volatility * 0.1  # Default risk adjustment factor
        
        breakdown.risk_adjustment = risk_adjustment
        breakdown.total_score -= risk_adjustment
    
    def _apply_consistency_penalty(self, backtest_result: Dict[str, Any], breakdown: ScoreBreakdown) -> None:
        """Apply consistency penalty to scores"""
        
        if not self.config.enable_consistency_penalty:
            return
        
        # Calculate consistency penalty based on consistency score
        consistency_score = backtest_result.get('consistency_score', 0)
        penalty = (1 - consistency_score) * self.config.consistency_penalty_factor * 100
        
        breakdown.consistency_penalty = penalty
        breakdown.total_score -= penalty
    
    def _apply_efficiency_bonus(self, backtest_result: Dict[str, Any], breakdown: ScoreBreakdown) -> None:
        """Apply efficiency bonus to scores"""
        
        if not self.config.enable_efficiency_bonus:
            return
        
        # Calculate efficiency bonus based on efficiency score
        efficiency_score = backtest_result.get('efficiency_score', 0)
        bonus = efficiency_score * self.config.efficiency_bonus_factor * 100
        
        breakdown.efficiency_bonus = bonus
        breakdown.total_score += bonus
    
    def _calculate_total_score(self, breakdown: ScoreBreakdown) -> None:
        """Calculate total weighted score"""
        
        breakdown.total_score = (
            breakdown.performance_score * self.weights.performance_weight +
            breakdown.risk_score * self.weights.risk_weight +
            breakdown.consistency_score * self.weights.consistency_weight +
            breakdown.efficiency_score * self.weights.efficiency_weight +
            breakdown.robustness_score * 0.1 +  # Small weight for robustness
            breakdown.benchmark_score * self.weights.benchmark_weight
        )
    
    def _normalize_score(self, breakdown: ScoreBreakdown) -> None:
        """Normalize score to 0-100 range"""
        
        if self.config.normalization_method == "min_max":
            # Min-max normalization
            breakdown.normalized_score = max(0, min(100, breakdown.total_score))
        elif self.config.normalization_method == "z_score":
            # Z-score normalization (would need population statistics)
            breakdown.normalized_score = max(0, min(100, 50 + breakdown.total_score))
        elif self.config.normalization_method == "percentile":
            # Percentile normalization (would need population statistics)
            breakdown.normalized_score = max(0, min(100, breakdown.total_score))
        else:
            breakdown.normalized_score = breakdown.total_score
    
    def _add_rankings_and_percentiles(self, results: List[ScoringResult]) -> None:
        """Add rankings and percentiles to results"""
        
        if not results:
            return
        
        # Sort by normalized score
        results.sort(key=lambda x: x.normalized_score, reverse=True)
        
        # Add rankings
        for i, result in enumerate(results):
            result.overall_rank = i + 1
        
        # Add percentiles
        n = len(results)
        for i, result in enumerate(results):
            result.overall_percentile = (n - i) / n * 100
        
        # Component rankings and percentiles
        self._add_component_rankings(results)
    
    def _add_component_rankings(self, results: List[ScoringResult]) -> None:
        """Add component-specific rankings and percentiles"""
        
        if not results:
            return
        
        # Performance rankings
        performance_scores = [r.breakdown.performance_score for r in results]
        performance_ranks = self._calculate_ranks(performance_scores, reverse=True)
        for i, result in enumerate(results):
            result.performance_rank = performance_ranks[i]
            result.performance_percentile = (len(results) - performance_ranks[i] + 1) / len(results) * 100
        
        # Risk rankings
        risk_scores = [r.breakdown.risk_score for r in results]
        risk_ranks = self._calculate_ranks(risk_scores, reverse=True)
        for i, result in enumerate(results):
            result.risk_rank = risk_ranks[i]
            result.risk_percentile = (len(results) - risk_ranks[i] + 1) / len(results) * 100
        
        # Consistency rankings
        consistency_scores = [r.breakdown.consistency_score for r in results]
        consistency_ranks = self._calculate_ranks(consistency_scores, reverse=True)
        for i, result in enumerate(results):
            result.consistency_rank = consistency_ranks[i]
            result.consistency_percentile = (len(results) - consistency_ranks[i] + 1) / len(results) * 100
        
        # Efficiency rankings
        efficiency_scores = [r.breakdown.efficiency_score for r in results]
        efficiency_ranks = self._calculate_ranks(efficiency_scores, reverse=True)
        for i, result in enumerate(results):
            result.efficiency_rank = efficiency_ranks[i]
            result.efficiency_percentile = (len(results) - efficiency_ranks[i] + 1) / len(results) * 100
    
    def _calculate_ranks(self, scores: List[float], reverse: bool = True) -> List[int]:
        """Calculate ranks for a list of scores"""
        
        # Create list of (score, index) tuples
        score_index_pairs = [(score, i) for i, score in enumerate(scores)]
        
        # Sort by score
        score_index_pairs.sort(key=lambda x: x[0], reverse=reverse)
        
        # Assign ranks
        ranks = [0] * len(scores)
        for rank, (_, index) in enumerate(score_index_pairs, 1):
            ranks[index] = rank
        
        return ranks
    
    def _find_backtest_result(
        self,
        strategy: Dict[str, Any],
        backtest_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find backtest result for a strategy"""
        
        strategy_id = strategy.get('strategy_id')
        if not strategy_id:
            return None
        
        for result in backtest_results:
            if result.get('strategy_id') == strategy_id:
                return result
        
        return None
