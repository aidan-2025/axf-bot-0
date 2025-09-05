"""
Scoring Engine

Comprehensive scoring system for trading strategies based on multiple
criteria including performance, risk, consistency, and robustness.
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

logger = logging.getLogger(__name__)


class ScoreComponent(Enum):
    """Components of the scoring system"""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    DRAWDOWN_PROFILE = "drawdown_profile"
    CONSISTENCY = "consistency"
    TRADE_FREQUENCY = "trade_frequency"
    ROBUSTNESS = "robustness"
    EXECUTION_FEASIBILITY = "execution_feasibility"
    COMPLEXITY_PLAUSIBILITY = "complexity_plausibility"


@dataclass
class ScoringWeights:
    """Weights for different scoring components"""
    
    statistical_significance: float = 0.20
    risk_adjusted_return: float = 0.20
    drawdown_profile: float = 0.15
    consistency: float = 0.10
    trade_frequency: float = 0.10
    robustness: float = 0.15
    execution_feasibility: float = 0.05
    complexity_plausibility: float = 0.05
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = sum([
            self.statistical_significance,
            self.risk_adjusted_return,
            self.drawdown_profile,
            self.consistency,
            self.trade_frequency,
            self.robustness,
            self.execution_feasibility,
            self.complexity_plausibility
        ])
        
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


@dataclass
class ScoringResult:
    """Result of strategy scoring"""
    
    strategy_id: str
    total_score: float
    max_score: float = 100.0
    normalized_score: float = 0.0
    
    # Component scores
    component_scores: Dict[ScoreComponent, float] = field(default_factory=dict)
    
    # Detailed metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Scoring details
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    scored_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate normalized score"""
        self.normalized_score = (self.total_score / self.max_score) * 100.0


@dataclass
class ScoringConfig:
    """Configuration for scoring engine"""
    
    # Scoring weights
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    
    # Statistical significance settings
    significance_level: float = 0.05
    min_sample_size: int = 30
    benchmark_return: float = 0.08  # 8% annual benchmark
    
    # Risk-adjusted return settings
    risk_free_rate: float = 0.02  # 2% risk-free rate
    target_sharpe: float = 1.0
    target_sortino: float = 1.5
    target_calmar: float = 1.0
    
    # Drawdown settings
    max_acceptable_drawdown: float = 0.15  # 15%
    max_acceptable_drawdown_duration: int = 180  # 180 days
    
    # Consistency settings
    min_win_rate: float = 0.4  # 40%
    min_profitable_months: float = 0.6  # 60%
    max_volatility: float = 0.25  # 25%
    
    # Trade frequency settings
    min_trades_per_month: float = 2.0
    max_trades_per_month: float = 50.0
    min_trade_duration_hours: float = 1.0
    max_trade_duration_days: float = 30.0
    
    # Robustness settings
    min_walk_forward_periods: int = 5
    min_monte_carlo_runs: int = 100
    min_correlation_with_benchmark: float = 0.1
    
    # Execution feasibility settings
    max_slippage_impact: float = 0.001  # 0.1%
    max_commission_impact: float = 0.002  # 0.2%
    max_spread_impact: float = 0.0005  # 0.05%
    
    # Complexity settings
    max_parameters: int = 10
    max_indicators: int = 5
    max_rule_complexity: int = 3


class ScoringEngine:
    """Comprehensive scoring engine for trading strategies"""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def score_strategy(
        self, 
        strategy: Dict[str, Any], 
        backtest_results: Dict[str, Any],
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        """Score a strategy based on multiple criteria"""
        
        strategy_id = strategy.get('strategy_id', 'unknown')
        
        # Calculate component scores
        component_scores = {}
        metrics = {}
        details = {}
        
        # Statistical significance
        stat_sig_score, stat_sig_metrics = self._score_statistical_significance(
            backtest_results, benchmark_results
        )
        component_scores[ScoreComponent.STATISTICAL_SIGNIFICANCE] = stat_sig_score
        metrics.update(stat_sig_metrics)
        details['statistical_significance'] = stat_sig_metrics
        
        # Risk-adjusted return
        risk_adj_score, risk_adj_metrics = self._score_risk_adjusted_return(backtest_results)
        component_scores[ScoreComponent.RISK_ADJUSTED_RETURN] = risk_adj_score
        metrics.update(risk_adj_metrics)
        details['risk_adjusted_return'] = risk_adj_metrics
        
        # Drawdown profile
        drawdown_score, drawdown_metrics = self._score_drawdown_profile(backtest_results)
        component_scores[ScoreComponent.DRAWDOWN_PROFILE] = drawdown_score
        metrics.update(drawdown_metrics)
        details['drawdown_profile'] = drawdown_metrics
        
        # Consistency
        consistency_score, consistency_metrics = self._score_consistency(backtest_results)
        component_scores[ScoreComponent.CONSISTENCY] = consistency_score
        metrics.update(consistency_metrics)
        details['consistency'] = consistency_metrics
        
        # Trade frequency
        trade_freq_score, trade_freq_metrics = self._score_trade_frequency(backtest_results)
        component_scores[ScoreComponent.TRADE_FREQUENCY] = trade_freq_score
        metrics.update(trade_freq_metrics)
        details['trade_frequency'] = trade_freq_metrics
        
        # Robustness
        robustness_score, robustness_metrics = self._score_robustness(backtest_results, strategy)
        component_scores[ScoreComponent.ROBUSTNESS] = robustness_score
        metrics.update(robustness_metrics)
        details['robustness'] = robustness_metrics
        
        # Execution feasibility
        execution_score, execution_metrics = self._score_execution_feasibility(backtest_results, strategy)
        component_scores[ScoreComponent.EXECUTION_FEASIBILITY] = execution_score
        metrics.update(execution_metrics)
        details['execution_feasibility'] = execution_metrics
        
        # Complexity/plausibility
        complexity_score, complexity_metrics = self._score_complexity_plausibility(strategy)
        component_scores[ScoreComponent.COMPLEXITY_PLAUSIBILITY] = complexity_score
        metrics.update(complexity_metrics)
        details['complexity_plausibility'] = complexity_metrics
        
        # Calculate total score
        total_score = 0.0
        for component, score in component_scores.items():
            weight = getattr(self.config.weights, component.value)
            total_score += score * weight
        
        return ScoringResult(
            strategy_id=strategy_id,
            total_score=total_score,
            component_scores=component_scores,
            metrics=metrics,
            details=details
        )
    
    def score_strategies(
        self, 
        strategies: List[Dict[str, Any]], 
        backtest_results: List[Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> List[ScoringResult]:
        """Score multiple strategies"""
        
        results = []
        
        for i, strategy in enumerate(strategies):
            bt_results = backtest_results[i] if i < len(backtest_results) else {}
            result = self.score_strategy(strategy, bt_results, benchmark_results)
            results.append(result)
        
        return results
    
    def _score_statistical_significance(
        self, 
        backtest_results: Dict[str, Any], 
        benchmark_results: Optional[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """Score statistical significance of outperformance"""
        
        score = 0.0
        metrics = {}
        
        if not benchmark_results:
            # No benchmark to compare against
            metrics['p_value'] = None
            metrics['t_statistic'] = None
            metrics['is_significant'] = False
            metrics['note'] = 'No benchmark data available'
            return 0.5, metrics  # Neutral score
        
        # Extract returns
        strategy_returns = backtest_results.get('returns', [])
        benchmark_returns = benchmark_results.get('returns', [])
        
        if not strategy_returns or not benchmark_returns:
            metrics['p_value'] = None
            metrics['t_statistic'] = None
            metrics['is_significant'] = False
            metrics['note'] = 'Insufficient return data'
            return 0.5, metrics
        
        try:
            # Calculate excess returns
            strategy_returns = np.array(strategy_returns)
            benchmark_returns = np.array(benchmark_returns)
            
            # Align lengths
            min_length = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]
            
            excess_returns = strategy_returns - benchmark_returns
            
            # Perform t-test
            t_statistic, p_value = stats.ttest_1samp(excess_returns, 0)
            
            metrics['p_value'] = float(p_value)
            metrics['t_statistic'] = float(t_statistic)
            metrics['is_significant'] = p_value < self.config.significance_level
            metrics['excess_return_mean'] = float(np.mean(excess_returns))
            metrics['excess_return_std'] = float(np.std(excess_returns))
            
            # Score based on significance and effect size
            if p_value < self.config.significance_level:
                # Significant outperformance
                effect_size = abs(np.mean(excess_returns)) / np.std(excess_returns)
                if effect_size > 0.5:  # Large effect
                    score = 5.0
                elif effect_size > 0.2:  # Medium effect
                    score = 4.0
                else:  # Small effect
                    score = 3.0
            else:
                # Not significant
                score = 1.0
            
        except Exception as e:
            self.logger.warning(f"Statistical significance calculation failed: {e}")
            metrics['error'] = str(e)
            score = 0.0
        
        return score, metrics
    
    def _score_risk_adjusted_return(self, backtest_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score risk-adjusted return metrics"""
        
        score = 0.0
        metrics = {}
        
        # Sharpe ratio
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Sortino ratio
        sortino_ratio = backtest_results.get('sortino_ratio', 0)
        metrics['sortino_ratio'] = sortino_ratio
        
        # Calmar ratio
        calmar_ratio = backtest_results.get('calmar_ratio', 0)
        metrics['calmar_ratio'] = calmar_ratio
        
        # Information ratio
        information_ratio = backtest_results.get('information_ratio', 0)
        metrics['information_ratio'] = information_ratio
        
        # Score based on ratios
        sharpe_score = min(5.0, max(0.0, sharpe_ratio / self.config.target_sharpe * 5.0))
        sortino_score = min(5.0, max(0.0, sortino_ratio / self.config.target_sortino * 5.0))
        calmar_score = min(5.0, max(0.0, calmar_ratio / self.config.target_calmar * 5.0))
        
        # Weighted average
        score = (sharpe_score * 0.4 + sortino_score * 0.3 + calmar_score * 0.3)
        
        metrics['sharpe_score'] = sharpe_score
        metrics['sortino_score'] = sortino_score
        metrics['calmar_score'] = calmar_score
        
        return score, metrics
    
    def _score_drawdown_profile(self, backtest_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score drawdown profile"""
        
        score = 0.0
        metrics = {}
        
        # Max drawdown
        max_drawdown = backtest_results.get('max_drawdown', 1.0)
        metrics['max_drawdown'] = max_drawdown
        
        # Average drawdown
        avg_drawdown = backtest_results.get('avg_drawdown', 1.0)
        metrics['avg_drawdown'] = avg_drawdown
        
        # Drawdown duration
        max_dd_duration = backtest_results.get('max_dd_duration_days', 365)
        metrics['max_dd_duration_days'] = max_dd_duration
        
        # Recovery time
        recovery_time = backtest_results.get('recovery_time_days', 365)
        metrics['recovery_time_days'] = recovery_time
        
        # Score based on drawdown severity and duration
        if max_drawdown <= self.config.max_acceptable_drawdown:
            dd_score = 5.0
        elif max_drawdown <= self.config.max_acceptable_drawdown * 1.5:
            dd_score = 3.0
        else:
            dd_score = 1.0
        
        if max_dd_duration <= self.config.max_acceptable_drawdown_duration:
            duration_score = 5.0
        elif max_dd_duration <= self.config.max_acceptable_drawdown_duration * 1.5:
            duration_score = 3.0
        else:
            duration_score = 1.0
        
        score = (dd_score * 0.6 + duration_score * 0.4)
        
        metrics['dd_score'] = dd_score
        metrics['duration_score'] = duration_score
        
        return score, metrics
    
    def _score_consistency(self, backtest_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score consistency metrics"""
        
        score = 0.0
        metrics = {}
        
        # Win rate
        win_rate = backtest_results.get('win_rate', 0)
        metrics['win_rate'] = win_rate
        
        # Profitable months percentage
        profitable_months = backtest_results.get('profitable_months_pct', 0)
        metrics['profitable_months_pct'] = profitable_months
        
        # Volatility
        volatility = backtest_results.get('volatility', 1.0)
        metrics['volatility'] = volatility
        
        # Rolling Sharpe consistency
        rolling_sharpe_std = backtest_results.get('rolling_sharpe_std', 1.0)
        metrics['rolling_sharpe_std'] = rolling_sharpe_std
        
        # Score win rate
        if win_rate >= 0.6:
            win_rate_score = 5.0
        elif win_rate >= 0.5:
            win_rate_score = 4.0
        elif win_rate >= self.config.min_win_rate:
            win_rate_score = 3.0
        else:
            win_rate_score = 1.0
        
        # Score profitable months
        if profitable_months >= 0.8:
            months_score = 5.0
        elif profitable_months >= 0.7:
            months_score = 4.0
        elif profitable_months >= self.config.min_profitable_months:
            months_score = 3.0
        else:
            months_score = 1.0
        
        # Score volatility (lower is better)
        if volatility <= 0.1:
            vol_score = 5.0
        elif volatility <= 0.15:
            vol_score = 4.0
        elif volatility <= self.config.max_volatility:
            vol_score = 3.0
        else:
            vol_score = 1.0
        
        score = (win_rate_score * 0.4 + months_score * 0.3 + vol_score * 0.3)
        
        metrics['win_rate_score'] = win_rate_score
        metrics['months_score'] = months_score
        metrics['vol_score'] = vol_score
        
        return score, metrics
    
    def _score_trade_frequency(self, backtest_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score trade frequency"""
        
        score = 0.0
        metrics = {}
        
        total_trades = backtest_results.get('total_trades', 0)
        backtest_days = backtest_results.get('backtest_days', 1)
        
        if backtest_days > 0:
            trades_per_month = (total_trades / backtest_days) * 30
            trades_per_day = total_trades / backtest_days
        else:
            trades_per_month = 0
            trades_per_day = 0
        
        metrics['total_trades'] = total_trades
        metrics['trades_per_month'] = trades_per_month
        metrics['trades_per_day'] = trades_per_day
        
        # Score based on frequency
        if self.config.min_trades_per_month <= trades_per_month <= self.config.max_trades_per_month:
            freq_score = 5.0
        elif trades_per_month < self.config.min_trades_per_month:
            # Too few trades
            freq_score = max(1.0, trades_per_month / self.config.min_trades_per_month * 3.0)
        else:
            # Too many trades
            freq_score = max(1.0, self.config.max_trades_per_month / trades_per_month * 3.0)
        
        score = freq_score
        metrics['freq_score'] = freq_score
        
        return score, metrics
    
    def _score_robustness(self, backtest_results: Dict[str, Any], strategy: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score robustness metrics"""
        
        score = 0.0
        metrics = {}
        
        # Walk-forward analysis results
        wf_periods = backtest_results.get('walk_forward_periods', 0)
        wf_consistency = backtest_results.get('walk_forward_consistency', 0)
        metrics['walk_forward_periods'] = wf_periods
        metrics['walk_forward_consistency'] = wf_consistency
        
        # Monte Carlo results
        mc_runs = backtest_results.get('monte_carlo_runs', 0)
        mc_consistency = backtest_results.get('monte_carlo_consistency', 0)
        metrics['monte_carlo_runs'] = mc_runs
        metrics['monte_carlo_consistency'] = mc_consistency
        
        # Parameter sensitivity
        param_sensitivity = backtest_results.get('parameter_sensitivity', 1.0)
        metrics['parameter_sensitivity'] = param_sensitivity
        
        # Score walk-forward
        if wf_periods >= self.config.min_walk_forward_periods:
            wf_score = min(5.0, wf_consistency * 5.0)
        else:
            wf_score = 2.0  # Penalty for insufficient walk-forward analysis
        
        # Score Monte Carlo
        if mc_runs >= self.config.min_monte_carlo_runs:
            mc_score = min(5.0, mc_consistency * 5.0)
        else:
            mc_score = 2.0  # Penalty for insufficient Monte Carlo analysis
        
        # Score parameter sensitivity (lower is better)
        if param_sensitivity <= 0.1:
            param_score = 5.0
        elif param_sensitivity <= 0.3:
            param_score = 3.0
        else:
            param_score = 1.0
        
        score = (wf_score * 0.4 + mc_score * 0.4 + param_score * 0.2)
        
        metrics['wf_score'] = wf_score
        metrics['mc_score'] = mc_score
        metrics['param_score'] = param_score
        
        return score, metrics
    
    def _score_execution_feasibility(self, backtest_results: Dict[str, Any], strategy: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score execution feasibility"""
        
        score = 0.0
        metrics = {}
        
        # Slippage impact
        slippage_impact = backtest_results.get('slippage_impact', 0)
        metrics['slippage_impact'] = slippage_impact
        
        # Commission impact
        commission_impact = backtest_results.get('commission_impact', 0)
        metrics['commission_impact'] = commission_impact
        
        # Spread impact
        spread_impact = backtest_results.get('spread_impact', 0)
        metrics['spread_impact'] = spread_impact
        
        # Score based on execution costs
        slippage_score = 5.0 if slippage_impact <= self.config.max_slippage_impact else max(1.0, 5.0 - (slippage_impact / self.config.max_slippage_impact) * 4.0)
        commission_score = 5.0 if commission_impact <= self.config.max_commission_impact else max(1.0, 5.0 - (commission_impact / self.config.max_commission_impact) * 4.0)
        spread_score = 5.0 if spread_impact <= self.config.max_spread_impact else max(1.0, 5.0 - (spread_impact / self.config.max_spread_impact) * 4.0)
        
        score = (slippage_score * 0.4 + commission_score * 0.3 + spread_score * 0.3)
        
        metrics['slippage_score'] = slippage_score
        metrics['commission_score'] = commission_score
        metrics['spread_score'] = spread_score
        
        return score, metrics
    
    def _score_complexity_plausibility(self, strategy: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score complexity and plausibility"""
        
        score = 0.0
        metrics = {}
        
        # Parameter count
        param_count = len(strategy.get('parameters', {}))
        metrics['parameter_count'] = param_count
        
        # Indicator count
        indicator_count = len(strategy.get('indicators', []))
        metrics['indicator_count'] = indicator_count
        
        # Rule complexity (simplified)
        entry_rules = strategy.get('entry_rules', '')
        exit_rules = strategy.get('exit_rules', '')
        rule_complexity = len(entry_rules.split()) + len(exit_rules.split())
        metrics['rule_complexity'] = rule_complexity
        
        # Score parameter count (fewer is better, up to a point)
        if param_count <= 5:
            param_score = 5.0
        elif param_count <= self.config.max_parameters:
            param_score = 4.0
        else:
            param_score = max(1.0, 5.0 - (param_count - self.config.max_parameters) * 0.5)
        
        # Score indicator count (fewer is better, up to a point)
        if indicator_count <= 3:
            indicator_score = 5.0
        elif indicator_count <= self.config.max_indicators:
            indicator_score = 4.0
        else:
            indicator_score = max(1.0, 5.0 - (indicator_count - self.config.max_indicators) * 0.5)
        
        # Score rule complexity (simpler is better)
        if rule_complexity <= 20:
            rule_score = 5.0
        elif rule_complexity <= 50:
            rule_score = 4.0
        else:
            rule_score = max(1.0, 5.0 - (rule_complexity - 50) * 0.1)
        
        score = (param_score * 0.4 + indicator_score * 0.3 + rule_score * 0.3)
        
        metrics['param_score'] = param_score
        metrics['indicator_score'] = indicator_score
        metrics['rule_score'] = rule_score
        
        return score, metrics

