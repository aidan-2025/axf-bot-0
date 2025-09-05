#!/usr/bin/env python3
"""
Strategy Scorer

Comprehensive scoring system for trading strategies based on multiple performance dimensions.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import logging
from datetime import datetime

from .scoring_weights import ScoringWeights, ScoringCategory
from ..criteria.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ScoringMetrics:
    """Detailed scoring metrics for a strategy"""
    
    # Overall scores
    overall_score: float = 0.0
    performance_score: float = 0.0
    risk_score: float = 0.0
    consistency_score: float = 0.0
    efficiency_score: float = 0.0
    robustness_score: float = 0.0
    
    # Detailed sub-scores
    return_score: float = 0.0
    profit_factor_score: float = 0.0
    win_rate_score: float = 0.0
    drawdown_score: float = 0.0
    volatility_score: float = 0.0
    var_score: float = 0.0
    stability_score: float = 0.0
    consistency_sub_score: float = 0.0
    consecutive_losses_score: float = 0.0
    sharpe_score: float = 0.0
    sortino_score: float = 0.0
    calmar_score: float = 0.0
    event_avoidance_score: float = 0.0
    recovery_factor_score: float = 0.0
    trade_frequency_score: float = 0.0
    
    # Additional metrics
    total_trades: int = 0
    backtest_duration_days: int = 0
    scoring_timestamp: datetime = None
    
    def __post_init__(self):
        if self.scoring_timestamp is None:
            self.scoring_timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scoring metrics to dictionary"""
        return {
            'overall_score': self.overall_score,
            'performance_score': self.performance_score,
            'risk_score': self.risk_score,
            'consistency_score': self.consistency_score,
            'efficiency_score': self.efficiency_score,
            'robustness_score': self.robustness_score,
            'return_score': self.return_score,
            'profit_factor_score': self.profit_factor_score,
            'win_rate_score': self.win_rate_score,
            'drawdown_score': self.drawdown_score,
            'volatility_score': self.volatility_score,
            'var_score': self.var_score,
            'stability_score': self.stability_score,
            'consistency_sub_score': self.consistency_sub_score,
            'consecutive_losses_score': self.consecutive_losses_score,
            'sharpe_score': self.sharpe_score,
            'sortino_score': self.sortino_score,
            'calmar_score': self.calmar_score,
            'event_avoidance_score': self.event_avoidance_score,
            'recovery_factor_score': self.recovery_factor_score,
            'trade_frequency_score': self.trade_frequency_score,
            'total_trades': self.total_trades,
            'backtest_duration_days': self.backtest_duration_days,
            'scoring_timestamp': self.scoring_timestamp.isoformat()
        }


class StrategyScorer:
    """Comprehensive strategy scoring system"""
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()
        if not self.weights.validate_weights():
            raise ValueError("Invalid scoring weights - weights must sum to 1.0 within categories")
        
        logger.info("StrategyScorer initialized with custom weights")
    
    def score_strategy(self, performance_metrics: PerformanceMetrics, 
                      trades: List[Dict[str, Any]] = None,
                      backtest_duration_days: int = 0) -> ScoringMetrics:
        """Score a strategy based on performance metrics"""
        logger.info("Starting strategy scoring")
        
        scoring_metrics = ScoringMetrics()
        scoring_metrics.total_trades = performance_metrics.total_trades
        scoring_metrics.backtest_duration_days = backtest_duration_days
        
        # Calculate performance score
        scoring_metrics.performance_score = self._calculate_performance_score(performance_metrics)
        
        # Calculate risk score
        scoring_metrics.risk_score = self._calculate_risk_score(performance_metrics)
        
        # Calculate consistency score
        scoring_metrics.consistency_score = self._calculate_consistency_score(performance_metrics)
        
        # Calculate efficiency score
        scoring_metrics.efficiency_score = self._calculate_efficiency_score(performance_metrics)
        
        # Calculate robustness score
        scoring_metrics.robustness_score = self._calculate_robustness_score(
            performance_metrics, trades, backtest_duration_days
        )
        
        # Calculate overall score
        scoring_metrics.overall_score = self._calculate_overall_score(scoring_metrics)
        
        logger.info(f"Strategy scoring completed. Overall score: {scoring_metrics.overall_score:.3f}")
        
        return scoring_metrics
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate performance score (0.0 to 1.0)"""
        # Return score (normalized annualized return)
        return_score = self._normalize_return(metrics.annualized_return)
        
        # Profit factor score
        profit_factor_score = self._normalize_profit_factor(metrics.profit_factor)
        
        # Win rate score
        win_rate_score = self._normalize_win_rate(metrics.win_rate)
        
        # Weighted performance score
        performance_score = (
            return_score * self.weights.return_weight +
            profit_factor_score * self.weights.profit_factor_weight +
            win_rate_score * self.weights.win_rate_weight
        )
        
        return min(1.0, max(0.0, performance_score))
    
    def _calculate_risk_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate risk score (0.0 to 1.0, higher is better)"""
        # Drawdown score (inverse of drawdown)
        drawdown_score = self._normalize_drawdown(metrics.max_drawdown)
        
        # Volatility score (inverse of volatility)
        volatility_score = self._normalize_volatility(metrics.volatility)
        
        # VaR score (inverse of VaR)
        var_score = self._normalize_var(metrics.var_95)
        
        # Weighted risk score
        risk_score = (
            drawdown_score * self.weights.drawdown_weight +
            volatility_score * self.weights.volatility_weight +
            var_score * self.weights.var_weight
        )
        
        return min(1.0, max(0.0, risk_score))
    
    def _calculate_consistency_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate consistency score (0.0 to 1.0)"""
        # Stability score
        stability_score = metrics.stability_score
        
        # Consistency score
        consistency_sub_score = metrics.consistency_score
        
        # Consecutive losses score (inverse of consecutive losses)
        consecutive_losses_score = self._normalize_consecutive_losses(metrics.max_consecutive_losses)
        
        # Weighted consistency score
        consistency_score = (
            stability_score * self.weights.stability_weight +
            consistency_sub_score * self.weights.consistency_weight +
            consecutive_losses_score * self.weights.consecutive_losses_weight
        )
        
        return min(1.0, max(0.0, consistency_score))
    
    def _calculate_efficiency_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate efficiency score (0.0 to 1.0)"""
        # Sharpe ratio score
        sharpe_score = self._normalize_sharpe_ratio(metrics.sharpe_ratio)
        
        # Sortino ratio score
        sortino_score = self._normalize_sortino_ratio(metrics.sortino_ratio)
        
        # Calmar ratio score
        calmar_score = self._normalize_calmar_ratio(metrics.calmar_ratio)
        
        # Weighted efficiency score
        efficiency_score = (
            sharpe_score * self.weights.sharpe_weight +
            sortino_score * self.weights.sortino_weight +
            calmar_score * self.weights.calmar_weight
        )
        
        return min(1.0, max(0.0, efficiency_score))
    
    def _calculate_robustness_score(self, metrics: PerformanceMetrics, 
                                  trades: List[Dict[str, Any]] = None,
                                  backtest_duration_days: int = 0) -> float:
        """Calculate robustness score (0.0 to 1.0)"""
        # Event avoidance score (placeholder - would need economic event data)
        event_avoidance_score = metrics.event_avoidance_score
        
        # Recovery factor score
        recovery_factor_score = self._normalize_recovery_factor(metrics.recovery_factor)
        
        # Trade frequency score
        trade_frequency_score = self._normalize_trade_frequency(
            metrics.total_trades, backtest_duration_days
        )
        
        # Weighted robustness score
        robustness_score = (
            event_avoidance_score * self.weights.event_avoidance_weight +
            recovery_factor_score * self.weights.recovery_factor_weight +
            trade_frequency_score * self.weights.trade_frequency_weight
        )
        
        return min(1.0, max(0.0, robustness_score))
    
    def _calculate_overall_score(self, scoring_metrics: ScoringMetrics) -> float:
        """Calculate overall weighted score"""
        category_weights = self.weights.get_category_weights()
        
        overall_score = (
            scoring_metrics.performance_score * category_weights[ScoringCategory.PERFORMANCE] +
            scoring_metrics.risk_score * category_weights[ScoringCategory.RISK] +
            scoring_metrics.consistency_score * category_weights[ScoringCategory.CONSISTENCY] +
            scoring_metrics.efficiency_score * category_weights[ScoringCategory.EFFICIENCY] +
            scoring_metrics.robustness_score * category_weights[ScoringCategory.ROBUSTNESS]
        )
        
        return min(1.0, max(0.0, overall_score))
    
    # Normalization functions
    def _normalize_return(self, annualized_return: float) -> float:
        """Normalize annualized return to 0-1 scale"""
        # Target: 20% annual return = 1.0, 0% = 0.5, negative = 0.0
        if annualized_return <= 0:
            return 0.0
        elif annualized_return >= 0.20:
            return 1.0
        else:
            return 0.5 + (annualized_return / 0.20) * 0.5
    
    def _normalize_profit_factor(self, profit_factor: float) -> float:
        """Normalize profit factor to 0-1 scale"""
        # Target: 2.0 = 1.0, 1.0 = 0.0, 3.0+ = 1.0
        if profit_factor <= 1.0:
            return 0.0
        elif profit_factor >= 3.0:
            return 1.0
        else:
            return (profit_factor - 1.0) / 2.0
    
    def _normalize_win_rate(self, win_rate: float) -> float:
        """Normalize win rate to 0-1 scale"""
        # Target: 60% = 1.0, 30% = 0.0, 80%+ = 1.0
        if win_rate <= 0.30:
            return 0.0
        elif win_rate >= 0.80:
            return 1.0
        else:
            return (win_rate - 0.30) / 0.50
    
    def _normalize_drawdown(self, max_drawdown: float) -> float:
        """Normalize drawdown to 0-1 scale (inverse)"""
        # Target: 0% = 1.0, 20% = 0.0, 5% = 0.75
        if max_drawdown <= 0.05:
            return 1.0
        elif max_drawdown >= 0.20:
            return 0.0
        else:
            return 1.0 - (max_drawdown - 0.05) / 0.15
    
    def _normalize_volatility(self, volatility: float) -> float:
        """Normalize volatility to 0-1 scale (inverse)"""
        # Target: 5% = 1.0, 30% = 0.0, 15% = 0.5
        if volatility <= 0.05:
            return 1.0
        elif volatility >= 0.30:
            return 0.0
        else:
            return 1.0 - (volatility - 0.05) / 0.25
    
    def _normalize_var(self, var_95: float) -> float:
        """Normalize VaR to 0-1 scale (inverse)"""
        # Target: -1% = 1.0, -10% = 0.0
        if var_95 >= -0.01:
            return 1.0
        elif var_95 <= -0.10:
            return 0.0
        else:
            return 1.0 - abs(var_95 + 0.01) / 0.09
    
    def _normalize_consecutive_losses(self, consecutive_losses: int) -> float:
        """Normalize consecutive losses to 0-1 scale (inverse)"""
        # Target: 0 = 1.0, 10+ = 0.0
        if consecutive_losses == 0:
            return 1.0
        elif consecutive_losses >= 10:
            return 0.0
        else:
            return 1.0 - consecutive_losses / 10.0
    
    def _normalize_sharpe_ratio(self, sharpe_ratio: float) -> float:
        """Normalize Sharpe ratio to 0-1 scale"""
        # Target: 2.0 = 1.0, 0.0 = 0.0, 3.0+ = 1.0
        if sharpe_ratio <= 0.0:
            return 0.0
        elif sharpe_ratio >= 3.0:
            return 1.0
        else:
            return sharpe_ratio / 3.0
    
    def _normalize_sortino_ratio(self, sortino_ratio: float) -> float:
        """Normalize Sortino ratio to 0-1 scale"""
        # Target: 2.0 = 1.0, 0.0 = 0.0, 3.0+ = 1.0
        if sortino_ratio <= 0.0:
            return 0.0
        elif sortino_ratio >= 3.0:
            return 1.0
        else:
            return sortino_ratio / 3.0
    
    def _normalize_calmar_ratio(self, calmar_ratio: float) -> float:
        """Normalize Calmar ratio to 0-1 scale"""
        # Target: 1.0 = 1.0, 0.0 = 0.0, 2.0+ = 1.0
        if calmar_ratio <= 0.0:
            return 0.0
        elif calmar_ratio >= 2.0:
            return 1.0
        else:
            return calmar_ratio / 2.0
    
    def _normalize_recovery_factor(self, recovery_factor: float) -> float:
        """Normalize recovery factor to 0-1 scale"""
        # Target: 5.0 = 1.0, 0.0 = 0.0, 10.0+ = 1.0
        if recovery_factor <= 0.0:
            return 0.0
        elif recovery_factor >= 10.0:
            return 1.0
        else:
            return recovery_factor / 10.0
    
    def _normalize_trade_frequency(self, total_trades: int, backtest_days: int) -> float:
        """Normalize trade frequency to 0-1 scale"""
        if backtest_days <= 0:
            return 0.0
        
        trades_per_year = (total_trades / backtest_days) * 365
        
        # Target: 50-200 trades/year = 1.0, 0-10 = 0.0, 500+ = 0.5
        if trades_per_year <= 10:
            return 0.0
        elif 50 <= trades_per_year <= 200:
            return 1.0
        elif trades_per_year >= 500:
            return 0.5
        elif trades_per_year < 50:
            return trades_per_year / 50.0
        else:  # 200 < trades_per_year < 500
            return 1.0 - (trades_per_year - 200) / 300.0
    
    def get_score_breakdown(self, scoring_metrics: ScoringMetrics) -> Dict[str, Any]:
        """Get detailed score breakdown for analysis"""
        category_weights = self.weights.get_category_weights()
        
        return {
            'overall_score': scoring_metrics.overall_score,
            'category_scores': {
                'performance': {
                    'score': scoring_metrics.performance_score,
                    'weight': category_weights[ScoringCategory.PERFORMANCE],
                    'contribution': scoring_metrics.performance_score * category_weights[ScoringCategory.PERFORMANCE]
                },
                'risk': {
                    'score': scoring_metrics.risk_score,
                    'weight': category_weights[ScoringCategory.RISK],
                    'contribution': scoring_metrics.risk_score * category_weights[ScoringCategory.RISK]
                },
                'consistency': {
                    'score': scoring_metrics.consistency_score,
                    'weight': category_weights[ScoringCategory.CONSISTENCY],
                    'contribution': scoring_metrics.consistency_score * category_weights[ScoringCategory.CONSISTENCY]
                },
                'efficiency': {
                    'score': scoring_metrics.efficiency_score,
                    'weight': category_weights[ScoringCategory.EFFICIENCY],
                    'contribution': scoring_metrics.efficiency_score * category_weights[ScoringCategory.EFFICIENCY]
                },
                'robustness': {
                    'score': scoring_metrics.robustness_score,
                    'weight': category_weights[ScoringCategory.ROBUSTNESS],
                    'contribution': scoring_metrics.robustness_score * category_weights[ScoringCategory.ROBUSTNESS]
                }
            },
            'weights': self.weights.to_dict()
        }

