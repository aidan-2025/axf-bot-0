#!/usr/bin/env python3
"""
Scoring Weights Configuration

Defines weights and scoring parameters for strategy evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum


class ScoringCategory(Enum):
    """Scoring categories for different aspects of strategy performance"""
    PERFORMANCE = "performance"
    RISK = "risk"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"


@dataclass
class ScoringWeights:
    """Weights for different scoring categories and metrics"""
    
    # Category weights (must sum to 1.0)
    performance_weight: float = 0.30
    risk_weight: float = 0.25
    consistency_weight: float = 0.20
    efficiency_weight: float = 0.15
    robustness_weight: float = 0.10
    
    # Performance sub-weights
    return_weight: float = 0.40
    profit_factor_weight: float = 0.30
    win_rate_weight: float = 0.30
    
    # Risk sub-weights
    drawdown_weight: float = 0.50
    volatility_weight: float = 0.30
    var_weight: float = 0.20
    
    # Consistency sub-weights
    stability_weight: float = 0.40
    consistency_sub_weight: float = 0.35
    consecutive_losses_weight: float = 0.25
    
    # Efficiency sub-weights
    sharpe_weight: float = 0.40
    sortino_weight: float = 0.30
    calmar_weight: float = 0.30
    
    # Robustness sub-weights
    event_avoidance_weight: float = 0.50
    recovery_factor_weight: float = 0.30
    trade_frequency_weight: float = 0.20
    
    def validate_weights(self) -> bool:
        """Validate that all weights sum to 1.0 within their categories"""
        # Check category weights
        category_sum = (self.performance_weight + self.risk_weight + 
                       self.consistency_weight + self.efficiency_weight + 
                       self.robustness_weight)
        
        if not abs(category_sum - 1.0) < 0.001:
            return False
        
        # Check performance sub-weights
        performance_sum = (self.return_weight + self.profit_factor_weight + 
                          self.win_rate_weight)
        if not abs(performance_sum - 1.0) < 0.001:
            return False
        
        # Check risk sub-weights
        risk_sum = (self.drawdown_weight + self.volatility_weight + self.var_weight)
        if not abs(risk_sum - 1.0) < 0.001:
            return False
        
        # Check consistency sub-weights
        consistency_sum = (self.stability_weight + self.consistency_sub_weight + 
                          self.consecutive_losses_weight)
        if not abs(consistency_sum - 1.0) < 0.001:
            return False
        
        # Check efficiency sub-weights
        efficiency_sum = (self.sharpe_weight + self.sortino_weight + self.calmar_weight)
        if not abs(efficiency_sum - 1.0) < 0.001:
            return False
        
        # Check robustness sub-weights
        robustness_sum = (self.event_avoidance_weight + self.recovery_factor_weight + 
                         self.trade_frequency_weight)
        if not abs(robustness_sum - 1.0) < 0.001:
            return False
        
        return True
    
    def get_category_weights(self) -> Dict[ScoringCategory, float]:
        """Get category weights as dictionary"""
        return {
            ScoringCategory.PERFORMANCE: self.performance_weight,
            ScoringCategory.RISK: self.risk_weight,
            ScoringCategory.CONSISTENCY: self.consistency_weight,
            ScoringCategory.EFFICIENCY: self.efficiency_weight,
            ScoringCategory.ROBUSTNESS: self.robustness_weight
        }
    
    def get_performance_weights(self) -> Dict[str, float]:
        """Get performance sub-weights as dictionary"""
        return {
            'return': self.return_weight,
            'profit_factor': self.profit_factor_weight,
            'win_rate': self.win_rate_weight
        }
    
    def get_risk_weights(self) -> Dict[str, float]:
        """Get risk sub-weights as dictionary"""
        return {
            'drawdown': self.drawdown_weight,
            'volatility': self.volatility_weight,
            'var': self.var_weight
        }
    
    def get_consistency_weights(self) -> Dict[str, float]:
        """Get consistency sub-weights as dictionary"""
        return {
            'stability': self.stability_weight,
            'consistency': self.consistency_sub_weight,
            'consecutive_losses': self.consecutive_losses_weight
        }
    
    def get_efficiency_weights(self) -> Dict[str, float]:
        """Get efficiency sub-weights as dictionary"""
        return {
            'sharpe': self.sharpe_weight,
            'sortino': self.sortino_weight,
            'calmar': self.calmar_weight
        }
    
    def get_robustness_weights(self) -> Dict[str, float]:
        """Get robustness sub-weights as dictionary"""
        return {
            'event_avoidance': self.event_avoidance_weight,
            'recovery_factor': self.recovery_factor_weight,
            'trade_frequency': self.trade_frequency_weight
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert weights to dictionary"""
        return {
            'category_weights': {
                'performance': self.performance_weight,
                'risk': self.risk_weight,
                'consistency': self.consistency_weight,
                'efficiency': self.efficiency_weight,
                'robustness': self.robustness_weight
            },
            'performance_weights': self.get_performance_weights(),
            'risk_weights': self.get_risk_weights(),
            'consistency_weights': self.get_consistency_weights(),
            'efficiency_weights': self.get_efficiency_weights(),
            'robustness_weights': self.get_robustness_weights()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoringWeights':
        """Create weights from dictionary"""
        category_weights = data.get('category_weights', {})
        performance_weights = data.get('performance_weights', {})
        risk_weights = data.get('risk_weights', {})
        consistency_weights = data.get('consistency_weights', {})
        efficiency_weights = data.get('efficiency_weights', {})
        robustness_weights = data.get('robustness_weights', {})
        
        return cls(
            # Category weights
            performance_weight=category_weights.get('performance', 0.30),
            risk_weight=category_weights.get('risk', 0.25),
            consistency_weight=category_weights.get('consistency', 0.20),
            efficiency_weight=category_weights.get('efficiency', 0.15),
            robustness_weight=category_weights.get('robustness', 0.10),
            
            # Performance sub-weights
            return_weight=performance_weights.get('return', 0.40),
            profit_factor_weight=performance_weights.get('profit_factor', 0.30),
            win_rate_weight=performance_weights.get('win_rate', 0.30),
            
            # Risk sub-weights
            drawdown_weight=risk_weights.get('drawdown', 0.50),
            volatility_weight=risk_weights.get('volatility', 0.30),
            var_weight=risk_weights.get('var', 0.20),
            
            # Consistency sub-weights
            stability_weight=consistency_weights.get('stability', 0.40),
            consistency_sub_weight=consistency_weights.get('consistency', 0.35),
            consecutive_losses_weight=consistency_weights.get('consecutive_losses', 0.25),
            
            # Efficiency sub-weights
            sharpe_weight=efficiency_weights.get('sharpe', 0.40),
            sortino_weight=efficiency_weights.get('sortino', 0.30),
            calmar_weight=efficiency_weights.get('calmar', 0.30),
            
            # Robustness sub-weights
            event_avoidance_weight=robustness_weights.get('event_avoidance', 0.50),
            recovery_factor_weight=robustness_weights.get('recovery_factor', 0.30),
            trade_frequency_weight=robustness_weights.get('trade_frequency', 0.20)
        )
