#!/usr/bin/env python3
"""
Risk Metrics

Comprehensive risk metrics calculation and analysis for portfolio
risk assessment and monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import statistics

from ..models import RiskMetrics, PortfolioData, PositionData

logger = logging.getLogger(__name__)


@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics calculation"""
    # Calculation settings
    var_confidence_levels: List[float] = None
    lookback_periods: int = 252  # 1 year of trading days
    min_data_points: int = 30
    
    # Volatility calculation
    volatility_lookback: int = 20
    volatility_annualization_factor: int = 252
    
    # Correlation calculation
    correlation_lookback: int = 60
    min_correlation_periods: int = 10
    
    # Performance metrics
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_return: float = 0.08  # 8% annual benchmark return
    
    def __post_init__(self):
        if self.var_confidence_levels is None:
            self.var_confidence_levels = [0.95, 0.99]


class RiskMetricsCalculator:
    """
    Comprehensive risk metrics calculator.
    
    Calculates various risk metrics including:
    - Value at Risk (VaR)
    - Expected Shortfall (ES)
    - Maximum Drawdown
    - Sharpe Ratio, Sortino Ratio, Calmar Ratio
    - Volatility and correlation metrics
    - Performance attribution
    """
    
    def __init__(self, config: RiskMetricsConfig = None):
        """Initialize risk metrics calculator"""
        self.config = config or RiskMetricsConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.calculations_count = 0
        self.last_calculation = None
        
        self.logger.info("RiskMetricsCalculator initialized")
    
    def calculate_risk_metrics(self, portfolio_data: PortfolioData, 
                             returns_history: List[float] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            portfolio_data: Current portfolio state
            returns_history: Historical returns (if available)
            
        Returns:
            Comprehensive risk metrics
        """
        try:
            self.logger.debug("Calculating risk metrics")
            
            # Use provided returns or calculate from portfolio data
            if returns_history is None:
                returns_history = self._estimate_returns_from_portfolio(portfolio_data)
            
            # Calculate basic metrics
            portfolio_value = portfolio_data.total_value
            total_pnl = portfolio_data.total_pnl
            unrealized_pnl = portfolio_data.unrealized_pnl
            realized_pnl = portfolio_data.realized_pnl
            
            # Calculate drawdown metrics
            max_drawdown, current_drawdown = self._calculate_drawdowns(returns_history)
            
            # Calculate VaR metrics
            var_95 = self._calculate_var(returns_history, 0.95)
            var_99 = self._calculate_var(returns_history, 0.99)
            
            # Calculate performance ratios
            sharpe_ratio = self._calculate_sharpe_ratio(returns_history)
            sortino_ratio = self._calculate_sortino_ratio(returns_history)
            calmar_ratio = self._calculate_calmar_ratio(returns_history, max_drawdown)
            
            # Calculate other metrics
            win_rate = self._calculate_win_rate(returns_history)
            profit_factor = self._calculate_profit_factor(returns_history)
            recovery_factor = self._calculate_recovery_factor(returns_history, max_drawdown)
            consecutive_losses = self._calculate_consecutive_losses(returns_history)
            
            # Calculate total risk
            total_risk = self._calculate_total_risk(portfolio_data)
            risk_per_trade = total_risk / max(len(portfolio_data.positions), 1)
            
            # Create risk metrics object
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_risk=total_risk,
                risk_per_trade=risk_per_trade,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                var_95=var_95,
                var_99=var_99,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_consecutive_losses=consecutive_losses,
                win_rate=win_rate,
                profit_factor=profit_factor,
                recovery_factor=recovery_factor
            )
            
            self.calculations_count += 1
            self.last_calculation = datetime.utcnow()
            
            self.logger.debug(f"Risk metrics calculated: Sharpe={sharpe_ratio:.2f}, VaR95={var_95:.2%}")
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            # Return basic metrics in case of error
            return RiskMetrics(portfolio_value=portfolio_data.total_value)
    
    def _estimate_returns_from_portfolio(self, portfolio_data: PortfolioData) -> List[float]:
        """Estimate returns from portfolio data"""
        # This is a simplified estimation
        # In practice, you'd have actual historical returns
        
        if not portfolio_data.positions:
            return []
        
        # Estimate returns based on position PnL
        total_pnl = portfolio_data.total_pnl
        portfolio_value = portfolio_data.total_value
        
        if portfolio_value == 0:
            return []
        
        # Simple return estimation
        estimated_return = total_pnl / portfolio_value
        
        # Generate some mock historical returns for demonstration
        import random
        returns = []
        for _ in range(min(30, self.config.lookback_periods)):
            # Add some randomness around the estimated return
            noise = random.gauss(0, 0.02)  # 2% volatility
            returns.append(estimated_return + noise)
        
        return returns
    
    def _calculate_drawdowns(self, returns: List[float]) -> Tuple[float, float]:
        """Calculate maximum and current drawdowns"""
        if not returns:
            return 0.0, 0.0
        
        # Calculate cumulative returns
        cumulative = [1.0]
        for ret in returns:
            cumulative.append(cumulative[-1] * (1 + ret))
        
        # Calculate running maximum
        running_max = [cumulative[0]]
        for i in range(1, len(cumulative)):
            running_max.append(max(running_max[-1], cumulative[i]))
        
        # Calculate drawdowns
        drawdowns = []
        for i in range(len(cumulative)):
            drawdown = (cumulative[i] - running_max[i]) / running_max[i]
            drawdowns.append(drawdown)
        
        max_drawdown = abs(min(drawdowns))
        current_drawdown = abs(drawdowns[-1]) if drawdowns else 0.0
        
        return max_drawdown, current_drawdown
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        
        if index < len(sorted_returns):
            return abs(sorted_returns[index])
        else:
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize the ratio
        return (mean_return - self.config.risk_free_rate / self.config.volatility_annualization_factor) / std_return * (self.config.volatility_annualization_factor ** 0.5)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = statistics.stdev(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        return (mean_return - self.config.risk_free_rate / self.config.volatility_annualization_factor) / downside_std * (self.config.volatility_annualization_factor ** 0.5)
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if not returns or max_drawdown == 0:
            return 0.0
        
        mean_return = statistics.mean(returns)
        annual_return = mean_return * self.config.volatility_annualization_factor
        
        return annual_return / max_drawdown
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """Calculate win rate"""
        if not returns:
            return 0.0
        
        winning_trades = sum(1 for r in returns if r > 0)
        return winning_trades / len(returns)
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor"""
        if not returns:
            return 0.0
        
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_recovery_factor(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate recovery factor"""
        if not returns or max_drawdown == 0:
            return 0.0
        
        total_return = sum(returns)
        return total_return / max_drawdown
    
    def _calculate_consecutive_losses(self, returns: List[float]) -> int:
        """Calculate maximum consecutive losses"""
        if not returns:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_total_risk(self, portfolio_data: PortfolioData) -> float:
        """Calculate total portfolio risk"""
        if not portfolio_data.positions:
            return 0.0
        
        # Simple risk calculation based on position sizes
        total_risk = 0.0
        
        for position in portfolio_data.positions:
            position_value = abs(position.size * position.current_price)
            estimated_volatility = 0.15  # 15% default volatility
            position_risk = position_value * estimated_volatility
            total_risk += position_risk
        
        return total_risk / portfolio_data.total_value if portfolio_data.total_value > 0 else 0.0
    
    def calculate_position_risk(self, position: PositionData, 
                              portfolio_value: float) -> Dict[str, float]:
        """Calculate risk metrics for a single position"""
        position_value = abs(position.size * position.current_price)
        position_weight = position_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # Estimate position volatility (simplified)
        estimated_volatility = 0.15  # 15% default
        position_risk = position_value * estimated_volatility
        
        return {
            "position_value": position_value,
            "position_weight": position_weight,
            "estimated_volatility": estimated_volatility,
            "position_risk": position_risk,
            "risk_percentage": position_risk / portfolio_value if portfolio_value > 0 else 0.0
        }
    
    def calculate_correlation_matrix(self, returns_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for multiple assets"""
        if not returns_data or len(returns_data) < 2:
            return {}
        
        # This is a simplified correlation calculation
        # In practice, you'd use a proper correlation function
        assets = list(returns_data.keys())
        correlation_matrix = {}
        
        for asset1 in assets:
            correlation_matrix[asset1] = {}
            for asset2 in assets:
                if asset1 == asset2:
                    correlation_matrix[asset1][asset2] = 1.0
                else:
                    # Simplified correlation calculation
                    correlation_matrix[asset1][asset2] = 0.3  # Default correlation
        
        return correlation_matrix
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """Get calculation summary"""
        return {
            "calculations_count": self.calculations_count,
            "last_calculation": self.last_calculation.isoformat() if self.last_calculation else None,
            "config": {
                "var_confidence_levels": self.config.var_confidence_levels,
                "lookback_periods": self.config.lookback_periods,
                "min_data_points": self.config.min_data_points,
                "volatility_lookback": self.config.volatility_lookback,
                "risk_free_rate": self.config.risk_free_rate
            }
        }

