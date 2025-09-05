#!/usr/bin/env python3
"""
Performance Metrics for Strategy Validation

Defines comprehensive performance metrics for trading strategies.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for trading strategies"""
    
    # Basic Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    # Risk Metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Risk-Adjusted Returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Trading Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Consecutive Statistics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_consecutive_wins: int = 0
    current_consecutive_losses: int = 0
    
    # Time-based Metrics
    avg_trade_duration: float = 0.0
    avg_winning_trade_duration: float = 0.0
    avg_losing_trade_duration: float = 0.0
    
    # Consistency Metrics
    consistency_score: float = 0.0
    stability_score: float = 0.0
    recovery_factor: float = 0.0
    
    # Economic Event Metrics
    event_avoidance_score: float = 0.0
    event_impact_drawdown: float = 0.0
    
    # Additional Metrics
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_daily_loss: float = 0.0
    max_daily_profit: float = 0.0
    
    def calculate_from_trades(self, trades: List[Dict[str, Any]], 
                            initial_capital: float = 10000.0,
                            risk_free_rate: float = 0.02) -> 'PerformanceMetrics':
        """Calculate performance metrics from trade data"""
        if not trades:
            return self
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', datetime.min))
        
        # Extract trade data
        trade_returns = [trade.get('return', 0.0) for trade in sorted_trades]
        trade_pnls = [trade.get('pnl', 0.0) for trade in sorted_trades]
        trade_durations = [trade.get('duration', 0.0) for trade in sorted_trades]
        
        # Basic calculations
        self.total_trades = len(trades)
        self.winning_trades = len([r for r in trade_returns if r > 0])
        self.losing_trades = len([r for r in trade_returns if r < 0])
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        # P&L calculations
        self.gross_profit = sum([pnl for pnl in trade_pnls if pnl > 0])
        self.gross_loss = abs(sum([pnl for pnl in trade_pnls if pnl < 0]))
        self.net_profit = self.gross_profit - self.gross_loss
        
        if self.gross_loss > 0:
            self.profit_factor = self.gross_profit / self.gross_loss
        
        # Win/Loss ratios
        if self.winning_trades > 0:
            self.avg_win = self.gross_profit / self.winning_trades
        if self.losing_trades > 0:
            self.avg_loss = self.gross_loss / self.losing_trades
        if self.avg_loss > 0:
            self.avg_win_loss_ratio = self.avg_win / self.avg_loss
        
        # Return calculations
        if initial_capital > 0:
            self.total_return = self.net_profit / initial_capital
        
        # Calculate annualized return (assuming 252 trading days)
        if len(trade_returns) > 1:
            days = (sorted_trades[-1].get('timestamp', datetime.now()) - 
                   sorted_trades[0].get('timestamp', datetime.now())).days
            if days > 0:
                self.annualized_return = (1 + self.total_return) ** (365 / days) - 1
        
        # Risk metrics
        if trade_returns:
            self.volatility = np.std(trade_returns) * np.sqrt(252)  # Annualized
            self.max_drawdown = self._calculate_max_drawdown(trade_returns)
            self.current_drawdown = self._calculate_current_drawdown(trade_returns)
            
            # VaR and CVaR (95% confidence)
            sorted_returns = sorted(trade_returns)
            var_index = int(0.05 * len(sorted_returns))
            self.var_95 = sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0
            self.cvar_95 = np.mean(sorted_returns[:var_index]) if var_index > 0 else 0.0
        
        # Risk-adjusted returns
        if self.volatility > 0:
            self.sharpe_ratio = (self.annualized_return - risk_free_rate) / self.volatility
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in trade_returns if r < 0]
        if downside_returns:
            downside_volatility = np.std(downside_returns) * np.sqrt(252)
            if downside_volatility > 0:
                self.sortino_ratio = (self.annualized_return - risk_free_rate) / downside_volatility
        
        # Calmar ratio
        if self.max_drawdown > 0:
            self.calmar_ratio = self.annualized_return / self.max_drawdown
        
        # Consecutive statistics
        self._calculate_consecutive_stats(trade_returns)
        
        # Duration statistics
        if trade_durations:
            self.avg_trade_duration = np.mean(trade_durations)
            winning_durations = [d for d, r in zip(trade_durations, trade_returns) if r > 0]
            losing_durations = [d for d, r in zip(trade_durations, trade_returns) if r < 0]
            
            if winning_durations:
                self.avg_winning_trade_duration = np.mean(winning_durations)
            if losing_durations:
                self.avg_losing_trade_duration = np.mean(losing_durations)
        
        # Additional metrics
        if trade_pnls:
            self.largest_win = max(trade_pnls)
            self.largest_loss = min(trade_pnls)
        
        # Consistency and stability scores
        self.consistency_score = self._calculate_consistency_score(trade_returns)
        self.stability_score = self._calculate_stability_score(trade_returns)
        
        # Recovery factor
        if self.max_drawdown > 0:
            self.recovery_factor = self.net_profit / self.max_drawdown
        
        return self
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        return abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    def _calculate_current_drawdown(self, returns: List[float]) -> float:
        """Calculate current drawdown"""
        if not returns:
            return 0.0
        
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        current_drawdown = (cumulative_returns[-1] - running_max[-1]) / running_max[-1]
        
        return abs(current_drawdown) if current_drawdown < 0 else 0.0
    
    def _calculate_consecutive_stats(self, returns: List[float]) -> None:
        """Calculate consecutive win/loss statistics"""
        if not returns:
            return
        
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for ret in returns:
            if ret > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif ret < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        self.max_consecutive_wins = max_wins
        self.max_consecutive_losses = max_losses
        self.current_consecutive_wins = current_wins
        self.current_consecutive_losses = current_losses
    
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score (0.0 to 1.0)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0:
            return 0.0
        
        cv = std_return / abs(mean_return)
        
        # Convert to 0-1 scale (lower CV = higher consistency)
        consistency = max(0.0, 1.0 - cv)
        return min(1.0, consistency)
    
    def _calculate_stability_score(self, returns: List[float]) -> float:
        """Calculate stability score (0.0 to 1.0)"""
        if not returns or len(returns) < 3:
            return 0.0
        
        # Calculate rolling volatility stability
        window_size = min(10, len(returns) // 3)
        if window_size < 2:
            return 0.0
        
        rolling_vols = []
        for i in range(len(returns) - window_size + 1):
            window_returns = returns[i:i + window_size]
            rolling_vols.append(np.std(window_returns))
        
        if not rolling_vols:
            return 0.0
        
        # Stability is inverse of volatility of volatility
        vol_of_vol = np.std(rolling_vols)
        mean_vol = np.mean(rolling_vols)
        
        if mean_vol == 0:
            return 0.0
        
        stability = max(0.0, 1.0 - (vol_of_vol / mean_vol))
        return min(1.0, stability)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'net_profit': self.net_profit,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'current_drawdown': self.current_drawdown,
            'volatility': self.volatility,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_win_loss_ratio': self.avg_win_loss_ratio,
            'profit_factor': self.profit_factor,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'current_consecutive_wins': self.current_consecutive_wins,
            'current_consecutive_losses': self.current_consecutive_losses,
            'avg_trade_duration': self.avg_trade_duration,
            'avg_winning_trade_duration': self.avg_winning_trade_duration,
            'avg_losing_trade_duration': self.avg_losing_trade_duration,
            'consistency_score': self.consistency_score,
            'stability_score': self.stability_score,
            'recovery_factor': self.recovery_factor,
            'event_avoidance_score': self.event_avoidance_score,
            'event_impact_drawdown': self.event_impact_drawdown,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'max_daily_loss': self.max_daily_loss,
            'max_daily_profit': self.max_daily_profit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create metrics from dictionary"""
        return cls(**data)

