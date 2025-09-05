#!/usr/bin/env python3
"""
High-Fidelity Position Sizer

Implements realistic position sizing logic that accounts for variable spreads,
market impact, and risk management for high-fidelity backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import backtrader as bt
    from backtrader import SizerBase, Sizer
    from backtrader.sizer import SizerBase
except ImportError:
    bt = None
    # Create mock classes for testing when backtrader is not available
    class MockSizerBase:
        def __init__(self, *args, **kwargs):
            self.broker = None
        def getsizing(self, data, isbuy):
            return 1.0
    
    SizerBase = MockSizerBase
    Sizer = MockSizerBase

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLATILITY = "volatility"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"
    ADAPTIVE = "adaptive"


@dataclass
class SizingConfig:
    """Configuration for position sizing"""
    
    # Basic sizing
    method: SizingMethod = SizingMethod.PERCENTAGE
    fixed_size: float = 1.0
    percentage: float = 0.1  # 10% of portfolio
    
    # Risk management
    max_position_size: float = 0.2  # Maximum 20% of portfolio
    min_position_size: float = 0.01  # Minimum 1% of portfolio
    max_risk_per_trade: float = 0.02  # Maximum 2% risk per trade
    
    # Volatility-based sizing
    volatility_lookback: int = 20  # Days for volatility calculation
    target_volatility: float = 0.15  # 15% target volatility
    volatility_factor: float = 1.0
    
    # Kelly criterion
    kelly_fraction: float = 0.25  # 25% of Kelly optimal
    min_kelly_fraction: float = 0.01
    max_kelly_fraction: float = 0.5
    
    # Risk parity
    risk_budget: float = 0.1  # 10% risk budget per asset
    correlation_lookback: int = 60  # Days for correlation calculation
    
    # Adaptive sizing
    performance_lookback: int = 30  # Days for performance calculation
    min_adaptive_size: float = 0.01
    max_adaptive_size: float = 0.3
    
    # Spread considerations
    spread_impact_factor: float = 0.5  # Reduce size based on spread
    min_spread_threshold: float = 0.0001  # 1 pip minimum spread
    max_spread_threshold: float = 0.001  # 10 pips maximum spread
    
    # Market impact
    market_impact_factor: float = 0.1  # Reduce size based on market impact
    max_market_impact: float = 0.0005  # 5 pips maximum market impact


class HighFidelitySizer(SizerBase):
    """
    High-fidelity position sizer that accounts for variable spreads,
    market impact, and risk management.
    """
    
    def __init__(self, config: SizingConfig = None):
        super().__init__()
        
        self.config = config or SizingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.position_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Risk metrics
        self.current_risk: float = 0.0
        self.portfolio_risk: float = 0.0
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        logger.info(f"HighFidelitySizer initialized with {self.config.method.value} method")
    
    def _get_current_price(self, data) -> float:
        """Get current price from data feed"""
        try:
            if hasattr(data, 'close'):
                return data.close[0]
            elif hasattr(data, 'price'):
                return data.price[0]
            else:
                return data[0]
        except (IndexError, AttributeError):
            return 0.0
    
    def _get_current_spread(self, data) -> float:
        """Get current spread from data feed"""
        try:
            if hasattr(data, 'spread'):
                return data.spread[0]
            elif hasattr(data, 'ask') and hasattr(data, 'bid'):
                return data.ask[0] - data.bid[0]
            else:
                return self.config.min_spread_threshold
        except (IndexError, AttributeError):
            return self.config.min_spread_threshold
    
    def _get_current_volume(self, data) -> float:
        """Get current volume from data feed"""
        try:
            if hasattr(data, 'volume'):
                return data.volume[0]
            else:
                return 1.0
        except (IndexError, AttributeError):
            return 1.0
    
    def _calculate_volatility(self, data, lookback: int = None) -> float:
        """Calculate historical volatility"""
        
        if lookback is None:
            lookback = self.config.volatility_lookback
        
        try:
            if hasattr(data, 'close'):
                prices = [data.close[-i] for i in range(min(lookback, len(data)))]
            else:
                prices = [data[-i] for i in range(min(lookback, len(data)))]
            
            if len(prices) < 2:
                return 0.15  # Default volatility
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Calculate volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)
            
            return max(volatility, 0.01)  # Minimum 1% volatility
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {e}")
            return 0.15  # Default volatility
    
    def _calculate_kelly_fraction(self, data, lookback: int = 30) -> float:
        """Calculate Kelly criterion fraction"""
        
        try:
            if hasattr(data, 'close'):
                prices = [data.close[-i] for i in range(min(lookback, len(data)))]
            else:
                prices = [data[-i] for i in range(min(lookback, len(data)))]
            
            if len(prices) < 10:
                return self.config.kelly_fraction
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Calculate win rate and average win/loss
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return self.config.kelly_fraction
            
            win_rate = len(positive_returns) / len(returns)
            avg_win = np.mean(positive_returns)
            avg_loss = abs(np.mean(negative_returns))
            
            if avg_loss == 0:
                return self.config.kelly_fraction
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply constraints
            kelly_fraction = max(self.config.min_kelly_fraction,
                               min(self.config.max_kelly_fraction, kelly_fraction))
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.warning(f"Error calculating Kelly fraction: {e}")
            return self.config.kelly_fraction
    
    def _calculate_risk_parity_size(self, data, portfolio_value: float) -> float:
        """Calculate position size using risk parity approach"""
        
        try:
            # Get current price and volatility
            current_price = self._get_current_price(data)
            volatility = self._calculate_volatility(data)
            
            if current_price == 0 or volatility == 0:
                return 0.0
            
            # Risk parity: size = risk_budget / (price * volatility)
            risk_budget = portfolio_value * self.config.risk_budget
            position_size = risk_budget / (current_price * volatility)
            
            return position_size
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk parity size: {e}")
            return 0.0
    
    def _calculate_adaptive_size(self, data, portfolio_value: float) -> float:
        """Calculate adaptive position size based on recent performance"""
        
        try:
            # Get recent performance
            if len(self.position_history) < 5:
                return self.config.percentage * portfolio_value
            
            # Calculate recent performance metrics
            recent_positions = self.position_history[-self.config.performance_lookback:]
            
            if not recent_positions:
                return self.config.percentage * portfolio_value
            
            # Calculate performance score
            total_pnl = sum(pos.get('pnl', 0) for pos in recent_positions)
            win_rate = sum(1 for pos in recent_positions if pos.get('pnl', 0) > 0) / len(recent_positions)
            
            # Adjust size based on performance
            if total_pnl > 0 and win_rate > 0.5:
                # Good performance - increase size
                performance_factor = min(1.5, 1.0 + (total_pnl / portfolio_value) * 10)
            elif total_pnl < 0 and win_rate < 0.4:
                # Poor performance - decrease size
                performance_factor = max(0.5, 1.0 + (total_pnl / portfolio_value) * 10)
            else:
                performance_factor = 1.0
            
            # Calculate base size
            base_size = self.config.percentage * portfolio_value * performance_factor
            
            return base_size
            
        except Exception as e:
            self.logger.warning(f"Error calculating adaptive size: {e}")
            return self.config.percentage * portfolio_value
    
    def _apply_spread_adjustment(self, size: float, spread: float) -> float:
        """Adjust position size based on current spread"""
        
        if spread <= self.config.min_spread_threshold:
            return size  # No adjustment for tight spreads
        
        # Calculate spread impact
        spread_ratio = spread / self.config.max_spread_threshold
        spread_impact = min(1.0, spread_ratio) * self.config.spread_impact_factor
        
        # Reduce size based on spread
        adjusted_size = size * (1.0 - spread_impact)
        
        return max(adjusted_size, size * 0.1)  # Minimum 10% of original size
    
    def _apply_market_impact_adjustment(self, size: float, current_price: float) -> float:
        """Adjust position size based on estimated market impact"""
        
        # Estimate market impact based on position size
        estimated_impact = self.config.market_impact_factor * size * current_price
        
        if estimated_impact > self.config.max_market_impact:
            # Reduce size to limit market impact
            impact_ratio = self.config.max_market_impact / estimated_impact
            adjusted_size = size * impact_ratio
            return adjusted_size
        
        return size
    
    def _apply_risk_constraints(self, size: float, portfolio_value: float, 
                              current_price: float) -> float:
        """Apply risk management constraints"""
        
        if current_price == 0:
            return 0.0
        
        # Calculate position value
        position_value = size * current_price
        
        # Apply maximum position size constraint
        max_position_value = portfolio_value * self.config.max_position_size
        if position_value > max_position_value:
            size = max_position_value / current_price
        
        # Apply minimum position size constraint
        min_position_value = portfolio_value * self.config.min_position_size
        if position_value < min_position_value:
            size = min_position_value / current_price
        
        # Apply maximum risk per trade constraint
        max_risk_value = portfolio_value * self.config.max_risk_per_trade
        # Estimate risk as position value * volatility (use default volatility if no data)
        try:
            volatility = self._calculate_volatility(data)
        except:
            volatility = 0.15  # Default volatility
        estimated_risk = position_value * volatility
        
        if estimated_risk > max_risk_value:
            size = max_risk_value / (current_price * volatility)
        
        return size
    
    def _record_position(self, size: float, price: float, timestamp: datetime):
        """Record position for performance tracking"""
        
        position_record = {
            'timestamp': timestamp,
            'size': size,
            'price': price,
            'value': size * price,
            'pnl': 0.0  # Will be updated when position is closed
        }
        
        self.position_history.append(position_record)
        
        # Keep only recent history
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-500:]
    
    def _getsizing(self, data, isbuy: bool) -> float:
        """Calculate position size for the given data and direction"""
        
        try:
            # Get current market data
            current_price = self._get_current_price(data)
            current_spread = self._get_current_spread(data)
            current_volume = self._get_current_volume(data)
            
            if current_price == 0:
                return 0.0
            
            # Get portfolio value
            portfolio_value = self.broker.getvalue()
            
            if portfolio_value <= 0:
                return 0.0
            
            # Calculate base position size based on method
            if self.config.method == SizingMethod.FIXED:
                base_size = self.config.fixed_size
            elif self.config.method == SizingMethod.PERCENTAGE:
                base_size = self.config.percentage * portfolio_value / current_price
            elif self.config.method == SizingMethod.VOLATILITY:
                volatility = self._calculate_volatility(data)
                target_risk = portfolio_value * self.config.target_volatility
                base_size = target_risk / (current_price * volatility * self.config.volatility_factor)
            elif self.config.method == SizingMethod.KELLY:
                kelly_fraction = self._calculate_kelly_fraction(data)
                base_size = kelly_fraction * portfolio_value / current_price
            elif self.config.method == SizingMethod.RISK_PARITY:
                base_size = self._calculate_risk_parity_size(data, portfolio_value)
            elif self.config.method == SizingMethod.ADAPTIVE:
                base_size = self._calculate_adaptive_size(data, portfolio_value) / current_price
            else:
                base_size = self.config.percentage * portfolio_value / current_price
            
            # Apply spread adjustment
            adjusted_size = self._apply_spread_adjustment(base_size, current_spread)
            
            # Apply market impact adjustment
            adjusted_size = self._apply_market_impact_adjustment(adjusted_size, current_price)
            
            # Apply risk constraints
            final_size = self._apply_risk_constraints(adjusted_size, portfolio_value, current_price)
            
            # Ensure positive size
            final_size = max(0.0, final_size)
            
            # Record position for tracking
            self._record_position(final_size, current_price, datetime.now())
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def getsizing(self, data, isbuy: bool) -> float:
        """Backtrader interface for position sizing"""
        return self._getsizing(data, isbuy)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for adaptive sizing"""
        
        if not self.position_history:
            return {
                'total_positions': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Calculate metrics
        total_positions = len(self.position_history)
        profitable_positions = sum(1 for pos in self.position_history if pos.get('pnl', 0) > 0)
        win_rate = profitable_positions / total_positions if total_positions > 0 else 0.0
        
        pnls = [pos.get('pnl', 0) for pos in self.position_history]
        avg_pnl = np.mean(pnls) if pnls else 0.0
        total_pnl = sum(pnls)
        
        # Calculate Sharpe ratio
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_positions': total_positions,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio
        }
    
    def update_position_pnl(self, position_id: str, pnl: float):
        """Update P&L for a specific position"""
        
        # Find and update position
        for pos in self.position_history:
            if pos.get('id') == position_id:
                pos['pnl'] = pnl
                break


class HighFidelitySizerFactory:
    """Factory for creating high-fidelity sizers"""
    
    @staticmethod
    def create_fixed_sizer(size: float = 1.0) -> HighFidelitySizer:
        """Create fixed size sizer"""
        config = SizingConfig(
            method=SizingMethod.FIXED,
            fixed_size=size
        )
        return HighFidelitySizer(config)
    
    @staticmethod
    def create_percentage_sizer(percentage: float = 0.1) -> HighFidelitySizer:
        """Create percentage-based sizer"""
        config = SizingConfig(
            method=SizingMethod.PERCENTAGE,
            percentage=percentage
        )
        return HighFidelitySizer(config)
    
    @staticmethod
    def create_volatility_sizer(target_volatility: float = 0.15) -> HighFidelitySizer:
        """Create volatility-based sizer"""
        config = SizingConfig(
            method=SizingMethod.VOLATILITY,
            target_volatility=target_volatility
        )
        return HighFidelitySizer(config)
    
    @staticmethod
    def create_kelly_sizer(kelly_fraction: float = 0.25) -> HighFidelitySizer:
        """Create Kelly criterion sizer"""
        config = SizingConfig(
            method=SizingMethod.KELLY,
            kelly_fraction=kelly_fraction
        )
        return HighFidelitySizer(config)
    
    @staticmethod
    def create_adaptive_sizer(base_percentage: float = 0.1) -> HighFidelitySizer:
        """Create adaptive sizer"""
        config = SizingConfig(
            method=SizingMethod.ADAPTIVE,
            percentage=base_percentage
        )
        return HighFidelitySizer(config)
