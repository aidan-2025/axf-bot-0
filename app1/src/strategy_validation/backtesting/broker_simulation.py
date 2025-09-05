#!/usr/bin/env python3
"""
Broker Simulation for Backtrader Integration

Provides realistic broker simulation with spreads, commissions, and slippage.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    """Configuration for broker simulation"""
    
    # Commission and fees
    commission: float = 0.0001  # 0.01% commission per trade
    commission_type: str = 'percentage'  # 'percentage' or 'fixed'
    
    # Spread simulation
    spread: float = 0.0001  # 0.01% spread
    spread_type: str = 'fixed'  # 'fixed' or 'variable'
    variable_spread_range: tuple = (0.00005, 0.0002)  # Min/max spread
    
    # Slippage simulation
    slippage: float = 0.0001  # 0.01% slippage
    slippage_type: str = 'fixed'  # 'fixed' or 'variable'
    variable_slippage_range: tuple = (0.00005, 0.0003)  # Min/max slippage
    
    # Market hours simulation
    market_hours: Dict[str, List[tuple]] = None  # Symbol -> [(start_hour, end_hour), ...]
    
    # Liquidity simulation
    liquidity_threshold: float = 0.1  # 10% of position size
    max_position_size: float = 0.1  # 10% of account balance
    
    def __post_init__(self):
        if self.market_hours is None:
            # Default 24/7 market hours for major forex pairs
            self.market_hours = {
                'EURUSD': [(0, 24)],
                'GBPUSD': [(0, 24)],
                'USDJPY': [(0, 24)],
                'AUDUSD': [(0, 24)],
                'USDCAD': [(0, 24)]
            }


class ForexBroker:
    """Realistic forex broker simulation"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Trading state
        self.positions = {}  # symbol -> position info
        self.trades = []  # List of executed trades
        self.account_balance = 10000.0  # Starting balance
        self.equity = 10000.0  # Current equity
        
        self.logger.info("ForexBroker initialized")
    
    def calculate_spread(self, symbol: str, timestamp: datetime) -> float:
        """Calculate spread for a symbol at a given time"""
        if self.config.spread_type == 'fixed':
            return self.config.spread
        elif self.config.spread_type == 'variable':
            # Simulate variable spread based on volatility and time
            base_spread = np.random.uniform(*self.config.variable_spread_range)
            
            # Increase spread during volatile periods (simplified)
            hour = timestamp.hour
            if hour in [0, 1, 2, 3, 4, 5]:  # Low liquidity hours
                base_spread *= 1.5
            elif hour in [8, 9, 10, 11, 14, 15, 16, 17]:  # High liquidity hours
                base_spread *= 0.8
            
            return base_spread
        else:
            return self.config.spread
    
    def calculate_slippage(self, symbol: str, order_size: float, 
                          order_type: str) -> float:
        """Calculate slippage for an order"""
        if self.config.slippage_type == 'fixed':
            return self.config.slippage
        elif self.config.slippage_type == 'variable':
            # Larger orders have more slippage
            size_factor = min(1.0, order_size / self.config.liquidity_threshold)
            base_slippage = np.random.uniform(*self.config.variable_slippage_range)
            return base_slippage * (1 + size_factor)
        else:
            return self.config.slippage
    
    def calculate_commission(self, order_size: float, price: float) -> float:
        """Calculate commission for an order"""
        if self.config.commission_type == 'percentage':
            return order_size * price * self.config.commission
        else:  # fixed
            return self.config.commission
    
    def is_market_open(self, symbol: str, timestamp: datetime) -> bool:
        """Check if market is open for a symbol"""
        if symbol not in self.config.market_hours:
            return True  # Assume 24/7 if not specified
        
        hour = timestamp.hour
        for start_hour, end_hour in self.config.market_hours[symbol]:
            if start_hour <= hour < end_hour:
                return True
        return False
    
    def execute_order(self, symbol: str, order_type: str, size: float, 
                     price: float, timestamp: datetime) -> Dict[str, Any]:
        """Execute a trading order"""
        if not self.is_market_open(symbol, timestamp):
            return {
                'success': False,
                'error': 'Market closed',
                'executed_price': None,
                'commission': 0.0,
                'slippage': 0.0
            }
        
        # Check position size limits
        if size > self.config.max_position_size * self.account_balance:
            return {
                'success': False,
                'error': 'Position size exceeds limit',
                'executed_price': None,
                'commission': 0.0,
                'slippage': 0.0
            }
        
        # Calculate spread and slippage
        spread = self.calculate_spread(symbol, timestamp)
        slippage = self.calculate_slippage(symbol, size, order_type)
        
        # Calculate executed price
        if order_type.lower() == 'buy':
            executed_price = price + spread + slippage
        else:  # sell
            executed_price = price - spread - slippage
        
        # Calculate commission
        commission = self.calculate_commission(size, executed_price)
        
        # Update account
        self.account_balance -= commission
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'type': order_type,
            'size': size,
            'requested_price': price,
            'executed_price': executed_price,
            'spread': spread,
            'slippage': slippage,
            'commission': commission,
            'pnl': 0.0  # Will be calculated when position is closed
        }
        
        self.trades.append(trade)
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = {
                'size': 0.0,
                'avg_price': 0.0,
                'unrealized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        
        if order_type.lower() == 'buy':
            # Add to long position
            if position['size'] >= 0:  # Adding to long or starting new long
                new_size = position['size'] + size
                new_avg_price = ((position['size'] * position['avg_price']) + 
                               (size * executed_price)) / new_size
                position['size'] = new_size
                position['avg_price'] = new_avg_price
            else:  # Reducing short position
                if size <= abs(position['size']):
                    # Partially or fully close short
                    position['size'] += size
                else:
                    # Close short and open long
                    remaining_short = abs(position['size'])
                    new_long = size - remaining_short
                    position['size'] = new_long
                    position['avg_price'] = executed_price
        else:  # sell
            # Add to short position
            if position['size'] <= 0:  # Adding to short or starting new short
                new_size = position['size'] - size
                if position['size'] != 0:
                    new_avg_price = ((abs(position['size']) * position['avg_price']) + 
                                   (size * executed_price)) / abs(new_size)
                else:
                    new_avg_price = executed_price
                position['size'] = new_size
                position['avg_price'] = new_avg_price
            else:  # Reducing long position
                if size <= position['size']:
                    # Partially or fully close long
                    position['size'] -= size
                else:
                    # Close long and open short
                    remaining_long = position['size']
                    new_short = size - remaining_long
                    position['size'] = -new_short
                    position['avg_price'] = executed_price
        
        self.logger.info(f"Executed {order_type} order: {size} {symbol} at {executed_price}")
        
        return {
            'success': True,
            'executed_price': executed_price,
            'commission': commission,
            'slippage': slippage,
            'spread': spread,
            'trade': trade
        }
    
    def update_pnl(self, symbol: str, current_price: float) -> float:
        """Update P&L for a position"""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        if position['size'] == 0:
            return 0.0
        
        # Calculate unrealized P&L
        if position['size'] > 0:  # Long position
            pnl = (current_price - position['avg_price']) * position['size']
        else:  # Short position
            pnl = (position['avg_price'] - current_price) * abs(position['size'])
        
        position['unrealized_pnl'] = pnl
        return pnl
    
    def close_position(self, symbol: str, current_price: float, 
                      timestamp: datetime) -> Dict[str, Any]:
        """Close a position completely"""
        if symbol not in self.positions or self.positions[symbol]['size'] == 0:
            return {
                'success': False,
                'error': 'No position to close',
                'realized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        size = abs(position['size'])
        order_type = 'sell' if position['size'] > 0 else 'buy'
        
        # Execute closing order
        result = self.execute_order(symbol, order_type, size, current_price, timestamp)
        
        if result['success']:
            # Calculate realized P&L
            if position['size'] > 0:  # Was long
                realized_pnl = (current_price - position['avg_price']) * size
            else:  # Was short
                realized_pnl = (position['avg_price'] - current_price) * size
            
            # Update account
            self.account_balance += realized_pnl
            
            # Clear position
            self.positions[symbol] = {'size': 0.0, 'avg_price': 0.0, 'unrealized_pnl': 0.0}
            
            result['realized_pnl'] = realized_pnl
            self.logger.info(f"Closed position in {symbol}: P&L = {realized_pnl:.2f}")
        
        return result
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        self.equity = self.account_balance + total_unrealized_pnl
        
        return {
            'account_balance': self.account_balance,
            'equity': self.equity,
            'unrealized_pnl': total_unrealized_pnl,
            'positions': self.positions.copy(),
            'total_trades': len(self.trades),
            'open_positions': len([p for p in self.positions.values() if p['size'] != 0])
        }
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trades.copy()
    
    def reset(self):
        """Reset broker state"""
        self.positions = {}
        self.trades = []
        self.account_balance = 10000.0
        self.equity = 10000.0
        self.logger.info("Broker state reset")

