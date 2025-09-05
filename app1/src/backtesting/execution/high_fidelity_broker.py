#!/usr/bin/env python3
"""
High-Fidelity Order Execution Broker

Implements realistic order execution logic using tick-level bid/ask prices
and variable spreads for 99% modeling quality in backtesting.
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
    from backtrader import BrokerBase, Order, BuyOrder, SellOrder
    from backtrader.broker import BrokerBase
    from backtrader.order import Order, BuyOrder, SellOrder, StopOrder, StopLimitOrder, LimitOrder
except ImportError:
    bt = None
    # Create mock classes for testing when backtrader is not available
    class MockBrokerBase:
        def __init__(self, *args, **kwargs):
            pass
        def getvalue(self):
            return 10000.0
        def setbroker(self, broker):
            pass
        def addsizer(self, sizer):
            pass
        def adddata(self, data, name=None):
            pass
        def addstrategy(self, strategy, **kwargs):
            pass
        def run(self):
            return []
    
    class MockOrder:
        # Status constants
        Created = 0
        Submitted = 1
        Accepted = 2
        Partial = 3
        Completed = 4
        Canceled = 5
        Expired = 6
        Margin = 7
        Rejected = 8
        
        # Order type constants
        Market = 0
        Limit = 1
        Stop = 2
        StopLimit = 3
        
        def __init__(self, *args, **kwargs):
            self.ref = None
            self.status = 0
            self.size = 0
            self.price = None
            self.exectype = 0
            self.executed = None
        def isbuy(self):
            return True
        def getstatusname(self):
            return "Submitted"
        def submit(self):
            pass
        def accept(self):
            pass
        def cancel(self):
            pass
        def execute(self, *args, **kwargs):
            pass
        def completed(self):
            pass
    
    BrokerBase = MockBrokerBase
    Order = MockOrder
    BuyOrder = MockOrder
    SellOrder = MockOrder
    StopOrder = MockOrder
    StopLimitOrder = MockOrder
    LimitOrder = MockOrder

from ..tick_data.variable_spread_simulator import VariableSpreadSimulator, SpreadConfig, SpreadModel
from ..tick_data.enhanced_tick_feed import EnhancedTickDataFeed

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Supported order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class ExecutionConfig:
    """Configuration for high-fidelity execution"""
    
    # Spread simulation
    spread_model: SpreadModel = SpreadModel.HYBRID
    base_spread: float = 0.0001  # 1 pip base spread
    
    # Execution parameters
    slippage_model: str = "realistic"  # "none", "fixed", "realistic", "aggressive"
    slippage_factor: float = 0.1  # 10% of spread as slippage
    min_slippage: float = 0.00001  # 0.1 pip minimum slippage
    max_slippage: float = 0.0005   # 5 pips maximum slippage
    
    # Market impact
    market_impact_enabled: bool = True
    impact_factor: float = 0.0001  # Price impact per unit volume
    max_impact: float = 0.001      # Maximum price impact (10 pips)
    
    # Latency simulation
    latency_enabled: bool = True
    min_latency_ms: int = 10       # Minimum latency in milliseconds
    max_latency_ms: int = 100      # Maximum latency in milliseconds
    avg_latency_ms: int = 50       # Average latency in milliseconds
    
    # Partial fills
    partial_fills_enabled: bool = True
    min_fill_size: float = 0.1     # Minimum 10% fill
    max_fill_size: float = 1.0     # Maximum 100% fill
    fill_probability: float = 0.95  # 95% chance of full fill
    
    # Quality control
    enforce_spread_constraints: bool = True
    validate_execution_prices: bool = True
    log_execution_details: bool = True


@dataclass
class ExecutionResult:
    """Result of order execution"""
    
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    executed_quantity: float
    fill_price: float
    spread_at_execution: float
    slippage: float
    market_impact: float
    execution_time: datetime
    latency_ms: int
    partial_fill: bool
    execution_quality_score: float
    raw_bid: float
    raw_ask: float
    simulated_bid: float
    simulated_ask: float


class HighFidelityBroker(BrokerBase):
    """
    High-fidelity broker that simulates realistic order execution
    using tick-level bid/ask prices and variable spreads.
    """
    
    def __init__(self, execution_config: ExecutionConfig = None):
        super().__init__()
        
        self.config = execution_config or ExecutionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize spread simulator
        spread_config = SpreadConfig(
            model=self.config.spread_model,
            base_spread=self.config.base_spread
        )
        self.spread_simulator = VariableSpreadSimulator(spread_config)
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.execution_results: List[ExecutionResult] = []
        self.next_order_id = 1
        
        # Market data tracking
        self.current_tick_data: Optional[Dict[str, Any]] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.avg_slippage = 0.0
        self.avg_execution_time_ms = 0.0
        
        logger.info("HighFidelityBroker initialized with realistic execution modeling")
    
    def set_market_data(self, tick_data: Dict[str, Any]):
        """Set current market data for execution"""
        self.current_tick_data = tick_data
        
        # Update spread simulator with current market conditions
        if tick_data:
            self.spread_simulator.update_market_conditions(
                volatility=tick_data.get('volatility', 1.0),
                liquidity=tick_data.get('liquidity', 1.0)
            )
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        order_id = f"HF_{self.next_order_id:06d}"
        self.next_order_id += 1
        return order_id
    
    def _calculate_slippage(self, spread: float, quantity: float, side: str) -> float:
        """Calculate realistic slippage based on spread and quantity"""
        
        if self.config.slippage_model == "none":
            return 0.0
        elif self.config.slippage_model == "fixed":
            return self.config.slippage_factor * spread
        elif self.config.slippage_model == "realistic":
            # Realistic slippage based on spread and quantity
            base_slippage = self.config.slippage_factor * spread
            
            # Quantity impact (larger orders = more slippage)
            quantity_factor = min(quantity / 100.0, 2.0)  # Cap at 2x
            quantity_slippage = base_slippage * quantity_factor
            
            # Random component (market conditions)
            random_factor = np.random.uniform(0.5, 1.5)
            total_slippage = quantity_slippage * random_factor
            
            # Apply constraints
            total_slippage = max(self.config.min_slippage, 
                               min(self.config.max_slippage, total_slippage))
            
            return total_slippage
        elif self.config.slippage_model == "aggressive":
            # More aggressive slippage for testing
            return min(spread * 0.5, self.config.max_slippage)
        else:
            return 0.0
    
    def _calculate_market_impact(self, quantity: float, current_price: float) -> float:
        """Calculate market impact based on order size"""
        
        if not self.config.market_impact_enabled:
            return 0.0
        
        # Linear impact model (can be enhanced with more sophisticated models)
        impact = self.config.impact_factor * quantity * current_price
        impact = min(impact, self.config.max_impact)
        
        return impact
    
    def _simulate_latency(self) -> int:
        """Simulate realistic execution latency"""
        
        if not self.config.latency_enabled:
            return 0
        
        # Normal distribution around average latency
        latency = np.random.normal(self.config.avg_latency_ms, 
                                 (self.config.max_latency_ms - self.config.min_latency_ms) / 4)
        
        # Apply bounds
        latency = max(self.config.min_latency_ms, 
                    min(self.config.max_latency_ms, latency))
        
        return int(latency)
    
    def _simulate_partial_fill(self, requested_quantity: float) -> float:
        """Simulate partial fills based on market conditions"""
        
        if not self.config.partial_fills_enabled:
            return requested_quantity
        
        # Probability of full fill
        if np.random.random() < self.config.fill_probability:
            return requested_quantity
        
        # Partial fill
        min_fill = requested_quantity * self.config.min_fill_size
        max_fill = requested_quantity * self.config.max_fill_size
        
        fill_ratio = np.random.uniform(self.config.min_fill_size, self.config.max_fill_size)
        executed_quantity = requested_quantity * fill_ratio
        
        return max(min_fill, min(max_fill, executed_quantity))
    
    def _get_execution_prices(self, side: str, current_tick: Dict[str, Any]) -> Tuple[float, float]:
        """Get execution prices using variable spread simulation"""
        
        if not current_tick:
            raise ValueError("No current tick data available for execution")
        
        # Get current mid price and volume
        mid_price = current_tick.get('mid', 0.0)
        volume = current_tick.get('volume', 1.0)
        timestamp = current_tick.get('timestamp', datetime.now())
        
        # Simulate spread for current market conditions
        spread_data = self.spread_simulator.simulate_spread(
            timestamp=timestamp,
            mid_price=mid_price,
            volume=volume
        )
        
        return spread_data.bid, spread_data.ask
    
    def _execute_market_order(self, order: Order, current_tick: Dict[str, Any]) -> ExecutionResult:
        """Execute market order with realistic pricing"""
        
        # Get execution prices
        bid, ask = self._get_execution_prices(order.isbuy(), current_tick)
        
        # Determine execution price based on order side
        if order.isbuy():
            base_price = ask  # Buy at ask price
            side = "buy"
        else:
            base_price = bid  # Sell at bid price
            side = "sell"
        
        # Calculate slippage
        spread = ask - bid
        slippage = self._calculate_slippage(spread, abs(order.size), side)
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(abs(order.size), base_price)
        
        # Apply slippage and market impact
        if order.isbuy():
            execution_price = base_price + slippage + market_impact
        else:
            execution_price = base_price - slippage - market_impact
        
        # Simulate partial fills
        executed_quantity = self._simulate_partial_fill(abs(order.size))
        
        # Simulate latency
        latency_ms = self._simulate_latency()
        
        # Calculate execution quality score
        quality_score = self._calculate_execution_quality(
            spread, slippage, market_impact, executed_quantity, abs(order.size)
        )
        
        # Create execution result
        result = ExecutionResult(
            order_id=order.ref,
            symbol=getattr(order.data, '_name', 'UNKNOWN'),
            side=side,
            quantity=abs(order.size),
            executed_quantity=executed_quantity,
            fill_price=execution_price,
            spread_at_execution=spread,
            slippage=slippage,
            market_impact=market_impact,
            execution_time=datetime.now(),
            latency_ms=latency_ms,
            partial_fill=executed_quantity < abs(order.size),
            execution_quality_score=quality_score,
            raw_bid=current_tick.get('bid', 0.0),
            raw_ask=current_tick.get('ask', 0.0),
            simulated_bid=bid,
            simulated_ask=ask
        )
        
        return result
    
    def _execute_limit_order(self, order: Order, current_tick: Dict[str, Any]) -> Optional[ExecutionResult]:
        """Execute limit order if price conditions are met"""
        
        # Get execution prices
        bid, ask = self._get_execution_prices(order.isbuy(), current_tick)
        
        # Check if limit price can be executed
        if order.isbuy():
            # Buy limit: execute if ask <= limit price
            if ask <= order.price:
                # Execute at better price (limit price or better)
                execution_price = min(order.price, ask)
                side = "buy"
            else:
                return None  # Cannot execute
        else:
            # Sell limit: execute if bid >= limit price
            if bid >= order.price:
                # Execute at better price (limit price or better)
                execution_price = max(order.price, bid)
                side = "sell"
            else:
                return None  # Cannot execute
        
        # Calculate spread and slippage
        spread = ask - bid
        slippage = self._calculate_slippage(spread, abs(order.size), side)
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(abs(order.size), execution_price)
        
        # Apply market impact
        if order.isbuy():
            execution_price += market_impact
        else:
            execution_price -= market_impact
        
        # Simulate partial fills
        executed_quantity = self._simulate_partial_fill(abs(order.size))
        
        # Simulate latency
        latency_ms = self._simulate_latency()
        
        # Calculate execution quality score
        quality_score = self._calculate_execution_quality(
            spread, slippage, market_impact, executed_quantity, abs(order.size)
        )
        
        # Create execution result
        result = ExecutionResult(
            order_id=order.ref,
            symbol=getattr(order.data, '_name', 'UNKNOWN'),
            side=side,
            quantity=abs(order.size),
            executed_quantity=executed_quantity,
            fill_price=execution_price,
            spread_at_execution=spread,
            slippage=slippage,
            market_impact=market_impact,
            execution_time=datetime.now(),
            latency_ms=latency_ms,
            partial_fill=executed_quantity < abs(order.size),
            execution_quality_score=quality_score,
            raw_bid=current_tick.get('bid', 0.0),
            raw_ask=current_tick.get('ask', 0.0),
            simulated_bid=bid,
            simulated_ask=ask
        )
        
        return result
    
    def _execute_stop_order(self, order: Order, current_tick: Dict[str, Any]) -> Optional[ExecutionResult]:
        """Execute stop order if stop price is triggered"""
        
        # Get execution prices
        bid, ask = self._get_execution_prices(order.isbuy(), current_tick)
        
        # Check if stop price is triggered
        if order.isbuy():
            # Buy stop: execute if ask >= stop price
            if ask >= order.price:
                execution_price = ask  # Execute at market
                side = "buy"
            else:
                return None  # Stop not triggered
        else:
            # Sell stop: execute if bid <= stop price
            if bid <= order.price:
                execution_price = bid  # Execute at market
                side = "sell"
            else:
                return None  # Stop not triggered
        
        # Calculate spread and slippage
        spread = ask - bid
        slippage = self._calculate_slippage(spread, abs(order.size), side)
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(abs(order.size), execution_price)
        
        # Apply slippage and market impact
        if order.isbuy():
            execution_price += slippage + market_impact
        else:
            execution_price -= slippage + market_impact
        
        # Simulate partial fills
        executed_quantity = self._simulate_partial_fill(abs(order.size))
        
        # Simulate latency
        latency_ms = self._simulate_latency()
        
        # Calculate execution quality score
        quality_score = self._calculate_execution_quality(
            spread, slippage, market_impact, executed_quantity, abs(order.size)
        )
        
        # Create execution result
        result = ExecutionResult(
            order_id=order.ref,
            symbol=getattr(order.data, '_name', 'UNKNOWN'),
            side=side,
            quantity=abs(order.size),
            executed_quantity=executed_quantity,
            fill_price=execution_price,
            spread_at_execution=spread,
            slippage=slippage,
            market_impact=market_impact,
            execution_time=datetime.now(),
            latency_ms=latency_ms,
            partial_fill=executed_quantity < abs(order.size),
            execution_quality_score=quality_score,
            raw_bid=current_tick.get('bid', 0.0),
            raw_ask=current_tick.get('ask', 0.0),
            simulated_bid=bid,
            simulated_ask=ask
        )
        
        return result
    
    def _calculate_execution_quality(self, spread: float, slippage: float, 
                                   market_impact: float, executed_quantity: float, 
                                   requested_quantity: float) -> float:
        """Calculate execution quality score (0-100)"""
        
        # Base score
        quality_score = 100.0
        
        # Slippage penalty (relative to spread)
        if spread > 0:
            slippage_ratio = slippage / spread
            if slippage_ratio > 0.5:  # More than 50% of spread
                quality_score -= 20
            elif slippage_ratio > 0.25:  # More than 25% of spread
                quality_score -= 10
        
        # Market impact penalty
        if market_impact > spread * 0.1:  # More than 10% of spread
            quality_score -= 15
        
        # Partial fill penalty
        fill_ratio = executed_quantity / requested_quantity
        if fill_ratio < 1.0:
            quality_score -= (1.0 - fill_ratio) * 30
        
        # Ensure score is within bounds
        quality_score = max(0.0, min(100.0, quality_score))
        
        return quality_score
    
    def submit(self, order: Order) -> Order:
        """Submit order for execution"""
        
        # Generate order ID
        order.ref = self._generate_order_id()
        self.orders[order.ref] = order
        
        # Set order status
        order.submit()
        order.accept()
        
        self.total_orders += 1
        
        logger.debug(f"Order {order.ref} submitted: {order.getstatusname()}")
        
        return order
    
    def cancel(self, order: Order) -> Order:
        """Cancel order"""
        
        if order.ref in self.orders:
            order.cancel()
            logger.debug(f"Order {order.ref} cancelled")
        
        return order
    
    def buy(self, owner, data, size, price=None, plimit=None, exectype=None, valid=None, 
            tradeid=0, oco=None, trailamount=None, trailpercent=None, 
            parent=None, transmit=True, **kwargs) -> Order:
        """Create buy order"""
        
        order = BuyOrder(owner, data, size, price=price, plimit=plimit, 
                        exectype=exectype, valid=valid, tradeid=tradeid, 
                        oco=oco, trailamount=trailamount, trailpercent=trailpercent,
                        parent=parent, transmit=transmit, **kwargs)
        
        return self.submit(order)
    
    def sell(self, owner, data, size, price=None, plimit=None, exectype=None, valid=None,
             tradeid=0, oco=None, trailamount=None, trailpercent=None,
             parent=None, transmit=True, **kwargs) -> Order:
        """Create sell order"""
        
        order = SellOrder(owner, data, size, price=price, plimit=plimit,
                         exectype=exectype, valid=valid, tradeid=tradeid,
                         oco=oco, trailamount=trailamount, trailpercent=trailpercent,
                         parent=parent, transmit=transmit, **kwargs)
        
        return self.submit(order)
    
    def process_orders(self, current_tick: Dict[str, Any]):
        """Process all pending orders with current tick data"""
        
        if not current_tick:
            return
        
        # Set current market data
        self.set_market_data(current_tick)
        
        # Process each pending order
        for order_ref, order in list(self.orders.items()):
            if order.status in [Order.Submitted, Order.Accepted]:
                try:
                    # Execute order based on type
                    if order.exectype == Order.Market:
                        result = self._execute_market_order(order, current_tick)
                    elif order.exectype == Order.Limit:
                        result = self._execute_limit_order(order, current_tick)
                    elif order.exectype == Order.Stop:
                        result = self._execute_stop_order(order, current_tick)
                    elif order.exectype == Order.StopLimit:
                        # For now, treat as stop order (can be enhanced)
                        result = self._execute_stop_order(order, current_tick)
                    else:
                        logger.warning(f"Unsupported order type: {order.exectype}")
                        continue
                    
                    if result:
                        # Order was executed
                        self._complete_order(order, result)
                        self.execution_results.append(result)
                        self.successful_orders += 1
                        
                        if self.config.log_execution_details:
                            logger.info(f"Order {order.ref} executed: {result.executed_quantity} @ {result.fill_price:.5f}")
                    else:
                        # Order not executed (e.g., limit not hit, stop not triggered)
                        pass
                        
                except Exception as e:
                    logger.error(f"Error executing order {order.ref}: {e}")
                    order.cancel()
                    self.failed_orders += 1
    
    def _complete_order(self, order: Order, result: ExecutionResult):
        """Complete order execution"""
        
        # Update order with execution details
        order.execute(result.execution_time, result.executed_quantity, 
                     result.fill_price, 0, 0, 0, 0, 0, 0, 0)
        
        # Mark as completed
        order.completed()
        
        # Remove from pending orders
        if order.ref in self.orders:
            del self.orders[order.ref]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        
        if self.total_orders == 0:
            return {
                'total_orders': 0,
                'successful_orders': 0,
                'failed_orders': 0,
                'success_rate': 0.0,
                'avg_slippage': 0.0,
                'avg_execution_time_ms': 0.0,
                'avg_quality_score': 0.0
            }
        
        # Calculate statistics
        success_rate = (self.successful_orders / self.total_orders) * 100
        
        if self.execution_results:
            avg_slippage = np.mean([r.slippage for r in self.execution_results])
            avg_execution_time = np.mean([r.latency_ms for r in self.execution_results])
            avg_quality_score = np.mean([r.execution_quality_score for r in self.execution_results])
        else:
            avg_slippage = 0.0
            avg_execution_time = 0.0
            avg_quality_score = 0.0
        
        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'success_rate': success_rate,
            'avg_slippage': avg_slippage,
            'avg_execution_time_ms': avg_execution_time,
            'avg_quality_score': avg_quality_score,
            'execution_results': len(self.execution_results)
        }
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get execution history"""
        return self.execution_results.copy()
    
    def reset_statistics(self):
        """Reset execution statistics"""
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.execution_results.clear()
        self.execution_history.clear()
