"""
Base strategy template and types
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta


class StrategyType(Enum):
    """Strategy type enumeration"""
    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    SENTIMENT = "sentiment"
    NEWS = "news"
    MULTI_TIMEFRAME = "multi_timeframe"
    PAIRS = "pairs"


@dataclass
class Signal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyParameters:
    """Base strategy parameters"""
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    parameters: Dict[str, Any]
    risk_level: str = "medium"  # low, medium, high
    market_conditions: List[str] = None
    timeframes: List[str] = None
    symbols: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.market_conditions is None:
            self.market_conditions = []
        if self.timeframes is None:
            self.timeframes = ["H1"]
        if self.symbols is None:
            self.symbols = ["EURUSD"]
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: timedelta
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    recovery_factor: float
    stability: float  # Strategy stability score
    robustness: float  # Strategy robustness score


class StrategyTemplate(ABC):
    """
    Abstract base class for all strategy templates
    
    This class defines the interface that all strategy modules must implement
    to ensure consistency and modularity across the strategy generation engine.
    """
    
    def __init__(self, parameters: StrategyParameters):
        self.parameters = parameters
        self.is_initialized = False
        self.performance_history: List[StrategyPerformance] = []
        self.signal_history: List[Signal] = []
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the strategy with given parameters
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on market data
        
        Args:
            market_data: Dictionary containing market data (OHLCV, indicators, etc.)
            
        Returns:
            List[Signal]: List of generated trading signals
        """
        pass
    
    @abstractmethod
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """
        Validate strategy parameters for logical consistency
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """
        Get the parameter space definition for optimization
        
        Returns:
            Dict[str, Any]: Parameter space configuration
        """
        pass
    
    @abstractmethod
    def update_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """
        Update strategy parameters
        
        Args:
            new_parameters: New parameter values
            
        Returns:
            bool: True if update successful, False otherwise
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information
        
        Returns:
            Dict[str, Any]: Strategy information dictionary
        """
        return {
            "strategy_id": self.parameters.strategy_id,
            "strategy_type": self.parameters.strategy_type.value,
            "name": self.parameters.name,
            "description": self.parameters.description,
            "risk_level": self.parameters.risk_level,
            "market_conditions": self.parameters.market_conditions,
            "timeframes": self.parameters.timeframes,
            "symbols": self.parameters.symbols,
            "is_initialized": self.is_initialized,
            "created_at": self.parameters.created_at.isoformat(),
            "updated_at": self.parameters.updated_at.isoformat()
        }
    
    def add_performance_metrics(self, performance: StrategyPerformance):
        """Add performance metrics to history"""
        self.performance_history.append(performance)
        
    def get_latest_performance(self) -> Optional[StrategyPerformance]:
        """Get the most recent performance metrics"""
        return self.performance_history[-1] if self.performance_history else None
    
    def get_average_performance(self) -> Optional[StrategyPerformance]:
        """Get average performance across all history"""
        if not self.performance_history:
            return None
            
        # Calculate averages
        total_trades = sum(p.total_trades for p in self.performance_history)
        winning_trades = sum(p.winning_trades for p in self.performance_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        profit_factor = np.mean([p.profit_factor for p in self.performance_history])
        sharpe_ratio = np.mean([p.sharpe_ratio for p in self.performance_history])
        max_drawdown = max(p.max_drawdown for p in self.performance_history)
        
        return StrategyPerformance(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_trade_duration=timedelta(0),  # Calculate if needed
            total_return=0.0,  # Calculate if needed
            volatility=0.0,  # Calculate if needed
            calmar_ratio=0.0,  # Calculate if needed
            sortino_ratio=0.0,  # Calculate if needed
            var_95=0.0,  # Calculate if needed
            cvar_95=0.0,  # Calculate if needed
            recovery_factor=0.0,  # Calculate if needed
            stability=0.0,  # Calculate if needed
            robustness=0.0  # Calculate if needed
        )
    
    def reset(self):
        """Reset strategy state"""
        self.is_initialized = False
        self.performance_history.clear()
        self.signal_history.clear()
    
    def __str__(self) -> str:
        return f"{self.parameters.strategy_type.value.title()}Strategy({self.parameters.name})"
    
    def __repr__(self) -> str:
        return (f"StrategyTemplate(id={self.parameters.strategy_id}, "
                f"type={self.parameters.strategy_type.value}, "
                f"name={self.parameters.name})")

