#!/usr/bin/env python3
"""
Strategy Generator

Generates real, backtestable trading strategies using the strategy templates.
"""

import random
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .core.strategy_template import StrategyTemplate
from .templates.trend_strategy import TrendStrategy
from .templates.range_strategy import RangeStrategy
from .templates.breakout_strategy import BreakoutStrategy
from .templates.sentiment_strategy import SentimentStrategy
from .templates.news_strategy import NewsStrategy
from .templates.multi_timeframe_strategy import MultiTimeframeStrategy
from .templates.pairs_strategy import PairsStrategy

@dataclass
class StrategyGenerationRequest:
    """Request for strategy generation"""
    strategy_types: List[str]  # ['trend', 'range', 'breakout', etc.]
    symbols: List[str]  # ['EURUSD', 'GBPUSD', etc.]
    timeframes: List[str]  # ['1h', '4h', '1d', etc.]
    count: int = 5  # Number of strategies to generate
    market_conditions: Optional[str] = None  # 'trending', 'ranging', 'volatile'
    risk_level: Optional[str] = None  # 'low', 'medium', 'high'

@dataclass
class GeneratedStrategy:
    """Generated strategy definition"""
    strategy_id: str
    strategy_name: str
    strategy_type: str
    module_path: str
    class_name: str
    parameters: Dict[str, Any]
    description: str
    market_conditions: str
    risk_level: str
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type,
            'module_path': self.module_path,
            'class_name': self.class_name,
            'parameters': self.parameters,
            'description': self.description,
            'market_conditions': self.market_conditions,
            'risk_level': self.risk_level,
            'created_at': self.created_at.isoformat()
        }

class StrategyGenerator:
    """Generates real trading strategies"""
    
    def __init__(self):
        self.strategy_classes = {
            'trend': TrendStrategy,
            'range': RangeStrategy,
            'breakout': BreakoutStrategy,
            'sentiment': SentimentStrategy,
            'news': NewsStrategy,
            'multi_timeframe': MultiTimeframeStrategy,
            'pairs': PairsStrategy
        }
        
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY']
        self.timeframes = ['1h', '4h', '1d', '1w']
        self.market_conditions = ['trending', 'ranging', 'volatile', 'consolidating']
        self.risk_levels = ['low', 'medium', 'high']
    
    def generate_strategies(self, request: StrategyGenerationRequest) -> List[GeneratedStrategy]:
        """Generate multiple strategies based on request"""
        strategies = []
        
        for i in range(request.count):
            strategy_type = random.choice(request.strategy_types)
            symbol = random.choice(request.symbols)
            timeframe = random.choice(request.timeframes)
            
            strategy = self._generate_single_strategy(
                strategy_type=strategy_type,
                symbol=symbol,
                timeframe=timeframe,
                market_conditions=request.market_conditions or random.choice(self.market_conditions),
                risk_level=request.risk_level or random.choice(self.risk_levels)
            )
            
            strategies.append(strategy)
        
        return strategies
    
    def _generate_single_strategy(self, strategy_type: str, symbol: str, timeframe: str, 
                                 market_conditions: str, risk_level: str) -> GeneratedStrategy:
        """Generate a single strategy"""
        
        # Generate unique ID and name
        strategy_id = f"GEN_{strategy_type.upper()}_{uuid.uuid4().hex[:8].upper()}"
        strategy_name = self._generate_strategy_name(strategy_type, symbol, timeframe)
        
        # Generate parameters based on strategy type
        parameters = self._generate_parameters(strategy_type, symbol, timeframe, risk_level)
        
        # Generate description
        description = self._generate_description(strategy_type, symbol, timeframe, parameters, market_conditions, risk_level)
        
        return GeneratedStrategy(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            module_path=f"strategy_generation.templates.{strategy_type}_strategy",
            class_name=f"{strategy_type.title()}Strategy",
            parameters=parameters,
            description=description,
            market_conditions=market_conditions,
            risk_level=risk_level,
            created_at=datetime.now()
        )
    
    def _generate_strategy_name(self, strategy_type: str, symbol: str, timeframe: str) -> str:
        """Generate a descriptive strategy name"""
        type_names = {
            'trend': 'Trend Following',
            'range': 'Range Trading',
            'breakout': 'Breakout',
            'sentiment': 'Sentiment Based',
            'news': 'News Driven',
            'multi_timeframe': 'Multi-Timeframe',
            'pairs': 'Pairs Trading'
        }
        
        base_name = type_names.get(strategy_type, strategy_type.title())
        return f"{base_name} {symbol} {timeframe}"
    
    def _generate_parameters(self, strategy_type: str, symbol: str, timeframe: str, risk_level: str) -> Dict[str, Any]:
        """Generate parameters based on strategy type and risk level"""
        base_params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'name': f"{strategy_type}_{symbol}_{timeframe}"
        }
        
        # Risk-based parameter scaling
        risk_multipliers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.5
        }
        risk_mult = risk_multipliers.get(risk_level, 1.0)
        
        if strategy_type == 'trend':
            return {
                **base_params,
                'short_period': int(10 * risk_mult),
                'long_period': int(20 * risk_mult),
                'threshold': 0.001 * risk_mult,
                'stop_loss': 0.01 * risk_mult,
                'take_profit': 0.02 * risk_mult
            }
        
        elif strategy_type == 'range':
            return {
                **base_params,
                'period': int(20 * risk_mult),
                'std_dev': 2.0,
                'support_level': 1.2500,
                'resistance_level': 1.2800,
                'entry_threshold': 0.0005 * risk_mult,
                'exit_threshold': 0.001 * risk_mult
            }
        
        elif strategy_type == 'breakout':
            return {
                **base_params,
                'lookback_period': int(50 * risk_mult),
                'breakout_threshold': 0.002 * risk_mult,
                'volume_threshold': 1.5,
                'confirmation_periods': int(3 * risk_mult),
                'stop_loss': 0.015 * risk_mult
            }
        
        elif strategy_type == 'sentiment':
            return {
                **base_params,
                'sentiment_threshold': 0.6,
                'confidence_threshold': 0.7,
                'lookback_hours': int(24 * risk_mult),
                'weight_news': 0.4,
                'weight_social': 0.3,
                'weight_technical': 0.3
            }
        
        elif strategy_type == 'news':
            return {
                **base_params,
                'impact_threshold': 'medium',
                'time_window_minutes': int(30 * risk_mult),
                'sentiment_weight': 0.6,
                'volatility_threshold': 0.01 * risk_mult,
                'max_positions': int(3 * risk_mult)
            }
        
        elif strategy_type == 'multi_timeframe':
            return {
                **base_params,
                'primary_timeframe': timeframe,
                'secondary_timeframe': '4h' if timeframe == '1h' else '1d',
                'trend_confirmation': True,
                'entry_timeframe': '15m',
                'risk_per_trade': 0.02 * risk_mult
            }
        
        elif strategy_type == 'pairs':
            return {
                **base_params,
                'pair_symbol': random.choice([s for s in self.symbols if s != symbol]),
                'correlation_threshold': 0.7,
                'zscore_threshold': 2.0 * risk_mult,
                'lookback_period': int(100 * risk_mult),
                'rebalance_frequency': 'daily'
            }
        
        return base_params
    
    def _generate_description(self, strategy_type: str, symbol: str, timeframe: str, 
                            parameters: Dict[str, Any], market_conditions: str, risk_level: str) -> str:
        """Generate a descriptive strategy description"""
        
        descriptions = {
            'trend': f"Moving average crossover strategy for {symbol} on {timeframe} timeframe. "
                    f"Uses {parameters.get('short_period', 10)}-period short MA and {parameters.get('long_period', 20)}-period long MA. "
                    f"Optimized for {market_conditions} market conditions with {risk_level} risk profile.",
            
            'range': f"Bollinger Bands range trading strategy for {symbol} on {timeframe} timeframe. "
                    f"Trades between support and resistance levels using {parameters.get('period', 20)}-period BB with "
                    f"{parameters.get('std_dev', 2.0)} standard deviations. Suitable for {market_conditions} markets.",
            
            'breakout': f"Support/Resistance breakout strategy for {symbol} on {timeframe} timeframe. "
                       f"Identifies breakouts using {parameters.get('lookback_period', 50)}-period lookback with "
                       f"{parameters.get('breakout_threshold', 0.002)} threshold. Designed for {market_conditions} conditions.",
            
            'sentiment': f"Sentiment-driven strategy for {symbol} combining news, social media, and technical analysis. "
                        f"Uses {parameters.get('sentiment_threshold', 0.6)} sentiment threshold with "
                        f"{parameters.get('confidence_threshold', 0.7)} confidence level. Optimized for {risk_level} risk.",
            
            'news': f"News event trading strategy for {symbol} that reacts to high-impact economic events. "
                   f"Trades within {parameters.get('time_window_minutes', 30)} minutes of news release with "
                   f"{parameters.get('impact_threshold', 'medium')} impact threshold. {risk_level.title()} risk profile.",
            
            'multi_timeframe': f"Multi-timeframe analysis strategy for {symbol} using {timeframe} and secondary timeframes. "
                              f"Combines trend analysis across multiple timeframes with {parameters.get('risk_per_trade', 0.02)} risk per trade. "
                              f"Designed for {market_conditions} market conditions.",
            
            'pairs': f"Pairs trading strategy between {symbol} and {parameters.get('pair_symbol', 'EURUSD')}. "
                    f"Uses statistical arbitrage with {parameters.get('correlation_threshold', 0.7)} correlation threshold and "
                    f"{parameters.get('zscore_threshold', 2.0)} z-score entry. {risk_level.title()} risk profile."
        }
        
        return descriptions.get(strategy_type, f"Custom {strategy_type} strategy for {symbol} on {timeframe} timeframe.")
    
    def get_available_strategy_types(self) -> List[str]:
        """Get list of available strategy types"""
        return list(self.strategy_classes.keys())
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        return self.symbols.copy()
    
    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes"""
        return self.timeframes.copy()
    
    def get_available_market_conditions(self) -> List[str]:
        """Get list of available market conditions"""
        return self.market_conditions.copy()
    
    def get_available_risk_levels(self) -> List[str]:
        """Get list of available risk levels"""
        return self.risk_levels.copy()
