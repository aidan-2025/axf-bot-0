"""
Strategy Generation Engine

A modular system for generating, optimizing, and validating trading strategies
using genetic algorithms, technical analysis, sentiment analysis, and economic data.
"""

from .core.strategy_template import StrategyTemplate, StrategyType, StrategyParameters
from .core.strategy_engine import StrategyGenerationEngine
from .core.parameter_space import ParameterSpace, ParameterType
from .templates.trend_strategy import TrendStrategy
from .templates.range_strategy import RangeStrategy
from .templates.breakout_strategy import BreakoutStrategy
from .templates.sentiment_strategy import SentimentStrategy
from .templates.news_strategy import NewsStrategy
from .templates.multi_timeframe_strategy import MultiTimeframeStrategy
from .templates.pairs_strategy import PairsStrategy
from .optimization.genetic_optimizer import GeneticOptimizer
from .optimization.monte_carlo import MonteCarloSimulator
from .optimization.walk_forward import WalkForwardTester
from .validation.strategy_validator import StrategyValidator
from .modules.signal_processor import SignalProcessor
from .modules.feature_extractor import FeatureExtractor

__all__ = [
    # Core classes
    'StrategyTemplate',
    'StrategyType',
    'StrategyParameters',
    'StrategyGenerationEngine',
    'ParameterSpace',
    'ParameterType',
    
    # Strategy templates
    'TrendStrategy',
    'RangeStrategy', 
    'BreakoutStrategy',
    'SentimentStrategy',
    'NewsStrategy',
    'MultiTimeframeStrategy',
    'PairsStrategy',
    
    # Optimization
    'GeneticOptimizer',
    'MonteCarloSimulator',
    'WalkForwardTester',
    
    # Validation
    'StrategyValidator',
    
    # Modules
    'SignalProcessor',
    'FeatureExtractor'
]
