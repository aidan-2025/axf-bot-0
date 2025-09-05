"""
Main strategy generation engine
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json

from .strategy_template import StrategyTemplate, StrategyType, StrategyParameters
from .parameter_space import ParameterSpace
from ..templates.trend_strategy import TrendStrategy
from ..templates.range_strategy import RangeStrategy
from ..templates.breakout_strategy import BreakoutStrategy
from ..templates.sentiment_strategy import SentimentStrategy
from ..templates.news_strategy import NewsStrategy
from ..templates.multi_timeframe_strategy import MultiTimeframeStrategy
from ..templates.pairs_strategy import PairsStrategy
from ..optimization.genetic_optimizer import GeneticOptimizer
from ..optimization.monte_carlo import MonteCarloSimulator
from ..optimization.walk_forward import WalkForwardTester
from ..validation.strategy_validator import StrategyValidator
from ..modules.signal_processor import SignalProcessor
from ..modules.feature_extractor import FeatureExtractor


class StrategyGenerationEngine:
    """
    Main engine for generating, optimizing, and validating trading strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.signal_processor = SignalProcessor()
        self.feature_extractor = FeatureExtractor()
        self.genetic_optimizer = GeneticOptimizer()
        self.monte_carlo_simulator = MonteCarloSimulator()
        self.walk_forward_tester = WalkForwardTester()
        self.strategy_validator = StrategyValidator()
        
        # Strategy template registry
        self.strategy_templates = {
            StrategyType.TREND: TrendStrategy,
            StrategyType.RANGE: RangeStrategy,
            StrategyType.BREAKOUT: BreakoutStrategy,
            StrategyType.SENTIMENT: SentimentStrategy,
            StrategyType.NEWS: NewsStrategy,
            StrategyType.MULTI_TIMEFRAME: MultiTimeframeStrategy,
            StrategyType.PAIRS: PairsStrategy
        }
        
        # Generated strategies cache
        self.generated_strategies: List[StrategyTemplate] = []
        self.optimization_results: Dict[str, Any] = {}
        
    def create_strategy(self, strategy_type: StrategyType, parameters: StrategyParameters) -> Optional[StrategyTemplate]:
        """
        Create a new strategy instance
        
        Args:
            strategy_type: Type of strategy to create
            parameters: Strategy parameters
            
        Returns:
            StrategyTemplate: Created strategy instance or None if failed
        """
        try:
            if strategy_type not in self.strategy_templates:
                self.logger.error(f"Unknown strategy type: {strategy_type}")
                return None
            
            strategy_class = self.strategy_templates[strategy_type]
            strategy = strategy_class(parameters)
            
            # Initialize the strategy
            if strategy.initialize():
                self.generated_strategies.append(strategy)
                self.logger.info(f"Created strategy: {strategy}")
                return strategy
            else:
                self.logger.error(f"Failed to initialize strategy: {strategy}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating strategy: {e}")
            return None
    
    def generate_strategies(self, 
                          strategy_types: List[StrategyType],
                          parameter_ranges: Dict[str, Dict[str, Any]],
                          count: int = 10) -> List[StrategyTemplate]:
        """
        Generate multiple strategies using random parameter sampling
        
        Args:
            strategy_types: List of strategy types to generate
            parameter_ranges: Parameter ranges for each strategy type
            count: Number of strategies to generate per type
            
        Returns:
            List[StrategyTemplate]: Generated strategies
        """
        generated = []
        
        for strategy_type in strategy_types:
            for i in range(count):
                try:
                    # Generate random parameters
                    params = self._generate_random_parameters(
                        strategy_type, 
                        parameter_ranges.get(strategy_type.value, {})
                    )
                    
                    # Create strategy
                    strategy = self.create_strategy(strategy_type, params)
                    if strategy:
                        generated.append(strategy)
                        
                except Exception as e:
                    self.logger.error(f"Error generating strategy {i} of type {strategy_type}: {e}")
        
        self.logger.info(f"Generated {len(generated)} strategies")
        return generated
    
    def optimize_strategy(self, 
                         strategy: StrategyTemplate,
                         market_data: Dict[str, Any],
                         optimization_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize a strategy using genetic algorithms
        
        Args:
            strategy: Strategy to optimize
            market_data: Historical market data for optimization
            optimization_config: Optimization configuration
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        try:
            self.logger.info(f"Starting optimization for strategy: {strategy}")
            
            # Get parameter space
            param_space = strategy.get_parameter_space()
            
            # Run genetic optimization
            optimization_results = self.genetic_optimizer.optimize(
                strategy=strategy,
                parameter_space=param_space,
                market_data=market_data,
                config=optimization_config or {}
            )
            
            # Update strategy with optimized parameters
            if optimization_results.get('success', False):
                best_params = optimization_results.get('best_parameters', {})
                strategy.update_parameters(best_params)
                
                # Re-initialize with optimized parameters
                strategy.initialize()
            
            self.optimization_results[strategy.parameters.strategy_id] = optimization_results
            self.logger.info(f"Optimization completed for strategy: {strategy}")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_strategy(self, strategy: StrategyTemplate) -> Tuple[bool, List[str]]:
        """
        Validate a strategy for logical consistency and requirements
        
        Args:
            strategy: Strategy to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        return self.strategy_validator.validate(strategy)
    
    def backtest_strategy(self, 
                         strategy: StrategyTemplate,
                         market_data: Dict[str, Any],
                         config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Backtest a strategy on historical data
        
        Args:
            strategy: Strategy to backtest
            market_data: Historical market data
            config: Backtesting configuration
            
        Returns:
            Dict[str, Any]: Backtesting results
        """
        try:
            self.logger.info(f"Starting backtest for strategy: {strategy}")
            
            # Process market data
            processed_data = self.signal_processor.process_market_data(market_data)
            
            # Generate signals
            signals = strategy.generate_signals(processed_data)
            
            # Simulate trading
            backtest_results = self._simulate_trading(signals, processed_data, config or {})
            
            self.logger.info(f"Backtest completed for strategy: {strategy}")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error backtesting strategy: {e}")
            return {"success": False, "error": str(e)}
    
    def run_monte_carlo_simulation(self, 
                                  strategy: StrategyTemplate,
                                  market_data: Dict[str, Any],
                                  iterations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on a strategy
        
        Args:
            strategy: Strategy to simulate
            market_data: Historical market data
            iterations: Number of simulation iterations
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        return self.monte_carlo_simulator.simulate(
            strategy=strategy,
            market_data=market_data,
            iterations=iterations
        )
    
    def run_walk_forward_analysis(self, 
                                 strategy: StrategyTemplate,
                                 market_data: Dict[str, Any],
                                 config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run walk-forward analysis on a strategy
        
        Args:
            strategy: Strategy to analyze
            market_data: Historical market data
            config: Walk-forward configuration
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        return self.walk_forward_tester.test(
            strategy=strategy,
            market_data=market_data,
            config=config or {}
        )
    
    def get_strategy_performance(self, strategy: StrategyTemplate) -> Dict[str, Any]:
        """
        Get performance metrics for a strategy
        
        Args:
            strategy: Strategy to analyze
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        latest_performance = strategy.get_latest_performance()
        if latest_performance:
            return {
                "total_trades": latest_performance.total_trades,
                "win_rate": latest_performance.win_rate,
                "profit_factor": latest_performance.profit_factor,
                "sharpe_ratio": latest_performance.sharpe_ratio,
                "max_drawdown": latest_performance.max_drawdown,
                "total_return": latest_performance.total_return,
                "volatility": latest_performance.volatility
            }
        return {}
    
    def export_strategy(self, strategy: StrategyTemplate, format: str = "json") -> str:
        """
        Export strategy to specified format
        
        Args:
            strategy: Strategy to export
            format: Export format (json, xml, etc.)
            
        Returns:
            str: Exported strategy data
        """
        if format.lower() == "json":
            return json.dumps({
                "strategy_info": strategy.get_strategy_info(),
                "parameters": strategy.parameters.parameters,
                "performance": self.get_strategy_performance(strategy)
            }, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_random_parameters(self, 
                                  strategy_type: StrategyType,
                                  parameter_ranges: Dict[str, Any]) -> StrategyParameters:
        """Generate random parameters for a strategy type"""
        import random
        import uuid
        
        # Generate basic parameters
        strategy_id = f"{strategy_type.value}_{uuid.uuid4().hex[:8]}"
        name = f"{strategy_type.value.title()} Strategy {random.randint(1000, 9999)}"
        description = f"Generated {strategy_type.value} strategy"
        
        # Generate strategy-specific parameters
        parameters = {}
        for param_name, param_config in parameter_ranges.items():
            if param_config.get("type") == "int":
                min_val = param_config.get("min", 1)
                max_val = param_config.get("max", 100)
                parameters[param_name] = random.randint(min_val, max_val)
            elif param_config.get("type") == "float":
                min_val = param_config.get("min", 0.0)
                max_val = param_config.get("max", 1.0)
                parameters[param_name] = random.uniform(min_val, max_val)
            elif param_config.get("type") == "bool":
                parameters[param_name] = random.choice([True, False])
            elif param_config.get("type") == "categorical":
                categories = param_config.get("categories", ["option1"])
                parameters[param_name] = random.choice(categories)
        
        return StrategyParameters(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            name=name,
            description=description,
            parameters=parameters
        )
    
    def _simulate_trading(self, 
                         signals: List[Any],
                         market_data: Dict[str, Any],
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate trading based on signals"""
        # This is a simplified simulation - implement full trading simulation
        return {
            "success": True,
            "total_trades": len(signals),
            "winning_trades": len([s for s in signals if s.signal_type == "buy"]),
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and statistics"""
        return {
            "total_strategies": len(self.generated_strategies),
            "strategy_types": list(self.strategy_templates.keys()),
            "optimization_results": len(self.optimization_results),
            "engine_initialized": True,
            "timestamp": datetime.now().isoformat()
        }

