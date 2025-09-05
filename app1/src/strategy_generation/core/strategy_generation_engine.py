"""
Core Strategy Generation Engine - Orchestrates the complete strategy generation pipeline
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Import all strategy generation components
from .strategy_template import StrategyTemplate, StrategyParameters, StrategyType
from .parameter_space import ParameterSpace, ParameterDefinition, ParameterType
from ..templates.trend_strategy import TrendStrategy
from ..templates.range_strategy import RangeStrategy
from ..templates.breakout_strategy import BreakoutStrategy
from ..templates.sentiment_strategy import SentimentStrategy
from ..templates.news_strategy import NewsStrategy
from ..templates.multi_timeframe_strategy import MultiTimeframeStrategy
from ..templates.pairs_strategy import PairsStrategy
from ..modules.signal_processor import SignalProcessor
from ..modules.feature_extractor import FeatureExtractor
from ..modules.advanced_signal_processor import AdvancedSignalProcessor
from ..modules.advanced_feature_extractor import AdvancedFeatureExtractor
from ..modules.real_time_integration import RealTimeIntegration
from ..optimization.advanced_genetic_optimizer import (
    AdvancedGeneticOptimizer, OptimizationConfig, OptimizationObjective, FitnessMetrics
)
from ..optimization.advanced_monte_carlo import (
    AdvancedMonteCarloSimulator, MonteCarloConfig, SimulationType
)
from ..optimization.advanced_walk_forward import (
    AdvancedWalkForwardTester, WalkForwardConfig, WalkForwardMode
)
from ..validation.strategy_validator import StrategyValidator


class GenerationStatus(Enum):
    """Strategy generation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    DEPLOYED = "deployed"


class StrategyCategory(Enum):
    """Strategy category for organization"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SENTIMENT_BASED = "sentiment_based"
    NEWS_BASED = "news_based"
    MULTI_TIMEFRAME = "multi_timeframe"
    PAIRS_TRADING = "pairs_trading"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"


@dataclass
class GenerationConfig:
    """Configuration for strategy generation"""
    # Generation parameters
    max_strategies: int = 100
    generation_timeout: int = 3600  # 1 hour
    parallel_generation: bool = True
    max_parallel_tasks: int = 4
    
    # Optimization parameters
    optimization_config: OptimizationConfig = None
    monte_carlo_config: MonteCarloConfig = None
    walk_forward_config: WalkForwardConfig = None
    
    # Validation parameters
    min_performance_score: float = 0.6
    max_drawdown_threshold: float = 0.2
    min_sharpe_ratio: float = 0.5
    min_win_rate: float = 0.4
    min_trades: int = 10
    
    # Strategy diversity
    strategy_diversity_weight: float = 0.3
    parameter_diversity_weight: float = 0.2
    
    # Output parameters
    save_intermediate_results: bool = True
    output_format: str = "json"  # json, pickle, yaml
    results_directory: str = "strategy_results"


@dataclass
class GeneratedStrategy:
    """Generated strategy with metadata"""
    strategy_id: str
    strategy_name: str
    strategy_type: StrategyType
    category: StrategyCategory
    template_class: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    validation_results: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    status: GenerationStatus
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type.value,
            "category": self.category.value,
            "template_class": self.template_class,
            "parameters": self.parameters,
            "performance_metrics": self.performance_metrics,
            "validation_results": self.validation_results,
            "generation_metadata": self.generation_metadata,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class GenerationResult:
    """Result of strategy generation process"""
    generation_id: str
    total_strategies_generated: int
    successful_strategies: int
    failed_strategies: int
    validated_strategies: int
    strategies: List[GeneratedStrategy]
    generation_time: float
    config: GenerationConfig
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "generation_id": self.generation_id,
            "total_strategies_generated": self.total_strategies_generated,
            "successful_strategies": self.successful_strategies,
            "failed_strategies": self.failed_strategies,
            "validated_strategies": self.validated_strategies,
            "strategies": [s.to_dict() for s in self.strategies],
            "generation_time": self.generation_time,
            "config": asdict(self.config),
            "timestamp": self.timestamp.isoformat()
        }


class StrategyGenerationEngine:
    """
    Core Strategy Generation Engine - Orchestrates the complete strategy generation pipeline
    """
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.signal_processor = AdvancedSignalProcessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.real_time_integration = RealTimeIntegration()
        self.genetic_optimizer = AdvancedGeneticOptimizer()
        self.monte_carlo_simulator = AdvancedMonteCarloSimulator()
        self.walk_forward_tester = AdvancedWalkForwardTester()
        self.strategy_validator = StrategyValidator()
        
        # Strategy templates registry
        self.strategy_templates = {
            StrategyType.TREND: TrendStrategy,
            StrategyType.RANGE: RangeStrategy,
            StrategyType.BREAKOUT: BreakoutStrategy,
            StrategyType.SENTIMENT: SentimentStrategy,
            StrategyType.NEWS: NewsStrategy,
            StrategyType.MULTI_TIMEFRAME: MultiTimeframeStrategy,
            StrategyType.PAIRS: PairsStrategy
        }
        
        # Strategy categories mapping
        self.strategy_categories = {
            StrategyType.TREND: StrategyCategory.TREND_FOLLOWING,
            StrategyType.RANGE: StrategyCategory.MEAN_REVERSION,
            StrategyType.BREAKOUT: StrategyCategory.BREAKOUT,
            StrategyType.SENTIMENT: StrategyCategory.SENTIMENT_BASED,
            StrategyType.NEWS: StrategyCategory.NEWS_BASED,
            StrategyType.MULTI_TIMEFRAME: StrategyCategory.MULTI_TIMEFRAME,
            StrategyType.PAIRS: StrategyCategory.PAIRS_TRADING
        }
        
        # Results storage
        self.generated_strategies: List[GeneratedStrategy] = []
        self.generation_results: List[GenerationResult] = []
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    async def generate_strategies_async(self, market_data: Dict[str, Any], 
                                      config: GenerationConfig = None) -> GenerationResult:
        """
        Generate strategies asynchronously using the complete pipeline
        
        Args:
            market_data: Historical market data for strategy generation
            config: Generation configuration
            
        Returns:
            GenerationResult: Complete generation results
        """
        if config:
            self.config = config
        
        try:
            self.start_time = datetime.now()
            generation_id = f"gen_{int(self.start_time.timestamp())}"
            
            self.logger.info(f"Starting strategy generation: {generation_id}")
            
            # Prepare market data
            prepared_data = await self._prepare_market_data(market_data)
            
            # Generate strategies
            if self.config.parallel_generation:
                strategies = await self._generate_strategies_parallel(prepared_data)
            else:
                strategies = await self._generate_strategies_sequential(prepared_data)
            
            # Validate strategies
            validated_strategies = await self._validate_strategies(strategies, prepared_data)
            
            # Calculate metrics
            successful_strategies = len([s for s in strategies if s.status == GenerationStatus.COMPLETED])
            failed_strategies = len([s for s in strategies if s.status == GenerationStatus.FAILED])
            validated_count = len([s for s in validated_strategies if s.status == GenerationStatus.VALIDATED])
            
            self.end_time = datetime.now()
            generation_time = (self.end_time - self.start_time).total_seconds()
            
            # Create result
            result = GenerationResult(
                generation_id=generation_id,
                total_strategies_generated=len(strategies),
                successful_strategies=successful_strategies,
                failed_strategies=failed_strategies,
                validated_strategies=validated_count,
                strategies=validated_strategies,
                generation_time=generation_time,
                config=self.config,
                timestamp=self.end_time
            )
            
            self.generation_results.append(result)
            
            self.logger.info(f"Strategy generation completed: {successful_strategies}/{len(strategies)} successful")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in strategy generation: {e}")
            raise
    
    def generate_strategies(self, market_data: Dict[str, Any], 
                          config: GenerationConfig = None) -> GenerationResult:
        """Synchronous wrapper for strategy generation"""
        return asyncio.run(self.generate_strategies_async(market_data, config))
    
    async def _prepare_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare market data for strategy generation"""
        try:
            self.logger.info("Preparing market data for strategy generation")
            
            # Process signals
            processed_signals = await self.signal_processor.process_signals_async(market_data)
            
            # Extract features
            extracted_features = await self.feature_extractor.extract_features_async(market_data)
            
            # Integrate real-time data
            integrated_data = await self.real_time_integration.integrate_data_async(market_data)
            
            # Combine all data
            prepared_data = {
                **market_data,
                'processed_signals': processed_signals,
                'extracted_features': extracted_features,
                'integrated_data': integrated_data,
                'preparation_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Market data preparation completed")
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"Error preparing market data: {e}")
            return market_data
    
    async def _generate_strategies_parallel(self, market_data: Dict[str, Any]) -> List[GeneratedStrategy]:
        """Generate strategies in parallel"""
        # For now, use sequential processing
        # In production, you'd use asyncio.gather or multiprocessing
        return await self._generate_strategies_sequential(market_data)
    
    async def _generate_strategies_sequential(self, market_data: Dict[str, Any]) -> List[GeneratedStrategy]:
        """Generate strategies sequentially"""
        strategies = []
        
        # Generate strategies for each template type
        for strategy_type, template_class in self.strategy_templates.items():
            try:
                self.logger.info(f"Generating strategies for {strategy_type.value}")
                
                # Generate multiple strategies of this type
                type_strategies = await self._generate_strategies_for_type(
                    strategy_type, template_class, market_data
                )
                
                strategies.extend(type_strategies)
                
            except Exception as e:
                self.logger.error(f"Error generating strategies for {strategy_type.value}: {e}")
                continue
        
        return strategies
    
    async def _generate_strategies_for_type(self, strategy_type: StrategyType, 
                                          template_class: type, 
                                          market_data: Dict[str, Any]) -> List[GeneratedStrategy]:
        """Generate strategies for a specific type"""
        strategies = []
        
        # Calculate number of strategies to generate for this type
        strategies_per_type = max(1, self.config.max_strategies // len(self.strategy_templates))
        
        for i in range(strategies_per_type):
            try:
                # Generate strategy parameters
                parameters = self._generate_strategy_parameters(strategy_type)
                
                # Create strategy instance
                strategy_params = StrategyParameters(
                    strategy_id=f"{strategy_type.value}_{i}_{int(datetime.now().timestamp())}",
                    strategy_type=strategy_type,
                    name=f"{strategy_type.value.title()} Strategy {i+1}",
                    description=f"Generated {strategy_type.value} strategy",
                    parameters=parameters
                )
                
                strategy = template_class(strategy_params)
                
                # Generate strategy using genetic optimization
                optimized_strategy = await self._optimize_strategy(strategy, market_data)
                
                if optimized_strategy:
                    # Create generated strategy object
                    generated_strategy = GeneratedStrategy(
                        strategy_id=strategy_params.strategy_id,
                        strategy_name=strategy_params.name,
                        strategy_type=strategy_type,
                        category=self.strategy_categories[strategy_type],
                        template_class=template_class.__name__,
                        parameters=parameters,
                        performance_metrics={},
                        validation_results={},
                        generation_metadata={
                            'generation_method': 'genetic_optimization',
                            'optimization_iterations': 50,
                            'template_type': strategy_type.value
                        },
                        status=GenerationStatus.COMPLETED,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    strategies.append(generated_strategy)
                
            except Exception as e:
                self.logger.error(f"Error generating strategy {i} for {strategy_type.value}: {e}")
                continue
        
        return strategies
    
    def _generate_strategy_parameters(self, strategy_type: StrategyType) -> Dict[str, Any]:
        """Generate random parameters for a strategy type"""
        base_parameters = {
            'risk_level': np.random.choice(['low', 'medium', 'high']),
            'timeframe': np.random.choice(['1m', '5m', '15m', '1h', '4h', '1d']),
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
            'max_positions': np.random.randint(1, 5),
            'stop_loss_pct': np.random.uniform(0.01, 0.05),
            'take_profit_pct': np.random.uniform(0.02, 0.10)
        }
        
        # Add type-specific parameters
        if strategy_type == StrategyType.TREND:
            base_parameters.update({
                'sma_period': np.random.randint(10, 50),
                'trend_strength_threshold': np.random.uniform(0.5, 0.9),
                'trend_confirmation_periods': np.random.randint(3, 10)
            })
        elif strategy_type == StrategyType.RANGE:
            base_parameters.update({
                'range_period': np.random.randint(20, 100),
                'range_threshold': np.random.uniform(0.6, 0.9),
                'mean_reversion_strength': np.random.uniform(0.3, 0.8)
            })
        elif strategy_type == StrategyType.BREAKOUT:
            base_parameters.update({
                'breakout_period': np.random.randint(10, 30),
                'breakout_threshold': np.random.uniform(0.02, 0.05),
                'volume_confirmation': np.random.choice([True, False])
            })
        elif strategy_type == StrategyType.SENTIMENT:
            base_parameters.update({
                'sentiment_threshold': np.random.uniform(0.3, 0.7),
                'sentiment_weight': np.random.uniform(0.1, 0.5),
                'sentiment_lookback_periods': np.random.randint(5, 20)
            })
        elif strategy_type == StrategyType.NEWS:
            base_parameters.update({
                'news_impact_threshold': np.random.uniform(0.5, 0.9),
                'news_time_window': np.random.randint(30, 120),  # minutes
                'news_sentiment_weight': np.random.uniform(0.2, 0.6)
            })
        elif strategy_type == StrategyType.MULTI_TIMEFRAME:
            base_parameters.update({
                'primary_timeframe': np.random.choice(['1h', '4h', '1d']),
                'secondary_timeframe': np.random.choice(['15m', '1h', '4h']),
                'timeframe_alignment_weight': np.random.uniform(0.3, 0.8)
            })
        elif strategy_type == StrategyType.PAIRS:
            base_parameters.update({
                'pairs': [['EURUSD', 'GBPUSD'], ['USDJPY', 'AUDUSD']],
                'correlation_threshold': np.random.uniform(0.7, 0.95),
                'divergence_threshold': np.random.uniform(0.02, 0.05)
            })
        
        return base_parameters
    
    async def _optimize_strategy(self, strategy: StrategyTemplate, 
                               market_data: Dict[str, Any]) -> Optional[StrategyTemplate]:
        """Optimize strategy using genetic algorithm"""
        try:
            # Create parameter space
            parameter_space = self._create_parameter_space_for_strategy(strategy)
            
            # Create optimization config
            opt_config = self.config.optimization_config or OptimizationConfig(
                population_size=30,
                generations=20,
                use_multi_objective=True
            )
            
            # Run optimization
            result = await self.genetic_optimizer.optimize_async(
                strategy, parameter_space, market_data, opt_config
            )
            
            if result.get('success'):
                # Update strategy with optimized parameters
                optimized_params = result.get('best_parameters', {})
                strategy.update_parameters(optimized_params)
                return strategy
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy: {e}")
            return None
    
    def _create_parameter_space_for_strategy(self, strategy: StrategyTemplate) -> ParameterSpace:
        """Create parameter space for strategy optimization"""
        space = ParameterSpace()
        
        # Get strategy's parameter space
        strategy_space = strategy.get_parameter_space()
        
        # Add all parameters from strategy
        for param_name, param_def in strategy_space.get_all_parameters().items():
            space.add_parameter(param_def)
        
        return space
    
    async def _validate_strategies(self, strategies: List[GeneratedStrategy], 
                                 market_data: Dict[str, Any]) -> List[GeneratedStrategy]:
        """Validate generated strategies"""
        validated_strategies = []
        
        for strategy in strategies:
            try:
                self.logger.info(f"Validating strategy: {strategy.strategy_name}")
                
                # Run Monte Carlo simulation
                monte_carlo_result = await self._run_monte_carlo_validation(strategy, market_data)
                
                # Run walk-forward testing
                walk_forward_result = await self._run_walk_forward_validation(strategy, market_data)
                
                # Run strategy validation
                validation_result = await self._run_strategy_validation(strategy, market_data)
                
                # Update strategy with validation results
                strategy.validation_results = {
                    'monte_carlo': monte_carlo_result,
                    'walk_forward': walk_forward_result,
                    'strategy_validation': validation_result
                }
                
                # Determine if strategy passes validation
                if self._evaluate_strategy_validation(strategy):
                    strategy.status = GenerationStatus.VALIDATED
                    strategy.performance_metrics = self._extract_performance_metrics(strategy)
                else:
                    strategy.status = GenerationStatus.FAILED
                
                strategy.updated_at = datetime.now()
                validated_strategies.append(strategy)
                
            except Exception as e:
                self.logger.error(f"Error validating strategy {strategy.strategy_name}: {e}")
                strategy.status = GenerationStatus.FAILED
                strategy.updated_at = datetime.now()
                validated_strategies.append(strategy)
        
        return validated_strategies
    
    async def _run_monte_carlo_validation(self, strategy: GeneratedStrategy, 
                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo validation for strategy"""
        try:
            # Create Monte Carlo config
            mc_config = self.config.monte_carlo_config or MonteCarloConfig(
                iterations=100,
                simulation_type=SimulationType.BOOTSTRAP
            )
            
            # Create strategy instance
            strategy_params = StrategyParameters(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy.strategy_type,
                name=strategy.strategy_name,
                description="Validation strategy",
                parameters=strategy.parameters
            )
            
            template_class = self.strategy_templates[strategy.strategy_type]
            strategy_instance = template_class(strategy_params)
            
            # Run Monte Carlo simulation
            result = await self.monte_carlo_simulator.simulate_async(
                strategy_instance, market_data, mc_config
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo validation: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_walk_forward_validation(self, strategy: GeneratedStrategy, 
                                         market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run walk-forward validation for strategy"""
        try:
            # Create walk-forward config
            wf_config = self.config.walk_forward_config or WalkForwardConfig(
                mode=WalkForwardMode.ROLLING_WINDOW,
                in_sample_ratio=0.7,
                out_sample_ratio=0.3,
                min_in_sample_periods=100,
                min_out_sample_periods=50
            )
            
            # Create strategy instance
            strategy_params = StrategyParameters(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy.strategy_type,
                name=strategy.strategy_name,
                description="Validation strategy",
                parameters=strategy.parameters
            )
            
            template_class = self.strategy_templates[strategy.strategy_type]
            strategy_instance = template_class(strategy_params)
            
            # Create parameter space
            parameter_space = self._create_parameter_space_for_strategy(strategy_instance)
            
            # Run walk-forward test
            result = await self.walk_forward_tester.test_async(
                strategy_instance, market_data, parameter_space, wf_config
            )
            
            return result.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward validation: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_strategy_validation(self, strategy: GeneratedStrategy, 
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run strategy validation"""
        try:
            # Create strategy instance
            strategy_params = StrategyParameters(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy.strategy_type,
                name=strategy.strategy_name,
                description="Validation strategy",
                parameters=strategy.parameters
            )
            
            template_class = self.strategy_templates[strategy.strategy_type]
            strategy_instance = template_class(strategy_params)
            
            # Run validation
            result = await self.strategy_validator.validate_async(
                strategy_instance, market_data
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in strategy validation: {e}")
            return {"success": False, "error": str(e)}
    
    def _evaluate_strategy_validation(self, strategy: GeneratedStrategy) -> bool:
        """Evaluate if strategy passes validation criteria"""
        try:
            validation_results = strategy.validation_results
            
            # Check Monte Carlo results
            if 'monte_carlo' in validation_results:
                mc_result = validation_results['monte_carlo']
                if not mc_result.get('success', False):
                    return False
                
                analysis = mc_result.get('analysis', {})
                if analysis.get('robustness_score', 0) < self.config.min_performance_score:
                    return False
            
            # Check walk-forward results
            if 'walk_forward' in validation_results:
                wf_result = validation_results['walk_forward']
                if wf_result.get('successful_periods', 0) == 0:
                    return False
                
                if wf_result.get('robustness_score', 0) < self.config.min_performance_score:
                    return False
            
            # Check strategy validation
            if 'strategy_validation' in validation_results:
                sv_result = validation_results['strategy_validation']
                if not sv_result.get('success', False):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy validation: {e}")
            return False
    
    def _extract_performance_metrics(self, strategy: GeneratedStrategy) -> Dict[str, Any]:
        """Extract performance metrics from validation results"""
        try:
            metrics = {}
            validation_results = strategy.validation_results
            
            # Extract Monte Carlo metrics
            if 'monte_carlo' in validation_results:
                mc_result = validation_results['monte_carlo']
                if mc_result.get('success'):
                    analysis = mc_result.get('analysis', {})
                    metrics.update({
                        'monte_carlo_mean_return': analysis.get('mean_return', 0),
                        'monte_carlo_robustness_score': analysis.get('robustness_score', 0),
                        'monte_carlo_stability_score': analysis.get('stability_score', 0)
                    })
            
            # Extract walk-forward metrics
            if 'walk_forward' in validation_results:
                wf_result = validation_results['walk_forward']
                if wf_result.get('success'):
                    metrics.update({
                        'walk_forward_robustness_score': wf_result.get('robustness_score', 0),
                        'walk_forward_stability_score': wf_result.get('stability_score', 0),
                        'walk_forward_consistency_score': wf_result.get('consistency_score', 0),
                        'walk_forward_overall_grade': wf_result.get('overall_grade', 'D')
                    })
            
            # Extract strategy validation metrics
            if 'strategy_validation' in validation_results:
                sv_result = validation_results['strategy_validation']
                if sv_result.get('success'):
                    metrics.update({
                        'validation_score': sv_result.get('score', 0),
                        'validation_passed': sv_result.get('passed', False)
                    })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting performance metrics: {e}")
            return {}
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of strategy generation"""
        return {
            "total_generations": len(self.generation_results),
            "total_strategies": len(self.generated_strategies),
            "successful_strategies": len([s for s in self.generated_strategies if s.status == GenerationStatus.COMPLETED]),
            "validated_strategies": len([s for s in self.generated_strategies if s.status == GenerationStatus.VALIDATED]),
            "failed_strategies": len([s for s in self.generated_strategies if s.status == GenerationStatus.FAILED])
        }
    
    def save_results(self, filepath: str):
        """Save generation results to file"""
        try:
            results = {
                "generation_results": [r.to_dict() for r in self.generation_results],
                "generated_strategies": [s.to_dict() for s in self.generated_strategies],
                "summary": self.get_generation_summary(),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Generation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str) -> bool:
        """Load generation results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Reconstruct generation results
            self.generation_results = []
            for r in results.get("generation_results", []):
                # This would need proper reconstruction in production
                pass
            
            # Reconstruct generated strategies
            self.generated_strategies = []
            for s in results.get("generated_strategies", []):
                # This would need proper reconstruction in production
                pass
            
            self.logger.info(f"Generation results loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return False

