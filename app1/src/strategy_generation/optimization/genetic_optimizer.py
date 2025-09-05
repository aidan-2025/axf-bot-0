"""
Genetic algorithm optimizer for strategy parameters
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import random
import logging
from datetime import datetime
import json

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logging.warning("DEAP not available. Install with: pip install deap")

from ..core.strategy_template import StrategyTemplate
from ..core.parameter_space import ParameterSpace


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for strategy parameters using DEAP
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.population_size = self.config.get('population_size', 50)
        self.generations = self.config.get('generations', 100)
        self.crossover_prob = self.config.get('crossover_prob', 0.7)
        self.mutation_prob = self.config.get('mutation_prob', 0.3)
        self.tournament_size = self.config.get('tournament_size', 3)
        self.elite_size = self.config.get('elite_size', 5)
        
        # Results storage
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        if not DEAP_AVAILABLE:
            self.logger.error("DEAP library not available. Cannot perform genetic optimization.")
    
    def optimize(self, strategy: StrategyTemplate, parameter_space: ParameterSpace,
                market_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters using genetic algorithm
        
        Args:
            strategy: Strategy to optimize
            parameter_space: Parameter space definition
            market_data: Historical market data for evaluation
            config: Additional optimization configuration
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        if not DEAP_AVAILABLE:
            return {
                "success": False,
                "error": "DEAP library not available",
                "best_parameters": strategy.parameters.parameters,
                "best_fitness": 0.0
            }
        
        try:
            self.logger.info(f"Starting genetic optimization for strategy: {strategy}")
            
            # Update config with provided parameters
            if config:
                self.population_size = config.get('population_size', self.population_size)
                self.generations = config.get('generations', self.generations)
                self.crossover_prob = config.get('crossover_prob', self.crossover_prob)
                self.mutation_prob = config.get('mutation_prob', self.mutation_prob)
                self.tournament_size = config.get('tournament_size', self.tournament_size)
                self.elite_size = config.get('elite_size', self.elite_size)
            
            # Setup DEAP
            self._setup_deap()
            
            # Create toolbox
            toolbox = self._create_toolbox(strategy, parameter_space, market_data)
            
            # Initialize population
            population = toolbox.population(n=self.population_size)
            
            # Evaluate initial population
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Track best individual
            best_individual = tools.selBest(population, 1)[0]
            self.best_individual = best_individual
            self.best_fitness = best_individual.fitness.values[0]
            
            # Evolution loop
            for generation in range(self.generations):
                # Select parents
                parents = toolbox.select(population)
                
                # Clone parents for offspring
                offspring = list(map(toolbox.clone, parents))
                
                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.crossover_prob:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Apply mutation
                for mutant in offspring:
                    if random.random() < self.mutation_prob:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = list(map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Replace population
                population[:] = offspring
                
                # Track best individual
                current_best = tools.selBest(population, 1)[0]
                if current_best.fitness.values[0] > self.best_fitness:
                    self.best_individual = current_best
                    self.best_fitness = current_best.fitness.values[0]
                
                # Log progress
                if generation % 10 == 0:
                    self.logger.info(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}")
                
                # Store generation data
                self.optimization_history.append({
                    "generation": generation,
                    "best_fitness": self.best_fitness,
                    "avg_fitness": np.mean([ind.fitness.values[0] for ind in population]),
                    "std_fitness": np.std([ind.fitness.values[0] for ind in population])
                })
            
            # Decode best parameters
            best_parameters = parameter_space.decode_parameters(self.best_individual)
            
            # Prepare results
            results = {
                "success": True,
                "best_parameters": best_parameters,
                "best_fitness": self.best_fitness,
                "generations": self.generations,
                "population_size": self.population_size,
                "optimization_history": self.optimization_history,
                "convergence": self._calculate_convergence()
            }
            
            self.logger.info(f"Optimization completed. Best fitness: {self.best_fitness:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during genetic optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "best_parameters": strategy.parameters.parameters,
                "best_fitness": 0.0
            }
    
    def _setup_deap(self):
        """Setup DEAP creator and base classes"""
        if not DEAP_AVAILABLE:
            return
        
        # Create fitness class
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Create individual class
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    def _create_toolbox(self, strategy: StrategyTemplate, parameter_space: ParameterSpace,
                       market_data: Dict[str, Any]):
        """Create DEAP toolbox with operators"""
        if not DEAP_AVAILABLE:
            return None
        toolbox = base.Toolbox()
        
        # Register individual and population creators
        toolbox.register("attr_float", random.uniform, 0.0, 1.0)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attr_float, parameter_space.get_parameter_count())
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register evaluation function
        toolbox.register("evaluate", self._evaluate_individual, strategy, parameter_space, market_data)
        
        # Register genetic operators
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=15.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        return toolbox
    
    def _evaluate_individual(self, individual: List[float], strategy: StrategyTemplate,
                           parameter_space: ParameterSpace, market_data: Dict[str, Any]) -> Tuple[float]:
        """
        Evaluate an individual (parameter set) and return fitness
        
        Args:
            individual: Encoded parameter values
            strategy: Strategy template
            parameter_space: Parameter space definition
            market_data: Market data for evaluation
            
        Returns:
            Tuple[float]: Fitness value
        """
        try:
            # Decode parameters
            parameters = parameter_space.decode_parameters(individual)
            
            # Validate parameters
            is_valid, errors = parameter_space.validate_parameters(parameters)
            if not is_valid:
                return (0.0,)  # Invalid parameters get zero fitness
            
            # Update strategy with new parameters
            strategy_copy = self._create_strategy_copy(strategy, parameters)
            if not strategy_copy:
                return (0.0,)
            
            # Evaluate strategy performance
            fitness = self._calculate_fitness(strategy_copy, market_data)
            
            return (fitness,)
            
        except Exception as e:
            self.logger.warning(f"Error evaluating individual: {e}")
            return (0.0,)
    
    def _create_strategy_copy(self, original_strategy: StrategyTemplate, 
                            parameters: Dict[str, Any]) -> Optional[StrategyTemplate]:
        """Create a copy of strategy with new parameters"""
        try:
            # Create new parameters object
            new_parameters = original_strategy.parameters
            new_parameters.parameters.update(parameters)
            new_parameters.updated_at = datetime.now()
            
            # Create new strategy instance
            strategy_class = type(original_strategy)
            new_strategy = strategy_class(new_parameters)
            
            # Initialize with new parameters
            if new_strategy.initialize():
                return new_strategy
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Error creating strategy copy: {e}")
            return None
    
    def _calculate_fitness(self, strategy: StrategyTemplate, market_data: Dict[str, Any]) -> float:
        """
        Calculate fitness score for a strategy
        
        This is a simplified fitness function. In practice, this would involve:
        - Backtesting the strategy
        - Calculating performance metrics (Sharpe ratio, profit factor, etc.)
        - Penalizing for drawdown, overfitting, etc.
        """
        try:
            # Generate signals
            signals = strategy.generate_signals(market_data)
            
            if not signals:
                return 0.0
            
            # Simple fitness based on signal quality
            total_signals = len(signals)
            avg_strength = np.mean([s.strength for s in signals])
            avg_confidence = np.mean([s.confidence for s in signals])
            
            # Combine metrics (can be enhanced with actual backtesting)
            fitness = (avg_strength * 0.4 + avg_confidence * 0.4 + 
                     min(1.0, total_signals / 100) * 0.2)
            
            return max(0.0, min(1.0, fitness))
            
        except Exception as e:
            self.logger.warning(f"Error calculating fitness: {e}")
            return 0.0
    
    def _calculate_convergence(self) -> Dict[str, Any]:
        """Calculate convergence metrics"""
        if len(self.optimization_history) < 2:
            return {"converged": False, "convergence_rate": 0.0}
        
        # Calculate convergence rate
        fitness_values = [gen["best_fitness"] for gen in self.optimization_history]
        improvement = fitness_values[-1] - fitness_values[0]
        convergence_rate = improvement / max(1e-8, fitness_values[0])
        
        # Check if converged (no improvement in last 20% of generations)
        last_20_percent = int(len(fitness_values) * 0.8)
        recent_fitness = fitness_values[last_20_percent:]
        recent_improvement = max(recent_fitness) - min(recent_fitness)
        converged = recent_improvement < 0.01  # Less than 1% improvement
        
        return {
            "converged": converged,
            "convergence_rate": convergence_rate,
            "total_improvement": improvement,
            "final_fitness": fitness_values[-1]
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        return {
            "best_fitness": self.best_fitness,
            "generations_completed": len(self.optimization_history),
            "convergence": self._calculate_convergence(),
            "best_parameters": self.best_individual.tolist() if self.best_individual else None
        }
    
    def save_results(self, filepath: str):
        """Save optimization results to file"""
        try:
            results = {
                "optimization_history": self.optimization_history,
                "best_individual": self.best_individual.tolist() if self.best_individual else None,
                "best_fitness": self.best_fitness,
                "config": self.config,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str) -> bool:
        """Load optimization results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            self.optimization_history = results.get("optimization_history", [])
            self.best_individual = np.array(results.get("best_individual", []))
            self.best_fitness = results.get("best_fitness", 0.0)
            
            self.logger.info(f"Optimization results loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return False
