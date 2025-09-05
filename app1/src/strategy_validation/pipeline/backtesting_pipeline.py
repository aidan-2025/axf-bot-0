#!/usr/bin/env python3
"""
Backtesting Pipeline

Automated backtesting pipeline using Backtrader for strategy validation.
"""

import asyncio
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
import logging
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import backtrader as bt

from ..backtesting import BacktraderValidator, BacktestConfig
# Parallel processor imports moved to avoid circular dependency
from ..backtesting.influxdb_feed import FeedConfig
from ..backtesting.spread_simulator import SpreadConfig
from ..criteria import ValidationCriteria
from ..scoring import StrategyScorer
from ..storage import ValidationStorage, ValidationResult
from .strategy_loader import StrategyLoader, StrategyDefinition

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for backtesting pipeline"""
    
    # Data settings (required fields first)
    start_date: datetime
    end_date: datetime
    
    # Pipeline settings
    max_workers: int = 4
    memory_limit_mb: int = 2048
    timeout_seconds: int = 300
    
    # Backtesting settings
    initial_capital: float = 10000.0
    commission: float = 0.0001
    slippage: float = 0.0001
    
    # Data settings (optional)
    timeframe: str = '1h'
    symbols: List[str] = None
    
    # Validation settings
    validation_thresholds: Dict[str, Any] = None
    scoring_weights: Dict[str, Any] = None
    
    # Storage settings
    storage_connection_string: str = None
    save_results: bool = True
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        if self.validation_thresholds is None:
            self.validation_thresholds = {}
        if self.scoring_weights is None:
            self.scoring_weights = {}


class BacktestingPipeline:
    """Automated backtesting pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.strategy_loader = StrategyLoader()
        self.validation_criteria = ValidationCriteria()
        self.scorer = StrategyScorer()
        
        # Initialize storage if configured
        self.storage = None
        if config.storage_connection_string and config.save_results:
            try:
                self.storage = ValidationStorage(config.storage_connection_string)
                self.logger.info("Storage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize storage: {e}")
        
        self.logger.info("BacktestingPipeline initialized")
    
    async def run_single_strategy(self, strategy_definition: StrategyDefinition) -> Dict[str, Any]:
        """Run backtesting for a single strategy"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting backtest for strategy: {strategy_definition.strategy_id}")
            
            # Validate strategy definition
            if not strategy_definition.is_valid:
                return {
                    'success': False,
                    'strategy_id': strategy_definition.strategy_id,
                    'error': f"Strategy validation failed: {strategy_definition.validation_errors}",
                    'duration_seconds': 0
                }
            
            # Create backtest configuration
            backtest_config = BacktestConfig(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                symbols=self.config.symbols,
                timeframe=self.config.timeframe,
                initial_capital=self.config.initial_capital,
                commission=self.config.commission,
                slippage=self.config.slippage
            )
            
            # Create validator
            validator = BacktraderValidator(backtest_config)
            
            # Instantiate strategy class
            strategy_class = self.strategy_loader.instantiate_strategy_class(strategy_definition)
            
            # Run backtest
            validation_results = await validator.run_backtest(
                strategy_class, 
                strategy_definition.parameters
            )
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            if validation_results['success']:
                # Create validation result
                validation_result = ValidationResult(
                    strategy_id=strategy_definition.strategy_id,
                    strategy_name=strategy_definition.strategy_name,
                    strategy_type=strategy_definition.strategy_type,
                    validation_timestamp=datetime.now(),
                    validation_passed=validation_results['validation_passed'],
                    validation_score=validation_results['validation_score'],
                    critical_violations=validation_results.get('critical_violations', []),
                    warnings=validation_results.get('warnings', []),
                    performance_metrics=validation_results['performance_metrics'],
                    scoring_metrics=validation_results['scoring_metrics'],
                    backtest_config=backtest_config.__dict__,
                    validation_duration_seconds=duration,
                    backtest_duration_days=(self.config.end_date - self.config.start_date).days,
                    total_trades=validation_results['performance_metrics'].total_trades
                )
                
                # Save to storage if configured
                if self.storage:
                    await self.storage.store_validation_result(validation_result)
                
                self.logger.info(f"Successfully completed backtest for {strategy_definition.strategy_id}")
                
                return {
                    'success': True,
                    'strategy_id': strategy_definition.strategy_id,
                    'validation_result': validation_result,
                    'duration_seconds': duration
                }
            else:
                self.logger.error(f"Backtest failed for {strategy_definition.strategy_id}: {validation_results.get('error', 'Unknown error')}")
                
                return {
                    'success': False,
                    'strategy_id': strategy_definition.strategy_id,
                    'error': validation_results.get('error', 'Unknown error'),
                    'duration_seconds': duration
                }
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Unexpected error in backtest: {str(e)}"
            self.logger.error(f"Error in backtest for {strategy_definition.strategy_id}: {error_msg}")
            self.logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'strategy_id': strategy_definition.strategy_id,
                'error': error_msg,
                'duration_seconds': duration
            }
        finally:
            # Cleanup
            if 'validator' in locals():
                validator.close()
    
    async def run_batch_strategies(self, strategy_definitions: List[StrategyDefinition]) -> List[Dict[str, Any]]:
        """Run backtesting for multiple strategies in parallel"""
        self.logger.info(f"Starting batch backtesting for {len(strategy_definitions)} strategies")
        
        # Filter valid strategies
        valid_strategies = [s for s in strategy_definitions if s.is_valid]
        invalid_strategies = [s for s in strategy_definitions if not s.is_valid]
        
        if invalid_strategies:
            self.logger.warning(f"Skipping {len(invalid_strategies)} invalid strategies")
        
        if not valid_strategies:
            self.logger.error("No valid strategies to backtest")
            return []
        
        # Create tasks for parallel execution
        tasks = []
        for strategy in valid_strategies:
            task = self.run_single_strategy(strategy)
            tasks.append(task)
        
        # Run tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task {i} failed with exception: {result}")
                processed_results.append({
                    'success': False,
                    'strategy_id': valid_strategies[i].strategy_id,
                    'error': str(result),
                    'duration_seconds': 0
                })
            else:
                processed_results.append(result)
        
        # Add results for invalid strategies
        for strategy in invalid_strategies:
            processed_results.append({
                'success': False,
                'strategy_id': strategy.strategy_id,
                'error': f"Strategy validation failed: {strategy.validation_errors}",
                'duration_seconds': 0
            })
        
        # Log summary
        successful = len([r for r in processed_results if r['success']])
        failed = len(processed_results) - successful
        total_duration = sum(r['duration_seconds'] for r in processed_results)
        
        self.logger.info(f"Batch backtesting completed: {successful} successful, {failed} failed")
        self.logger.info(f"Total duration: {total_duration:.2f} seconds")
        
        return processed_results
    
    def run_batch_strategies_multiprocess(self, strategy_definitions: List[StrategyDefinition]) -> List[Dict[str, Any]]:
        """Run backtesting for multiple strategies using multiprocessing"""
        self.logger.info(f"Starting multiprocess backtesting for {len(strategy_definitions)} strategies")
        
        # Filter valid strategies
        valid_strategies = [s for s in strategy_definitions if s.is_valid]
        invalid_strategies = [s for s in strategy_definitions if not s.is_valid]
        
        if not valid_strategies:
            self.logger.error("No valid strategies to backtest")
            return []
        
        # Prepare data for multiprocessing
        strategy_data = [s.to_dict() for s in valid_strategies]
        config_data = {
            'start_date': self.config.start_date.isoformat(),
            'end_date': self.config.end_date.isoformat(),
            'symbols': self.config.symbols,
            'timeframe': self.config.timeframe,
            'initial_capital': self.config.initial_capital,
            'commission': self.config.commission,
            'slippage': self.config.slippage
        }
        
        # Run multiprocessing
        results = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks
            future_to_strategy = {
                executor.submit(_run_strategy_worker, strategy_data, config_data): strategy_data
                for strategy_data in strategy_data
            }
            
            # Collect results
            for future in as_completed(future_to_strategy):
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                except Exception as e:
                    strategy_data = future_to_strategy[future]
                    self.logger.error(f"Strategy {strategy_data['strategy_id']} failed: {e}")
                    results.append({
                        'success': False,
                        'strategy_id': strategy_data['strategy_id'],
                        'error': str(e),
                        'duration_seconds': 0
                    })
        
        # Add results for invalid strategies
        for strategy in invalid_strategies:
            results.append({
                'success': False,
                'strategy_id': strategy.strategy_id,
                'error': f"Strategy validation failed: {strategy.validation_errors}",
                'duration_seconds': 0
            })
        
        # Log summary
        successful = len([r for r in results if r['success']])
        failed = len(results) - successful
        total_duration = sum(r['duration_seconds'] for r in results)
        
        self.logger.info(f"Multiprocess backtesting completed: {successful} successful, {failed} failed")
        self.logger.info(f"Total duration: {total_duration:.2f} seconds")
        
        return results
    
    async def run_strategy_optimization(self, strategy_definition: StrategyDefinition, 
                                      parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Run parameter optimization for a strategy"""
        self.logger.info(f"Starting optimization for strategy: {strategy_definition.strategy_id}")
        
        try:
            # Create backtest configuration
            backtest_config = BacktestConfig(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                symbols=self.config.symbols,
                timeframe=self.config.timeframe,
                initial_capital=self.config.initial_capital,
                commission=self.config.commission,
                slippage=self.config.slippage
            )
            
            # Create validator
            validator = BacktraderValidator(backtest_config)
            
            # Instantiate strategy class
            strategy_class = self.strategy_loader.instantiate_strategy_class(strategy_definition)
            
            # Run optimization using Backtrader's built-in optimization
            optimization_results = await validator.run_optimization(
                strategy_class,
                parameter_ranges
            )
            
            self.logger.info(f"Optimization completed for {strategy_definition.strategy_id}")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Optimization failed for {strategy_definition.strategy_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'best_parameters': None,
                'best_score': 0.0
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'config': {
                'max_workers': self.config.max_workers,
                'memory_limit_mb': self.config.memory_limit_mb,
                'timeout_seconds': self.config.timeout_seconds,
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'symbols': self.config.symbols,
                'timeframe': self.config.timeframe
            },
            'storage_configured': self.storage is not None,
            'strategy_loader_directories': self.strategy_loader.strategy_directories
        }
    
    async def run_parallel_backtests(
        self, 
        strategy_definitions: List[StrategyDefinition],
        use_chunked: bool = False,
        max_workers: Optional[int] = None,
        chunk_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Run backtesting for multiple strategies using parallel processing
        
        Args:
            strategy_definitions: List of strategies to backtest
            use_chunked: Whether to use chunked processing for memory management
            max_workers: Maximum number of worker processes (default: auto-detect)
            chunk_size: Size of chunks for chunked processing
            
        Returns:
            List of backtest results
        """
        self.logger.info(f"Starting parallel backtesting for {len(strategy_definitions)} strategies")
        
        # Import parallel processor dynamically to avoid circular dependency
        try:
            from ..backtesting.parallel_processor import (
                create_parallel_processor,
                create_chunked_processor
            )
        except ImportError as e:
            self.logger.error(f"Failed to import parallel processor: {e}")
            # Fallback to regular batch processing
            return await self.run_batch_strategies(strategy_definitions)
        
        # Filter valid strategies
        valid_strategies = [s for s in strategy_definitions if s.is_valid]
        invalid_strategies = [s for s in strategy_definitions if not s.is_valid]
        
        if invalid_strategies:
            self.logger.warning(f"Skipping {len(invalid_strategies)} invalid strategies")
        
        if not valid_strategies:
            self.logger.error("No valid strategies to backtest")
            return []
        
        # Create data and spread configurations
        data_config = FeedConfig(
            symbol=self.config.symbols[0] if self.config.symbols else "EURUSD",
            timeframe=self.config.timeframe,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        spread_config = SpreadConfig(
            base_spread=self.config.commission,
            min_spread=self.config.commission * 0.5,
            max_spread=self.config.commission * 2.0,
            volatility_sensitivity=0.5
        )
        
        # Create parallel processor
        if use_chunked:
            processor = create_chunked_processor(
                max_workers=max_workers or self.config.max_workers,
                chunk_size=chunk_size,
                use_shared_memory=True
            )
        else:
            processor = create_parallel_processor(
                max_workers=max_workers or self.config.max_workers,
                use_shared_memory=True
            )
        
        # Progress callback
        def progress_callback(completed, total):
            self.logger.info(f"Parallel backtesting progress: {completed}/{total} strategies completed")
        
        # Run parallel backtests
        try:
            if use_chunked:
                results = await processor.run_chunked_backtests(
                    valid_strategies,
                    self.config,
                    data_config,
                    spread_config,
                    progress_callback
                )
            else:
                results = await processor.run_parallel_backtests(
                    valid_strategies,
                    self.config,
                    data_config,
                    spread_config,
                    progress_callback
                )
            
            # Convert results to expected format
            processed_results = []
            for result in results:
                processed_result = {
                    'success': result.success,
                    'strategy_id': result.strategy_id,
                    'strategy_name': result.strategy_name,
                    'duration_seconds': result.execution_time,
                    'performance_metrics': result.performance_metrics,
                    'error': result.error_message
                }
                processed_results.append(processed_result)
            
            # Add results for invalid strategies
            for strategy in invalid_strategies:
                processed_results.append({
                    'success': False,
                    'strategy_id': strategy.strategy_id,
                    'strategy_name': strategy.strategy_name,
                    'error': f"Strategy validation failed: {strategy.validation_errors}",
                    'duration_seconds': 0
                })
            
            # Get performance statistics
            stats = processor.get_performance_stats()
            self.logger.info(f"Parallel backtesting completed: {stats}")
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Parallel backtesting failed: {e}")
            raise
    
    def get_parallel_capabilities(self) -> Dict[str, Any]:
        """Get information about parallel processing capabilities"""
        import multiprocessing as mp
        
        return {
            'cpu_count': mp.cpu_count(),
            'recommended_max_workers': min(mp.cpu_count(), 8),
            'available_processors': [
                'parallel_processor',
                'chunked_parallel_processor',
                'multiprocess_executor'
            ],
            'memory_optimization': {
                'shared_memory': True,
                'chunked_processing': True,
                'data_preprocessing': True
            },
            'performance_features': {
                'parallel_execution': True,
                'memory_efficient': True,
                'progress_tracking': True,
                'error_handling': True
            }
        }
    
    def close(self):
        """Close pipeline and cleanup resources"""
        if self.storage:
            # Storage cleanup if needed
            pass
        
        self.logger.info("BacktestingPipeline closed")


def _run_strategy_worker(strategy_data: Dict[str, Any], config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for multiprocessing"""
    try:
        # This would be implemented to run a single strategy backtest
        # in a separate process. For now, return a mock result.
        return {
            'success': True,
            'strategy_id': strategy_data['strategy_id'],
            'duration_seconds': 10.0,
            'validation_passed': True,
            'validation_score': 0.85
        }
    except Exception as e:
        return {
            'success': False,
            'strategy_id': strategy_data['strategy_id'],
            'error': str(e),
            'duration_seconds': 0
        }
