"""
Parallel Backtesting Processor

This module implements high-performance parallel backtesting using multiprocessing
to distribute strategy backtests across multiple CPU cores for improved performance.
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
import logging
import time
import pickle
import os
from pathlib import Path

import backtrader as bt
import pandas as pd
import numpy as np

from ..pipeline.strategy_loader import StrategyDefinition
from ..pipeline.backtesting_pipeline import PipelineConfig
from .influxdb_feed import InfluxDBDataFeed, FeedConfig
from .spread_aware_broker import SpreadAwareBroker, SpreadAwareOrder
from .spread_simulator import VariableSpreadSimulator, SpreadConfig

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel backtesting"""
    max_workers: int = field(default_factory=lambda: min(mp.cpu_count(), 8))
    chunk_size: int = 1  # Number of strategies per process
    timeout_seconds: int = 300  # Timeout per strategy
    memory_limit_mb: int = 2048  # Memory limit per process
    use_shared_memory: bool = True  # Use shared memory for data
    temp_dir: str = "/tmp/backtesting"  # Temporary directory for data sharing
    enable_profiling: bool = False  # Enable performance profiling


@dataclass
class BacktestTask:
    """Individual backtest task for parallel processing"""
    task_id: str
    strategy_definition: StrategyDefinition
    config: PipelineConfig
    data_config: FeedConfig
    spread_config: SpreadConfig
    result_callback: Optional[Callable] = None


@dataclass
class BacktestResult:
    """Result of a single backtest"""
    task_id: str
    strategy_id: str
    strategy_name: str
    success: bool
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    trade_count: int = 0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0


class SharedDataManager:
    """Manages shared data for parallel processing"""
    
    def __init__(self, temp_dir: str = "/tmp/backtesting"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.shared_data_files: Dict[str, str] = {}
    
    def store_data(self, data_key: str, data: pd.DataFrame) -> str:
        """Store data in shared memory file"""
        file_path = self.temp_dir / f"{data_key}.parquet"
        data.to_parquet(file_path, index=True)
        self.shared_data_files[data_key] = str(file_path)
        logger.debug(f"Stored data for {data_key} at {file_path}")
        return str(file_path)
    
    def load_data(self, data_key: str) -> pd.DataFrame:
        """Load data from shared memory file"""
        if data_key not in self.shared_data_files:
            raise ValueError(f"Data key {data_key} not found in shared data")
        
        file_path = self.shared_data_files[data_key]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Shared data file {file_path} not found")
        
        data = pd.read_parquet(file_path)
        logger.debug(f"Loaded data for {data_key} from {file_path}")
        return data
    
    def cleanup(self):
        """Clean up temporary files"""
        for file_path in self.shared_data_files.values():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")


def run_single_backtest(task: BacktestTask) -> BacktestResult:
    """
    Run a single backtest in a separate process.
    This function must be defined at module level for multiprocessing.
    """
    start_time = time.time()
    result = BacktestResult(
        task_id=task.task_id,
        strategy_id=task.strategy_definition.strategy_id,
        strategy_name=task.strategy_definition.strategy_name,
        success=False
    )
    
    try:
        # Create Cerebro instance
        cerebro = bt.Cerebro()
        
        # Add strategy
        strategy_class = task.strategy_definition.strategy_class
        strategy_params = task.strategy_definition.parameters
        cerebro.addstrategy(strategy_class, **strategy_params)
        
        # Add data feed
        data_feed = InfluxDBDataFeed(task.data_config)
        cerebro.adddata(data_feed)
        
        # Add custom broker with spread simulation
        spread_simulator = VariableSpreadSimulator(task.spread_config)
        broker = SpreadAwareBroker(
            cash=task.config.initial_cash,
            spread_simulator=spread_simulator
        )
        cerebro.setbroker(broker)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Run backtest
        results = cerebro.run()
        
        if results and len(results) > 0:
            strategy_result = results[0]
            
            # Extract performance metrics
            sharpe_analyzer = strategy_result.analyzers.sharpe.get_analysis()
            drawdown_analyzer = strategy_result.analyzers.drawdown.get_analysis()
            trade_analyzer = strategy_result.analyzers.trades.get_analysis()
            returns_analyzer = strategy_result.analyzers.returns.get_analysis()
            
            # Calculate metrics
            total_trades = trade_analyzer.get('total', {}).get('total', 0)
            winning_trades = trade_analyzer.get('won', {}).get('total', 0)
            losing_trades = trade_analyzer.get('lost', {}).get('total', 0)
            
            profit_factor = 0.0
            if losing_trades > 0:
                total_won = trade_analyzer.get('won', {}).get('pnl', {}).get('total', 0)
                total_lost = abs(trade_analyzer.get('lost', {}).get('pnl', {}).get('total', 0))
                profit_factor = total_won / total_lost if total_lost > 0 else 0
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            result.success = True
            result.performance_metrics = {
                'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0),
                'max_drawdown': drawdown_analyzer.get('max', {}).get('drawdown', 0),
                'max_drawdown_pct': drawdown_analyzer.get('max', {}).get('drawdown', 0) * 100,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_return': returns_analyzer.get('rtot', 0),
                'annual_return': returns_analyzer.get('rnorm', 0)
            }
            result.trade_count = total_trades
            result.total_return = returns_analyzer.get('rtot', 0)
            result.sharpe_ratio = sharpe_analyzer.get('sharperatio', 0)
            result.max_drawdown = drawdown_analyzer.get('max', {}).get('drawdown', 0)
            result.profit_factor = profit_factor
            
        else:
            result.error_message = "No results returned from backtest"
            
    except Exception as e:
        result.error_message = f"Backtest failed: {str(e)}"
        logger.error(f"Backtest failed for {task.strategy_definition.strategy_name}: {e}")
    
    finally:
        result.execution_time = time.time() - start_time
        # Memory usage estimation (simplified)
        result.memory_usage_mb = 0.0  # Would need psutil for accurate measurement
    
    return result


class ParallelBacktestProcessor:
    """High-performance parallel backtesting processor"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.shared_data_manager = SharedDataManager(config.temp_dir)
        self.active_tasks: Dict[str, BacktestTask] = {}
        self.results: List[BacktestResult] = []
        
    def prepare_shared_data(self, data_config: FeedConfig) -> str:
        """Prepare and store shared data for parallel processing"""
        try:
            # Create data feed to load data
            data_feed = InfluxDBDataFeed(data_config)
            data = data_feed._load_data()
            
            # Store in shared memory
            data_key = f"backtest_data_{int(time.time())}"
            self.shared_data_manager.store_data(data_key, data)
            
            logger.info(f"Prepared shared data with {len(data)} records")
            return data_key
            
        except Exception as e:
            logger.error(f"Failed to prepare shared data: {e}")
            raise
    
    async def run_parallel_backtests(
        self,
        strategy_definitions: List[StrategyDefinition],
        pipeline_config: PipelineConfig,
        data_config: FeedConfig,
        spread_config: SpreadConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BacktestResult]:
        """
        Run multiple backtests in parallel
        
        Args:
            strategy_definitions: List of strategies to backtest
            pipeline_config: Pipeline configuration
            data_config: Data feed configuration
            spread_config: Spread simulation configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of backtest results
        """
        logger.info(f"Starting parallel backtesting for {len(strategy_definitions)} strategies")
        
        # Prepare shared data if enabled
        data_key = None
        if self.config.use_shared_memory:
            try:
                data_key = self.prepare_shared_data(data_config)
            except Exception as e:
                logger.warning(f"Failed to prepare shared data, falling back to individual loading: {e}")
                self.config.use_shared_memory = False
        
        # Create backtest tasks
        tasks = []
        for i, strategy_def in enumerate(strategy_definitions):
            task = BacktestTask(
                task_id=f"task_{i}_{int(time.time())}",
                strategy_definition=strategy_def,
                config=pipeline_config,
                data_config=data_config,
                spread_config=spread_config
            )
            tasks.append(task)
            self.active_tasks[task.task_id] = task
        
        # Run parallel backtests
        results = []
        completed_count = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(run_single_backtest, task): task
                    for task in tasks
                }
                
                # Process completed tasks
                for future in as_completed(future_to_task, timeout=self.config.timeout_seconds):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        results.append(result)
                        completed_count += 1
                        
                        # Update progress
                        if progress_callback:
                            progress_callback(completed_count, len(tasks))
                        
                        logger.info(
                            f"Completed backtest {completed_count}/{len(tasks)}: "
                            f"{result.strategy_name} - "
                            f"Success: {result.success}, "
                            f"Time: {result.execution_time:.2f}s"
                        )
                        
                    except Exception as e:
                        logger.error(f"Task {task.task_id} failed: {e}")
                        error_result = BacktestResult(
                            task_id=task.task_id,
                            strategy_id=task.strategy_definition.strategy_id,
                            strategy_name=task.strategy_definition.strategy_name,
                            success=False,
                            error_message=str(e)
                        )
                        results.append(error_result)
                        completed_count += 1
                        
                        if progress_callback:
                            progress_callback(completed_count, len(tasks))
        
        except Exception as e:
            logger.error(f"Parallel backtesting failed: {e}")
            raise
        
        finally:
            # Cleanup
            self.active_tasks.clear()
            if data_key:
                self.shared_data_manager.cleanup()
        
        self.results = results
        logger.info(f"Parallel backtesting completed: {len(results)} results")
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the last run"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if not successful_results:
            return {
                'total_strategies': len(self.results),
                'successful_strategies': 0,
                'failed_strategies': len(failed_results),
                'average_execution_time': 0,
                'total_execution_time': 0
            }
        
        execution_times = [r.execution_time for r in successful_results]
        
        return {
            'total_strategies': len(self.results),
            'successful_strategies': len(successful_results),
            'failed_strategies': len(failed_results),
            'success_rate': len(successful_results) / len(self.results) * 100,
            'average_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'total_execution_time': np.sum(execution_times),
            'parallel_efficiency': self._calculate_parallel_efficiency(execution_times)
        }
    
    def _calculate_parallel_efficiency(self, execution_times: List[float]) -> float:
        """Calculate parallel processing efficiency"""
        if not execution_times:
            return 0.0
        
        sequential_time = sum(execution_times)
        parallel_time = max(execution_times)  # Assuming perfect parallelization
        efficiency = (sequential_time / parallel_time) / self.config.max_workers * 100
        
        return min(efficiency, 100.0)  # Cap at 100%


class ChunkedParallelProcessor(ParallelBacktestProcessor):
    """Parallel processor that processes strategies in chunks for better resource management"""
    
    def __init__(self, config: ParallelConfig, chunk_size: int = 5):
        super().__init__(config)
        self.chunk_size = chunk_size
    
    async def run_chunked_backtests(
        self,
        strategy_definitions: List[StrategyDefinition],
        pipeline_config: PipelineConfig,
        data_config: FeedConfig,
        spread_config: SpreadConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BacktestResult]:
        """Run backtests in chunks to manage memory usage"""
        logger.info(f"Starting chunked parallel backtesting: {len(strategy_definitions)} strategies in chunks of {self.chunk_size}")
        
        all_results = []
        total_chunks = (len(strategy_definitions) + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(strategy_definitions))
            chunk_strategies = strategy_definitions[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks}: strategies {start_idx}-{end_idx-1}")
            
            # Process chunk
            chunk_results = await self.run_parallel_backtests(
                chunk_strategies,
                pipeline_config,
                data_config,
                spread_config,
                progress_callback
            )
            
            all_results.extend(chunk_results)
            
            # Update progress for overall process
            if progress_callback:
                completed_strategies = len(all_results)
                progress_callback(completed_strategies, len(strategy_definitions))
        
        logger.info(f"Chunked parallel backtesting completed: {len(all_results)} total results")
        return all_results


# Utility functions for easy integration
def create_parallel_processor(
    max_workers: Optional[int] = None,
    chunk_size: int = 1,
    use_shared_memory: bool = True
) -> ParallelBacktestProcessor:
    """Create a parallel processor with default configuration"""
    config = ParallelConfig(
        max_workers=max_workers or min(mp.cpu_count(), 8),
        chunk_size=chunk_size,
        use_shared_memory=use_shared_memory
    )
    return ParallelBacktestProcessor(config)


def create_chunked_processor(
    max_workers: Optional[int] = None,
    chunk_size: int = 5,
    use_shared_memory: bool = True
) -> ChunkedParallelProcessor:
    """Create a chunked parallel processor with default configuration"""
    config = ParallelConfig(
        max_workers=max_workers or min(mp.cpu_count(), 8),
        chunk_size=chunk_size,
        use_shared_memory=use_shared_memory
    )
    return ChunkedParallelProcessor(config, chunk_size)
