"""
Test script for parallel backtesting processor

This script tests the parallel processing capabilities for backtesting
multiple strategies simultaneously.
"""

import asyncio
import time
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.strategy_validation.backtesting.parallel_processor import (
    ParallelBacktestProcessor,
    ChunkedParallelProcessor,
    ParallelConfig,
    BacktestTask,
    BacktestResult,
    create_parallel_processor,
    create_chunked_processor
)
from src.strategy_validation.pipeline.strategy_loader import StrategyDefinition
from src.strategy_validation.pipeline.backtesting_pipeline import PipelineConfig
from src.strategy_validation.backtesting.influxdb_feed import FeedConfig
from src.strategy_validation.backtesting.spread_simulator import SpreadConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockStrategy:
    """Mock strategy for testing"""
    
    def __init__(self, **params):
        self.params = params
        self.name = params.get('name', 'MockStrategy')
    
    def next(self):
        """Mock next method"""
        pass


def create_mock_strategy_definitions(count: int = 10) -> list:
    """Create mock strategy definitions for testing"""
    strategies = []
    
    for i in range(count):
        strategy_def = StrategyDefinition(
            strategy_id=f"mock_strategy_{i}",
            strategy_name=f"Mock Strategy {i}",
            strategy_type="mock",
            description=f"Mock strategy for testing {i}",
            class_name="MockStrategy",
            module_path="test_module",
            parameters={
                'name': f'Mock Strategy {i}',
                'param1': 10 + i,
                'param2': 20 + i * 2
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        strategies.append(strategy_def)
    
    return strategies


def create_test_configs():
    """Create test configurations"""
    pipeline_config = PipelineConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        initial_capital=100000,
        max_workers=4,
        timeout_seconds=60
    )
    
    data_config = FeedConfig(
        symbol="EURUSD",
        timeframe="1h",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )
    
    spread_config = SpreadConfig(
        base_spread=0.0002,
        min_spread=0.0001,
        max_spread=0.0005,
        volatility_sensitivity=0.5
    )
    
    return pipeline_config, data_config, spread_config


async def test_parallel_processor():
    """Test the parallel processor"""
    logger.info("Testing Parallel Backtest Processor")
    
    # Create test data
    strategies = create_mock_strategy_definitions(8)
    pipeline_config, data_config, spread_config = create_test_configs()
    
    # Create parallel processor
    parallel_config = ParallelConfig(
        max_workers=4,
        chunk_size=1,
        use_shared_memory=False,  # Disable for testing
        timeout_seconds=30
    )
    processor = ParallelBacktestProcessor(parallel_config)
    
    # Progress callback
    def progress_callback(completed, total):
        logger.info(f"Progress: {completed}/{total} strategies completed")
    
    # Run parallel backtests
    start_time = time.time()
    results = await processor.run_parallel_backtests(
        strategies,
        pipeline_config,
        data_config,
        spread_config,
        progress_callback
    )
    execution_time = time.time() - start_time
    
    # Analyze results
    logger.info(f"Parallel processing completed in {execution_time:.2f} seconds")
    logger.info(f"Processed {len(results)} strategies")
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        for result in failed:
            logger.warning(f"Failed strategy {result.strategy_name}: {result.error_message}")
    
    # Get performance stats
    stats = processor.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
    
    return results


async def test_chunked_processor():
    """Test the chunked parallel processor"""
    logger.info("Testing Chunked Parallel Backtest Processor")
    
    # Create test data
    strategies = create_mock_strategy_definitions(12)
    pipeline_config, data_config, spread_config = create_test_configs()
    
    # Create chunked processor
    processor = create_chunked_processor(
        max_workers=3,
        chunk_size=4,
        use_shared_memory=False
    )
    
    # Progress callback
    def progress_callback(completed, total):
        logger.info(f"Chunked progress: {completed}/{total} strategies completed")
    
    # Run chunked backtests
    start_time = time.time()
    results = await processor.run_chunked_backtests(
        strategies,
        pipeline_config,
        data_config,
        spread_config,
        progress_callback
    )
    execution_time = time.time() - start_time
    
    # Analyze results
    logger.info(f"Chunked processing completed in {execution_time:.2f} seconds")
    logger.info(f"Processed {len(results)} strategies")
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    # Get performance stats
    stats = processor.get_performance_stats()
    logger.info(f"Chunked performance stats: {stats}")
    
    return results


def test_shared_data_manager():
    """Test the shared data manager"""
    logger.info("Testing Shared Data Manager")
    
    from src.strategy_validation.backtesting.parallel_processor import SharedDataManager
    import pandas as pd
    import numpy as np
    
    # Create test data
    data = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=100, freq='1h'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    data.set_index('datetime', inplace=True)
    
    # Test shared data manager
    manager = SharedDataManager(temp_dir="/tmp/test_backtesting")
    
    try:
        # Store data
        data_key = "test_data"
        file_path = manager.store_data(data_key, data)
        logger.info(f"Stored data at: {file_path}")
        
        # Load data
        loaded_data = manager.load_data(data_key)
        logger.info(f"Loaded data shape: {loaded_data.shape}")
        
        # Verify data integrity
        assert len(loaded_data) == len(data), "Data length mismatch"
        assert list(loaded_data.columns) == list(data.columns), "Column mismatch"
        
        logger.info("Shared data manager test passed")
        
    finally:
        # Cleanup
        manager.cleanup()
        logger.info("Cleaned up shared data")


def test_backtest_task():
    """Test the BacktestTask dataclass"""
    logger.info("Testing BacktestTask")
    
    # Create test task
    strategy_def = create_mock_strategy_definitions(1)[0]
    pipeline_config, data_config, spread_config = create_test_configs()
    
    task = BacktestTask(
        task_id="test_task_1",
        strategy_definition=strategy_def,
        config=pipeline_config,
        data_config=data_config,
        spread_config=spread_config
    )
    
    # Verify task properties
    assert task.task_id == "test_task_1"
    assert task.strategy_definition.strategy_id == "mock_strategy_0"
    assert task.strategy_definition.strategy_name == "Mock Strategy 0"
    
    logger.info("BacktestTask test passed")


def test_performance_comparison():
    """Compare performance between parallel and sequential processing"""
    logger.info("Testing Performance Comparison")
    
    # This would require actual backtesting implementation
    # For now, just test the structure
    strategies = create_mock_strategy_definitions(6)
    pipeline_config, data_config, spread_config = create_test_configs()
    
    # Test parallel processor creation
    parallel_processor = create_parallel_processor(max_workers=2)
    chunked_processor = create_chunked_processor(max_workers=2, chunk_size=3)
    
    assert parallel_processor.config.max_workers == 2
    assert chunked_processor.config.max_workers == 2
    assert chunked_processor.chunk_size == 3
    
    logger.info("Performance comparison test structure verified")


async def run_all_tests():
    """Run all tests"""
    logger.info("Starting Parallel Processor Tests")
    
    try:
        # Test individual components
        test_shared_data_manager()
        test_backtest_task()
        test_performance_comparison()
        
        # Test parallel processing
        logger.info("\n" + "="*50)
        logger.info("Testing Parallel Processing")
        logger.info("="*50)
        parallel_results = await test_parallel_processor()
        
        # Test chunked processing
        logger.info("\n" + "="*50)
        logger.info("Testing Chunked Processing")
        logger.info("="*50)
        chunked_results = await test_chunked_processor()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("Test Summary")
        logger.info("="*50)
        logger.info(f"Parallel processing: {len(parallel_results)} results")
        logger.info(f"Chunked processing: {len(chunked_results)} results")
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_all_tests())
