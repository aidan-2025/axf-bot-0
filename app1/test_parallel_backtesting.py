#!/usr/bin/env python3
"""
Test Suite for Parallel Backtesting Framework

Comprehensive tests for parallel backtesting functionality including
performance benchmarking, resource management, and result aggregation.
"""

import unittest
import time
import tempfile
import os
import shutil
import logging
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp

# Import the modules to test
import sys
sys.path.append('app1/src')

from backtesting.parallel.parallel_backtester import (
    ParallelBacktester, ParallelConfig, BacktestConfig, BacktestResult,
    ParallelBacktestManager
)
from backtesting.parallel.parallel_manager import (
    PerformanceBenchmark, ResourceManager, ParallelBacktestManager as AdvancedManager
)


class TestParallelConfig(unittest.TestCase):
    """Test ParallelConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ParallelConfig()
        
        self.assertIsNotNone(config.max_workers)
        self.assertLessEqual(config.max_workers, mp.cpu_count())
        self.assertEqual(config.chunk_size, 1)
        self.assertEqual(config.timeout, 300.0)
        self.assertEqual(config.memory_limit, 1024)
        self.assertTrue(config.result_aggregation)
        self.assertTrue(config.save_intermediate_results)
        self.assertTrue(config.cleanup_temp_files)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ParallelConfig(
            max_workers=4,
            chunk_size=2,
            timeout=600.0,
            memory_limit=2048,
            result_aggregation=False,
            save_intermediate_results=False,
            cleanup_temp_files=False
        )
        
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.chunk_size, 2)
        self.assertEqual(config.timeout, 600.0)
        self.assertEqual(config.memory_limit, 2048)
        self.assertFalse(config.result_aggregation)
        self.assertFalse(config.save_intermediate_results)
        self.assertFalse(config.cleanup_temp_files)


class TestBacktestConfig(unittest.TestCase):
    """Test BacktestConfig dataclass"""
    
    def test_backtest_config_creation(self):
        """Test creating a backtest configuration"""
        config = BacktestConfig(
            strategy_class="TestStrategy",
            strategy_params={"param1": 1.0, "param2": "test"},
            data_config={"symbol": "EURUSD", "source": "test"},
            execution_config={"slippage": 0.001},
            sizing_config={"method": "fixed", "size": 1000},
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        self.assertEqual(config.strategy_class, "TestStrategy")
        self.assertEqual(config.strategy_params["param1"], 1.0)
        self.assertEqual(config.data_config["symbol"], "EURUSD")
        self.assertEqual(config.start_date, "2023-01-01")
        self.assertEqual(config.end_date, "2023-12-31")
        self.assertIsNotNone(config.run_id)
    
    def test_auto_generated_run_id(self):
        """Test automatic run ID generation"""
        # Add small delay to ensure different timestamps
        config1 = BacktestConfig(
            strategy_class="TestStrategy",
            strategy_params={},
            data_config={},
            execution_config={},
            sizing_config={},
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        time.sleep(0.001)  # Small delay to ensure different timestamps
        
        config2 = BacktestConfig(
            strategy_class="TestStrategy",
            strategy_params={},
            data_config={},
            execution_config={},
            sizing_config={},
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        self.assertNotEqual(config1.run_id, config2.run_id)
        self.assertTrue(config1.run_id.startswith("backtest_"))


class TestBacktestResult(unittest.TestCase):
    """Test BacktestResult dataclass"""
    
    def test_backtest_result_creation(self):
        """Test creating a backtest result"""
        result = BacktestResult(
            run_id="test_001",
            success=True,
            execution_time=1.5,
            results={"total_return": 0.1},
            performance_metrics={"sharpe_ratio": 1.2},
            execution_statistics={"total_orders": 100}
        )
        
        self.assertEqual(result.run_id, "test_001")
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 1.5)
        self.assertEqual(result.results["total_return"], 0.1)
        self.assertIsNone(result.error)
    
    def test_failed_backtest_result(self):
        """Test creating a failed backtest result"""
        result = BacktestResult(
            run_id="test_002",
            success=False,
            execution_time=0.5,
            error="Test error message"
        )
        
        self.assertEqual(result.run_id, "test_002")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Test error message")
        self.assertIsNone(result.results)
    
    def test_to_dict(self):
        """Test converting result to dictionary"""
        result = BacktestResult(
            run_id="test_003",
            success=True,
            execution_time=2.0,
            results={"total_return": 0.15}
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["run_id"], "test_003")
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["execution_time"], 2.0)


class TestParallelBacktester(unittest.TestCase):
    """Test ParallelBacktester class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ParallelConfig(max_workers=2, timeout=10.0)
        self.backtester = ParallelBacktester(self.config)
        
        # Create mock backtest configs
        self.mock_configs = [
            BacktestConfig(
                strategy_class="MockStrategy",
                strategy_params={"param1": 1.0},
                data_config={"symbol": "EURUSD"},
                execution_config={"slippage": 0.001},
                sizing_config={"method": "fixed"},
                start_date="2023-01-01",
                end_date="2023-01-02"
            ) for _ in range(3)
        ]
    
    def test_initialization(self):
        """Test parallel backtester initialization"""
        self.assertEqual(self.backtester.config.max_workers, 2)
        self.assertEqual(self.backtester.total_runs, 0)
        self.assertEqual(self.backtester.completed_runs, 0)
        self.assertEqual(self.backtester.failed_runs, 0)
        self.assertEqual(len(self.backtester.results), 0)
    
    def test_run_parallel_backtests_success(self):
        """Test successful parallel backtest execution"""
        # Mock the _execute_single_backtest method directly on the instance
        def mock_execute(config):
            return BacktestResult(
                run_id=config.run_id,
                success=True,
                execution_time=1.0,
                results={"total_return": 0.1}
            )
        
        # Patch the static method
        with patch.object(ParallelBacktester, '_execute_single_backtest', side_effect=mock_execute):
            results = self.backtester.run_parallel_backtests(self.mock_configs)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(self.backtester.completed_runs, 3)
        self.assertEqual(self.backtester.failed_runs, 0)
        self.assertTrue(all(r.success for r in results))
    
    def test_run_parallel_backtests_failure(self):
        """Test parallel backtest execution with failures"""
        # Mock mixed success/failure
        def mock_execute_side_effect(config):
            if "backtest_0" in config.run_id:
                return BacktestResult(
                    run_id=config.run_id,
                    success=True,
                    execution_time=1.0
                )
            else:
                return BacktestResult(
                    run_id=config.run_id,
                    success=False,
                    execution_time=0.5,
                    error="Test error"
                )
        
        # Patch the static method
        with patch.object(ParallelBacktester, '_execute_single_backtest', side_effect=mock_execute_side_effect):
            results = self.backtester.run_parallel_backtests(self.mock_configs)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(self.backtester.completed_runs, 1)
        self.assertEqual(self.backtester.failed_runs, 2)
        self.assertEqual(sum(1 for r in results if r.success), 1)
    
    def test_aggregate_results_empty(self):
        """Test result aggregation with no results"""
        aggregated = self.backtester._aggregate_results()
        self.assertEqual(aggregated, {})
    
    def test_aggregate_results_with_data(self):
        """Test result aggregation with data"""
        # Add mock results
        self.backtester.results = [
            BacktestResult(
                run_id="test_001",
                success=True,
                execution_time=1.0,
                performance_metrics={"total_return": 0.1, "sharpe_ratio": 1.2},
                execution_statistics={"total_orders": 100, "avg_slippage": 0.001}
            ),
            BacktestResult(
                run_id="test_002",
                success=True,
                execution_time=2.0,
                performance_metrics={"total_return": 0.15, "sharpe_ratio": 1.5},
                execution_statistics={"total_orders": 150, "avg_slippage": 0.002}
            ),
            BacktestResult(
                run_id="test_003",
                success=False,
                execution_time=0.5,
                error="Test error"
            )
        ]
        
        aggregated = self.backtester._aggregate_results()
        
        self.assertEqual(aggregated["total_runs"], 3)
        self.assertEqual(aggregated["successful_runs"], 2)
        self.assertEqual(aggregated["failed_runs"], 1)
        self.assertEqual(aggregated["success_rate"], 2/3)
        self.assertEqual(aggregated["total_execution_time"], 3.5)
        self.assertEqual(aggregated["average_execution_time"], 3.5/3)
        
        # Check performance metrics aggregation
        self.assertIn("avg_total_return", aggregated["performance_metrics"])
        self.assertIn("avg_sharpe_ratio", aggregated["performance_metrics"])
        
        # Check execution statistics aggregation
        self.assertIn("total_total_orders", aggregated["execution_statistics"])
        self.assertIn("avg_avg_slippage", aggregated["execution_statistics"])
    
    def test_calculate_parallel_efficiency(self):
        """Test parallel efficiency calculation"""
        # Set up mock timing data
        self.backtester.start_time = 0.0
        self.backtester.end_time = 10.0
        self.backtester.results = [
            BacktestResult(run_id=f"test_{i}", success=True, execution_time=5.0)
            for i in range(4)
        ]
        
        efficiency = self.backtester._calculate_parallel_efficiency()
        
        # With 4 results taking 5s each (20s total CPU time) and 10s wall time with 2 workers
        # Efficiency = 20 / (10 * 2) = 1.0
        self.assertEqual(efficiency, 1.0)
    
    def test_get_performance_summary(self):
        """Test performance summary generation"""
        # Set up mock data
        self.backtester.total_runs = 5
        self.backtester.completed_runs = 4
        self.backtester.failed_runs = 1
        self.backtester.start_time = 0.0
        self.backtester.end_time = 10.0
        self.backtester.results = [
            BacktestResult(run_id=f"test_{i}", success=True, execution_time=2.0)
            for i in range(4)
        ]
        self.backtester.results.append(
            BacktestResult(run_id="test_4", success=False, execution_time=1.0)
        )
        
        summary = self.backtester.get_performance_summary()
        
        self.assertEqual(summary["total_runs"], 5)
        self.assertEqual(summary["completed_runs"], 4)
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual(summary["success_rate"], 0.8)
        self.assertEqual(summary["total_execution_time"], 9.0)
        self.assertEqual(summary["wall_time"], 10.0)
        self.assertIn("parallel_efficiency", summary)
        self.assertIn("speedup_factor", summary)


class TestPerformanceBenchmark(unittest.TestCase):
    """Test PerformanceBenchmark class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.benchmark = PerformanceBenchmark()
        
        # Create mock configs
        self.mock_configs = [
            BacktestConfig(
                strategy_class="MockStrategy",
                strategy_params={},
                data_config={},
                execution_config={},
                sizing_config={},
                start_date="2023-01-01",
                end_date="2023-01-02"
            ) for _ in range(2)
        ]
    
    @patch('backtesting.parallel.parallel_manager.ParallelBacktester')
    @patch('backtesting.parallel.parallel_manager.HighFidelityExecutionIntegration')
    def test_benchmark_parallel_vs_sequential(self, mock_integration, mock_backtester):
        """Test parallel vs sequential benchmarking"""
        # Mock parallel execution
        mock_backtester_instance = Mock()
        mock_backtester_instance.run_parallel_backtests.return_value = []
        mock_backtester.return_value = mock_backtester_instance
        
        # Mock sequential execution
        mock_integration_instance = Mock()
        mock_integration_instance.setup_cerebro.return_value = Mock()
        mock_integration_instance.run_backtest.return_value = {}
        mock_integration.return_value = mock_integration_instance
        
        # Mock time to control execution times
        with patch('time.time', side_effect=[0, 5, 10, 15]):  # 5s parallel, 10s sequential
            results = self.benchmark.benchmark_parallel_vs_sequential(
                self.mock_configs, 
                ParallelConfig(max_workers=2),
                num_runs=1
            )
        
        self.assertIn("parallel_times", results)
        self.assertIn("sequential_times", results)
        self.assertIn("speedup_factor", results)
        self.assertIn("efficiency", results)
        self.assertEqual(results["num_backtests"], 2)
        self.assertEqual(results["num_workers"], 2)


class TestResourceManager(unittest.TestCase):
    """Test ResourceManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.resource_manager = ResourceManager(max_memory_mb=1000, max_cpu_percent=80.0)
    
    def test_initialization(self):
        """Test resource manager initialization"""
        self.assertEqual(self.resource_manager.max_memory_mb, 1000)
        self.assertEqual(self.resource_manager.max_cpu_percent, 80.0)
        self.assertIsInstance(self.resource_manager.resource_data.cpu_usage, list)
        self.assertIsInstance(self.resource_manager.resource_data.memory_usage, list)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_monitor_resources(self, mock_memory, mock_cpu):
        """Test resource monitoring"""
        # Mock resource data
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(used=500 * 1024 * 1024)  # 500MB
        
        # Start monitoring briefly
        self.resource_manager.start_monitoring()
        time.sleep(0.1)  # Brief monitoring
        self.resource_manager.stop_monitoring()
        
        # Check that data was collected
        self.assertGreater(len(self.resource_manager.resource_data.cpu_usage), 0)
        self.assertGreater(len(self.resource_manager.resource_data.memory_usage), 0)
    
    def test_get_resource_summary_empty(self):
        """Test resource summary with no data"""
        summary = self.resource_manager.get_resource_summary()
        self.assertEqual(summary, {})
    
    def test_get_resource_summary_with_data(self):
        """Test resource summary with data"""
        # Add mock data
        self.resource_manager.resource_data.cpu_usage = [50.0, 60.0, 70.0]
        self.resource_manager.resource_data.memory_usage = [500.0, 600.0, 700.0]
        self.resource_manager.resource_data.start_time = 0.0
        self.resource_manager.resource_data.end_time = 10.0
        
        summary = self.resource_manager.get_resource_summary()
        
        self.assertIn("duration", summary)
        self.assertIn("avg_cpu_usage", summary)
        self.assertIn("max_cpu_usage", summary)
        self.assertIn("avg_memory_usage_mb", summary)
        self.assertIn("max_memory_usage_mb", summary)


class TestParallelBacktestManager(unittest.TestCase):
    """Test ParallelBacktestManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = AdvancedManager(ParallelConfig(max_workers=2))
        
        # Create mock configs
        self.mock_configs = [
            BacktestConfig(
                strategy_class="MockStrategy",
                strategy_params={},
                data_config={},
                execution_config={},
                sizing_config={},
                start_date="2023-01-01",
                end_date="2023-01-02"
            ) for _ in range(3)
        ]
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsInstance(self.manager.backtester, ParallelBacktester)
        self.assertIsInstance(self.manager.benchmark, PerformanceBenchmark)
        self.assertIsInstance(self.manager.resource_manager, ResourceManager)
    
    @patch('backtesting.parallel.parallel_manager.ParallelBacktester.run_parallel_backtests')
    @patch('backtesting.parallel.parallel_manager.ResourceManager.start_monitoring')
    @patch('backtesting.parallel.parallel_manager.ResourceManager.stop_monitoring')
    @patch('backtesting.parallel.parallel_manager.ResourceManager.get_resource_summary')
    def test_run_with_monitoring(self, mock_get_summary, mock_stop, mock_start, mock_run):
        """Test running backtests with monitoring"""
        # Mock results
        mock_results = [
            BacktestResult(run_id="test_1", success=True, execution_time=1.0),
            BacktestResult(run_id="test_2", success=True, execution_time=2.0)
        ]
        mock_run.return_value = mock_results
        mock_get_summary.return_value = {"avg_cpu_usage": 50.0}
        
        results, monitoring_data = self.manager.run_with_monitoring(
            self.mock_configs, 
            enable_resource_monitoring=True
        )
        
        self.assertEqual(len(results), 2)
        self.assertIn("resource_summary", monitoring_data)
        self.assertIn("performance_summary", monitoring_data)
        self.assertIn("system_info", monitoring_data)
        
        # Verify monitoring was started and stopped
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
    
    def test_generate_parameter_combinations(self):
        """Test parameter combination generation"""
        base_config = BacktestConfig(
            strategy_class="TestStrategy",
            strategy_params={"param1": 1.0},
            data_config={},
            execution_config={},
            sizing_config={},
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        parameter_ranges = {
            "param1": [1.0, 2.0],
            "param2": ["A", "B"]
        }
        
        configs = self.manager._generate_parameter_combinations(base_config, parameter_ranges)
        
        self.assertEqual(len(configs), 4)  # 2 * 2 combinations
        
        # Check that parameters are correctly set
        param1_values = [config.strategy_params["param1"] for config in configs]
        param2_values = [config.strategy_params["param2"] for config in configs]
        
        self.assertEqual(set(param1_values), {1.0, 2.0})
        self.assertEqual(set(param2_values), {"A", "B"})
    
    def test_find_best_result(self):
        """Test finding best result"""
        results = [
            BacktestResult(
                run_id="test_1",
                success=True,
                execution_time=1.0,
                performance_metrics={"total_return": 0.1}
            ),
            BacktestResult(
                run_id="test_2",
                success=True,
                execution_time=2.0,
                performance_metrics={"total_return": 0.2}
            ),
            BacktestResult(
                run_id="test_3",
                success=False,
                execution_time=0.5,
                error="Test error"
            )
        ]
        
        best_result = self.manager._find_best_result(results)
        
        self.assertIsNotNone(best_result)
        self.assertEqual(best_result.run_id, "test_2")
        self.assertEqual(best_result.performance_metrics["total_return"], 0.2)
    
    def test_find_best_result_no_successful(self):
        """Test finding best result when no successful results"""
        results = [
            BacktestResult(run_id="test_1", success=False, execution_time=1.0, error="Error 1"),
            BacktestResult(run_id="test_2", success=False, execution_time=2.0, error="Error 2")
        ]
        
        best_result = self.manager._find_best_result(results)
        
        self.assertIsNone(best_result)


class TestIntegration(unittest.TestCase):
    """Integration tests for parallel backtesting"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple mock strategy for testing
        class MockStrategy:
            def __init__(self):
                self.order_count = 0
            
            def next(self):
                pass
            
            def notify_order(self, order):
                self.order_count += 1
            
            def notify_trade(self, trade):
                pass
        
        self.MockStrategy = MockStrategy
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('backtesting.parallel.parallel_backtester.HighFidelityExecutionIntegration')
    @patch('backtesting.parallel.parallel_backtester.EnhancedTickDataFeedFactory')
    def test_end_to_end_parallel_execution(self, mock_factory, mock_integration):
        """Test end-to-end parallel execution"""
        # Mock the integration and data feed
        mock_integration_instance = Mock()
        mock_integration_instance.setup_cerebro.return_value = Mock()
        mock_integration_instance.run_backtest.return_value = {
            "results": [],
            "performance_metrics": {"total_return": 0.1},
            "execution_statistics": {"total_orders": 10}
        }
        mock_integration.return_value = mock_integration_instance
        
        mock_feed = Mock()
        mock_factory.create_feed.return_value = mock_feed
        
        # Create configs
        configs = [
            BacktestConfig(
                strategy_class="MockStrategy",
                strategy_params={},
                data_config={"symbol": "EURUSD"},
                execution_config={},
                sizing_config={},
                start_date="2023-01-01",
                end_date="2023-01-02"
            ) for _ in range(2)
        ]
        
        # Run parallel backtests
        backtester = ParallelBacktester(ParallelConfig(max_workers=2, timeout=30.0))
        results = backtester.run_parallel_backtests(configs)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(backtester.completed_runs, 2)
        self.assertEqual(backtester.failed_runs, 0)
    
    def test_save_and_load_results(self):
        """Test saving and loading results"""
        # Create mock results
        backtester = ParallelBacktester()
        backtester.results = [
            BacktestResult(
                run_id="test_001",
                success=True,
                execution_time=1.0,
                results={"total_return": 0.1}
            )
        ]
        backtester.aggregated_results = {"total_runs": 1}
        
        # Save results
        results_file = os.path.join(self.temp_dir, "test_results.json")
        backtester.save_results(results_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(results_file))
        
        # Load results into new backtester
        new_backtester = ParallelBacktester()
        new_backtester.load_results(results_file)
        
        # Verify results were loaded
        self.assertEqual(len(new_backtester.results), 1)
        self.assertEqual(new_backtester.results[0].run_id, "test_001")
        self.assertEqual(new_backtester.aggregated_results["total_runs"], 1)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)
