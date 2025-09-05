#!/usr/bin/env python3
"""
Test Validation Workflow

Tests the end-to-end validation workflow orchestrator.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.strategy_validation.workflow.validation_workflow import (
    ValidationWorkflow, WorkflowConfig, WorkflowStatus, WorkflowStage
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_config() -> WorkflowConfig:
    """Create a test workflow configuration"""
    return WorkflowConfig(
        strategy_count=5,  # Small number for testing
        strategy_types=["trend_following", "mean_reversion"],
        currency_pairs=["EURUSD", "GBPUSD"],
        timeframes=["H1", "H4"],
        enable_parallel_processing=True,
        max_workers=2,
        batch_size=3,
        timeout_seconds=60.0,
        store_results=True,
        cleanup_old_results=False,  # Don't cleanup during tests
        days_to_keep=30,
        enable_logging=True,
        log_level="INFO",
        enable_metrics=True
    )


async def test_workflow_initialization():
    """Test workflow initialization"""
    logger.info("Testing workflow initialization...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create workflow configuration
    config = create_test_config()
    
    # Create workflow instance
    workflow = ValidationWorkflow(config, database_url)
    
    # Verify initialization
    assert workflow.config.strategy_count == 5
    assert workflow.config.strategy_types == ["trend_following", "mean_reversion"]
    assert workflow.config.currency_pairs == ["EURUSD", "GBPUSD"]
    assert workflow.config.timeframes == ["H1", "H4"]
    assert workflow.config.enable_parallel_processing == True
    assert workflow.config.max_workers == 2
    assert workflow.config.batch_size == 3
    assert workflow.config.timeout_seconds == 60.0
    assert workflow.config.store_results == True
    assert workflow.config.cleanup_old_results == False
    assert workflow.config.days_to_keep == 30
    assert workflow.config.enable_logging == True
    assert workflow.config.log_level == "INFO"
    assert workflow.config.enable_metrics == True
    
    logger.info("✅ Workflow initialization test passed")


async def test_workflow_execution():
    """Test complete workflow execution"""
    logger.info("Testing complete workflow execution...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create workflow configuration
    config = create_test_config()
    
    # Create workflow instance
    workflow = ValidationWorkflow(config, database_url)
    
    # Run workflow
    result = await workflow.run_workflow()
    
    # Verify workflow completed
    assert result.status == WorkflowStatus.COMPLETED, f"Workflow failed with status: {result.status}"
    assert result.workflow_id is not None, "Workflow ID is None"
    assert result.message == "Workflow completed successfully", f"Unexpected message: {result.message}"
    
    # Verify metrics
    assert result.metrics.strategies_generated > 0, "No strategies generated"
    assert result.metrics.strategies_validated > 0, "No strategies validated"
    assert result.metrics.strategies_passed >= 0, "Invalid strategies passed count"
    assert result.metrics.strategies_stored >= 0, "Invalid strategies stored count"
    assert result.metrics.duration > 0, "Invalid duration"
    
    # Verify stage timings
    assert len(result.metrics.stage_timings) > 0, "No stage timings recorded"
    assert "strategy_generation" in result.metrics.stage_timings, "Missing strategy generation timing"
    assert "validation" in result.metrics.stage_timings, "Missing validation timing"
    assert "storage" in result.metrics.stage_timings, "Missing storage timing"
    
    # Verify generated strategies
    assert len(result.generated_strategies) > 0, "No strategies generated"
    assert len(result.generated_strategies) <= config.strategy_count, "Too many strategies generated"
    
    # Verify validation results
    assert result.validation_results is not None, "No validation results"
    assert result.validation_results.input_strategies > 0, "No input strategies for validation"
    assert result.validation_results.passing_strategies >= 0, "Invalid passing strategies count"
    
    # Verify stored results
    if config.store_results:
        assert len(result.stored_results) >= 0, "Invalid stored results count"
        assert len(result.stored_results) <= result.validation_results.passing_strategies, "Too many stored results"
    
    logger.info("✅ Complete workflow execution test passed")


async def test_workflow_error_handling():
    """Test workflow error handling"""
    logger.info("Testing workflow error handling...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create workflow configuration with invalid settings to trigger errors
    config = WorkflowConfig(
        strategy_count=0,  # This should cause an error
        strategy_types=[],
        currency_pairs=[],
        timeframes=[],
        enable_parallel_processing=True,
        max_workers=1,
        batch_size=1,
        timeout_seconds=1.0,  # Very short timeout
        store_results=True,
        cleanup_old_results=False,
        days_to_keep=30,
        enable_logging=True,
        log_level="INFO",
        enable_metrics=True
    )
    
    # Create workflow instance
    workflow = ValidationWorkflow(config, database_url)
    
    # Run workflow (should fail)
    result = await workflow.run_workflow()
    
    # Verify workflow failed
    assert result.status == WorkflowStatus.FAILED, f"Workflow should have failed but got status: {result.status}"
    assert len(result.errors) > 0, "No errors recorded"
    assert result.metrics.total_errors > 0, "No error metrics recorded"
    
    logger.info("✅ Workflow error handling test passed")


async def test_workflow_status_tracking():
    """Test workflow status tracking"""
    logger.info("Testing workflow status tracking...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create workflow configuration
    config = create_test_config()
    
    # Create workflow instance
    workflow = ValidationWorkflow(config, database_url)
    
    # Start workflow
    result = await workflow.run_workflow()
    
    # Verify status tracking
    assert result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED], f"Invalid final status: {result.status}"
    assert result.workflow_id is not None, "Workflow ID is None"
    assert result.message is not None, "Message is None"
    assert result.created_at is not None, "Created at is None"
    assert result.updated_at is not None, "Updated at is None"
    
    # Verify metrics
    assert result.metrics.start_time is not None, "Start time is None"
    assert result.metrics.duration >= 0, "Invalid duration"
    assert result.metrics.strategies_generated >= 0, "Invalid strategies generated"
    assert result.metrics.strategies_validated >= 0, "Invalid strategies validated"
    assert result.metrics.strategies_passed >= 0, "Invalid strategies passed"
    assert result.metrics.strategies_stored >= 0, "Invalid strategies stored"
    
    logger.info("✅ Workflow status tracking test passed")


async def test_workflow_metrics():
    """Test workflow metrics calculation"""
    logger.info("Testing workflow metrics calculation...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create workflow configuration
    config = create_test_config()
    
    # Create workflow instance
    workflow = ValidationWorkflow(config, database_url)
    
    # Run workflow
    result = await workflow.run_workflow()
    
    # Verify metrics calculation
    if result.metrics.duration > 0:
        assert result.metrics.strategies_per_second >= 0, "Invalid strategies per second"
    
    if result.metrics.strategies_validated > 0:
        assert result.metrics.average_validation_time >= 0, "Invalid average validation time"
    
    if result.metrics.strategies_stored > 0:
        assert result.metrics.average_storage_time >= 0, "Invalid average storage time"
    
    # Verify stage timings
    for stage in result.metrics.stage_timings:
        assert result.metrics.stage_timings[stage] >= 0, f"Invalid timing for stage {stage}"
    
    # Verify error metrics
    assert result.metrics.total_errors >= 0, "Invalid total errors"
    assert result.metrics.retry_count >= 0, "Invalid retry count"
    
    logger.info("✅ Workflow metrics calculation test passed")


async def test_workflow_cancellation():
    """Test workflow cancellation"""
    logger.info("Testing workflow cancellation...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create workflow configuration
    config = create_test_config()
    
    # Create workflow instance
    workflow = ValidationWorkflow(config, database_url)
    
    # Test cancellation of non-existent workflow
    success = await workflow.cancel_workflow("non_existent_workflow")
    assert not success, "Cancellation of non-existent workflow should fail"
    
    # Test cancellation of completed workflow
    result = await workflow.run_workflow()
    success = await workflow.cancel_workflow(result.workflow_id)
    assert not success, "Cancellation of completed workflow should fail"
    
    logger.info("✅ Workflow cancellation test passed")


async def test_workflow_history():
    """Test workflow history retrieval"""
    logger.info("Testing workflow history retrieval...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create workflow configuration
    config = create_test_config()
    
    # Create workflow instance
    workflow = ValidationWorkflow(config, database_url)
    
    # Run a few workflows
    results = []
    for i in range(3):
        result = await workflow.run_workflow()
        results.append(result)
    
    # Get workflow history
    history = await workflow.get_workflow_history(limit=10)
    
    # Verify history
    assert isinstance(history, list), "History should be a list"
    assert len(history) >= 0, "History should not be negative"
    
    logger.info("✅ Workflow history retrieval test passed")


async def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("VALIDATION WORKFLOW TEST")
    logger.info("=" * 80)
    
    try:
        # Test workflow initialization
        await test_workflow_initialization()
        
        # Test complete workflow execution
        await test_workflow_execution()
        
        # Test workflow error handling
        await test_workflow_error_handling()
        
        # Test workflow status tracking
        await test_workflow_status_tracking()
        
        # Test workflow metrics
        await test_workflow_metrics()
        
        # Test workflow cancellation
        await test_workflow_cancellation()
        
        # Test workflow history
        await test_workflow_history()
        
        logger.info("=" * 80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

