#!/usr/bin/env python3
"""
Test Validation Storage Integration

Tests the integration between validation framework and PostgreSQL storage.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.strategy_validation.storage.simple_validation_storage import (
    SimpleValidationStorage, ValidationResultRecord
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_validation_record() -> ValidationResultRecord:
    """Create a test validation record"""
    return ValidationResultRecord(
        strategy_id="test_strategy_001",
        strategy_name="Test Strategy 001",
        strategy_type="trend_following",
        validation_timestamp=datetime.now(),
        validation_passed=True,
        validation_score=0.85,
        critical_violations=[],
        warnings=["Low trade count"],
        performance_metrics={
            "total_return": 15.5,
            "annualized_return": 12.3,
            "sharpe_ratio": 1.45,
            "sortino_ratio": 1.67,
            "calmar_ratio": 1.23,
            "max_drawdown": 8.2,
            "win_rate": 62.5,
            "profit_factor": 1.45,
            "total_trades": 156,
            "volatility": 12.1,
            "consistency_score": 75.0,
            "stability_score": 80.0
        },
        scoring_metrics={
            "total_score": 85.0,
            "performance_score": 90.0,
            "risk_score": 80.0,
            "consistency_score": 75.0,
            "efficiency_score": 85.0,
            "robustness_score": 90.0,
            "breakdown": {
                "return_score": 85.0,
                "profit_factor_score": 90.0,
                "win_rate_score": 75.0,
                "drawdown_score": 80.0,
                "var_score": 85.0,
                "volatility_score": 90.0,
                "stability_score": 80.0,
                "reliability_score": 85.0,
                "predictability_score": 75.0,
                "sharpe_score": 90.0,
                "calmar_score": 85.0,
                "sortino_score": 90.0
            }
        },
        backtest_config={
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
            "currency_pairs": ["EURUSD", "GBPUSD"],
            "timeframes": ["H1", "H4"]
        },
        validation_duration_seconds=45.2,
        backtest_duration_days=365,
        total_trades=156
    )


async def test_storage_service():
    """Test the validation storage service"""
    logger.info("Testing SimpleValidationStorage...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create storage service
    storage_service = SimpleValidationStorage(database_url)
    
    # Create test record
    test_record = create_test_validation_record()
    
    # Test storing validation result
    logger.info("Testing store_validation_result...")
    success = await storage_service.store_validation_result(test_record)
    assert success, "Failed to store validation result"
    logger.info("✅ store_validation_result passed")
    
    # Test getting validation results
    logger.info("Testing get_validation_results...")
    results = await storage_service.get_validation_results(limit=10)
    assert len(results) > 0, "No validation results found"
    assert results[0]['strategy_id'] == test_record.strategy_id, "Wrong strategy ID"
    logger.info("✅ get_validation_results passed")
    
    # Test getting performance summary
    logger.info("Testing get_strategy_performance_summary...")
    summary = await storage_service.get_strategy_performance_summary()
    assert len(summary) > 0, "No performance summary found"
    logger.info("✅ get_strategy_performance_summary passed")
    
    # Test getting top strategies
    logger.info("Testing get_top_strategies...")
    top_strategies = await storage_service.get_top_strategies(limit=5)
    assert len(top_strategies) > 0, "No top strategies found"
    logger.info("✅ get_top_strategies passed")
    
    # Test getting validation statistics
    logger.info("Testing get_validation_statistics...")
    stats = await storage_service.get_validation_statistics()
    assert 'total_validations' in stats, "Missing total_validations in stats"
    assert stats['total_validations'] > 0, "No validations in statistics"
    logger.info("✅ get_validation_statistics passed")
    
    logger.info("✅ SimpleValidationStorage tests passed")


async def test_database_schema():
    """Test database schema creation"""
    logger.info("Testing database schema creation...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create storage service (this will initialize the schema)
    storage_service = SimpleValidationStorage(database_url)
    
    # Test that we can query the validation_results table
    import asyncpg
    conn = await asyncpg.connect(database_url)
    try:
        # Check if table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'validation_results'
            )
        """)
        assert table_exists, "validation_results table does not exist"
        
        # Check if view exists
        view_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.views 
                WHERE table_name = 'strategy_performance_summary'
            )
        """)
        assert view_exists, "strategy_performance_summary view does not exist"
        
        # Check if indexes exist
        indexes = await conn.fetch("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'validation_results'
        """)
        index_names = [idx['indexname'] for idx in indexes]
        assert 'idx_validation_results_strategy_id' in index_names, "Missing strategy_id index"
        assert 'idx_validation_results_timestamp' in index_names, "Missing timestamp index"
        assert 'idx_validation_results_passed' in index_names, "Missing passed index"
        assert 'idx_validation_results_score' in index_names, "Missing score index"
    finally:
        await conn.close()
    
    logger.info("✅ Database schema tests passed")


async def test_cleanup():
    """Test cleanup functionality"""
    logger.info("Testing cleanup functionality...")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
    
    # Create storage service
    storage_service = SimpleValidationStorage(database_url)
    
    # Test cleanup (should not fail even if no old results)
    deleted_count = await storage_service.cleanup_old_results(days_to_keep=0)
    assert isinstance(deleted_count, int), "Deleted count should be an integer"
    logger.info(f"Cleaned up {deleted_count} old results")
    
    logger.info("✅ Cleanup tests passed")


async def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("VALIDATION STORAGE INTEGRATION TEST")
    logger.info("=" * 80)
    
    try:
        # Test database schema
        await test_database_schema()
        
        # Test storage service
        await test_storage_service()
        
        # Test cleanup
        await test_cleanup()
        
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
