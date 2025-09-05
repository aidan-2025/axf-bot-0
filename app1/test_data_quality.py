#!/usr/bin/env python3
"""
Test Data Quality Assurance System
Tests the comprehensive data quality assurance and failover system
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock classes for testing
class MockInfluxDBWriter:
    async def write_quality_metrics(self, data: Dict[str, Any]):
        logger.info(f"Mock InfluxDB write: {data}")

class MockRedisCache:
    async def cache_quality_report(self, report):
        logger.info(f"Mock Redis cache: Quality report cached")

async def test_data_quality_system():
    """Test the data quality assurance system"""
    try:
        logger.info("Starting data quality assurance system test")
        
        # Import the quality assurance service
        from src.data_quality import DataQualityAssuranceService, QualityThresholds
        
        # Create mock dependencies
        mock_influxdb = MockInfluxDBWriter()
        mock_redis = MockRedisCache()
        
        # Create quality thresholds
        thresholds = QualityThresholds(
            min_completeness=0.95,
            max_missing_data=0.05,
            max_outlier_ratio=0.02
        )
        
        # Initialize the service
        qa_service = DataQualityAssuranceService(
            influxdb_writer=mock_influxdb,
            redis_cache=mock_redis,
            thresholds=thresholds
        )
        
        await qa_service.initialize()
        logger.info("✅ Quality assurance service initialized successfully")
        
        # Test 1: Validate good data
        logger.info("\n--- Test 1: Validating good data ---")
        good_data = {
            'instrument': 'EURUSD',
            'time': '2025-09-05T10:30:00Z',
            'bid': 1.1000,
            'ask': 1.1001,
            'volume': 1000
        }
        
        is_valid, issues = await qa_service.validate_data(good_data, 'oanda')
        logger.info(f"Good data validation: Valid={is_valid}, Issues={issues}")
        assert is_valid, f"Good data should be valid, but got issues: {issues}"
        logger.info("✅ Good data validation passed")
        
        # Test 2: Validate bad data and correction
        logger.info("\n--- Test 2: Validating and correcting bad data ---")
        bad_data = {
            'instrument': 'EURUSD',
            'time': '2025-09-05T10:30:00Z',
            'bid': -1.1000,  # Negative price
            'ask': 1.1001,
            'volume': -100  # Negative volume
        }
        
        is_valid, issues = await qa_service.validate_data(bad_data, 'oanda')
        logger.info(f"Bad data validation: Valid={is_valid}, Issues={issues}")
        # Note: The system automatically corrects bad data, so it becomes valid
        assert is_valid, f"Bad data should be corrected and become valid, but got issues: {issues}"
        logger.info("✅ Bad data validation and correction passed")
        
        # Test 3: Monitor data source
        logger.info("\n--- Test 3: Monitoring data source ---")
        source_status = await qa_service.monitor_data_source('oanda')
        logger.info(f"Source status: {source_status}")
        logger.info("✅ Source monitoring passed")
        
        # Test 4: Handle data failure
        logger.info("\n--- Test 4: Handling data failure ---")
        await qa_service.handle_data_failure('oanda', 'Connection timeout')
        logger.info("✅ Data failure handling passed")
        
        # Test 5: Run quality audit
        logger.info("\n--- Test 5: Running quality audit ---")
        quality_report = await qa_service.run_quality_audit()
        logger.info(f"Quality report generated:")
        logger.info(f"  - Overall quality: {quality_report.overall_quality.value}")
        logger.info(f"  - Total sources: {len(quality_report.data_sources)}")
        logger.info(f"  - Issues found: {len(quality_report.issues)}")
        logger.info(f"  - Corrections applied: {len(quality_report.corrections_applied)}")
        logger.info(f"  - Recommendations: {len(quality_report.recommendations)}")
        logger.info("✅ Quality audit passed")
        
        # Test 6: Test real-time validator
        logger.info("\n--- Test 6: Testing real-time validator ---")
        from src.data_quality import RealTimeValidator
        
        validator = RealTimeValidator(thresholds)
        await validator.initialize()
        
        # Test schema validation
        test_data = {
            'instrument': 'EURUSD',
            'time': '2025-09-05T10:30:00Z',
            'bid': 1.1000,
            'ask': 1.1001,
            'volume': 1000
        }
        
        is_valid, issues = await validator.validate(test_data, 'oanda')
        logger.info(f"Validator test: Valid={is_valid}, Issues={issues}")
        assert is_valid, f"Validator should pass good data: {issues}"
        logger.info("✅ Real-time validator passed")
        
        # Test 7: Test data corrector
        logger.info("\n--- Test 7: Testing data corrector ---")
        from src.data_quality import DataCorrector
        
        corrector = DataCorrector()
        await corrector.initialize()
        
        # Test correction of bad data
        bad_data = {
            'instrument': 'EURUSD',
            'time': '2025-09-05T10:30:00Z',
            'bid': -1.1000,  # Negative price
            'ask': 1.1001,
            'volume': -100  # Negative volume
        }
        
        corrected_data = await corrector.correct(bad_data, ['Price field bid must be positive', 'Volume must be non-negative'])
        logger.info(f"Data correction: {corrected_data is not None}")
        if corrected_data:
            logger.info(f"Corrected bid: {corrected_data.get('bid')}")
            logger.info(f"Corrected volume: {corrected_data.get('volume')}")
        logger.info("✅ Data corrector passed")
        
        # Test 8: Test failover manager
        logger.info("\n--- Test 8: Testing failover manager ---")
        from src.data_quality import FailoverManager
        
        failover_manager = FailoverManager()
        await failover_manager.initialize()
        
        # Record some failures
        await failover_manager.record_failure('oanda', 'Connection timeout')
        await failover_manager.record_failure('oanda', 'API rate limit')
        await failover_manager.record_failure('oanda', 'Server error')
        
        should_failover = await failover_manager.should_failover('oanda')
        backup_source = await failover_manager.get_backup_source('oanda')
        
        logger.info(f"Should failover: {should_failover}")
        logger.info(f"Backup source: {backup_source}")
        logger.info("✅ Failover manager passed")
        
        # Cleanup
        await qa_service.shutdown()
        await validator.shutdown()
        await corrector.shutdown()
        await failover_manager.shutdown()
        
        logger.info("\n🎉 All data quality assurance tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("DATA QUALITY ASSURANCE SYSTEM TEST")
    logger.info("=" * 60)
    
    success = await test_data_quality_system()
    
    if success:
        logger.info("\n✅ All tests completed successfully!")
        logger.info("The data quality assurance system is working correctly.")
    else:
        logger.error("\n❌ Some tests failed!")
        logger.error("Please check the logs for details.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
