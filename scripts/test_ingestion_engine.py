#!/usr/bin/env python3
"""
Test Data Ingestion Engine
Test script for the asyncio-based data ingestion engine
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app1.src.data_ingestion.engines.ingestion_service import DataIngestionService, ServiceConfig
from app1.src.data_ingestion.engines.ingestion_engine import IngestionConfig
from app1.src.data_ingestion.brokers.broker_manager import BrokerManager
from app1.src.data_ingestion.brokers.oanda_client import OANDAClient, Granularity
from app1.src.data_ingestion.brokers.fxcm_client import FXCMClient, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ingestion_engine_components():
    """Test individual components of the ingestion engine"""
    logger.info("Testing ingestion engine components...")
    
    # Test data processor
    from app1.src.data_ingestion.engines.data_processor import DataProcessor
    from app1.src.data_ingestion.brokers.oanda_client import PriceData
    
    processor = DataProcessor()
    
    # Simulate price data
    base_time = datetime.now()
    for i in range(50):
        price_data = PriceData(
            instrument="EUR_USD",
            time=base_time + timedelta(minutes=i),
            bid=1.1000 + i * 0.0001,
            ask=1.1002 + i * 0.0001,
            spread=0.0002
        )
        await processor.process_price_data(price_data)
    
    # Get processor status
    status = processor.get_status()
    logger.info(f"Data processor status: {status}")
    
    # Test price history
    history = processor.get_price_history("EUR_USD")
    logger.info(f"Price history length: {len(history)}")
    
    logger.info("✅ Data processor test completed")

async def test_ingestion_service():
    """Test the complete ingestion service"""
    logger.info("Testing ingestion service...")
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Add mock brokers (since we don't have real API keys)
    # In a real scenario, you would add actual broker clients here
    
    # Create service configuration
    service_config = ServiceConfig(
        instruments=["EUR_USD", "GBP_USD"],
        granularities=["M1", "M5"],
        enable_real_time=False,  # Disable real-time for testing
        enable_historical=False,  # Disable historical for testing
        processing_enabled=True
    )
    
    # Create ingestion configuration
    ingestion_config = IngestionConfig(
        max_concurrent_requests=5,
        batch_size=50,
        flush_interval=1.0,
        health_check_interval=10.0
    )
    
    # Create service
    service = DataIngestionService(
        broker_manager, 
        service_config, 
        ingestion_config
    )
    
    # Add callbacks
    def data_callback(data):
        logger.info(f"Data received: {type(data)}")
    
    def error_callback(error):
        logger.error(f"Error received: {error}")
    
    service.add_data_callback(data_callback)
    service.add_error_callback(error_callback)
    
    # Start service
    await service.start()
    
    try:
        # Wait a bit
        await asyncio.sleep(2)
        
        # Get service status
        status = service.get_service_status()
        logger.info(f"Service status: {status}")
        
        # Test cached price (will be None without real brokers)
        price = await service.get_cached_price("EUR_USD")
        logger.info(f"Cached price: {price}")
        
        # Test historical data (will be empty without real brokers)
        historical = await service.get_historical_data("EUR_USD", "M1", 10)
        logger.info(f"Historical data: {len(historical)} points")
        
        logger.info("✅ Ingestion service test completed")
        
    finally:
        await service.stop()

async def test_performance_metrics():
    """Test performance metrics and monitoring"""
    logger.info("Testing performance metrics...")
    
    from app1.src.data_ingestion.engines.ingestion_engine import DataIngestionEngine, IngestionConfig
    from app1.src.data_ingestion.brokers.broker_manager import BrokerManager
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Create ingestion engine
    config = IngestionConfig(
        max_concurrent_requests=10,
        batch_size=100,
        flush_interval=0.5,
        health_check_interval=5.0
    )
    
    engine = DataIngestionEngine(broker_manager, config)
    
    # Start engine
    await engine.start()
    
    try:
        # Simulate some work
        await asyncio.sleep(2)
        
        # Get metrics
        metrics = engine.get_metrics()
        logger.info(f"Engine metrics: {metrics}")
        
        # Test queue operations
        from app1.src.data_ingestion.brokers.oanda_client import PriceData
        
        # Add some test data to queues
        for i in range(10):
            price_data = PriceData(
                instrument="EUR_USD",
                time=datetime.now(),
                bid=1.1000 + i * 0.0001,
                ask=1.1002 + i * 0.0001,
                spread=0.0002
            )
            await engine.price_queue.put(price_data)
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Get updated metrics
        updated_metrics = engine.get_metrics()
        logger.info(f"Updated metrics: {updated_metrics}")
        
        logger.info("✅ Performance metrics test completed")
        
    finally:
        await engine.stop()

async def test_error_handling():
    """Test error handling and recovery"""
    logger.info("Testing error handling...")
    
    from app1.src.data_ingestion.engines.ingestion_engine import DataIngestionEngine, IngestionConfig
    from app1.src.data_ingestion.brokers.broker_manager import BrokerManager
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Create ingestion engine with aggressive settings for testing
    config = IngestionConfig(
        max_concurrent_requests=1,
        batch_size=1,
        flush_interval=0.1,
        health_check_interval=1.0
    )
    
    engine = DataIngestionEngine(broker_manager, config)
    
    # Add error callback
    error_count = 0
    def error_callback(error):
        nonlocal error_count
        error_count += 1
        logger.info(f"Error callback triggered: {error}")
    
    # Start engine
    await engine.start()
    
    try:
        # Simulate some work
        await asyncio.sleep(1)
        
        # Get initial metrics
        initial_metrics = engine.get_metrics()
        logger.info(f"Initial metrics: {initial_metrics}")
        
        # Wait a bit more
        await asyncio.sleep(2)
        
        # Get final metrics
        final_metrics = engine.get_metrics()
        logger.info(f"Final metrics: {final_metrics}")
        
        logger.info(f"Error count: {error_count}")
        logger.info("✅ Error handling test completed")
        
    finally:
        await engine.stop()

async def main():
    """Main test function"""
    logger.info("🚀 Starting Data Ingestion Engine Tests")
    logger.info("=" * 60)
    
    # Test individual components
    await test_ingestion_engine_components()
    logger.info("")
    
    # Test performance metrics
    await test_performance_metrics()
    logger.info("")
    
    # Test error handling
    await test_error_handling()
    logger.info("")
    
    # Test complete service
    await test_ingestion_service()
    logger.info("")
    
    logger.info("🎉 Data Ingestion Engine Tests Completed!")

if __name__ == "__main__":
    asyncio.run(main())
