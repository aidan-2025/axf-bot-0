#!/usr/bin/env python3
"""
Test InfluxDB Integration
Tests the InfluxDB storage functionality for the data ingestion system
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app1.src.data_ingestion.storage.influxdb_writer import InfluxDBWriter, InfluxDBConfig
from app1.src.data_ingestion.storage.storage_manager import StorageManager, StorageConfig
from app1.src.data_ingestion.cache.redis_cache import RedisCacheManager, CacheConfig
from app1.src.data_ingestion.brokers.broker_manager import PriceData, CandleData, Granularity
from app1.src.data_ingestion.brokers.free_forex_client import FreeForexClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_influxdb_connection():
    """Test basic InfluxDB connection"""
    logger.info("Testing InfluxDB connection...")
    
    influxdb_config = InfluxDBConfig()
    writer = InfluxDBWriter(influxdb_config)
    
    try:
        connected = await writer.connect()
        if connected:
            logger.info("✅ InfluxDB connection successful")
            
            # Test bucket info
            bucket_info = await writer.get_bucket_info()
            logger.info(f"✅ Bucket info: {bucket_info}")
            
            await writer.disconnect()
            logger.info("✅ InfluxDB disconnection successful")
            return True
        else:
            logger.error("❌ InfluxDB connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ InfluxDB connection error: {e}")
        return False

async def test_price_data_writing():
    """Test price data writing to InfluxDB"""
    logger.info("Testing price data writing...")
    
    influxdb_config = InfluxDBConfig()
    writer = InfluxDBWriter(influxdb_config)
    
    try:
        await writer.connect()
        
        # Create test price data
        test_prices = []
        for i in range(10):
            bid = 1.0800 + i * 0.0001 + random.uniform(-0.00005, 0.00005)
            ask = 1.0802 + i * 0.0001 + random.uniform(-0.00005, 0.00005)
            price_data = PriceData(
                instrument="EUR_USD",
                time=datetime.now() - timedelta(minutes=i),
                bid=bid,
                ask=ask,
                spread=ask - bid
            )
            test_prices.append(price_data)
        
        # Write price data
        for price_data in test_prices:
            await writer.write_price_data(price_data, {"broker": "test"})
        
        logger.info(f"✅ Wrote {len(test_prices)} price data points")
        
        # Query the data back
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15)
        queried_data = await writer.query_price_data("EUR_USD", start_time, end_time)
        logger.info(f"✅ Queried {len(queried_data)} price data points")
        
        await writer.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Price data writing test failed: {e}")
        return False

async def test_candle_data_writing():
    """Test candle data writing to InfluxDB"""
    logger.info("Testing candle data writing...")
    
    influxdb_config = InfluxDBConfig()
    writer = InfluxDBWriter(influxdb_config)
    
    try:
        await writer.connect()
        
        # Create test candle data
        test_candles = []
        for i in range(20):
            candle_data = CandleData(
                time=datetime.now() - timedelta(minutes=i),
                open=1.0800 + i * 0.0001,
                high=1.0800 + i * 0.0001 + 0.0002,
                low=1.0800 + i * 0.0001 - 0.0001,
                close=1.0800 + i * 0.0001 + random.uniform(-0.00005, 0.00005),
                volume=random.randint(100, 500),
                complete=True
            )
            test_candles.append(candle_data)
        
        # Write candle data
        for candle_data in test_candles:
            await writer.write_candle_data(candle_data, "EUR_USD", "M1", {"broker": "test"})
        
        logger.info(f"✅ Wrote {len(test_candles)} candle data points")
        
        # Query the data back
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=25)
        queried_data = await writer.query_candle_data("EUR_USD", "M1", start_time, end_time)
        logger.info(f"✅ Queried {len(queried_data)} candle data points")
        
        await writer.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Candle data writing test failed: {e}")
        return False

async def test_indicators_writing():
    """Test indicators writing to InfluxDB"""
    logger.info("Testing indicators writing...")
    
    influxdb_config = InfluxDBConfig()
    writer = InfluxDBWriter(influxdb_config)
    
    try:
        await writer.connect()
        
        # Create test indicators
        test_indicators = {
            'sma_20': 1.0805,
            'sma_50': 1.0800,
            'rsi_14': 65.5,
            'macd_line': 0.0001,
            'macd_signal': 0.0002,
            'bollinger_upper': 1.0810,
            'bollinger_lower': 1.0790
        }
        
        # Write indicators
        await writer.write_indicators("EUR_USD", "M1", test_indicators, {"broker": "test"})
        logger.info("✅ Wrote indicators")
        
        # Query the data back
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        queried_data = await writer.query_indicators("EUR_USD", "M1", start_time, end_time)
        logger.info(f"✅ Queried {len(queried_data)} indicator data points")
        
        await writer.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Indicators writing test failed: {e}")
        return False

async def test_storage_manager():
    """Test the storage manager integration"""
    logger.info("Testing storage manager...")
    
    try:
        # Create Redis cache manager
        cache_config = CacheConfig()
        cache_manager = RedisCacheManager(cache_config)
        await cache_manager.connect()
        
        # Create storage manager
        influxdb_config = InfluxDBConfig()
        storage_config = StorageConfig(
            influxdb_enabled=True,
            redis_enabled=True
        )
        storage_manager = StorageManager(
            influxdb_config=influxdb_config,
            storage_config=storage_config,
            cache_manager=cache_manager
        )
        
        # Start storage manager
        await storage_manager.start()
        logger.info("✅ Storage manager started")
        
        # Test storing price data
        price_data = PriceData(
            instrument="EUR_USD",
            time=datetime.now(),
            bid=1.0800,
            ask=1.0802,
            spread=0.0002
        )
        
        await storage_manager.store_price_data(price_data, {"broker": "test"})
        logger.info("✅ Stored price data via storage manager")
        
        # Test storing candle data
        candle_data = CandleData(
            time=datetime.now(),
            open=1.0800,
            high=1.0805,
            low=1.0795,
            close=1.0802,
            volume=1000,
            complete=True
        )
        
        await storage_manager.store_candle_data(candle_data, "EUR_USD", "M1", {"broker": "test"})
        logger.info("✅ Stored candle data via storage manager")
        
        # Test storing indicators
        indicators = {'sma_20': 1.0805, 'rsi_14': 65.5}
        await storage_manager.store_indicators("EUR_USD", "M1", indicators, {"broker": "test"})
        logger.info("✅ Stored indicators via storage manager")
        
        # Test data retrieval
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        
        price_data_retrieved = await storage_manager.get_price_data("EUR_USD", start_time, end_time)
        logger.info(f"✅ Retrieved {len(price_data_retrieved)} price data points")
        
        candle_data_retrieved = await storage_manager.get_candle_data("EUR_USD", "M1", start_time, end_time)
        logger.info(f"✅ Retrieved {len(candle_data_retrieved)} candle data points")
        
        indicators_retrieved = await storage_manager.get_indicators("EUR_USD", "M1", start_time, end_time)
        logger.info(f"✅ Retrieved {len(indicators_retrieved)} indicator data points")
        
        # Get health status
        health_status = await storage_manager.get_health_status()
        logger.info(f"✅ Health status: {health_status}")
        
        # Stop storage manager
        await storage_manager.stop()
        await cache_manager.disconnect()
        logger.info("✅ Storage manager stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Storage manager test failed: {e}")
        return False

async def test_integrated_service():
    """Test the integrated data ingestion service with InfluxDB storage"""
    logger.info("Testing integrated service with InfluxDB storage...")
    
    try:
        from app1.src.data_ingestion.engines.ingestion_service import DataIngestionService, ServiceConfig
        from app1.src.data_ingestion.brokers.broker_manager import BrokerManager
        
        # Create broker manager with free forex client
        broker_manager = BrokerManager()
        await broker_manager.add_broker("free_forex", FreeForexClient(), priority=1)
        
        # Create service configuration with storage enabled
        service_config = ServiceConfig(
            instruments=["EUR_USD"],
            granularities=["M1"],
            enable_real_time=True,
            enable_historical=True,
            historical_lookback_days=1,
            processing_enabled=True,
            storage_enabled=True,
            cache_enabled=True,
            influxdb_enabled=True
        )
        
        # Create service
        service = DataIngestionService(broker_manager, service_config)
        
        # Start service
        await service.start()
        logger.info("✅ Service started with InfluxDB storage")
        
        # Wait a bit for data to be stored
        await asyncio.sleep(3)
        
        # Test getting cached price
        cached_price = await service.get_cached_price("EUR_USD")
        if cached_price:
            logger.info(f"✅ Retrieved cached price: {cached_price}")
        else:
            logger.warning("⚠️ No cached price data available")
        
        # Test getting historical data
        historical_data = await service.get_historical_data("EUR_USD", "M1", 5)
        logger.info(f"✅ Retrieved {len(historical_data)} historical data points")
        
        # Get service status
        status = service.get_service_status()
        logger.info(f"✅ Service status: {status}")
        
        # Stop service
        await service.stop()
        logger.info("✅ Service stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated service test failed: {e}")
        return False

async def main():
    """Run all InfluxDB integration tests"""
    logger.info("🚀 Starting InfluxDB Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("InfluxDB Connection", test_influxdb_connection),
        ("Price Data Writing", test_price_data_writing),
        ("Candle Data Writing", test_candle_data_writing),
        ("Indicators Writing", test_indicators_writing),
        ("Storage Manager", test_storage_manager),
        ("Integrated Service", test_integrated_service)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name}...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All InfluxDB integration tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
