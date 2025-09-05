#!/usr/bin/env python3
"""
Test Redis Cache Integration
Tests the Redis cache functionality for the data ingestion system
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app1.src.data_ingestion.cache.redis_cache import RedisCacheManager, CacheConfig
from app1.src.data_ingestion.brokers.broker_manager import PriceData, CandleData, Granularity
from app1.src.data_ingestion.engines.ingestion_service import DataIngestionService, ServiceConfig
from app1.src.data_ingestion.brokers.broker_manager import BrokerManager
from app1.src.data_ingestion.brokers.free_forex_client import FreeForexClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_redis_connection():
    """Test basic Redis connection"""
    logger.info("Testing Redis connection...")
    
    cache_config = CacheConfig()
    cache_manager = RedisCacheManager(cache_config)
    
    try:
        connected = await cache_manager.connect()
        if connected:
            logger.info("✅ Redis connection successful")
            
            # Test basic operations
            await cache_manager.redis.set("test_key", "test_value", ex=60)
            value = await cache_manager.redis.get("test_key")
            assert value.decode() == "test_value"
            logger.info("✅ Basic Redis operations working")
            
            await cache_manager.disconnect()
            logger.info("✅ Redis disconnection successful")
            return True
        else:
            logger.error("❌ Redis connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ Redis connection error: {e}")
        return False

async def test_price_data_caching():
    """Test price data caching functionality"""
    logger.info("Testing price data caching...")
    
    cache_config = CacheConfig()
    cache_manager = RedisCacheManager(cache_config)
    
    try:
        await cache_manager.connect()
        
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
        
        # Cache price data
        for price_data in test_prices:
            await cache_manager.cache_price_data(price_data.instrument, price_data)
        
        logger.info(f"✅ Cached {len(test_prices)} price data points")
        
        # Retrieve cached data
        cached_prices = await cache_manager.get_cached_price_data("EUR_USD")
        logger.info(f"✅ Retrieved {len(cached_prices)} cached price data points")
        
        assert len(cached_prices) == len(test_prices)
        logger.info("✅ Price data caching test passed")
        
        await cache_manager.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Price data caching test failed: {e}")
        return False

async def test_candle_data_caching():
    """Test candle data caching functionality"""
    logger.info("Testing candle data caching...")
    
    cache_config = CacheConfig()
    cache_manager = RedisCacheManager(cache_config)
    
    try:
        await cache_manager.connect()
        
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
        
        # Cache candle data
        for candle_data in test_candles:
            await cache_manager.cache_candle_data("EUR_USD", "M1", candle_data)
        
        logger.info(f"✅ Cached {len(test_candles)} candle data points")
        
        # Retrieve cached data
        cached_candles = await cache_manager.get_cached_candle_data("EUR_USD", "M1")
        logger.info(f"✅ Retrieved {len(cached_candles)} cached candle data points")
        
        assert len(cached_candles) == len(test_candles)
        logger.info("✅ Candle data caching test passed")
        
        await cache_manager.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Candle data caching test failed: {e}")
        return False

async def test_indicators_caching():
    """Test indicators caching functionality"""
    logger.info("Testing indicators caching...")
    
    cache_config = CacheConfig()
    cache_manager = RedisCacheManager(cache_config)
    
    try:
        await cache_manager.connect()
        
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
        
        # Cache indicators
        await cache_manager.cache_indicators("EUR_USD", "M1", test_indicators)
        logger.info("✅ Cached indicators")
        
        # Retrieve cached indicators
        cached_indicators = await cache_manager.get_cached_indicators("EUR_USD", "M1")
        logger.info(f"✅ Retrieved cached indicators: {cached_indicators}")
        
        assert cached_indicators is not None
        assert len(cached_indicators) == len(test_indicators)
        logger.info("✅ Indicators caching test passed")
        
        await cache_manager.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Indicators caching test failed: {e}")
        return False

async def test_api_response_caching():
    """Test API response caching functionality"""
    logger.info("Testing API response caching...")
    
    cache_config = CacheConfig()
    cache_manager = RedisCacheManager(cache_config)
    
    try:
        await cache_manager.connect()
        
        # Create test API response
        test_response = {
            'rates': {
                'EUR': 0.92,
                'GBP': 0.79,
                'JPY': 156.0
            },
            'base': 'USD',
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        test_params = {'base': 'USD', 'symbols': 'EUR,GBP,JPY'}
        
        # Cache API response
        await cache_manager.cache_api_response('/latest', test_params, test_response)
        logger.info("✅ Cached API response")
        
        # Retrieve cached response
        cached_response = await cache_manager.get_cached_api_response('/latest', test_params)
        logger.info(f"✅ Retrieved cached API response: {cached_response is not None}")
        
        assert cached_response is not None
        assert cached_response['base'] == test_response['base']
        logger.info("✅ API response caching test passed")
        
        await cache_manager.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ API response caching test failed: {e}")
        return False

async def test_cache_stats():
    """Test cache statistics functionality"""
    logger.info("Testing cache statistics...")
    
    cache_config = CacheConfig()
    cache_manager = RedisCacheManager(cache_config)
    
    try:
        await cache_manager.connect()
        
        # Get cache stats
        stats = await cache_manager.get_cache_stats()
        logger.info(f"✅ Cache stats: {stats}")
        
        assert 'connected' in stats
        assert stats['connected'] == True
        logger.info("✅ Cache statistics test passed")
        
        await cache_manager.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache statistics test failed: {e}")
        return False

async def test_integrated_service():
    """Test the integrated data ingestion service with Redis caching"""
    logger.info("Testing integrated service with Redis caching...")
    
    try:
        # Create broker manager with free forex client
        broker_manager = BrokerManager()
        await broker_manager.add_broker("free_forex", FreeForexClient(), priority=1)
        
        # Create service configuration with caching enabled
        service_config = ServiceConfig(
            instruments=["EUR_USD"],
            granularities=["M1"],
            enable_real_time=True,
            enable_historical=True,
            historical_lookback_days=1,
            processing_enabled=True,
            storage_enabled=True,
            cache_enabled=True
        )
        
        # Create service
        service = DataIngestionService(broker_manager, service_config)
        
        # Start service
        await service.start()
        logger.info("✅ Service started with Redis caching")
        
        # Wait a bit for data to be cached
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
    """Run all Redis cache tests"""
    logger.info("🚀 Starting Redis Cache Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("Price Data Caching", test_price_data_caching),
        ("Candle Data Caching", test_candle_data_caching),
        ("Indicators Caching", test_indicators_caching),
        ("API Response Caching", test_api_response_caching),
        ("Cache Statistics", test_cache_stats),
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
        logger.info("🎉 All Redis cache tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
