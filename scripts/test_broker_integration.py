#!/usr/bin/env python3
"""
Test Broker Integration
Test script for broker API integrations
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app1.src.data_ingestion.brokers.broker_manager import BrokerManager
from app1.src.data_ingestion.brokers.oanda_client import OANDAClient, Granularity
from app1.src.data_ingestion.brokers.fxcm_client import FXCMClient, TimeFrame
from app1.src.data_ingestion.brokers.free_forex_client import FreeForexClient
from app1.src.data_ingestion.config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_individual_brokers():
    """Test individual broker clients"""
    logger.info("Testing individual broker clients...")
    
    # Test OANDA
    oanda_api_key = os.getenv("OANDA_API_KEY")
    oanda_account_id = os.getenv("OANDA_ACCOUNT_ID")
    
    if oanda_api_key and oanda_account_id:
        logger.info("Testing OANDA client...")
        try:
            async with OANDAClient(oanda_api_key, oanda_account_id) as client:
                # Test account info
                account_info = await client.get_account_info()
                logger.info(f"OANDA Account: {account_info.get('account', {}).get('id')}")
                
                # Test instruments
                instruments = await client.get_instruments()
                logger.info(f"OANDA Instruments: {len(instruments)}")
                
                # Test pricing
                pricing = await client.get_pricing(["EUR_USD", "GBP_USD"])
                logger.info(f"OANDA Pricing: {len(pricing.get('prices', []))} instruments")
                
                # Test candles
                candles = await client.get_candles("EUR_USD", Granularity.M1, count=5)
                logger.info(f"OANDA Candles: {len(candles)} retrieved")
                
                logger.info("✅ OANDA client test passed")
        except Exception as e:
            logger.error(f"❌ OANDA client test failed: {e}")
    else:
        logger.warning("⚠️  OANDA credentials not provided, skipping test")
    
    # Test FXCM
    fxcm_token = os.getenv("FXCM_TOKEN")
    
    if fxcm_token:
        logger.info("Testing FXCM client...")
        try:
            async with FXCMClient(fxcm_token) as client:
                # Test account info
                account_info = await client.get_account_info()
                logger.info(f"FXCM Accounts: {len(account_info.get('accounts', []))}")
                
                # Test symbols
                symbols = await client.get_symbols()
                logger.info(f"FXCM Symbols: {len(symbols)}")
                
                # Test offers
                offers = await client.get_offers(["EURUSD", "GBPUSD"])
                logger.info(f"FXCM Offers: {len(offers.get('offers', []))} symbols")
                
                # Test historical data
                candles = await client.get_historical_data("EURUSD", TimeFrame.M1, count=5)
                logger.info(f"FXCM Candles: {len(candles)} retrieved")
                
                logger.info("✅ FXCM client test passed")
        except Exception as e:
            logger.error(f"❌ FXCM client test failed: {e}")
    else:
        logger.warning("⚠️  FXCM credentials not provided, skipping test")
    
    # Test Free Forex Client
    logger.info("Testing Free Forex client...")
    try:
        async with FreeForexClient() as client:
            # Test health status
            health = await client.get_health_status()
            logger.info(f"Free Forex health: {health}")
            
            # Test latest rates
            rates = await client.get_latest_rates()
            logger.info(f"Free Forex rates: {len(rates.get('rates', {}))} currencies")
            
            # Test real-time prices
            prices = await client.get_real_time_prices(["EUR_USD", "GBP_USD"])
            logger.info(f"Free Forex prices: {len(prices)} symbols")
            
            # Test historical data
            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()
            historical = await client.get_historical_rates("USD", "EUR", start_date, end_date)
            logger.info(f"Free Forex historical: {len(historical)} candles")
            
            logger.info("✅ Free Forex client test passed")
    except Exception as e:
        logger.error(f"❌ Free Forex client test failed: {e}")

async def test_broker_manager():
    """Test broker manager functionality"""
    logger.info("Testing broker manager...")
    
    # Create broker manager
    manager = BrokerManager()
    
    # Add OANDA broker
    oanda_api_key = os.getenv("OANDA_API_KEY")
    oanda_account_id = os.getenv("OANDA_ACCOUNT_ID")
    
    if oanda_api_key and oanda_account_id:
        oanda_client = OANDAClient(oanda_api_key, oanda_account_id)
        await manager.add_broker("oanda", oanda_client, priority=1)
        logger.info("Added OANDA broker")
    
    # Add FXCM broker
    fxcm_token = os.getenv("FXCM_TOKEN")
    
    if fxcm_token:
        fxcm_client = FXCMClient(fxcm_token)
        await manager.add_broker("fxcm", fxcm_client, priority=2)
        logger.info("Added FXCM broker")
    
    # Add Free Forex broker (always available)
    free_forex_client = FreeForexClient()
    await manager.add_broker("free_forex", free_forex_client, priority=3)
    logger.info("Added Free Forex broker")
    
    if not manager.brokers:
        logger.warning("⚠️  No brokers configured, skipping manager test")
        return
    
    # Start health monitoring
    await manager.start_health_monitoring()
    
    try:
        # Wait for health checks
        await asyncio.sleep(5)
        
        # Test status
        status = await manager.get_status()
        logger.info(f"Broker Manager Status: {status}")
        
        # Test getting candles
        try:
            candles = await manager.get_candles("EUR_USD", Granularity.M1, count=5)
            logger.info(f"✅ Retrieved {len(candles)} candles via manager")
        except Exception as e:
            logger.error(f"❌ Failed to get candles via manager: {e}")
        
        # Test getting pricing
        try:
            pricing = await manager.get_pricing(["EUR_USD", "GBP_USD"])
            logger.info(f"✅ Retrieved pricing for {len(pricing.get('prices', []))} instruments via manager")
        except Exception as e:
            logger.error(f"❌ Failed to get pricing via manager: {e}")
        
        # Test price streaming (short test)
        try:
            async def price_callback(price_data):
                logger.info(f"Price update: {price_data.instrument} - Bid: {price_data.bid}, Ask: {price_data.ask}")
            
            # Start streaming for 5 seconds
            stream_task = asyncio.create_task(
                manager.stream_prices(["EUR_USD"], price_callback)
            )
            await asyncio.sleep(5)
            stream_task.cancel()
            logger.info("✅ Price streaming test completed")
        except Exception as e:
            logger.error(f"❌ Price streaming test failed: {e}")
        
    finally:
        await manager.stop_health_monitoring()
        logger.info("Stopped broker manager")

async def test_configuration():
    """Test configuration loading"""
    logger.info("Testing configuration...")
    
    logger.info(f"Environment: {CONFIG.environment}")
    logger.info(f"Health check interval: {CONFIG.health_check_interval}s")
    logger.info(f"Max retries: {CONFIG.max_retries}")
    logger.info(f"Cache TTL: {CONFIG.cache_ttl}s")
    logger.info(f"Max latency: {CONFIG.max_latency_ms}ms")
    
    for name, broker in CONFIG.brokers.items():
        logger.info(f"Broker {name}: enabled={broker.enabled}, priority={broker.priority}")
    
    logger.info("✅ Configuration test completed")

async def main():
    """Main test function"""
    logger.info("🚀 Starting Broker Integration Tests")
    logger.info("=" * 50)
    
    # Test configuration
    await test_configuration()
    logger.info("")
    
    # Test individual brokers
    await test_individual_brokers()
    logger.info("")
    
    # Test broker manager
    await test_broker_manager()
    logger.info("")
    
    logger.info("🎉 Broker Integration Tests Completed!")

if __name__ == "__main__":
    asyncio.run(main())
