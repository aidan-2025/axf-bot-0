import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .influxdb_writer import InfluxDBWriter, InfluxDBConfig
from ..brokers.broker_manager import PriceData, CandleData
from ..cache.redis_cache import RedisCacheManager

logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """Configuration for storage operations"""
    influxdb_enabled: bool = True
    redis_enabled: bool = True
    batch_size: int = 1000
    flush_interval: int = 5  # seconds
    retention_days: int = 30
    compression_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'StorageConfig':
        """Create configuration from environment variables"""
        import os
        return cls(
            influxdb_enabled=os.getenv('INFLUXDB_ENABLED', 'true').lower() == 'true',
            redis_enabled=os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            batch_size=int(os.getenv('STORAGE_BATCH_SIZE', '1000')),
            flush_interval=int(os.getenv('STORAGE_FLUSH_INTERVAL', '5')),
            retention_days=int(os.getenv('STORAGE_RETENTION_DAYS', '30')),
            compression_enabled=os.getenv('STORAGE_COMPRESSION_ENABLED', 'true').lower() == 'true'
        )

class StorageManager:
    """Manages data storage across Redis and InfluxDB"""
    
    def __init__(self, 
                 influxdb_config: InfluxDBConfig = InfluxDBConfig(),
                 storage_config: StorageConfig = StorageConfig(),
                 cache_manager: Optional[RedisCacheManager] = None):
        self.influxdb_config = influxdb_config
        self.storage_config = storage_config
        self.cache_manager = cache_manager
        
        # Initialize components
        self.influxdb_writer = InfluxDBWriter(influxdb_config) if storage_config.influxdb_enabled else None
        
        # State
        self.connected = False
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("StorageManager initialized")

    async def start(self) -> bool:
        """Starts the storage manager"""
        if self.running:
            logger.warning("Storage manager is already running")
            return True
        
        logger.info("Starting storage manager...")
        
        try:
            # Connect to InfluxDB if enabled
            if self.storage_config.influxdb_enabled and self.influxdb_writer:
                influxdb_connected = await self.influxdb_writer.connect()
                if not influxdb_connected:
                    logger.warning("Failed to connect to InfluxDB, continuing without it")
                    self.storage_config.influxdb_enabled = False
            
            # Start flush task
            if self.storage_config.influxdb_enabled:
                self.flush_task = asyncio.create_task(self._flush_loop())
            
            self.running = True
            self.connected = True
            logger.info("Storage manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start storage manager: {e}")
            self.connected = False
            return False

    async def stop(self) -> None:
        """Stops the storage manager"""
        if not self.running:
            return
        
        logger.info("Stopping storage manager...")
        self.running = False
        
        # Cancel flush task
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush any pending data
        if self.influxdb_writer:
            await self.influxdb_writer.flush()
            await self.influxdb_writer.disconnect()
        
        self.connected = False
        logger.info("Storage manager stopped")

    async def _flush_loop(self):
        """Periodically flushes data to InfluxDB"""
        while self.running:
            try:
                await asyncio.sleep(self.storage_config.flush_interval)
                if self.influxdb_writer:
                    await self.influxdb_writer.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    # Price Data Storage
    async def store_price_data(self, price_data: PriceData, 
                              tags: Optional[Dict[str, str]] = None) -> bool:
        """Stores price data in both Redis cache and InfluxDB"""
        success = True
        
        # Store in Redis cache if available
        if self.cache_manager and self.cache_manager.connected:
            try:
                await self.cache_manager.cache_price_data(price_data.instrument, price_data)
            except Exception as e:
                logger.warning(f"Failed to cache price data: {e}")
        
        # Store in InfluxDB if enabled
        if self.storage_config.influxdb_enabled and self.influxdb_writer:
            try:
                await self.influxdb_writer.write_price_data(price_data, tags)
            except Exception as e:
                logger.error(f"Failed to write price data to InfluxDB: {e}")
                success = False
        
        return success

    async def store_candle_data(self, candle_data: CandleData, instrument: str,
                               granularity: str, tags: Optional[Dict[str, str]] = None) -> bool:
        """Stores candle data in both Redis cache and InfluxDB"""
        success = True
        
        # Store in Redis cache if available
        if self.cache_manager and self.cache_manager.connected:
            try:
                await self.cache_manager.cache_candle_data(instrument, granularity, candle_data)
            except Exception as e:
                logger.warning(f"Failed to cache candle data: {e}")
        
        # Store in InfluxDB if enabled
        if self.storage_config.influxdb_enabled and self.influxdb_writer:
            try:
                await self.influxdb_writer.write_candle_data(candle_data, instrument, granularity, tags)
            except Exception as e:
                logger.error(f"Failed to write candle data to InfluxDB: {e}")
                success = False
        
        return success

    async def store_indicators(self, instrument: str, granularity: str,
                              indicators: Dict[str, Any],
                              tags: Optional[Dict[str, str]] = None) -> bool:
        """Stores technical indicators in both Redis cache and InfluxDB"""
        success = True
        
        # Store in Redis cache if available
        if self.cache_manager and self.cache_manager.connected:
            try:
                await self.cache_manager.cache_indicators(instrument, granularity, indicators)
            except Exception as e:
                logger.warning(f"Failed to cache indicators: {e}")
        
        # Store in InfluxDB if enabled
        if self.storage_config.influxdb_enabled and self.influxdb_writer:
            try:
                await self.influxdb_writer.write_indicators(instrument, granularity, indicators, tags)
            except Exception as e:
                logger.error(f"Failed to write indicators to InfluxDB: {e}")
                success = False
        
        return success

    async def store_market_snapshot(self, snapshot: Dict[str, Any],
                                   tags: Optional[Dict[str, str]] = None) -> bool:
        """Stores a complete market snapshot in InfluxDB"""
        if not self.storage_config.influxdb_enabled or not self.influxdb_writer:
            return False
        
        try:
            await self.influxdb_writer.write_market_snapshot(snapshot, tags)
            return True
        except Exception as e:
            logger.error(f"Failed to write market snapshot to InfluxDB: {e}")
            return False

    # Data Retrieval
    async def get_price_data(self, instrument: str, start_time: datetime,
                            end_time: Optional[datetime] = None,
                            limit: Optional[int] = None,
                            use_cache: bool = True) -> List[Dict[str, Any]]:
        """Retrieves price data from cache or InfluxDB"""
        
        # Try Redis cache first if enabled and requested
        if use_cache and self.cache_manager and self.cache_manager.connected:
            try:
                cached_data = await self.cache_manager.get_cached_price_data(
                    instrument, start_time, end_time
                )
                if cached_data:
                    # Convert to dict format
                    data = []
                    for price in cached_data:
                        data.append({
                            'time': price.time.isoformat(),
                            'instrument': price.instrument,
                            'bid': price.bid,
                            'ask': price.ask,
                            'spread': price.spread
                        })
                    logger.info(f"Retrieved {len(data)} price data points from cache for {instrument}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to get price data from cache: {e}")
        
        # Fallback to InfluxDB
        if self.storage_config.influxdb_enabled and self.influxdb_writer:
            try:
                data = await self.influxdb_writer.query_price_data(
                    instrument, start_time, end_time, limit
                )
                logger.info(f"Retrieved {len(data)} price data points from InfluxDB for {instrument}")
                return data
            except Exception as e:
                logger.error(f"Failed to get price data from InfluxDB: {e}")
        
        return []

    async def get_candle_data(self, instrument: str, granularity: str,
                             start_time: datetime, end_time: Optional[datetime] = None,
                             limit: Optional[int] = None,
                             use_cache: bool = True) -> List[Dict[str, Any]]:
        """Retrieves candle data from cache or InfluxDB"""
        
        # Try Redis cache first if enabled and requested
        if use_cache and self.cache_manager and self.cache_manager.connected:
            try:
                cached_data = await self.cache_manager.get_cached_candle_data(
                    instrument, granularity, start_time, end_time
                )
                if cached_data:
                    # Convert to dict format
                    data = []
                    for candle in cached_data:
                        data.append({
                            'time': candle.time.isoformat(),
                            'open': candle.open,
                            'high': candle.high,
                            'low': candle.low,
                            'close': candle.close,
                            'volume': candle.volume,
                            'complete': candle.complete
                        })
                    logger.info(f"Retrieved {len(data)} candle data points from cache for {instrument} ({granularity})")
                    return data
            except Exception as e:
                logger.warning(f"Failed to get candle data from cache: {e}")
        
        # Fallback to InfluxDB
        if self.storage_config.influxdb_enabled and self.influxdb_writer:
            try:
                data = await self.influxdb_writer.query_candle_data(
                    instrument, granularity, start_time, end_time, limit
                )
                logger.info(f"Retrieved {len(data)} candle data points from InfluxDB for {instrument} ({granularity})")
                return data
            except Exception as e:
                logger.error(f"Failed to get candle data from InfluxDB: {e}")
        
        return []

    async def get_indicators(self, instrument: str, granularity: str,
                            start_time: datetime, end_time: Optional[datetime] = None,
                            limit: Optional[int] = None,
                            use_cache: bool = True) -> List[Dict[str, Any]]:
        """Retrieves technical indicators from cache or InfluxDB"""
        
        # Try Redis cache first if enabled and requested
        if use_cache and self.cache_manager and self.cache_manager.connected:
            try:
                cached_indicators = await self.cache_manager.get_cached_indicators(
                    instrument, granularity
                )
                if cached_indicators:
                    # Convert to list format with timestamp
                    data = [{
                        'time': datetime.now().isoformat(),
                        'instrument': instrument,
                        'granularity': granularity,
                        **cached_indicators
                    }]
                    logger.info(f"Retrieved indicators from cache for {instrument} ({granularity})")
                    return data
            except Exception as e:
                logger.warning(f"Failed to get indicators from cache: {e}")
        
        # Fallback to InfluxDB
        if self.storage_config.influxdb_enabled and self.influxdb_writer:
            try:
                data = await self.influxdb_writer.query_indicators(
                    instrument, granularity, start_time, end_time, limit
                )
                logger.info(f"Retrieved {len(data)} indicator data points from InfluxDB for {instrument} ({granularity})")
                return data
            except Exception as e:
                logger.error(f"Failed to get indicators from InfluxDB: {e}")
        
        return []

    # Health and Status
    async def get_health_status(self) -> Dict[str, Any]:
        """Gets health status of all storage components"""
        status = {
            "connected": self.connected,
            "running": self.running,
            "influxdb_enabled": self.storage_config.influxdb_enabled,
            "redis_enabled": self.cache_manager is not None and self.cache_manager.connected
        }
        
        # InfluxDB status
        if self.storage_config.influxdb_enabled and self.influxdb_writer:
            try:
                influxdb_healthy = await self.influxdb_writer.is_healthy()
                status["influxdb_healthy"] = influxdb_healthy
                if influxdb_healthy:
                    status["influxdb_stats"] = await self.influxdb_writer.get_stats()
            except Exception as e:
                status["influxdb_healthy"] = False
                status["influxdb_error"] = str(e)
        
        # Redis status
        if self.cache_manager:
            try:
                redis_healthy = await self.cache_manager.is_healthy()
                status["redis_healthy"] = redis_healthy
                if redis_healthy:
                    status["redis_stats"] = await self.cache_manager.get_cache_stats()
            except Exception as e:
                status["redis_healthy"] = False
                status["redis_error"] = str(e)
        
        return status

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Gets comprehensive storage statistics"""
        stats = {
            "storage_manager": {
                "connected": self.connected,
                "running": self.running,
                "influxdb_enabled": self.storage_config.influxdb_enabled,
                "redis_enabled": self.cache_manager is not None
            }
        }
        
        # InfluxDB stats
        if self.storage_config.influxdb_enabled and self.influxdb_writer:
            try:
                stats["influxdb"] = await self.influxdb_writer.get_stats()
            except Exception as e:
                stats["influxdb"] = {"error": str(e)}
        
        # Redis stats
        if self.cache_manager:
            try:
                stats["redis"] = await self.cache_manager.get_cache_stats()
            except Exception as e:
                stats["redis"] = {"error": str(e)}
        
        return stats

    # Utility Methods
    async def flush_all(self) -> None:
        """Flushes all pending data to storage"""
        if self.influxdb_writer:
            await self.influxdb_writer.flush()
        logger.info("Flushed all pending data to storage")

    async def clear_cache(self, pattern: str = "*") -> int:
        """Clears Redis cache"""
        if self.cache_manager and self.cache_manager.connected:
            return await self.cache_manager.clear_cache(pattern)
        return 0

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
