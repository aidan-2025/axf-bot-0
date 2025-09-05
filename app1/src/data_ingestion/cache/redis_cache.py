import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import redis.asyncio as redis
from redis.asyncio import Redis

from ..brokers.broker_manager import PriceData, CandleData
from ..config import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for Redis caching"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    
    # Cache TTL settings (in seconds)
    price_data_ttl: int = 60  # 1 minute for real-time prices
    candle_data_ttl: int = 3600  # 1 hour for historical candles
    indicators_ttl: int = 300  # 5 minutes for calculated indicators
    api_response_ttl: int = 1800  # 30 minutes for API responses
    health_check_ttl: int = 30  # 30 seconds for health checks

class RedisCacheManager:
    """Redis cache manager for data ingestion system"""
    
    def __init__(self, config: CacheConfig = CacheConfig()):
        self.config = config
        self.redis: Optional[Redis] = None
        self.connected = False
        self.connection_pool = None
        logger.info(f"RedisCacheManager initialized with config: {config}")

    async def connect(self) -> bool:
        """Establishes connection to Redis"""
        try:
            if self.redis:
                await self.redis.close()
            
            self.connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout
            )
            
            self.redis = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis.ping()
            self.connected = True
            logger.info("Successfully connected to Redis")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Closes Redis connection"""
        if self.redis:
            await self.redis.close()
            self.redis = None
        if self.connection_pool:
            await self.connection_pool.disconnect()
            self.connection_pool = None
        self.connected = False
        logger.info("Disconnected from Redis")

    async def is_healthy(self) -> bool:
        """Checks if Redis connection is healthy"""
        if not self.connected or not self.redis:
            return False
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    # Price Data Caching
    async def cache_price_data(self, instrument: str, price_data: PriceData, ttl: Optional[int] = None) -> bool:
        """Caches real-time price data"""
        if not self.connected:
            return False
        
        try:
            key = f"price:{instrument}:{price_data.time.isoformat()}"
            ttl = ttl or self.config.price_data_ttl
            
            # Convert PriceData to dict for JSON serialization
            data = asdict(price_data)
            data['time'] = price_data.time.isoformat()
            
            await self.redis.setex(key, ttl, json.dumps(data))
            logger.debug(f"Cached price data for {instrument}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache price data for {instrument}: {e}")
            return False

    async def get_cached_price_data(self, instrument: str, from_time: Optional[datetime] = None, 
                                   to_time: Optional[datetime] = None) -> List[PriceData]:
        """Retrieves cached price data for an instrument"""
        if not self.connected:
            return []
        
        try:
            pattern = f"price:{instrument}:*"
            keys = await self.redis.keys(pattern)
            
            if not keys:
                return []
            
            # Get all price data
            pipe = self.redis.pipeline()
            for key in keys:
                pipe.get(key)
            results = await pipe.execute()
            
            price_data_list = []
            for result in results:
                if result:
                    data = json.loads(result)
                    data['time'] = datetime.fromisoformat(data['time'])
                    price_data = PriceData(**data)
                    
                    # Filter by time range if specified
                    if from_time and price_data.time < from_time:
                        continue
                    if to_time and price_data.time > to_time:
                        continue
                    
                    price_data_list.append(price_data)
            
            # Sort by time
            price_data_list.sort(key=lambda x: x.time)
            logger.debug(f"Retrieved {len(price_data_list)} cached price data points for {instrument}")
            return price_data_list
            
        except Exception as e:
            logger.error(f"Failed to get cached price data for {instrument}: {e}")
            return []

    # Candle Data Caching
    async def cache_candle_data(self, instrument: str, granularity: str, candle_data: CandleData, 
                               ttl: Optional[int] = None) -> bool:
        """Caches historical candle data"""
        if not self.connected:
            return False
        
        try:
            key = f"candle:{instrument}:{granularity}:{candle_data.time.isoformat()}"
            ttl = ttl or self.config.candle_data_ttl
            
            # Convert CandleData to dict for JSON serialization
            data = asdict(candle_data)
            data['time'] = candle_data.time.isoformat()
            
            await self.redis.setex(key, ttl, json.dumps(data))
            logger.debug(f"Cached candle data for {instrument} ({granularity})")
            return True
        except Exception as e:
            logger.error(f"Failed to cache candle data for {instrument} ({granularity}): {e}")
            return False

    async def get_cached_candle_data(self, instrument: str, granularity: str, 
                                    from_time: Optional[datetime] = None,
                                    to_time: Optional[datetime] = None) -> List[CandleData]:
        """Retrieves cached candle data for an instrument and granularity"""
        if not self.connected:
            return []
        
        try:
            pattern = f"candle:{instrument}:{granularity}:*"
            keys = await self.redis.keys(pattern)
            
            if not keys:
                return []
            
            # Get all candle data
            pipe = self.redis.pipeline()
            for key in keys:
                pipe.get(key)
            results = await pipe.execute()
            
            candle_data_list = []
            for result in results:
                if result:
                    data = json.loads(result)
                    data['time'] = datetime.fromisoformat(data['time'])
                    candle_data = CandleData(**data)
                    
                    # Filter by time range if specified
                    if from_time and candle_data.time < from_time:
                        continue
                    if to_time and candle_data.time > to_time:
                        continue
                    
                    candle_data_list.append(candle_data)
            
            # Sort by time
            candle_data_list.sort(key=lambda x: x.time)
            logger.debug(f"Retrieved {len(candle_data_list)} cached candle data points for {instrument} ({granularity})")
            return candle_data_list
            
        except Exception as e:
            logger.error(f"Failed to get cached candle data for {instrument} ({granularity}): {e}")
            return []

    # Indicators Caching
    async def cache_indicators(self, instrument: str, granularity: str, indicators: Dict[str, Any], 
                              ttl: Optional[int] = None) -> bool:
        """Caches calculated technical indicators"""
        if not self.connected:
            return False
        
        try:
            key = f"indicators:{instrument}:{granularity}:{datetime.now().isoformat()}"
            ttl = ttl or self.config.indicators_ttl
            
            # Add timestamp to indicators
            data = {
                'indicators': indicators,
                'timestamp': datetime.now().isoformat(),
                'instrument': instrument,
                'granularity': granularity
            }
            
            await self.redis.setex(key, ttl, json.dumps(data))
            logger.debug(f"Cached indicators for {instrument} ({granularity})")
            return True
        except Exception as e:
            logger.error(f"Failed to cache indicators for {instrument} ({granularity}): {e}")
            return False

    async def get_cached_indicators(self, instrument: str, granularity: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached indicators for an instrument and granularity"""
        if not self.connected:
            return None
        
        try:
            pattern = f"indicators:{instrument}:{granularity}:*"
            keys = await self.redis.keys(pattern)
            
            if not keys:
                return None
            
            # Get the most recent indicators
            def get_timestamp(key):
                if isinstance(key, bytes):
                    key = key.decode()
                return key.split(':')[-1]
            
            latest_key = max(keys, key=get_timestamp)
            if isinstance(latest_key, bytes):
                latest_key = latest_key.decode()
            result = await self.redis.get(latest_key)
            
            if result:
                data = json.loads(result)
                logger.debug(f"Retrieved cached indicators for {instrument} ({granularity})")
                return data.get('indicators', {})
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached indicators for {instrument} ({granularity}): {e}")
            return None

    # API Response Caching
    async def cache_api_response(self, endpoint: str, params: Dict[str, Any], response: Any, 
                                ttl: Optional[int] = None) -> bool:
        """Caches API responses to reduce external API calls"""
        if not self.connected:
            return False
        
        try:
            # Create a hash of the endpoint and parameters for the key
            import hashlib
            key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            key = f"api:{endpoint}:{key_hash}"
            
            ttl = ttl or self.config.api_response_ttl
            
            data = {
                'endpoint': endpoint,
                'params': params,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.redis.setex(key, ttl, json.dumps(data, default=str))
            logger.debug(f"Cached API response for {endpoint}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache API response for {endpoint}: {e}")
            return False

    async def get_cached_api_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Retrieves cached API response"""
        if not self.connected:
            return None
        
        try:
            # Create the same hash as in cache_api_response
            import hashlib
            key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            key = f"api:{endpoint}:{key_hash}"
            
            result = await self.redis.get(key)
            if result:
                data = json.loads(result)
                logger.debug(f"Retrieved cached API response for {endpoint}")
                return data.get('response')
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached API response for {endpoint}: {e}")
            return None

    # Health Check Caching
    async def cache_health_check(self, service: str, status: Dict[str, Any], 
                                ttl: Optional[int] = None) -> bool:
        """Caches health check results"""
        if not self.connected:
            return False
        
        try:
            key = f"health:{service}"
            ttl = ttl or self.config.health_check_ttl
            
            data = {
                'service': service,
                'status': status,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.redis.setex(key, ttl, json.dumps(data))
            logger.debug(f"Cached health check for {service}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache health check for {service}: {e}")
            return False

    async def get_cached_health_check(self, service: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached health check results"""
        if not self.connected:
            return None
        
        try:
            key = f"health:{service}"
            result = await self.redis.get(key)
            
            if result:
                data = json.loads(result)
                logger.debug(f"Retrieved cached health check for {service}")
                return data.get('status')
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached health check for {service}: {e}")
            return None

    # Cache Management
    async def clear_cache(self, pattern: str = "*") -> int:
        """Clears cache entries matching a pattern"""
        if not self.connected:
            return 0
        
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache with pattern {pattern}: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Returns cache statistics"""
        if not self.connected:
            return {"connected": False}
        
        try:
            info = await self.redis.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"connected": False, "error": str(e)}

    async def get_cache_keys(self, pattern: str = "*") -> List[str]:
        """Returns cache keys matching a pattern"""
        if not self.connected:
            return []
        
        try:
            keys = await self.redis.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Failed to get cache keys with pattern {pattern}: {e}")
            return []
