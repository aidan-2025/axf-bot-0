#!/usr/bin/env python3
"""
Data Ingestion Engine
High-performance asyncio-based data ingestion system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict, deque

from ..brokers.broker_manager import BrokerManager, PriceData, CandleData
from ..cache.redis_cache import RedisCacheManager
from ..config import CONFIG

logger = logging.getLogger(__name__)

class IngestionStatus(Enum):
    """Ingestion engine status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class IngestionMetrics:
    """Metrics for data ingestion"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    last_update: datetime = field(default_factory=datetime.now)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    data_points_ingested: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class IngestionConfig:
    """Configuration for data ingestion"""
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    flush_interval: float = 1.0  # seconds
    health_check_interval: float = 30.0  # seconds
    max_queue_size: int = 10000
    enable_caching: bool = True
    cache_ttl: float = 60.0  # seconds

class DataIngestionEngine:
    """High-performance data ingestion engine"""
    
    def __init__(self, 
                 broker_manager: BrokerManager,
                 config: Optional[IngestionConfig] = None,
                 cache_manager: Optional[RedisCacheManager] = None):
        """
        Initialize data ingestion engine
        
        Args:
            broker_manager: Broker manager instance
            config: Ingestion configuration
            cache_manager: Redis cache manager instance
        """
        self.broker_manager = broker_manager
        self.config = config or IngestionConfig()
        self.cache_manager = cache_manager
        
        # Engine state
        self.status = IngestionStatus.STOPPED
        self.metrics = IngestionMetrics()
        
        # Task management
        self.tasks: Set[asyncio.Task] = set()
        self.ingestion_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Data queues
        self.price_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.candle_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Callbacks
        self.price_callbacks: List[Callable] = []
        self.candle_callbacks: List[Callable] = []
        
        # Caching
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.candle_cache: Dict[str, List[CandleData]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Rate limiting
        self.request_times: deque = deque(maxlen=1000)
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        
    async def start(self) -> None:
        """Start the data ingestion engine"""
        if self.status != IngestionStatus.STOPPED:
            logger.warning("Engine is already running or starting")
            return
        
        logger.info("Starting data ingestion engine...")
        self.status = IngestionStatus.STARTING
        
        try:
            # Start broker manager health monitoring
            await self.broker_manager.start_health_monitoring()
            
            # Start ingestion tasks
            self.ingestion_task = asyncio.create_task(self._ingestion_loop())
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.status = IngestionStatus.RUNNING
            logger.info("Data ingestion engine started successfully")
            
        except Exception as e:
            self.status = IngestionStatus.ERROR
            logger.error(f"Failed to start ingestion engine: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the data ingestion engine"""
        if self.status == IngestionStatus.STOPPED:
            return
        
        logger.info("Stopping data ingestion engine...")
        self.status = IngestionStatus.STOPPING
        
        # Cancel all tasks
        if self.ingestion_task:
            self.ingestion_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self.ingestion_task, self.health_check_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop broker manager
        await self.broker_manager.stop_health_monitoring()
        
        self.status = IngestionStatus.STOPPED
        logger.info("Data ingestion engine stopped")
    
    async def _ingestion_loop(self) -> None:
        """Main ingestion loop"""
        while self.status == IngestionStatus.RUNNING:
            try:
                # Process price data
                await self._process_price_queue()
                
                # Process candle data
                await self._process_candle_queue()
                
                # Flush cached data
                await self._flush_cached_data()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ingestion loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_price_queue(self) -> None:
        """Process price data from queue"""
        try:
            # Get batch of price data
            batch = []
            for _ in range(min(self.config.batch_size, self.price_queue.qsize())):
                try:
                    price_data = self.price_queue.get_nowait()
                    batch.append(price_data)
                except asyncio.QueueEmpty:
                    break
            
            if batch:
                await self._process_price_batch(batch)
                
        except Exception as e:
            logger.error(f"Error processing price queue: {e}")
    
    async def _process_candle_queue(self) -> None:
        """Process candle data from queue"""
        try:
            # Get batch of candle data
            batch = []
            for _ in range(min(self.config.batch_size, self.candle_queue.qsize())):
                try:
                    candle_data = self.candle_queue.get_nowait()
                    batch.append(candle_data)
                except asyncio.QueueEmpty:
                    break
            
            if batch:
                await self._process_candle_batch(batch)
                
        except Exception as e:
            logger.error(f"Error processing candle queue: {e}")
    
    async def _process_price_batch(self, batch: List[PriceData]) -> None:
        """Process a batch of price data"""
        start_time = time.time()
        
        try:
            # Update cache
            if self.config.enable_caching:
                await self._update_price_cache(batch)
            
            # Call callbacks
            for price_data in batch:
                for callback in self.price_callbacks:
                    try:
                        await callback(price_data)
                    except Exception as e:
                        logger.error(f"Error in price callback: {e}")
            
            # Update metrics
            self.metrics.data_points_ingested += len(batch)
            self.metrics.successful_requests += 1
            
        except Exception as e:
            logger.error(f"Error processing price batch: {e}")
            self.metrics.failed_requests += 1
            self.error_counts[type(e).__name__] += 1
        
        finally:
            # Update latency metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_metrics(latency_ms)
    
    async def _process_candle_batch(self, batch: List[CandleData]) -> None:
        """Process a batch of candle data"""
        start_time = time.time()
        
        try:
            # Update cache
            if self.config.enable_caching:
                await self._update_candle_cache(batch)
            
            # Call callbacks
            for candle_data in batch:
                for callback in self.candle_callbacks:
                    try:
                        await callback(candle_data)
                    except Exception as e:
                        logger.error(f"Error in candle callback: {e}")
            
            # Update metrics
            self.metrics.data_points_ingested += len(batch)
            self.metrics.successful_requests += 1
            
        except Exception as e:
            logger.error(f"Error processing candle batch: {e}")
            self.metrics.failed_requests += 1
            self.error_counts[type(e).__name__] += 1
        
        finally:
            # Update latency metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_metrics(latency_ms)
    
    async def _update_price_cache(self, batch: List[PriceData]) -> None:
        """Update price cache with new data"""
        if not self.cache_manager or not self.cache_manager.connected:
            # Fallback to in-memory cache if Redis is not available
            current_time = datetime.now()
            for price_data in batch:
                cache_key = price_data.instrument
                self.price_cache[cache_key] = {
                    'bid': price_data.bid,
                    'ask': price_data.ask,
                    'spread': price_data.spread,
                    'time': price_data.time
                }
                self.cache_timestamps[cache_key] = current_time
            return
        
        # Use Redis cache
        for price_data in batch:
            try:
                await self.cache_manager.cache_price_data(
                    price_data.instrument, 
                    price_data,
                    ttl=self.config.cache_ttl
                )
            except Exception as e:
                logger.warning(f"Failed to cache price data in Redis: {e}")
                # Fallback to in-memory cache
                cache_key = price_data.instrument
                self.price_cache[cache_key] = {
                    'bid': price_data.bid,
                    'ask': price_data.ask,
                    'spread': price_data.spread,
                    'time': price_data.time
                }
                self.cache_timestamps[cache_key] = datetime.now()
    
    async def _update_candle_cache(self, batch: List[CandleData]) -> None:
        """Update candle cache with new data"""
        if not self.cache_manager or not self.cache_manager.connected:
            # Fallback to in-memory cache if Redis is not available
            current_time = datetime.now()
            for candle_data in batch:
                cache_key = f"{candle_data.time.strftime('%Y%m%d%H%M')}"
                if cache_key not in self.candle_cache:
                    self.candle_cache[cache_key] = []
                
                self.candle_cache[cache_key].append(candle_data)
                self.cache_timestamps[cache_key] = current_time
            return
        
        # Use Redis cache
        for candle_data in batch:
            try:
                # Get instrument and granularity from candle data attributes
                instrument = getattr(candle_data, 'instrument', 'unknown')
                granularity = getattr(candle_data, 'granularity', 'M1')
                
                await self.cache_manager.cache_candle_data(
                    instrument,
                    granularity,
                    candle_data,
                    ttl=self.config.cache_ttl
                )
            except Exception as e:
                logger.warning(f"Failed to cache candle data in Redis: {e}")
                # Fallback to in-memory cache
                cache_key = f"{candle_data.time.strftime('%Y%m%d%H%M')}"
                if cache_key not in self.candle_cache:
                    self.candle_cache[cache_key] = []
                self.candle_cache[cache_key].append(candle_data)
                self.cache_timestamps[cache_key] = datetime.now()
    
    async def _flush_cached_data(self) -> None:
        """Flush cached data to storage"""
        # This would typically flush to InfluxDB or other storage
        # For now, we'll just clean up old cache entries
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (current_time - timestamp).total_seconds() > self.config.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.price_cache.pop(key, None)
            self.candle_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    async def _health_check_loop(self) -> None:
        """Health check loop"""
        while self.status == IngestionStatus.RUNNING:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_check(self) -> None:
        """Perform health check"""
        try:
            # Check broker manager status
            broker_status = await self.broker_manager.get_status()
            healthy_brokers = broker_status.get('healthy_brokers', 0)
            
            if healthy_brokers == 0:
                logger.warning("No healthy brokers available")
                self.status = IngestionStatus.ERROR
                return
            
            # Check queue sizes
            price_queue_size = self.price_queue.qsize()
            candle_queue_size = self.candle_queue.qsize()
            
            if price_queue_size > self.config.max_queue_size * 0.8:
                logger.warning(f"Price queue is {price_queue_size} items (80% full)")
            
            if candle_queue_size > self.config.max_queue_size * 0.8:
                logger.warning(f"Candle queue is {candle_queue_size} items (80% full)")
            
            # Update metrics
            self.metrics.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _update_latency_metrics(self, latency_ms: float) -> None:
        """Update latency metrics"""
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * self.metrics.total_requests + latency_ms) /
            (self.metrics.total_requests + 1)
        )
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency_ms)
        self.metrics.total_requests += 1
    
    async def ingest_prices(self, instruments: List[str]) -> None:
        """Start ingesting real-time prices for instruments"""
        if self.status != IngestionStatus.RUNNING:
            raise Exception("Engine is not running")
        
        async def price_callback(price_data: PriceData):
            await self.price_queue.put(price_data)
        
        # Start price streaming
        task = asyncio.create_task(
            self.broker_manager.stream_prices(instruments, price_callback)
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
    
    async def ingest_candles(self, 
                           instrument: str,
                           granularity: str,
                           count: Optional[int] = None) -> None:
        """Ingest historical candles for an instrument"""
        if self.status != IngestionStatus.RUNNING:
            raise Exception("Engine is not running")
        
        try:
            # Convert granularity string to enum
            from ..brokers.oanda_client import Granularity
            gran_enum = getattr(Granularity, granularity.upper(), Granularity.M1)
            
            # Get candles from broker
            candles = await self.broker_manager.get_candles(
                instrument, gran_enum, count
            )
            
            # Add to queue
            for candle in candles:
                # Attach context for downstream validation/storage
                try:
                    setattr(candle, 'instrument', instrument)
                    setattr(candle, 'granularity', granularity)
                except Exception:
                    pass
                await self.candle_queue.put(candle)
                
        except Exception as e:
            logger.error(f"Error ingesting candles for {instrument}: {e}")
            raise
    
    def add_price_callback(self, callback: Callable) -> None:
        """Add a callback for price data"""
        self.price_callbacks.append(callback)
    
    def add_candle_callback(self, callback: Callable) -> None:
        """Add a callback for candle data"""
        self.candle_callbacks.append(callback)
    
    async def get_cached_price(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get cached price data for an instrument"""
        if not self.config.enable_caching:
            return None
        
        # Try Redis cache first
        if self.cache_manager and self.cache_manager.connected:
            try:
                cached_data = await self.cache_manager.get_cached_price_data(
                    instrument, 
                    from_time=datetime.now() - timedelta(minutes=5)
                )
                if cached_data:
                    # Get the most recent price data
                    latest_price = max(cached_data, key=lambda x: x.time)
                    self.metrics.cache_hits += 1
                    return {
                        'bid': latest_price.bid,
                        'ask': latest_price.ask,
                        'spread': latest_price.spread,
                        'time': latest_price.time
                    }
                else:
                    self.metrics.cache_misses += 1
                    return None
            except Exception as e:
                logger.warning(f"Failed to get price data from Redis: {e}")
        
        # Fallback to in-memory cache
        cache_key = instrument
        if cache_key in self.price_cache:
            self.metrics.cache_hits += 1
            return self.price_cache[cache_key]
        else:
            self.metrics.cache_misses += 1
            return None
    
    async def get_cached_candles(self, instrument: str, granularity: str, time_key: str = None) -> Optional[List[CandleData]]:
        """Get cached candle data for an instrument and granularity"""
        if not self.config.enable_caching:
            return None
        
        # Try Redis cache first
        if self.cache_manager and self.cache_manager.connected:
            try:
                cached_data = await self.cache_manager.get_cached_candle_data(
                    instrument,
                    granularity,
                    from_time=datetime.now() - timedelta(hours=1)
                )
                if cached_data:
                    self.metrics.cache_hits += 1
                    return cached_data
                else:
                    self.metrics.cache_misses += 1
                    return None
            except Exception as e:
                logger.warning(f"Failed to get candle data from Redis: {e}")
        
        # Fallback to in-memory cache
        if time_key and time_key in self.candle_cache:
            self.metrics.cache_hits += 1
            return self.candle_cache[time_key]
        else:
            self.metrics.cache_misses += 1
            return None
    
    async def cache_indicators(self, instrument: str, granularity: str, indicators: Dict[str, Any]) -> bool:
        """Cache calculated indicators for an instrument and granularity"""
        if not self.config.enable_caching or not self.cache_manager or not self.cache_manager.connected:
            return False
        
        try:
            return await self.cache_manager.cache_indicators(
                instrument, 
                granularity, 
                indicators,
                ttl=self.config.cache_ttl
            )
        except Exception as e:
            logger.warning(f"Failed to cache indicators in Redis: {e}")
            return False
    
    async def get_cached_indicators(self, instrument: str, granularity: str) -> Optional[Dict[str, Any]]:
        """Get cached indicators for an instrument and granularity"""
        if not self.config.enable_caching or not self.cache_manager or not self.cache_manager.connected:
            return None
        
        try:
            return await self.cache_manager.get_cached_indicators(instrument, granularity)
        except Exception as e:
            logger.warning(f"Failed to get cached indicators from Redis: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'status': self.status.value,
            'total_requests': self.metrics.total_requests,
            'successful_requests': self.metrics.successful_requests,
            'failed_requests': self.metrics.failed_requests,
            'success_rate': (
                self.metrics.successful_requests / max(self.metrics.total_requests, 1) * 100
            ),
            'avg_latency_ms': round(self.metrics.avg_latency_ms, 2),
            'max_latency_ms': round(self.metrics.max_latency_ms, 2),
            'min_latency_ms': round(self.metrics.min_latency_ms, 2),
            'data_points_ingested': self.metrics.data_points_ingested,
            'cache_hit_rate': (
                self.metrics.cache_hits / max(self.metrics.cache_hits + self.metrics.cache_misses, 1) * 100
            ),
            'queue_sizes': {
                'price_queue': self.price_queue.qsize(),
                'candle_queue': self.candle_queue.qsize()
            },
            'error_counts': dict(self.error_counts),
            'last_update': self.metrics.last_update.isoformat()
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

# Example usage and testing
async def test_ingestion_engine():
    """Test the data ingestion engine"""
    from ..brokers.broker_manager import BrokerManager
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Create ingestion engine
    config = IngestionConfig(
        max_concurrent_requests=5,
        batch_size=50,
        flush_interval=0.5
    )
    
    engine = DataIngestionEngine(broker_manager, config)
    
    # Add callbacks
    def price_callback(price_data):
        print(f"Price: {price_data.instrument} - Bid: {price_data.bid}, Ask: {price_data.ask}")
    
    def candle_callback(candle_data):
        print(f"Candle: {candle_data.time} - O: {candle_data.open}, H: {candle_data.high}, L: {candle_data.low}, C: {candle_data.close}")
    
    engine.add_price_callback(price_callback)
    engine.add_candle_callback(candle_callback)
    
    # Start engine
    await engine.start()
    
    try:
        # Test metrics
        metrics = engine.get_metrics()
        print(f"Engine metrics: {metrics}")
        
        # Test candle ingestion
        await engine.ingest_candles("EUR_USD", "M1", count=5)
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Get final metrics
        final_metrics = engine.get_metrics()
        print(f"Final metrics: {final_metrics}")
        
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_ingestion_engine())
