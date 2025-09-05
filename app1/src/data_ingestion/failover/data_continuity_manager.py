#!/usr/bin/env python3
"""
Data Continuity Manager
Handles data buffering, replay, and continuity during outages
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import deque

from ..brokers.broker_manager import PriceData, CandleData
from ..cache.redis_cache import RedisCacheManager
from ..storage.storage_manager import StorageManager
from ..config import CONFIG

logger = logging.getLogger(__name__)

class BufferState(Enum):
    """Buffer state"""
    ACTIVE = "active"
    PAUSED = "paused"
    FLUSHING = "flushing"
    EMPTY = "empty"

class DataType(Enum):
    """Types of data being buffered"""
    PRICE = "price"
    CANDLE = "candle"
    INDICATOR = "indicator"

@dataclass
class BufferedData:
    """Buffered data item"""
    data: Any
    timestamp: datetime
    instrument: str
    data_type: DataType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BufferConfig:
    """Configuration for data buffering"""
    max_buffer_size: int = 10000
    max_buffer_age_seconds: int = 3600  # 1 hour
    flush_interval_seconds: int = 30
    replay_batch_size: int = 100
    enable_persistence: bool = True
    persistence_ttl_seconds: int = 86400  # 24 hours

@dataclass
class ContinuityMetrics:
    """Metrics for data continuity"""
    total_buffered: int = 0
    total_replayed: int = 0
    total_dropped: int = 0
    avg_buffer_size: float = 0.0
    max_buffer_size: int = 0
    data_gaps_detected: int = 0
    replay_operations: int = 0
    last_flush: Optional[datetime] = None
    last_replay: Optional[datetime] = None

class DataContinuityManager:
    """Manages data continuity during outages and failovers"""
    
    def __init__(self, 
                 cache_manager: Optional[RedisCacheManager] = None,
                 storage_manager: Optional[StorageManager] = None,
                 config: Optional[BufferConfig] = None):
        """
        Initialize data continuity manager
        
        Args:
            cache_manager: Redis cache manager for persistence
            storage_manager: Storage manager for data replay
            config: Buffer configuration
        """
        self.cache_manager = cache_manager
        self.storage_manager = storage_manager
        self.config = config or BufferConfig()
        
        # Data buffers by instrument and type
        self.buffers: Dict[str, Dict[DataType, Deque[BufferedData]]] = {}
        self.buffer_states: Dict[str, BufferState] = {}
        
        # Metrics
        self.metrics = ContinuityMetrics()
        
        # Monitoring tasks
        self.flush_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
    
    async def start(self) -> None:
        """Start the data continuity manager"""
        if self.is_running:
            return
        
        logger.info("Starting data continuity manager...")
        self.is_running = True
        
        # Start monitoring tasks
        self.flush_task = asyncio.create_task(self._flush_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Data continuity manager started successfully")
    
    async def stop(self) -> None:
        """Stop the data continuity manager"""
        if not self.is_running:
            return
        
        logger.info("Stopping data continuity manager...")
        self.is_running = False
        
        # Cancel monitoring tasks
        if self.flush_task:
            self.flush_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self.flush_task, self.monitoring_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flush all remaining data
        await self._flush_all_buffers()
        
        logger.info("Data continuity manager stopped")
    
    async def buffer_data(self, 
                         data: Any,
                         instrument: str,
                         data_type: DataType,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Buffer data for continuity during outages"""
        try:
            # Get or create buffer for instrument
            if instrument not in self.buffers:
                self.buffers[instrument] = {}
                self.buffer_states[instrument] = BufferState.ACTIVE
            
            if data_type not in self.buffers[instrument]:
                self.buffers[instrument][data_type] = deque(maxlen=self.config.max_buffer_size)
            
            # Create buffered data item
            buffered_item = BufferedData(
                data=data,
                timestamp=datetime.now(),
                instrument=instrument,
                data_type=data_type,
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.buffers[instrument][data_type].append(buffered_item)
            
            # Update metrics
            self.metrics.total_buffered += 1
            current_size = len(self.buffers[instrument][data_type])
            self.metrics.max_buffer_size = max(self.metrics.max_buffer_size, current_size)
            
            # Persist to cache if enabled
            if self.config.enable_persistence and self.cache_manager:
                await self._persist_buffer(instrument, data_type)
            
            logger.debug(f"Buffered {data_type.value} data for {instrument}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to buffer data for {instrument}: {e}")
            return False
    
    async def get_buffered_data(self, 
                               instrument: str,
                               data_type: DataType,
                               from_time: Optional[datetime] = None,
                               to_time: Optional[datetime] = None) -> List[BufferedData]:
        """Get buffered data for an instrument and type"""
        if instrument not in self.buffers or data_type not in self.buffers[instrument]:
            return []
        
        buffer = self.buffers[instrument][data_type]
        
        # Filter by time range if specified
        if from_time or to_time:
            filtered_data = []
            for item in buffer:
                if from_time and item.timestamp < from_time:
                    continue
                if to_time and item.timestamp > to_time:
                    continue
                filtered_data.append(item)
            return filtered_data
        
        return list(buffer)
    
    async def replay_data(self, 
                         instrument: str,
                         data_type: DataType,
                         callback: Callable,
                         from_time: Optional[datetime] = None,
                         to_time: Optional[datetime] = None) -> int:
        """Replay buffered data through a callback"""
        try:
            buffered_data = await self.get_buffered_data(instrument, data_type, from_time, to_time)
            
            if not buffered_data:
                logger.info(f"No buffered data to replay for {instrument} {data_type.value}")
                return 0
            
            # Sort by timestamp
            buffered_data.sort(key=lambda x: x.timestamp)
            
            # Replay in batches
            replayed_count = 0
            for i in range(0, len(buffered_data), self.config.replay_batch_size):
                batch = buffered_data[i:i + self.config.replay_batch_size]
                
                for item in batch:
                    try:
                        await callback(item.data)
                        replayed_count += 1
                    except Exception as e:
                        logger.error(f"Error replaying data for {instrument}: {e}")
                
                # Small delay between batches
                await asyncio.sleep(0.01)
            
            # Update metrics
            self.metrics.total_replayed += replayed_count
            self.metrics.replay_operations += 1
            self.metrics.last_replay = datetime.now()
            
            logger.info(f"Replayed {replayed_count} {data_type.value} data points for {instrument}")
            return replayed_count
            
        except Exception as e:
            logger.error(f"Failed to replay data for {instrument}: {e}")
            return 0
    
    async def clear_buffer(self, 
                          instrument: str,
                          data_type: Optional[DataType] = None) -> int:
        """Clear buffered data for an instrument"""
        try:
            if instrument not in self.buffers:
                return 0
            
            cleared_count = 0
            
            if data_type:
                # Clear specific data type
                if data_type in self.buffers[instrument]:
                    cleared_count = len(self.buffers[instrument][data_type])
                    self.buffers[instrument][data_type].clear()
            else:
                # Clear all data types for instrument
                for dt, buffer in self.buffers[instrument].items():
                    cleared_count += len(buffer)
                    buffer.clear()
            
            logger.info(f"Cleared {cleared_count} buffered data points for {instrument}")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear buffer for {instrument}: {e}")
            return 0
    
    async def detect_data_gaps(self, 
                              instrument: str,
                              data_type: DataType,
                              expected_interval_seconds: int = 60) -> List[Dict[str, Any]]:
        """Detect data gaps in buffered data"""
        try:
            buffered_data = await self.get_buffered_data(instrument, data_type)
            
            if len(buffered_data) < 2:
                return []
            
            # Sort by timestamp
            buffered_data.sort(key=lambda x: x.timestamp)
            
            gaps = []
            for i in range(1, len(buffered_data)):
                prev_time = buffered_data[i-1].timestamp
                curr_time = buffered_data[i].timestamp
                gap_duration = (curr_time - prev_time).total_seconds()
                
                if gap_duration > expected_interval_seconds * 2:  # Allow some tolerance
                    gap = {
                        'instrument': instrument,
                        'data_type': data_type.value,
                        'start_time': prev_time.isoformat(),
                        'end_time': curr_time.isoformat(),
                        'duration_seconds': gap_duration,
                        'expected_interval': expected_interval_seconds
                    }
                    gaps.append(gap)
                    
                    self.metrics.data_gaps_detected += 1
                    logger.warning(f"Data gap detected for {instrument} {data_type.value}: {gap_duration:.1f}s")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to detect data gaps for {instrument}: {e}")
            return []
    
    async def _flush_loop(self) -> None:
        """Periodic flush loop"""
        while self.is_running:
            try:
                await self._flush_all_buffers()
                await asyncio.sleep(self.config.flush_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(5)
    
    async def _monitoring_loop(self) -> None:
        """Monitoring loop for buffer health"""
        while self.is_running:
            try:
                await self._monitor_buffer_health()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_buffer_health(self) -> None:
        """Monitor buffer health and clean up old data"""
        current_time = datetime.now()
        max_age = timedelta(seconds=self.config.max_buffer_age_seconds)
        
        for instrument, data_types in self.buffers.items():
            for data_type, buffer in data_types.items():
                # Remove old data
                while buffer and (current_time - buffer[0].timestamp) > max_age:
                    old_item = buffer.popleft()
                    self.metrics.total_dropped += 1
                    logger.debug(f"Dropped old {data_type.value} data for {instrument}")
                
                # Check for oversized buffers
                if len(buffer) > self.config.max_buffer_size * 0.9:
                    logger.warning(f"Buffer for {instrument} {data_type.value} is {len(buffer)} items (90% full)")
    
    async def _flush_all_buffers(self) -> None:
        """Flush all buffers to storage"""
        try:
            total_flushed = 0
            
            for instrument, data_types in self.buffers.items():
                for data_type, buffer in data_types.items():
                    if not buffer:
                        continue
                    
                    # Flush to storage if available
                    if self.storage_manager:
                        await self._flush_buffer_to_storage(instrument, data_type, buffer)
                    
                    total_flushed += len(buffer)
            
            if total_flushed > 0:
                self.metrics.last_flush = datetime.now()
                logger.debug(f"Flushed {total_flushed} buffered data points")
                
        except Exception as e:
            logger.error(f"Error flushing buffers: {e}")
    
    async def _flush_buffer_to_storage(self, 
                                     instrument: str,
                                     data_type: DataType,
                                     buffer: Deque[BufferedData]) -> None:
        """Flush a specific buffer to storage"""
        try:
            for item in buffer:
                if data_type == DataType.PRICE and isinstance(item.data, PriceData):
                    await self.storage_manager.store_price_data(item.data, item.metadata)
                elif data_type == DataType.CANDLE and isinstance(item.data, CandleData):
                    granularity = item.metadata.get('granularity', 'M1')
                    await self.storage_manager.store_candle_data(
                        item.data, instrument, granularity, item.metadata
                    )
                # Add other data types as needed
                
        except Exception as e:
            logger.error(f"Failed to flush buffer to storage for {instrument} {data_type.value}: {e}")
    
    async def _persist_buffer(self, instrument: str, data_type: DataType) -> None:
        """Persist buffer to cache for durability"""
        if not self.cache_manager or not self.cache_manager.connected:
            return
        
        try:
            buffer = self.buffers[instrument][data_type]
            if not buffer:
                return
            
            # Serialize buffer data
            buffer_data = []
            for item in buffer:
                buffer_data.append({
                    'data': item.data.__dict__ if hasattr(item.data, '__dict__') else item.data,
                    'timestamp': item.timestamp.isoformat(),
                    'instrument': item.instrument,
                    'data_type': item.data_type.value,
                    'metadata': item.metadata
                })
            
            # Store in cache
            key = f"buffer:{instrument}:{data_type.value}"
            await self.cache_manager.redis.setex(
                key, 
                self.config.persistence_ttl_seconds,
                json.dumps(buffer_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to persist buffer for {instrument} {data_type.value}: {e}")
    
    async def _load_buffer_from_cache(self, instrument: str, data_type: DataType) -> bool:
        """Load buffer from cache on startup"""
        if not self.cache_manager or not self.cache_manager.connected:
            return False
        
        try:
            key = f"buffer:{instrument}:{data_type.value}"
            cached_data = await self.cache_manager.redis.get(key)
            
            if not cached_data:
                return False
            
            buffer_data = json.loads(cached_data)
            
            # Recreate buffer
            if instrument not in self.buffers:
                self.buffers[instrument] = {}
                self.buffer_states[instrument] = BufferState.ACTIVE
            
            if data_type not in self.buffers[instrument]:
                self.buffers[instrument][data_type] = deque(maxlen=self.config.max_buffer_size)
            
            buffer = self.buffers[instrument][data_type]
            
            for item_data in buffer_data:
                # Recreate BufferedData object
                buffered_item = BufferedData(
                    data=item_data['data'],
                    timestamp=datetime.fromisoformat(item_data['timestamp']),
                    instrument=item_data['instrument'],
                    data_type=DataType(item_data['data_type']),
                    metadata=item_data['metadata']
                )
                buffer.append(buffered_item)
            
            logger.info(f"Loaded {len(buffer)} buffered {data_type.value} data points for {instrument}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load buffer from cache for {instrument} {data_type.value}: {e}")
            return False
    
    def add_event_callback(self, callback: Callable) -> None:
        """Add a callback for continuity events"""
        self.event_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get continuity metrics"""
        # Calculate average buffer size
        total_buffers = 0
        total_size = 0
        for data_types in self.buffers.values():
            for buffer in data_types.values():
                total_buffers += 1
                total_size += len(buffer)
        
        self.metrics.avg_buffer_size = total_size / max(total_buffers, 1)
        
        return {
            'total_buffered': self.metrics.total_buffered,
            'total_replayed': self.metrics.total_replayed,
            'total_dropped': self.metrics.total_dropped,
            'avg_buffer_size': round(self.metrics.avg_buffer_size, 2),
            'max_buffer_size': self.metrics.max_buffer_size,
            'data_gaps_detected': self.metrics.data_gaps_detected,
            'replay_operations': self.metrics.replay_operations,
            'last_flush': self.metrics.last_flush.isoformat() if self.metrics.last_flush else None,
            'last_replay': self.metrics.last_replay.isoformat() if self.metrics.last_replay else None,
            'buffer_states': {
                instrument: state.value for instrument, state in self.buffer_states.items()
            },
            'buffer_sizes': {
                instrument: {
                    data_type.value: len(buffer) 
                    for data_type, buffer in data_types.items()
                }
                for instrument, data_types in self.buffers.items()
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

# Example usage and testing
async def test_data_continuity_manager():
    """Test the data continuity manager"""
    from ..brokers.broker_manager import PriceData, CandleData
    
    # Create continuity manager
    continuity_manager = DataContinuityManager()
    
    # Start manager
    await continuity_manager.start()
    
    try:
        # Test buffering price data
        price_data = PriceData(
            instrument="EUR_USD",
            time=datetime.now(),
            bid=1.1000,
            ask=1.1002,
            spread=0.0002
        )
        
        await continuity_manager.buffer_data(
            price_data, "EUR_USD", DataType.PRICE, {"broker": "test"}
        )
        
        # Test buffering candle data
        candle_data = CandleData(
            time=datetime.now(),
            open=1.1000,
            high=1.1005,
            low=1.0995,
            close=1.1002,
            volume=1000,
            complete=True
        )
        
        await continuity_manager.buffer_data(
            candle_data, "EUR_USD", DataType.CANDLE, {"granularity": "M1", "broker": "test"}
        )
        
        # Test getting buffered data
        buffered_prices = await continuity_manager.get_buffered_data("EUR_USD", DataType.PRICE)
        print(f"Buffered {len(buffered_prices)} price data points")
        
        # Test replay
        replayed_count = await continuity_manager.replay_data(
            "EUR_USD", DataType.PRICE, lambda data: print(f"Replayed: {data}")
        )
        print(f"Replayed {replayed_count} data points")
        
        # Test metrics
        metrics = continuity_manager.get_metrics()
        print(f"Continuity metrics: {metrics}")
        
    finally:
        await continuity_manager.stop()

if __name__ == "__main__":
    asyncio.run(test_data_continuity_manager())

