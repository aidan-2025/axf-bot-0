"""
Real-time data integration module for strategy generation
"""

from typing import Dict, List, Any, Optional, Callable
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from ..core.strategy_template import Signal
from .advanced_signal_processor import AdvancedSignalProcessor, DataStreamConfig
from .advanced_feature_extractor import AdvancedFeatureExtractor


class DataSource(Enum):
    """Data source types"""
    MARKET_DATA = "market_data"
    NEWS = "news"
    SENTIMENT = "sentiment"
    ECONOMIC_EVENTS = "economic_events"
    TECHNICAL_INDICATORS = "technical_indicators"


@dataclass
class DataStream:
    """Data stream configuration"""
    source: DataSource
    symbol: str
    endpoint: str
    update_frequency: int  # milliseconds
    is_active: bool = True
    last_update: Optional[datetime] = None
    error_count: int = 0
    max_errors: int = 5


class RealTimeIntegration:
    """
    Real-time data integration system for strategy generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.signal_processor = AdvancedSignalProcessor(self.config.get('signal_processing', {}))
        self.feature_extractor = AdvancedFeatureExtractor(self.config.get('feature_extraction', {}))
        
        # Data streams
        self.data_streams: Dict[str, DataStream] = {}
        self.data_buffers: Dict[str, List[Dict[str, Any]]] = {}
        
        # Real-time processing
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        self.callbacks: List[Callable] = []
        
        # Performance monitoring
        self.integration_stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "average_latency_ms": 0.0,
            "active_streams": 0
        }
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def add_data_stream(self, stream_id: str, stream: DataStream):
        """Add a data stream for real-time processing"""
        self.data_streams[stream_id] = stream
        self.data_buffers[stream_id] = []
        
        # Configure signal processor for this stream
        if stream.source == DataSource.MARKET_DATA:
            config = DataStreamConfig(
                symbol=stream.symbol,
                data_types=['ohlcv', 'volume'],
                timeframes=['M1', 'M5', 'H1'],
                buffer_size=1000,
                quality_threshold=0.8
            )
            self.signal_processor.configure_stream(stream.symbol, config)
        
        self.logger.info(f"Added data stream: {stream_id} for {stream.symbol}")
    
    def add_callback(self, callback: Callable):
        """Add a callback function for processed data"""
        self.callbacks.append(callback)
        self.logger.info(f"Added callback: {callback.__name__}")
    
    async def start_real_time_processing(self):
        """Start real-time data processing"""
        if self.is_running:
            self.logger.warning("Real-time processing already running")
            return
        
        self.is_running = True
        self.logger.info("Starting real-time data processing")
        
        # Start processing tasks for each stream
        for stream_id, stream in self.data_streams.items():
            if stream.is_active:
                task = asyncio.create_task(self._process_stream(stream_id, stream))
                self.processing_tasks.append(task)
        
        # Start data integration task
        integration_task = asyncio.create_task(self._integrate_data())
        self.processing_tasks.append(integration_task)
        
        self.integration_stats["active_streams"] = len([s for s in self.data_streams.values() if s.is_active])
    
    async def stop_real_time_processing(self):
        """Stop real-time data processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping real-time data processing")
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks.clear()
        
        self.integration_stats["active_streams"] = 0
    
    async def _process_stream(self, stream_id: str, stream: DataStream):
        """Process a single data stream"""
        while self.is_running and stream.is_active:
            try:
                start_time = datetime.now()
                
                # Fetch data from stream
                data = await self._fetch_stream_data(stream)
                
                if data:
                    # Process data through signal processor
                    processed_data = await self.signal_processor.process_market_data_async(data)
                    
                    # Add to buffer
                    self.data_buffers[stream_id].append({
                        "timestamp": datetime.now(),
                        "data": processed_data,
                        "stream_id": stream_id,
                        "source": stream.source.value
                    })
                    
                    # Keep buffer size manageable
                    if len(self.data_buffers[stream_id]) > 1000:
                        self.data_buffers[stream_id] = self.data_buffers[stream_id][-500:]
                    
                    # Update statistics
                    stream.last_update = datetime.now()
                    stream.error_count = 0
                    self.integration_stats["successful_updates"] += 1
                    
                    # Calculate latency
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_average_latency(latency)
                
                else:
                    stream.error_count += 1
                    self.integration_stats["failed_updates"] += 1
                    
                    if stream.error_count >= stream.max_errors:
                        self.logger.error(f"Stream {stream_id} exceeded max errors, deactivating")
                        stream.is_active = False
                
                self.integration_stats["total_updates"] += 1
                
                # Wait for next update
                await asyncio.sleep(stream.update_frequency / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error processing stream {stream_id}: {e}")
                stream.error_count += 1
                self.integration_stats["failed_updates"] += 1
                
                if stream.error_count >= stream.max_errors:
                    stream.is_active = False
                
                await asyncio.sleep(1.0)  # Wait before retry
    
    async def _fetch_stream_data(self, stream: DataStream) -> Optional[Dict[str, Any]]:
        """Fetch data from a stream endpoint"""
        try:
            if stream.source == DataSource.MARKET_DATA:
                return await self._fetch_market_data(stream)
            elif stream.source == DataSource.NEWS:
                return await self._fetch_news_data(stream)
            elif stream.source == DataSource.SENTIMENT:
                return await self._fetch_sentiment_data(stream)
            elif stream.source == DataSource.ECONOMIC_EVENTS:
                return await self._fetch_economic_events(stream)
            elif stream.source == DataSource.TECHNICAL_INDICATORS:
                return await self._fetch_technical_indicators(stream)
            else:
                self.logger.warning(f"Unknown data source: {stream.source}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching data from {stream.endpoint}: {e}")
            return None
    
    async def _fetch_market_data(self, stream: DataStream) -> Optional[Dict[str, Any]]:
        """Fetch market data from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(stream.endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "ohlcv": {
                                "open": data.get("open", []),
                                "high": data.get("high", []),
                                "low": data.get("low", []),
                                "close": data.get("close", []),
                                "volume": data.get("volume", []),
                                "timestamp": data.get("timestamp", [])
                            }
                        }
                    else:
                        self.logger.warning(f"Market data API returned status {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return None
    
    async def _fetch_news_data(self, stream: DataStream) -> Optional[Dict[str, Any]]:
        """Fetch news data from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(stream.endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "sentiment": {
                                "news": data.get("news", [])
                            }
                        }
                    else:
                        self.logger.warning(f"News API returned status {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching news data: {e}")
            return None
    
    async def _fetch_sentiment_data(self, stream: DataStream) -> Optional[Dict[str, Any]]:
        """Fetch sentiment data from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(stream.endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "sentiment": {
                                "social": data.get("social", []),
                                "news": data.get("news", [])
                            }
                        }
                    else:
                        self.logger.warning(f"Sentiment API returned status {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching sentiment data: {e}")
            return None
    
    async def _fetch_economic_events(self, stream: DataStream) -> Optional[Dict[str, Any]]:
        """Fetch economic events from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(stream.endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "economic_events": data.get("events", [])
                        }
                    else:
                        self.logger.warning(f"Economic events API returned status {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching economic events: {e}")
            return None
    
    async def _fetch_technical_indicators(self, stream: DataStream) -> Optional[Dict[str, Any]]:
        """Fetch technical indicators from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(stream.endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "indicators": data
                        }
                    else:
                        self.logger.warning(f"Technical indicators API returned status {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error fetching technical indicators: {e}")
            return None
    
    async def _integrate_data(self):
        """Integrate data from all streams and extract features"""
        while self.is_running:
            try:
                # Collect data from all active streams
                integrated_data = {}
                
                for stream_id, buffer in self.data_buffers.items():
                    if buffer and self.data_streams[stream_id].is_active:
                        # Get latest data point
                        latest_data = buffer[-1]["data"]
                        
                        # Merge data based on source
                        source = self.data_streams[stream_id].source
                        if source == DataSource.MARKET_DATA:
                            integrated_data.update(latest_data)
                        elif source == DataSource.NEWS:
                            if "sentiment" not in integrated_data:
                                integrated_data["sentiment"] = {}
                            integrated_data["sentiment"].update(latest_data.get("sentiment", {}))
                        elif source == DataSource.SENTIMENT:
                            if "sentiment" not in integrated_data:
                                integrated_data["sentiment"] = {}
                            integrated_data["sentiment"].update(latest_data.get("sentiment", {}))
                        elif source == DataSource.ECONOMIC_EVENTS:
                            integrated_data.update(latest_data)
                        elif source == DataSource.TECHNICAL_INDICATORS:
                            integrated_data.update(latest_data)
                
                if integrated_data:
                    # Extract features from integrated data
                    features = await self.feature_extractor.extract_features_async(integrated_data)
                    
                    # Add features to integrated data
                    integrated_data["features"] = features
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(integrated_data)
                            else:
                                callback(integrated_data)
                        except Exception as e:
                            self.logger.error(f"Error in callback {callback.__name__}: {e}")
                
                # Wait before next integration cycle
                await asyncio.sleep(1.0)  # 1 second integration cycle
                
            except Exception as e:
                self.logger.error(f"Error in data integration: {e}")
                await asyncio.sleep(1.0)
    
    def _update_average_latency(self, new_latency: float):
        """Update average latency using exponential moving average"""
        current_avg = self.integration_stats["average_latency_ms"]
        if current_avg == 0:
            self.integration_stats["average_latency_ms"] = new_latency
        else:
            # Exponential moving average with alpha = 0.1
            self.integration_stats["average_latency_ms"] = 0.9 * current_avg + 0.1 * new_latency
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            **self.integration_stats,
            "signal_processor_stats": self.signal_processor.get_processing_stats(),
            "feature_extractor_stats": self.feature_extractor.get_extraction_stats(),
            "active_streams": len([s for s in self.data_streams.values() if s.is_active]),
            "total_streams": len(self.data_streams)
        }
    
    def get_data_buffer_sizes(self) -> Dict[str, int]:
        """Get current data buffer sizes"""
        return {stream_id: len(buffer) for stream_id, buffer in self.data_buffers.items()}
    
    def clear_data_buffers(self):
        """Clear all data buffers"""
        for buffer in self.data_buffers.values():
            buffer.clear()
        self.logger.info("All data buffers cleared")
    
    def reset_stats(self):
        """Reset all statistics"""
        self.integration_stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "average_latency_ms": 0.0,
            "active_streams": 0
        }
        self.signal_processor.reset_stats()
        self.feature_extractor.reset_stats()
        self.logger.info("All statistics reset")

