#!/usr/bin/env python3
"""
Data Ingestion Service
Main service that orchestrates data ingestion, processing, and storage
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from .ingestion_engine import DataIngestionEngine, IngestionConfig
from .data_processor import DataProcessor
from .data_validator import DataValidator
from ..brokers.broker_manager import BrokerManager
from ..cache.redis_cache import RedisCacheManager, CacheConfig
from ..storage.storage_manager import StorageManager, InfluxDBConfig, StorageConfig
from ..config import CONFIG

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class ServiceConfig:
    """Service configuration"""
    instruments: List[str]
    granularities: List[str]
    enable_real_time: bool = True
    enable_historical: bool = True
    historical_lookback_days: int = 7
    processing_enabled: bool = True
    storage_enabled: bool = True
    cache_enabled: bool = True
    influxdb_enabled: bool = True

class DataIngestionService:
    """Main data ingestion service"""
    
    def __init__(self, 
                 broker_manager: BrokerManager,
                 service_config: Optional[ServiceConfig] = None,
                 ingestion_config: Optional[IngestionConfig] = None):
        """
        Initialize data ingestion service
        
        Args:
            broker_manager: Broker manager instance
            service_config: Service configuration
            ingestion_config: Ingestion engine configuration
        """
        self.broker_manager = broker_manager
        self.service_config = service_config or ServiceConfig(
            instruments=["EUR_USD", "GBP_USD", "USD_JPY"],
            granularities=["M1", "M5", "M15", "H1"]
        )
        self.ingestion_config = ingestion_config or IngestionConfig()
        
        # Service state
        self.status = ServiceStatus.STOPPED
        
        # Components
        self.ingestion_engine: Optional[DataIngestionEngine] = None
        self.data_processor: Optional[DataProcessor] = None
        self.data_validator: Optional[DataValidator] = DataValidator()
        self.cache_manager: Optional[RedisCacheManager] = None
        self.storage_manager: Optional[StorageManager] = None
        
        # Tasks
        self.service_tasks: List[asyncio.Task] = []
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
    
    async def start(self) -> None:
        """Start the data ingestion service"""
        if self.status != ServiceStatus.STOPPED:
            logger.warning("Service is already running or starting")
            return
        
        logger.info("Starting data ingestion service...")
        self.status = ServiceStatus.STARTING
        
        try:
            # Initialize cache manager if enabled
            if self.service_config.cache_enabled:
                cache_config = CacheConfig()
                self.cache_manager = RedisCacheManager(cache_config)
                await self.cache_manager.connect()
                logger.info("Redis cache manager initialized")
            
            # Initialize storage manager if enabled
            if self.service_config.storage_enabled:
                influxdb_config = InfluxDBConfig.from_env()
                storage_config = StorageConfig.from_env()
                self.storage_manager = StorageManager(
                    influxdb_config=influxdb_config,
                    storage_config=storage_config,
                    cache_manager=self.cache_manager
                )
                await self.storage_manager.start()
                logger.info("Storage manager initialized")
            
            # Initialize components
            self.ingestion_engine = DataIngestionEngine(
                self.broker_manager, 
                self.ingestion_config,
                self.cache_manager
            )
            
            if self.service_config.processing_enabled:
                self.data_processor = DataProcessor()
                self._setup_processing_callbacks()
            
            # Start ingestion engine
            await self.ingestion_engine.start()
            
            # Start data ingestion tasks
            if self.service_config.enable_real_time:
                await self._start_real_time_ingestion()
            
            if self.service_config.enable_historical:
                await self._start_historical_ingestion()
            
            # Start monitoring task
            self.service_tasks.append(
                asyncio.create_task(self._monitoring_loop())
            )
            
            self.status = ServiceStatus.RUNNING
            logger.info("Data ingestion service started successfully")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"Failed to start ingestion service: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the data ingestion service"""
        if self.status == ServiceStatus.STOPPED:
            return
        
        logger.info("Stopping data ingestion service...")
        self.status = ServiceStatus.STOPPING
        
        # Cancel all tasks
        for task in self.service_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.service_tasks:
            await asyncio.gather(*self.service_tasks, return_exceptions=True)
        
        # Stop ingestion engine
        if self.ingestion_engine:
            await self.ingestion_engine.stop()
        
        # Stop storage manager
        if self.storage_manager:
            await self.storage_manager.stop()
            logger.info("Storage manager stopped")
        
        # Disconnect cache manager
        if self.cache_manager:
            await self.cache_manager.disconnect()
            logger.info("Redis cache manager disconnected")
        
        self.status = ServiceStatus.STOPPED
        logger.info("Data ingestion service stopped")
    
    def _setup_processing_callbacks(self) -> None:
        """Setup callbacks between ingestion engine and data processor"""
        if not self.data_processor:
            return
        
        # Connect price data processing
        async def process_price_data(price_data):
            # Validate
            if self.data_validator:
                valid = self.data_validator.validate_price(
                    price_data.instrument,
                    price_data.time,
                    price_data.bid,
                    price_data.ask
                )
                if not valid:
                    logger.warning("Rejected invalid price data; skipping storage")
                    return

            await self.data_processor.process_price_data(price_data)
            # Store price data if storage is enabled
            if self.storage_manager:
                await self.storage_manager.store_price_data(price_data, {"broker": "unknown"})
        
        self.ingestion_engine.add_price_callback(process_price_data)
        
        # Connect candle data processing
        async def process_candle_data(candle_data):
            # Validate
            if self.data_validator:
                valid = self.data_validator.validate_candle(
                    getattr(candle_data, 'instrument', 'unknown'),
                    candle_data.time,
                    candle_data.open,
                    candle_data.high,
                    candle_data.low,
                    candle_data.close,
                    candle_data.volume
                )
                if not valid:
                    logger.warning("Rejected invalid candle data; skipping processing/storage")
                    return

            await self.data_processor.process_candle_data(candle_data)
            # Store candle data if storage is enabled
            if self.storage_manager:
                instr = getattr(candle_data, 'instrument', None)
                gran = getattr(candle_data, 'granularity', None)
                if instr and gran:
                    await self.storage_manager.store_candle_data(
                        candle_data, instr, gran, {"broker": "unknown"}
                    )
        
        self.ingestion_engine.add_candle_callback(process_candle_data)
    
    async def _start_real_time_ingestion(self) -> None:
        """Start real-time data ingestion"""
        logger.info("Starting real-time data ingestion...")
        
        try:
            await self.ingestion_engine.ingest_prices(self.service_config.instruments)
            logger.info(f"Real-time ingestion started for {len(self.service_config.instruments)} instruments")
        except Exception as e:
            logger.error(f"Failed to start real-time ingestion: {e}")
            raise
    
    async def _start_historical_ingestion(self) -> None:
        """Start historical data ingestion"""
        logger.info("Starting historical data ingestion...")
        
        for instrument in self.service_config.instruments:
            for granularity in self.service_config.granularities:
                try:
                    # Calculate count based on granularity and lookback days
                    count = self._calculate_historical_count(granularity)
                    
                    await self.ingestion_engine.ingest_candles(
                        instrument, granularity, count
                    )
                    
                    logger.info(f"Historical data queued for {instrument} {granularity}")
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to queue historical data for {instrument} {granularity}: {e}")
    
    def _calculate_historical_count(self, granularity: str) -> int:
        """Calculate number of historical data points needed"""
        minutes_per_period = {
            'M1': 1,
            'M5': 5,
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H4': 240,
            'D1': 1440
        }
        
        period_minutes = minutes_per_period.get(granularity, 1)
        total_minutes = self.service_config.historical_lookback_days * 24 * 60
        count = total_minutes // period_minutes
        
        return min(count, 5000)  # Limit to prevent API overload
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.status == ServiceStatus.RUNNING:
            try:
                await self._perform_health_checks()
                await self._log_metrics()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks"""
        try:
            # Check broker manager
            broker_status = await self.broker_manager.get_status()
            healthy_brokers = broker_status.get('healthy_brokers', 0)
            
            if healthy_brokers == 0:
                logger.warning("No healthy brokers available")
                self.status = ServiceStatus.ERROR
                return
            
            # Check ingestion engine
            if self.ingestion_engine:
                engine_metrics = self.ingestion_engine.get_metrics()
                if engine_metrics['status'] == 'error':
                    logger.warning("Ingestion engine is in error state")
                    self.status = ServiceStatus.ERROR
                    return
            
            # Check data processor
            if self.data_processor:
                processor_status = self.data_processor.get_status()
                if processor_status['status'] == 'error':
                    logger.warning("Data processor is in error state")
                    self.status = ServiceStatus.ERROR
                    return
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _log_metrics(self) -> None:
        """Log current metrics"""
        try:
            if self.ingestion_engine:
                metrics = self.ingestion_engine.get_metrics()
                logger.info(f"Ingestion metrics: {metrics['data_points_ingested']} points, "
                           f"{metrics['avg_latency_ms']}ms avg latency, "
                           f"{metrics['success_rate']:.1f}% success rate")
            
            if self.data_processor:
                status = self.data_processor.get_status()
                logger.info(f"Processor status: {status['instruments_tracked']} instruments, "
                           f"{status['total_price_points']} price points")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def add_data_callback(self, callback: Callable) -> None:
        """Add a callback for processed data"""
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add a callback for errors"""
        self.error_callbacks.append(callback)
    
    async def get_cached_price(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get cached price data for an instrument"""
        # Use storage manager for unified data retrieval
        if self.storage_manager:
            try:
                # Get recent price data (last 5 minutes)
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=5)
                price_data = await self.storage_manager.get_price_data(
                    instrument, start_time, end_time, limit=1, use_cache=True
                )
                if price_data:
                    return price_data[0]
            except Exception as e:
                logger.warning(f"Failed to get price data from storage manager: {e}")
        
        # Fallback to ingestion engine cache
        if self.ingestion_engine:
            return await self.ingestion_engine.get_cached_price(instrument)
        
        return None
    
    async def get_historical_data(self, 
                                instrument: str, 
                                granularity: str,
                                count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical data for an instrument"""
        # Use storage manager for unified data retrieval
        if self.storage_manager:
            try:
                # Calculate time range based on granularity and count
                end_time = datetime.now()
                if count:
                    # Estimate start time based on granularity
                    minutes_per_period = {
                        'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                        'H1': 60, 'H4': 240, 'D1': 1440
                    }
                    period_minutes = minutes_per_period.get(granularity, 1)
                    start_time = end_time - timedelta(minutes=count * period_minutes)
                else:
                    start_time = end_time - timedelta(days=7)  # Default to 7 days
                
                data = await self.storage_manager.get_candle_data(
                    instrument, granularity, start_time, end_time, count, use_cache=True
                )
                if data:
                    logger.info(f"Retrieved {len(data)} historical data points for {instrument} ({granularity})")
                    return data
            except Exception as e:
                logger.warning(f"Failed to get historical data from storage manager: {e}")
        
        # Fallback to broker if no storage or insufficient data
        if not self.ingestion_engine:
            return []
        
        try:
            # Get candles from broker
            from ..brokers.oanda_client import Granularity
            gran_enum = getattr(Granularity, granularity.upper(), Granularity.M1)
            
            candles = await self.broker_manager.get_candles(
                instrument, gran_enum, count
            )
            
            # Store the new data
            if self.storage_manager:
                for candle in candles:
                    await self.storage_manager.store_candle_data(
                        candle, instrument, granularity, {"broker": "unknown"}
                    )
            
            # Convert to dictionary format
            data = []
            for candle in candles:
                data.append({
                    'time': candle.time.isoformat(),
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'complete': candle.complete
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        status = {
            'service_status': self.status.value,
            'instruments': self.service_config.instruments,
            'granularities': self.service_config.granularities,
            'real_time_enabled': self.service_config.enable_real_time,
            'historical_enabled': self.service_config.enable_historical,
            'processing_enabled': self.service_config.processing_enabled,
            'storage_enabled': self.service_config.storage_enabled
        }
        
        if self.ingestion_engine:
            status['ingestion_engine'] = self.ingestion_engine.get_metrics()
        
        if self.data_processor:
            status['data_processor'] = self.data_processor.get_status()
        
        if self.cache_manager:
            status['cache_manager'] = {
                'connected': self.cache_manager.connected,
                'enabled': self.service_config.cache_enabled
            }
        
        if self.storage_manager:
            status['storage_manager'] = {
                'enabled': self.service_config.storage_enabled,
                'influxdb_enabled': self.service_config.influxdb_enabled
            }
        
        return status
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

# Example usage and testing
async def test_ingestion_service():
    """Test the data ingestion service"""
    from ..brokers.broker_manager import BrokerManager
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Create service configuration
    service_config = ServiceConfig(
        instruments=["EUR_USD", "GBP_USD"],
        granularities=["M1", "M5"],
        enable_real_time=True,
        enable_historical=True,
        historical_lookback_days=1
    )
    
    # Create ingestion configuration
    ingestion_config = IngestionConfig(
        max_concurrent_requests=5,
        batch_size=50,
        flush_interval=1.0
    )
    
    # Create service
    service = DataIngestionService(
        broker_manager, 
        service_config, 
        ingestion_config
    )
    
    # Add callbacks
    def data_callback(data):
        print(f"Data received: {type(data)}")
    
    service.add_data_callback(data_callback)
    
    # Start service
    await service.start()
    
    try:
        # Wait a bit
        await asyncio.sleep(5)
        
        # Get status
        status = service.get_service_status()
        print(f"Service status: {status}")
        
        # Test cached price
        price = await service.get_cached_price("EUR_USD")
        print(f"Cached price: {price}")
        
        # Test historical data
        historical = await service.get_historical_data("EUR_USD", "M1", 10)
        print(f"Historical data: {len(historical)} points")
        
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(test_ingestion_service())
