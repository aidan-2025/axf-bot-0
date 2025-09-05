#!/usr/bin/env python3
"""
Validated Data Ingestion Service
Enhanced service with comprehensive data validation and quality monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from .validated_ingestion_engine import ValidatedDataIngestionEngine, ValidatedIngestionConfig
from .data_processor import DataProcessor
from ..brokers.broker_manager import BrokerManager
from ..cache.redis_cache import RedisCacheManager, CacheConfig
from ..storage.storage_manager import StorageManager, InfluxDBConfig, StorageConfig
from ..validation import ValidationConfig, QualityReport
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
class ValidatedServiceConfig:
    """Enhanced service configuration with validation"""
    instruments: List[str]
    granularities: List[str]
    enable_real_time: bool = True
    enable_historical: bool = True
    historical_lookback_days: int = 7
    processing_enabled: bool = True
    storage_enabled: bool = True
    cache_enabled: bool = True
    influxdb_enabled: bool = True
    
    # Validation configuration
    enable_validation: bool = True
    validation_config: Optional[ValidationConfig] = None
    enable_auto_correction: bool = True
    quarantine_invalid_data: bool = True
    
    # Quality monitoring
    enable_quality_monitoring: bool = True
    quality_report_interval: float = 300.0  # 5 minutes
    quality_threshold: float = 0.8
    
    # Cross-source validation
    enable_cross_source_validation: bool = True
    min_sources_for_validation: int = 2

class ValidatedDataIngestionService:
    """Enhanced data ingestion service with comprehensive validation"""
    
    def __init__(self, 
                 broker_manager: BrokerManager,
                 config: Optional[ValidatedServiceConfig] = None,
                 cache_manager: Optional[RedisCacheManager] = None,
                 storage_manager: Optional[StorageManager] = None):
        """
        Initialize validated data ingestion service
        
        Args:
            broker_manager: Broker manager instance
            config: Service configuration
            cache_manager: Redis cache manager instance
            storage_manager: Storage manager instance
        """
        self.broker_manager = broker_manager
        self.config = config or ValidatedServiceConfig(
            instruments=["EUR_USD", "GBP_USD", "USD_JPY"],
            granularities=["M1", "M5", "M15", "H1"]
        )
        self.cache_manager = cache_manager
        self.storage_manager = storage_manager
        
        # Service state
        self.status = ServiceStatus.STOPPED
        
        # Data processor
        self.data_processor = DataProcessor()
        
        # Validated ingestion engine
        ingestion_config = ValidatedIngestionConfig(
            enable_validation=self.config.enable_validation,
            validation_config=self.config.validation_config,
            enable_auto_correction=self.config.enable_auto_correction,
            quarantine_invalid_data=self.config.quarantine_invalid_data,
            enable_quality_monitoring=self.config.enable_quality_monitoring,
            quality_report_interval=self.config.quality_report_interval,
            quality_threshold=self.config.quality_threshold,
            enable_cross_source_validation=self.config.enable_cross_source_validation,
            min_sources_for_validation=self.config.min_sources_for_validation
        )
        
        self.ingestion_engine = ValidatedDataIngestionEngine(
            broker_manager=broker_manager,
            config=ingestion_config,
            cache_manager=cache_manager
        )
        
        # Service metrics
        self.service_metrics = {
            'start_time': None,
            'total_data_points': 0,
            'valid_data_points': 0,
            'invalid_data_points': 0,
            'corrected_data_points': 0,
            'quarantined_data_points': 0,
            'quality_reports_generated': 0,
            'validation_errors': 0,
            'storage_errors': 0,
            'cache_errors': 0
        }
        
        # Callbacks
        self.service_callbacks: List[Callable] = []
        
        logger.info("ValidatedDataIngestionService initialized with validation: %s", 
                   self.config.enable_validation)
    
    async def start(self) -> None:
        """Start the validated data ingestion service"""
        try:
            self.status = ServiceStatus.STARTING
            logger.info("Starting validated data ingestion service...")
            
            # Initialize cache manager
            if self.config.cache_enabled and self.cache_manager:
                await self._initialize_cache()
            
            # Initialize storage manager
            if self.config.storage_enabled and self.storage_manager:
                await self._initialize_storage()
            
            # Set up data processing callbacks
            self._setup_data_processing_callbacks()
            
            # Set up quality monitoring callbacks
            if self.config.enable_quality_monitoring:
                self._setup_quality_monitoring_callbacks()
            
            # Start ingestion engine
            await self.ingestion_engine.start()
            
            # Start data collection tasks
            if self.config.enable_real_time:
                await self._start_real_time_data_collection()
            
            if self.config.enable_historical:
                await self._start_historical_data_collection()
            
            self.status = ServiceStatus.RUNNING
            self.service_metrics['start_time'] = datetime.utcnow()
            logger.info("Validated data ingestion service started successfully")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"Failed to start validated ingestion service: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the validated data ingestion service"""
        try:
            self.status = ServiceStatus.STOPPING
            logger.info("Stopping validated data ingestion service...")
            
            # Stop ingestion engine
            await self.ingestion_engine.stop()
            
            self.status = ServiceStatus.STOPPED
            logger.info("Validated data ingestion service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping validated ingestion service: {e}")
            self.status = ServiceStatus.ERROR
    
    async def _initialize_cache(self) -> None:
        """Initialize cache manager"""
        try:
            if self.cache_manager and not self.cache_manager.connected:
                await self.cache_manager.connect()
                logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            self.service_metrics['cache_errors'] += 1
    
    async def _initialize_storage(self) -> None:
        """Initialize storage manager"""
        try:
            if self.storage_manager:
                await self.storage_manager.initialize()
                logger.info("Storage manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize storage manager: {e}")
            self.service_metrics['storage_errors'] += 1
    
    def _setup_data_processing_callbacks(self) -> None:
        """Set up data processing callbacks"""
        # Price data processing callback
        async def process_price_data(price_data):
            try:
                if self.config.processing_enabled:
                    await self.data_processor.process_price_data(price_data)
                
                # Store data if storage is enabled
                if self.config.storage_enabled and self.storage_manager:
                    await self.storage_manager.store_price_data(price_data)
                
                # Update service metrics
                self.service_metrics['total_data_points'] += 1
                self.service_metrics['valid_data_points'] += 1
                
            except Exception as e:
                logger.error(f"Error processing price data: {e}")
                self.service_metrics['storage_errors'] += 1
        
        # Candle data processing callback
        async def process_candle_data(candle_data):
            try:
                if self.config.processing_enabled:
                    await self.data_processor.process_candle_data(candle_data)
                
                # Store data if storage is enabled
                if self.config.storage_enabled and self.storage_manager:
                    await self.storage_manager.store_candle_data(candle_data)
                
                # Update service metrics
                self.service_metrics['total_data_points'] += 1
                self.service_metrics['valid_data_points'] += 1
                
            except Exception as e:
                logger.error(f"Error processing candle data: {e}")
                self.service_metrics['storage_errors'] += 1
        
        # Add callbacks to ingestion engine
        self.ingestion_engine.add_price_callback(process_price_data)
        self.ingestion_engine.add_candle_callback(process_candle_data)
    
    def _setup_quality_monitoring_callbacks(self) -> None:
        """Set up quality monitoring callbacks"""
        async def handle_quality_report(quality_report: QualityReport):
            try:
                # Log quality report
                logger.info(f"Quality Report - Overall: {quality_report.overall_quality_score:.2f}, "
                          f"Completeness: {quality_report.completeness_score:.2f}, "
                          f"Accuracy: {quality_report.accuracy_score:.2f}, "
                          f"Latency: {quality_report.latency_score:.2f}")
                
                # Check quality thresholds
                if quality_report.overall_quality_score < self.config.quality_threshold:
                    logger.warning(f"Quality score below threshold: {quality_report.overall_quality_score:.2f}")
                
                # Update service metrics
                self.service_metrics['quality_reports_generated'] += 1
                
                # Call service callbacks
                for callback in self.service_callbacks:
                    try:
                        await callback('quality_report', quality_report)
                    except Exception as e:
                        logger.error(f"Error in service callback: {e}")
                
            except Exception as e:
                logger.error(f"Error handling quality report: {e}")
        
        # Add quality callback to ingestion engine
        self.ingestion_engine.add_quality_callback(handle_quality_report)
    
    async def _start_real_time_data_collection(self) -> None:
        """Start real-time data collection"""
        try:
            logger.info("Starting real-time data collection...")
            
            # Start data collection for each instrument
            for instrument in self.config.instruments:
                await self._start_instrument_data_collection(instrument)
            
            logger.info("Real-time data collection started")
            
        except Exception as e:
            logger.error(f"Error starting real-time data collection: {e}")
            raise
    
    async def _start_historical_data_collection(self) -> None:
        """Start historical data collection"""
        try:
            logger.info("Starting historical data collection...")
            
            # Collect historical data for each instrument and granularity
            for instrument in self.config.instruments:
                for granularity in self.config.granularities:
                    await self._collect_historical_data(instrument, granularity)
            
            logger.info("Historical data collection completed")
            
        except Exception as e:
            logger.error(f"Error in historical data collection: {e}")
    
    async def _start_instrument_data_collection(self, instrument: str) -> None:
        """Start data collection for a specific instrument"""
        try:
            # Start price data collection
            price_task = asyncio.create_task(
                self._collect_real_time_price_data(instrument)
            )
            
            # Start candle data collection for each granularity
            candle_tasks = []
            for granularity in self.config.granularities:
                candle_task = asyncio.create_task(
                    self._collect_real_time_candle_data(instrument, granularity)
                )
                candle_tasks.append(candle_task)
            
            # Wait for tasks to complete (they run indefinitely)
            await asyncio.gather(price_task, *candle_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error collecting data for {instrument}: {e}")
    
    async def _collect_real_time_price_data(self, instrument: str) -> None:
        """Collect real-time price data for an instrument"""
        while self.status == ServiceStatus.RUNNING:
            try:
                # Get price data from broker manager
                price_data = await self.broker_manager.get_price_data(instrument)
                
                if price_data:
                    # Ingest data through validated engine
                    await self.ingestion_engine.ingest_price_data(price_data)
                
                # Small delay between requests
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error collecting price data for {instrument}: {e}")
                await asyncio.sleep(5.0)
    
    async def _collect_real_time_candle_data(self, instrument: str, granularity: str) -> None:
        """Collect real-time candle data for an instrument and granularity"""
        while self.status == ServiceStatus.RUNNING:
            try:
                # Get candle data from broker manager
                candle_data = await self.broker_manager.get_candle_data(instrument, granularity)
                
                if candle_data:
                    # Ingest data through validated engine
                    await self.ingestion_engine.ingest_candle_data(candle_data)
                
                # Delay based on granularity
                delay = self._get_granularity_delay(granularity)
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error collecting candle data for {instrument} {granularity}: {e}")
                await asyncio.sleep(10.0)
    
    async def _collect_historical_data(self, instrument: str, granularity: str) -> None:
        """Collect historical data for an instrument and granularity"""
        try:
            # Calculate start time for historical data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=self.config.historical_lookback_days)
            
            # Get historical data from broker manager
            historical_data = await self.broker_manager.get_historical_data(
                instrument, granularity, start_time, end_time
            )
            
            if historical_data:
                # Process historical data through validated engine
                for candle_data in historical_data:
                    await self.ingestion_engine.ingest_candle_data(candle_data)
                
                logger.info(f"Collected {len(historical_data)} historical data points for {instrument} {granularity}")
            
        except Exception as e:
            logger.error(f"Error collecting historical data for {instrument} {granularity}: {e}")
    
    def _get_granularity_delay(self, granularity: str) -> float:
        """Get delay based on granularity"""
        delays = {
            'M1': 60.0,    # 1 minute
            'M5': 300.0,   # 5 minutes
            'M15': 900.0,  # 15 minutes
            'M30': 1800.0, # 30 minutes
            'H1': 3600.0,  # 1 hour
            'H4': 14400.0, # 4 hours
            'D1': 86400.0  # 1 day
        }
        return delays.get(granularity, 60.0)
    
    # Public methods
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            # Get ingestion engine status
            engine_status = self.ingestion_engine.get_status()
            
            # Get validation stats
            validation_stats = self.ingestion_engine.get_validation_stats()
            
            # Get quality report
            quality_report = await self.ingestion_engine.get_quality_report()
            
            return {
                'service_status': self.status.value,
                'service_metrics': self.service_metrics,
                'engine_status': engine_status,
                'validation_stats': validation_stats,
                'quality_report': {
                    'overall_quality_score': quality_report.overall_quality_score,
                    'completeness_score': quality_report.completeness_score,
                    'accuracy_score': quality_report.accuracy_score,
                    'latency_score': quality_report.latency_score,
                    'consistency_score': quality_report.consistency_score,
                    'anomaly_rate': quality_report.anomaly_rate,
                    'total_data_points': quality_report.total_data_points,
                    'valid_data_points': quality_report.valid_data_points,
                    'invalid_data_points': quality_report.invalid_data_points,
                    'corrected_data_points': quality_report.corrected_data_points
                },
                'config': {
                    'instruments': self.config.instruments,
                    'granularities': self.config.granularities,
                    'validation_enabled': self.config.enable_validation,
                    'auto_correction_enabled': self.config.enable_auto_correction,
                    'quality_monitoring_enabled': self.config.enable_quality_monitoring,
                    'cross_source_validation_enabled': self.config.enable_cross_source_validation
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                'service_status': self.status.value,
                'error': str(e)
            }
    
    def add_service_callback(self, callback: Callable) -> None:
        """Add service callback"""
        self.service_callbacks.append(callback)
    
    async def get_quality_report(self) -> QualityReport:
        """Get current quality report"""
        return await self.ingestion_engine.get_quality_report()
    
    async def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.ingestion_engine.get_validation_stats()

# Example usage and testing
async def test_validated_ingestion_service():
    """Test the validated ingestion service"""
    from ..brokers.broker_manager import BrokerManager
    from ..cache.redis_cache import RedisCacheManager
    from ..storage.storage_manager import StorageManager
    
    # Create components
    broker_manager = BrokerManager()
    cache_manager = RedisCacheManager()
    storage_manager = StorageManager()
    
    # Create service config
    config = ValidatedServiceConfig(
        instruments=["EUR_USD", "GBP_USD"],
        granularities=["M1", "M5"],
        enable_validation=True,
        enable_auto_correction=True,
        enable_quality_monitoring=True
    )
    
    # Create service
    service = ValidatedDataIngestionService(
        broker_manager=broker_manager,
        config=config,
        cache_manager=cache_manager,
        storage_manager=storage_manager
    )
    
    # Add service callback
    async def service_callback(event_type, data):
        print(f"Service event: {event_type}")
        if event_type == 'quality_report':
            print(f"Quality score: {data.overall_quality_score:.2f}")
    
    service.add_service_callback(service_callback)
    
    # Start service
    await service.start()
    
    # Wait for some data collection
    await asyncio.sleep(10.0)
    
    # Get status
    status = await service.get_service_status()
    print(f"Service status: {status['service_status']}")
    print(f"Quality score: {status['quality_report']['overall_quality_score']:.2f}")
    
    # Stop service
    await service.stop()

if __name__ == "__main__":
    asyncio.run(test_validated_ingestion_service())

