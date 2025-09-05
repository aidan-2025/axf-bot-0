#!/usr/bin/env python3
"""
Validated Data Ingestion Engine
Enhanced ingestion engine with comprehensive data validation and quality checks
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict, deque

from ..brokers.broker_manager import BrokerManager, PriceData, CandleData
from ..cache.redis_cache import RedisCacheManager
from ..config import CONFIG
from ..validation import (
    ValidationEngine, ValidationConfig, ValidationResult,
    QualityReport, CorrectionResult
)

logger = logging.getLogger(__name__)

class IngestionStatus(Enum):
    """Ingestion engine status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class ValidatedIngestionMetrics:
    """Enhanced metrics for validated data ingestion"""
    # Basic ingestion metrics
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
    
    # Validation metrics
    total_validations: int = 0
    valid_data_points: int = 0
    invalid_data_points: int = 0
    corrected_data_points: int = 0
    quarantined_data_points: int = 0
    avg_validation_time_ms: float = 0.0
    validation_errors: int = 0
    validation_warnings: int = 0
    
    # Quality metrics
    avg_quality_score: float = 0.0
    quality_threshold_breaches: int = 0
    anomaly_detections: int = 0
    cross_source_disagreements: int = 0

@dataclass
class ValidatedIngestionConfig:
    """Configuration for validated data ingestion"""
    # Basic ingestion config
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    flush_interval: float = 1.0
    health_check_interval: float = 30.0
    max_queue_size: int = 10000
    enable_caching: bool = True
    cache_ttl: float = 60.0
    
    # Validation config
    enable_validation: bool = True
    validation_config: Optional[ValidationConfig] = None
    validation_timeout_ms: int = 50
    enable_auto_correction: bool = True
    quarantine_invalid_data: bool = True
    
    # Quality monitoring
    enable_quality_monitoring: bool = True
    quality_report_interval: float = 300.0  # 5 minutes
    quality_threshold: float = 0.8
    
    # Cross-source validation
    enable_cross_source_validation: bool = True
    min_sources_for_validation: int = 2

class ValidatedDataIngestionEngine:
    """Enhanced data ingestion engine with comprehensive validation"""
    
    def __init__(self, 
                 broker_manager: BrokerManager,
                 config: Optional[ValidatedIngestionConfig] = None,
                 cache_manager: Optional[RedisCacheManager] = None):
        """
        Initialize validated data ingestion engine
        
        Args:
            broker_manager: Broker manager instance
            config: Ingestion configuration
            cache_manager: Redis cache manager instance
        """
        self.broker_manager = broker_manager
        self.config = config or ValidatedIngestionConfig()
        self.cache_manager = cache_manager
        
        # Engine state
        self.status = IngestionStatus.STOPPED
        self.metrics = ValidatedIngestionMetrics()
        
        # Task management
        self.tasks: Set[asyncio.Task] = set()
        self.ingestion_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.quality_monitoring_task: Optional[asyncio.Task] = None
        
        # Data queues
        self.price_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.candle_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.quarantine_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Callbacks
        self.price_callbacks: List[Callable] = []
        self.candle_callbacks: List[Callable] = []
        self.validation_callbacks: List[Callable] = []
        self.quality_callbacks: List[Callable] = []
        
        # Validation engine
        self.validation_engine = ValidationEngine(self.config.validation_config)
        
        # Data history for cross-source validation
        self.price_history: Dict[str, List[PriceData]] = defaultdict(list)
        self.candle_history: Dict[str, List[CandleData]] = defaultdict(list)
        self.max_history_size = 1000
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("ValidatedDataIngestionEngine initialized with validation: %s", 
                   self.config.enable_validation)
    
    async def start(self) -> None:
        """Start the validated ingestion engine"""
        try:
            self.status = IngestionStatus.STARTING
            logger.info("Starting validated data ingestion engine...")
            
            # Start main ingestion task
            self.ingestion_task = asyncio.create_task(self._ingestion_loop())
            self.tasks.add(self.ingestion_task)
            
            # Start health check task
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.tasks.add(self.health_check_task)
            
            # Start quality monitoring task
            if self.config.enable_quality_monitoring:
                self.quality_monitoring_task = asyncio.create_task(self._quality_monitoring_loop())
                self.tasks.add(self.quality_monitoring_task)
            
            self.status = IngestionStatus.RUNNING
            logger.info("Validated data ingestion engine started successfully")
            
        except Exception as e:
            self.status = IngestionStatus.ERROR
            logger.error(f"Failed to start validated ingestion engine: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the validated ingestion engine"""
        try:
            self.status = IngestionStatus.STOPPING
            logger.info("Stopping validated data ingestion engine...")
            
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self.tasks.clear()
            self.status = IngestionStatus.STOPPED
            logger.info("Validated data ingestion engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping validated ingestion engine: {e}")
            self.status = IngestionStatus.ERROR
    
    async def _ingestion_loop(self) -> None:
        """Main ingestion loop with validation"""
        while self.status == IngestionStatus.RUNNING:
            try:
                # Process price data
                await self._process_price_queue()
                
                # Process candle data
                await self._process_candle_queue()
                
                # Process quarantined data
                await self._process_quarantine_queue()
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in ingestion loop: {e}")
                self.error_counts[type(e).__name__] += 1
                await asyncio.sleep(1.0)
    
    async def _process_price_queue(self) -> None:
        """Process price data queue with validation"""
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
                await self._process_validated_price_batch(batch)
                
        except Exception as e:
            logger.error(f"Error processing price queue: {e}")
    
    async def _process_candle_queue(self) -> None:
        """Process candle data queue with validation"""
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
                await self._process_validated_candle_batch(batch)
                
        except Exception as e:
            logger.error(f"Error processing candle queue: {e}")
    
    async def _process_quarantine_queue(self) -> None:
        """Process quarantined data for review"""
        try:
            # Process quarantined data (simplified for now)
            quarantined_count = 0
            while not self.quarantine_queue.empty() and quarantined_count < 10:
                try:
                    quarantined_data = self.quarantine_queue.get_nowait()
                    logger.warning(f"Processing quarantined data: {quarantined_data}")
                    quarantined_count += 1
                except asyncio.QueueEmpty:
                    break
                    
        except Exception as e:
            logger.error(f"Error processing quarantine queue: {e}")
    
    async def _process_validated_price_batch(self, batch: List[PriceData]) -> None:
        """Process a batch of price data with validation"""
        start_time = time.time()
        
        try:
            validated_batch = []
            cross_source_data = []
            
            for price_data in batch:
                # Get previous data for validation context
                previous_data = self._get_previous_price_data(price_data.instrument)
                
                # Get cross-source data if available
                if self.config.enable_cross_source_validation:
                    cross_source_data = self._get_cross_source_price_data(price_data)
                
                # Validate data
                if self.config.enable_validation:
                    validation_result = await self.validation_engine.validate_price_data(
                        price_data, previous_data, cross_source_data
                    )
                    
                    # Update validation metrics
                    self._update_validation_metrics(validation_result)
                    
                    # Handle validation result
                    if validation_result.is_valid:
                        validated_batch.append(price_data)
                        self._update_price_history(price_data)
                    elif self.config.enable_auto_correction and validation_result.corrections:
                        # Use corrected data
                        corrected_data = validation_result.corrections[0].corrected_data
                        if corrected_data:
                            validated_batch.append(corrected_data)
                            self._update_price_history(corrected_data)
                    elif self.config.quarantine_invalid_data:
                        # Quarantine invalid data
                        await self.quarantine_queue.put(price_data)
                    else:
                        # Skip invalid data
                        logger.warning(f"Skipping invalid price data: {price_data.instrument}")
                        continue
                else:
                    # No validation - use data as-is
                    validated_batch.append(price_data)
                    self._update_price_history(price_data)
            
            # Process validated batch
            if validated_batch:
                await self._process_validated_price_data(validated_batch)
            
        except Exception as e:
            logger.error(f"Error processing validated price batch: {e}")
            self.metrics.failed_requests += 1
            self.error_counts[type(e).__name__] += 1
        
        finally:
            # Update latency metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_metrics(latency_ms)
    
    async def _process_validated_candle_batch(self, batch: List[CandleData]) -> None:
        """Process a batch of candle data with validation"""
        start_time = time.time()
        
        try:
            validated_batch = []
            cross_source_data = []
            
            for candle_data in batch:
                # Get previous data for validation context
                previous_data = self._get_previous_candle_data(candle_data.instrument)
                
                # Get cross-source data if available
                if self.config.enable_cross_source_validation:
                    cross_source_data = self._get_cross_source_candle_data(candle_data)
                
                # Validate data
                if self.config.enable_validation:
                    validation_result = await self.validation_engine.validate_candle_data(
                        candle_data, previous_data, cross_source_data
                    )
                    
                    # Update validation metrics
                    self._update_validation_metrics(validation_result)
                    
                    # Handle validation result
                    if validation_result.is_valid:
                        validated_batch.append(candle_data)
                        self._update_candle_history(candle_data)
                    elif self.config.enable_auto_correction and validation_result.corrections:
                        # Use corrected data
                        corrected_data = validation_result.corrections[0].corrected_data
                        if corrected_data:
                            validated_batch.append(corrected_data)
                            self._update_candle_history(corrected_data)
                    elif self.config.quarantine_invalid_data:
                        # Quarantine invalid data
                        await self.quarantine_queue.put(candle_data)
                    else:
                        # Skip invalid data
                        logger.warning(f"Skipping invalid candle data: {candle_data.instrument}")
                        continue
                else:
                    # No validation - use data as-is
                    validated_batch.append(candle_data)
                    self._update_candle_history(candle_data)
            
            # Process validated batch
            if validated_batch:
                await self._process_validated_candle_data(validated_batch)
            
        except Exception as e:
            logger.error(f"Error processing validated candle batch: {e}")
            self.metrics.failed_requests += 1
            self.error_counts[type(e).__name__] += 1
        
        finally:
            # Update latency metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_metrics(latency_ms)
    
    async def _process_validated_price_data(self, batch: List[PriceData]) -> None:
        """Process validated price data"""
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
            logger.error(f"Error processing validated price data: {e}")
            self.metrics.failed_requests += 1
            self.error_counts[type(e).__name__] += 1
    
    async def _process_validated_candle_data(self, batch: List[CandleData]) -> None:
        """Process validated candle data"""
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
            logger.error(f"Error processing validated candle data: {e}")
            self.metrics.failed_requests += 1
            self.error_counts[type(e).__name__] += 1
    
    async def _update_price_cache(self, batch: List[PriceData]) -> None:
        """Update price cache with validated data"""
        if not self.cache_manager or not self.cache_manager.connected:
            # Fallback to in-memory cache
            current_time = datetime.now()
            for price_data in batch:
                cache_key = price_data.instrument
                # Store in memory cache (simplified)
                pass
            return
        
        # Use Redis cache
        for price_data in batch:
            try:
                await self.cache_manager.cache_price_data(
                    price_data.instrument, 
                    price_data,
                    ttl=self.config.cache_ttl
                )
                self.metrics.cache_hits += 1
            except Exception as e:
                logger.error(f"Error caching price data: {e}")
                self.metrics.cache_misses += 1
    
    async def _update_candle_cache(self, batch: List[CandleData]) -> None:
        """Update candle cache with validated data"""
        if not self.cache_manager or not self.cache_manager.connected:
            # Fallback to in-memory cache
            for candle_data in batch:
                # Store in memory cache (simplified)
                pass
            return
        
        # Use Redis cache
        for candle_data in batch:
            try:
                await self.cache_manager.cache_candle_data(
                    candle_data.instrument,
                    candle_data,
                    ttl=self.config.cache_ttl
                )
                self.metrics.cache_hits += 1
            except Exception as e:
                logger.error(f"Error caching candle data: {e}")
                self.metrics.cache_misses += 1
    
    def _get_previous_price_data(self, instrument: str) -> Optional[PriceData]:
        """Get previous price data for validation context"""
        if instrument in self.price_history and self.price_history[instrument]:
            return self.price_history[instrument][-1]
        return None
    
    def _get_previous_candle_data(self, instrument: str) -> Optional[CandleData]:
        """Get previous candle data for validation context"""
        if instrument in self.candle_history and self.candle_history[instrument]:
            return self.candle_history[instrument][-1]
        return None
    
    def _get_cross_source_price_data(self, price_data: PriceData) -> List[PriceData]:
        """Get cross-source price data for validation"""
        # This would typically fetch data from other brokers
        # For now, return empty list
        return []
    
    def _get_cross_source_candle_data(self, candle_data: CandleData) -> List[CandleData]:
        """Get cross-source candle data for validation"""
        # This would typically fetch data from other brokers
        # For now, return empty list
        return []
    
    def _update_price_history(self, price_data: PriceData):
        """Update price history for validation context"""
        instrument = price_data.instrument
        self.price_history[instrument].append(price_data)
        
        # Maintain history size
        if len(self.price_history[instrument]) > self.max_history_size:
            self.price_history[instrument] = self.price_history[instrument][-self.max_history_size:]
    
    def _update_candle_history(self, candle_data: CandleData):
        """Update candle history for validation context"""
        instrument = candle_data.instrument
        self.candle_history[instrument].append(candle_data)
        
        # Maintain history size
        if len(self.candle_history[instrument]) > self.max_history_size:
            self.candle_history[instrument] = self.candle_history[instrument][-self.max_history_size:]
    
    def _update_validation_metrics(self, validation_result: ValidationResult):
        """Update validation metrics"""
        self.metrics.total_validations += 1
        
        if validation_result.is_valid:
            self.metrics.valid_data_points += 1
        else:
            self.metrics.invalid_data_points += 1
        
        if validation_result.status == 'corrected':
            self.metrics.corrected_data_points += 1
        elif validation_result.status == 'quarantined':
            self.metrics.quarantined_data_points += 1
        
        # Update validation time
        if self.metrics.total_validations == 1:
            self.metrics.avg_validation_time_ms = validation_result.validation_time_ms
        else:
            self.metrics.avg_validation_time_ms = (
                (self.metrics.avg_validation_time_ms * (self.metrics.total_validations - 1) + 
                 validation_result.validation_time_ms) / self.metrics.total_validations
            )
        
        # Update error/warning counts
        if hasattr(validation_result, 'errors'):
            self.metrics.validation_errors += len(validation_result.errors)
        
        if hasattr(validation_result, 'warnings'):
            self.metrics.validation_warnings += len(validation_result.warnings)
        
        # Update quality score
        if hasattr(validation_result, 'quality_score'):
            if self.metrics.total_validations == 1:
                self.metrics.avg_quality_score = validation_result.quality_score
            else:
                self.metrics.avg_quality_score = (
                    (self.metrics.avg_quality_score * (self.metrics.total_validations - 1) + 
                     validation_result.quality_score) / self.metrics.total_validations
                )
    
    def _update_latency_metrics(self, latency_ms: float):
        """Update latency metrics"""
        if self.metrics.total_requests == 0:
            self.metrics.avg_latency_ms = latency_ms
            self.metrics.max_latency_ms = latency_ms
            self.metrics.min_latency_ms = latency_ms
        else:
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * self.metrics.total_requests + latency_ms) / 
                (self.metrics.total_requests + 1)
            )
            self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
            self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency_ms)
        
        self.metrics.total_requests += 1
    
    async def _health_check_loop(self) -> None:
        """Health check loop"""
        while self.status == IngestionStatus.RUNNING:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(10.0)
    
    async def _perform_health_check(self) -> None:
        """Perform health check"""
        try:
            # Check broker manager health
            if hasattr(self.broker_manager, 'get_health_status'):
                health_status = await self.broker_manager.get_health_status()
                if not health_status.get('healthy', False):
                    logger.warning("Broker manager health check failed")
            
            # Check cache health
            if self.cache_manager and hasattr(self.cache_manager, 'connected'):
                if not self.cache_manager.connected:
                    logger.warning("Cache manager is not connected")
            
            # Check validation engine health
            validation_stats = self.validation_engine.get_validation_stats()
            if validation_stats['total_validations'] > 0:
                success_rate = validation_stats['successful_validations'] / validation_stats['total_validations']
                if success_rate < 0.8:  # 80% success rate threshold
                    logger.warning(f"Validation success rate is low: {success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    async def _quality_monitoring_loop(self) -> None:
        """Quality monitoring loop"""
        while self.status == IngestionStatus.RUNNING:
            try:
                await self._generate_quality_report()
                await asyncio.sleep(self.config.quality_report_interval)
            except Exception as e:
                logger.error(f"Error in quality monitoring: {e}")
                await asyncio.sleep(60.0)
    
    async def _generate_quality_report(self) -> None:
        """Generate and process quality report"""
        try:
            # Get quality report from validation engine
            quality_report = await self.validation_engine.get_quality_report()
            
            # Check quality thresholds
            if quality_report.overall_quality_score < self.config.quality_threshold:
                self.metrics.quality_threshold_breaches += 1
                logger.warning(f"Quality score below threshold: {quality_report.overall_quality_score:.2f}")
            
            # Call quality callbacks
            for callback in self.quality_callbacks:
                try:
                    await callback(quality_report)
                except Exception as e:
                    logger.error(f"Error in quality callback: {e}")
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
    
    # Public methods for data ingestion
    async def ingest_price_data(self, price_data: PriceData) -> None:
        """Ingest price data with validation"""
        try:
            await self.price_queue.put(price_data)
        except asyncio.QueueFull:
            logger.warning("Price queue is full, dropping data point")
            self.metrics.failed_requests += 1
    
    async def ingest_candle_data(self, candle_data: CandleData) -> None:
        """Ingest candle data with validation"""
        try:
            await self.candle_queue.put(candle_data)
        except asyncio.QueueFull:
            logger.warning("Candle queue is full, dropping data point")
            self.metrics.failed_requests += 1
    
    # Callback management
    def add_price_callback(self, callback: Callable) -> None:
        """Add price data callback"""
        self.price_callbacks.append(callback)
    
    def add_candle_callback(self, callback: Callable) -> None:
        """Add candle data callback"""
        self.candle_callbacks.append(callback)
    
    def add_validation_callback(self, callback: Callable) -> None:
        """Add validation result callback"""
        self.validation_callbacks.append(callback)
    
    def add_quality_callback(self, callback: Callable) -> None:
        """Add quality report callback"""
        self.quality_callbacks.append(callback)
    
    # Status and metrics
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            'status': self.status.value,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'data_points_ingested': self.metrics.data_points_ingested,
                'total_validations': self.metrics.total_validations,
                'valid_data_points': self.metrics.valid_data_points,
                'invalid_data_points': self.metrics.invalid_data_points,
                'corrected_data_points': self.metrics.corrected_data_points,
                'quarantined_data_points': self.metrics.quarantined_data_points,
                'avg_quality_score': self.metrics.avg_quality_score,
                'avg_validation_time_ms': self.metrics.avg_validation_time_ms
            },
            'queue_sizes': {
                'price_queue': self.price_queue.qsize(),
                'candle_queue': self.candle_queue.qsize(),
                'quarantine_queue': self.quarantine_queue.qsize()
            },
            'validation_enabled': self.config.enable_validation,
            'auto_correction_enabled': self.config.enable_auto_correction,
            'quality_monitoring_enabled': self.config.enable_quality_monitoring
        }
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.validation_engine.get_validation_stats()
    
    async def get_quality_report(self) -> QualityReport:
        """Get current quality report"""
        return await self.validation_engine.get_quality_report()

# Example usage and testing
async def test_validated_ingestion_engine():
    """Test the validated ingestion engine"""
    from ..brokers.broker_manager import BrokerManager
    
    # Create mock broker manager
    broker_manager = BrokerManager()
    
    # Create validated ingestion engine
    config = ValidatedIngestionConfig(
        enable_validation=True,
        enable_auto_correction=True,
        enable_quality_monitoring=True
    )
    
    engine = ValidatedDataIngestionEngine(broker_manager, config)
    
    # Add callbacks
    def price_callback(price_data):
        print(f"Received validated price data: {price_data.instrument}")
    
    def quality_callback(quality_report):
        print(f"Quality report: {quality_report.overall_quality_score:.2f}")
    
    engine.add_price_callback(price_callback)
    engine.add_quality_callback(quality_callback)
    
    # Start engine
    await engine.start()
    
    # Test data ingestion
    price_data = PriceData(
        instrument="EUR_USD",
        time=datetime.utcnow(),
        bid=1.1000,
        ask=1.1002,
        spread=0.0002
    )
    
    await engine.ingest_price_data(price_data)
    
    # Wait a bit for processing
    await asyncio.sleep(2.0)
    
    # Get status
    status = engine.get_status()
    print(f"Engine status: {status}")
    
    # Stop engine
    await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_validated_ingestion_engine())

