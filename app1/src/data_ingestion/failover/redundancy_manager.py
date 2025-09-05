#!/usr/bin/env python3
"""
Redundancy Manager
Handles parallel data ingestion and cross-source validation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import time
from collections import defaultdict, deque

from ..brokers.broker_manager import BrokerManager, PriceData, CandleData, BrokerInfo
from ..cache.redis_cache import RedisCacheManager
from ..config import CONFIG

logger = logging.getLogger(__name__)

class ValidationStrategy(Enum):
    """Data validation strategies"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    LATEST_DATA = "latest_data"

class DataSource(Enum):
    """Data source types"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    BACKUP = "backup"

@dataclass
class DataPoint:
    """Data point with source information"""
    data: Any
    source: str
    timestamp: datetime
    confidence: float = 1.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of data validation"""
    validated_data: Any
    confidence: float
    sources_used: List[str]
    validation_method: str
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RedundancyConfig:
    """Configuration for redundancy management"""
    max_parallel_sources: int = 3
    validation_timeout_ms: int = 1000
    confidence_threshold: float = 0.8
    max_discrepancy_percent: float = 5.0
    enable_cross_validation: bool = True
    enable_deduplication: bool = True
    deduplication_window_ms: int = 100
    enable_source_weighting: bool = True

@dataclass
class RedundancyMetrics:
    """Metrics for redundancy operations"""
    total_requests: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    data_discrepancies: int = 0
    avg_validation_time_ms: float = 0.0
    max_validation_time_ms: float = 0.0
    min_validation_time_ms: float = float('inf')
    source_reliability: Dict[str, float] = field(default_factory=dict)
    validation_method_usage: Dict[str, int] = field(default_factory=dict)

class RedundancyManager:
    """Manages parallel data ingestion and cross-source validation"""
    
    def __init__(self, 
                 broker_manager: BrokerManager,
                 cache_manager: Optional[RedisCacheManager] = None,
                 config: Optional[RedundancyConfig] = None):
        """
        Initialize redundancy manager
        
        Args:
            broker_manager: Broker manager instance
            cache_manager: Redis cache manager for state persistence
            config: Redundancy configuration
        """
        self.broker_manager = broker_manager
        self.cache_manager = cache_manager
        self.config = config or RedundancyConfig()
        
        # Source configuration
        self.source_weights: Dict[str, float] = {}
        self.source_reliability: Dict[str, float] = {}
        
        # Data validation
        self.validation_strategies: Dict[str, ValidationStrategy] = {
            'price': ValidationStrategy.MAJORITY_VOTE,
            'candle': ValidationStrategy.HIGHEST_CONFIDENCE,
            'indicator': ValidationStrategy.WEIGHTED_AVERAGE
        }
        
        # Deduplication
        self.recent_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Metrics
        self.metrics = RedundancyMetrics()
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
        
        # Initialize source weights and reliability
        self._initialize_source_weights()
    
    def _initialize_source_weights(self) -> None:
        """Initialize source weights based on broker priorities"""
        for name, broker in self.broker_manager.brokers.items():
            # Weight based on priority (lower priority number = higher weight)
            weight = 1.0 / (broker.priority + 1)
            self.source_weights[name] = weight
            self.source_reliability[name] = 1.0
            self.metrics.source_reliability[name] = 1.0
    
    async def get_data_with_redundancy(self, 
                                     operation: str,
                                     operation_func: Callable,
                                     *args, **kwargs) -> ValidationResult:
        """Get data with redundancy and validation"""
        start_time = time.time()
        
        try:
            # Get data from multiple sources in parallel
            data_points = await self._collect_data_from_sources(
                operation, operation_func, *args, **kwargs
            )
            
            if not data_points:
                raise Exception("No data received from any source")
            
            # Validate and consolidate data
            validation_result = await self._validate_and_consolidate_data(
                data_points, operation
            )
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_validations += 1
            
            validation_time_ms = (time.time() - start_time) * 1000
            self._update_validation_timing_metrics(validation_time_ms)
            
            # Update source reliability
            self._update_source_reliability(data_points, validation_result)
            
            logger.debug(f"Successfully validated data from {len(validation_result.sources_used)} sources")
            return validation_result
            
        except Exception as e:
            self.metrics.failed_validations += 1
            logger.error(f"Failed to get data with redundancy: {e}")
            raise
    
    async def _collect_data_from_sources(self, 
                                       operation: str,
                                       operation_func: Callable,
                                       *args, **kwargs) -> List[DataPoint]:
        """Collect data from multiple sources in parallel"""
        # Get available brokers
        healthy_brokers = self.broker_manager.get_healthy_brokers()
        
        if not healthy_brokers:
            raise Exception("No healthy brokers available")
        
        # Limit to max parallel sources
        brokers_to_use = healthy_brokers[:self.config.max_parallel_sources]
        
        # Create tasks for parallel execution
        tasks = []
        for broker in brokers_to_use:
            task = asyncio.create_task(
                self._get_data_from_source(broker, operation, operation_func, *args, **kwargs)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.validation_timeout_ms / 1000
            )
        except asyncio.TimeoutError:
            logger.warning("Data collection timed out")
            results = []
        
        # Process results
        data_points = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Source {brokers_to_use[i].name} failed: {result}")
                continue
            
            if result:
                data_points.append(result)
        
        return data_points
    
    async def _get_data_from_source(self, 
                                  broker: BrokerInfo,
                                  operation: str,
                                  operation_func: Callable,
                                  *args, **kwargs) -> Optional[DataPoint]:
        """Get data from a specific source"""
        start_time = time.time()
        
        try:
            # Execute operation with broker-specific logic
            if hasattr(broker.client, operation):
                data = await getattr(broker.client, operation)(*args, **kwargs)
            else:
                # Fallback to generic operation
                data = await operation_func(*args, **kwargs)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate confidence based on source reliability
            confidence = self.source_reliability.get(broker.name, 1.0)
            
            # Create data point
            data_point = DataPoint(
                data=data,
                source=broker.name,
                timestamp=datetime.now(),
                confidence=confidence,
                latency_ms=latency_ms,
                metadata={'operation': operation}
            )
            
            return data_point
            
        except Exception as e:
            logger.warning(f"Failed to get data from {broker.name}: {e}")
            return None
    
    async def _validate_and_consolidate_data(self, 
                                           data_points: List[DataPoint],
                                           operation: str) -> ValidationResult:
        """Validate and consolidate data from multiple sources"""
        if len(data_points) == 1:
            # Single source - no validation needed
            point = data_points[0]
            return ValidationResult(
                validated_data=point.data,
                confidence=point.confidence,
                sources_used=[point.source],
                validation_method="single_source"
            )
        
        # Determine validation strategy
        data_type = self._get_data_type_from_operation(operation)
        strategy = self.validation_strategies.get(data_type, ValidationStrategy.MAJORITY_VOTE)
        
        # Apply validation strategy
        if strategy == ValidationStrategy.MAJORITY_VOTE:
            return await self._validate_majority_vote(data_points)
        elif strategy == ValidationStrategy.WEIGHTED_AVERAGE:
            return await self._validate_weighted_average(data_points)
        elif strategy == ValidationStrategy.HIGHEST_CONFIDENCE:
            return await self._validate_highest_confidence(data_points)
        elif strategy == ValidationStrategy.LATEST_DATA:
            return await self._validate_latest_data(data_points)
        else:
            # Default to majority vote
            return await self._validate_majority_vote(data_points)
    
    def _get_data_type_from_operation(self, operation: str) -> str:
        """Determine data type from operation name"""
        if 'price' in operation.lower():
            return 'price'
        elif 'candle' in operation.lower():
            return 'candle'
        elif 'indicator' in operation.lower():
            return 'indicator'
        else:
            return 'unknown'
    
    async def _validate_majority_vote(self, data_points: List[DataPoint]) -> ValidationResult:
        """Validate data using majority vote strategy"""
        # Group data points by their values (for price data)
        value_groups = defaultdict(list)
        
        for point in data_points:
            if isinstance(point.data, PriceData):
                # Round to avoid floating point precision issues
                key = (round(point.data.bid, 5), round(point.data.ask, 5))
                value_groups[key].append(point)
            else:
                # For other data types, use string representation
                key = str(point.data)
                value_groups[key].append(point)
        
        # Find the group with the most votes
        largest_group = max(value_groups.values(), key=len)
        
        # Calculate confidence based on majority size
        confidence = len(largest_group) / len(data_points)
        
        # Use the data point with highest confidence from the largest group
        best_point = max(largest_group, key=lambda p: p.confidence)
        
        # Calculate discrepancies
        discrepancies = []
        for key, group in value_groups.items():
            if key != (round(best_point.data.bid, 5), round(best_point.data.ask, 5)) if isinstance(best_point.data, PriceData) else str(best_point.data):
                discrepancies.append({
                    'value': key,
                    'sources': [p.source for p in group],
                    'count': len(group)
                })
        
        self.metrics.validation_method_usage['majority_vote'] = self.metrics.validation_method_usage.get('majority_vote', 0) + 1
        
        return ValidationResult(
            validated_data=best_point.data,
            confidence=confidence,
            sources_used=[p.source for p in largest_group],
            validation_method="majority_vote",
            discrepancies=discrepancies
        )
    
    async def _validate_weighted_average(self, data_points: List[DataPoint]) -> ValidationResult:
        """Validate data using weighted average strategy"""
        if not data_points:
            raise Exception("No data points to validate")
        
        # Calculate weighted average for price data
        if isinstance(data_points[0].data, PriceData):
            total_weight = sum(p.confidence for p in data_points)
            
            weighted_bid = sum(p.data.bid * p.confidence for p in data_points) / total_weight
            weighted_ask = sum(p.data.ask * p.confidence for p in data_points) / total_weight
            
            # Create new price data with weighted values
            validated_data = PriceData(
                instrument=data_points[0].data.instrument,
                time=data_points[0].data.time,
                bid=weighted_bid,
                ask=weighted_ask,
                spread=weighted_ask - weighted_bid
            )
            
            # Calculate confidence as average of source confidences
            confidence = sum(p.confidence for p in data_points) / len(data_points)
            
        else:
            # For non-price data, use highest confidence
            best_point = max(data_points, key=lambda p: p.confidence)
            validated_data = best_point.data
            confidence = best_point.confidence
        
        self.metrics.validation_method_usage['weighted_average'] = self.metrics.validation_method_usage.get('weighted_average', 0) + 1
        
        return ValidationResult(
            validated_data=validated_data,
            confidence=confidence,
            sources_used=[p.source for p in data_points],
            validation_method="weighted_average"
        )
    
    async def _validate_highest_confidence(self, data_points: List[DataPoint]) -> ValidationResult:
        """Validate data using highest confidence strategy"""
        best_point = max(data_points, key=lambda p: p.confidence)
        
        self.metrics.validation_method_usage['highest_confidence'] = self.metrics.validation_method_usage.get('highest_confidence', 0) + 1
        
        return ValidationResult(
            validated_data=best_point.data,
            confidence=best_point.confidence,
            sources_used=[best_point.source],
            validation_method="highest_confidence"
        )
    
    async def _validate_latest_data(self, data_points: List[DataPoint]) -> ValidationResult:
        """Validate data using latest data strategy"""
        latest_point = max(data_points, key=lambda p: p.timestamp)
        
        self.metrics.validation_method_usage['latest_data'] = self.metrics.validation_method_usage.get('latest_data', 0) + 1
        
        return ValidationResult(
            validated_data=latest_point.data,
            confidence=latest_point.confidence,
            sources_used=[latest_point.source],
            validation_method="latest_data"
        )
    
    def _update_source_reliability(self, 
                                 data_points: List[DataPoint],
                                 validation_result: ValidationResult) -> None:
        """Update source reliability based on validation results"""
        for point in data_points:
            source = point.source
            
            # Check if this source was used in validation
            if source in validation_result.sources_used:
                # Increase reliability slightly
                current_reliability = self.source_reliability.get(source, 1.0)
                self.source_reliability[source] = min(1.0, current_reliability + 0.01)
            else:
                # Decrease reliability slightly
                current_reliability = self.source_reliability.get(source, 1.0)
                self.source_reliability[source] = max(0.1, current_reliability - 0.01)
            
            # Update metrics
            self.metrics.source_reliability[source] = self.source_reliability[source]
    
    def _update_validation_timing_metrics(self, validation_time_ms: float) -> None:
        """Update validation timing metrics"""
        self.metrics.avg_validation_time_ms = (
            (self.metrics.avg_validation_time_ms * (self.metrics.total_requests - 1) + validation_time_ms) /
            self.metrics.total_requests
        )
        self.metrics.max_validation_time_ms = max(self.metrics.max_validation_time_ms, validation_time_ms)
        self.metrics.min_validation_time_ms = min(self.metrics.min_validation_time_ms, validation_time_ms)
    
    async def detect_data_discrepancies(self, 
                                      data_points: List[DataPoint],
                                      max_discrepancy_percent: float = 5.0) -> List[Dict[str, Any]]:
        """Detect discrepancies between data sources"""
        if len(data_points) < 2:
            return []
        
        discrepancies = []
        
        # Compare price data
        if isinstance(data_points[0].data, PriceData):
            prices = [(p.data.bid, p.data.ask) for p in data_points]
            
            for i, (bid1, ask1) in enumerate(prices):
                for j, (bid2, ask2) in enumerate(prices[i+1:], i+1):
                    # Calculate percentage difference
                    bid_diff = abs(bid1 - bid2) / bid1 * 100
                    ask_diff = abs(ask1 - ask2) / ask1 * 100
                    
                    if bid_diff > max_discrepancy_percent or ask_diff > max_discrepancy_percent:
                        discrepancy = {
                            'sources': [data_points[i].source, data_points[j].source],
                            'bid_diff_percent': bid_diff,
                            'ask_diff_percent': ask_diff,
                            'price1': (bid1, ask1),
                            'price2': (bid2, ask2),
                            'timestamp': data_points[i].timestamp.isoformat()
                        }
                        discrepancies.append(discrepancy)
                        
                        self.metrics.data_discrepancies += 1
                        logger.warning(f"Data discrepancy detected between {data_points[i].source} and {data_points[j].source}: {bid_diff:.2f}% bid, {ask_diff:.2f}% ask")
        
        return discrepancies
    
    def add_event_callback(self, callback: Callable) -> None:
        """Add a callback for redundancy events"""
        self.event_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get redundancy metrics"""
        return {
            'total_requests': self.metrics.total_requests,
            'successful_validations': self.metrics.successful_validations,
            'failed_validations': self.metrics.failed_validations,
            'data_discrepancies': self.metrics.data_discrepancies,
            'avg_validation_time_ms': round(self.metrics.avg_validation_time_ms, 2),
            'max_validation_time_ms': round(self.metrics.max_validation_time_ms, 2),
            'min_validation_time_ms': round(self.metrics.min_validation_time_ms, 2),
            'source_reliability': self.metrics.source_reliability,
            'validation_method_usage': self.metrics.validation_method_usage,
            'success_rate': (
                self.metrics.successful_validations / max(self.metrics.total_requests, 1) * 100
            )
        }
    
    def set_validation_strategy(self, data_type: str, strategy: ValidationStrategy) -> None:
        """Set validation strategy for a data type"""
        self.validation_strategies[data_type] = strategy
        logger.info(f"Set validation strategy for {data_type} to {strategy.value}")
    
    def set_source_weight(self, source: str, weight: float) -> None:
        """Set weight for a data source"""
        self.source_weights[source] = weight
        logger.info(f"Set weight for {source} to {weight}")

# Example usage and testing
async def test_redundancy_manager():
    """Test the redundancy manager"""
    from ..brokers.broker_manager import BrokerManager
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Create redundancy manager
    redundancy_manager = RedundancyManager(broker_manager)
    
    # Test getting data with redundancy
    try:
        # This would be called with actual broker operations
        # result = await redundancy_manager.get_data_with_redundancy(
        #     "get_pricing", broker_manager.get_pricing, ["EUR_USD"]
        # )
        # print(f"Validation result: {result}")
        pass
    except Exception as e:
        print(f"Test failed: {e}")
    
    # Test metrics
    metrics = redundancy_manager.get_metrics()
    print(f"Redundancy metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(test_redundancy_manager())

