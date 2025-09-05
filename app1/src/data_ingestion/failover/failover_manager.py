#!/usr/bin/env python3
"""
Failover Manager
Advanced failover and redundancy mechanisms for data ingestion
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import random
import json
from collections import defaultdict, deque

from ..brokers.broker_manager import BrokerManager, BrokerInfo, BrokerStatus
from ..cache.redis_cache import RedisCacheManager
from ..config import CONFIG

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back

class FailoverEvent(Enum):
    """Types of failover events"""
    BROKER_FAILED = "broker_failed"
    BROKER_RECOVERED = "broker_recovered"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    FAILOVER_TRIGGERED = "failover_triggered"
    FAILBACK_TRIGGERED = "failback_triggered"
    DATA_GAP_DETECTED = "data_gap_detected"
    DATA_REPLAY_STARTED = "data_replay_started"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: int = 60  # Seconds before trying to close circuit
    success_threshold: int = 3  # Successes needed to close circuit
    timeout: int = 30  # Request timeout in seconds
    max_retries: int = 3  # Max retries per request
    retry_delay: float = 1.0  # Base delay between retries
    jitter: float = 0.1  # Jitter factor for retry delays

@dataclass
class CircuitBreakerState:
    """State of a circuit breaker"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None

@dataclass
class FailoverMetrics:
    """Metrics for failover operations"""
    total_failovers: int = 0
    total_failbacks: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    data_gaps_detected: int = 0
    data_replays: int = 0
    avg_failover_time_ms: float = 0.0
    max_failover_time_ms: float = 0.0
    min_failover_time_ms: float = float('inf')
    last_failover: Optional[datetime] = None
    last_failback: Optional[datetime] = None

@dataclass
class DataBuffer:
    """Buffer for data during outages"""
    data: deque = field(default_factory=lambda: deque(maxlen=10000))
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    instrument: str = ""
    data_type: str = ""

class FailoverManager:
    """Advanced failover and redundancy manager"""
    
    def __init__(self, 
                 broker_manager: BrokerManager,
                 cache_manager: Optional[RedisCacheManager] = None,
                 config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize failover manager
        
        Args:
            broker_manager: Broker manager instance
            cache_manager: Redis cache manager for state persistence
            config: Circuit breaker configuration
        """
        self.broker_manager = broker_manager
        self.cache_manager = cache_manager
        self.config = config or CircuitBreakerConfig()
        
        # Circuit breakers for each broker
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Failover state
        self.current_primary_broker: Optional[str] = None
        self.failover_history: deque = deque(maxlen=1000)
        self.metrics = FailoverMetrics()
        
        # Data buffering during outages
        self.data_buffers: Dict[str, DataBuffer] = {}
        self.buffering_enabled = True
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
        
        # Monitoring tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Initialize circuit breakers for existing brokers
        self._initialize_circuit_breakers()
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for all brokers"""
        for name in self.broker_manager.brokers.keys():
            self.circuit_breakers[name] = CircuitBreakerState()
    
    async def start(self) -> None:
        """Start the failover manager"""
        if self.is_running:
            return
        
        logger.info("Starting failover manager...")
        self.is_running = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Set initial primary broker
        await self._select_primary_broker()
        
        logger.info("Failover manager started successfully")
    
    async def stop(self) -> None:
        """Stop the failover manager"""
        if not self.is_running:
            return
        
        logger.info("Stopping failover manager...")
        self.is_running = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Flush any remaining buffered data
        await self._flush_all_buffers()
        
        logger.info("Failover manager stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for failover management"""
        while self.is_running:
            try:
                await self._check_circuit_breakers()
                await self._monitor_data_continuity()
                await self._check_primary_broker_health()
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in failover monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_circuit_breakers(self) -> None:
        """Check and update circuit breaker states"""
        for broker_name, circuit in self.circuit_breakers.items():
            if circuit.state == CircuitState.OPEN:
                # Check if it's time to try half-open
                if (circuit.next_attempt_time and 
                    datetime.now() >= circuit.next_attempt_time):
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.success_count = 0
                    logger.info(f"Circuit breaker for {broker_name} moved to HALF_OPEN")
                    await self._emit_event(FailoverEvent.CIRCUIT_CLOSED, {
                        'broker': broker_name,
                        'state': 'half_open'
                    })
    
    async def _monitor_data_continuity(self) -> None:
        """Monitor for data gaps and continuity issues"""
        if not self.buffering_enabled:
            return
        
        current_time = datetime.now()
        
        # Check for data gaps in buffers
        for buffer_key, buffer in self.data_buffers.items():
            if buffer.end_time and (current_time - buffer.end_time).total_seconds() > 30:
                # Data gap detected
                self.metrics.data_gaps_detected += 1
                logger.warning(f"Data gap detected for {buffer_key}")
                
                await self._emit_event(FailoverEvent.DATA_GAP_DETECTED, {
                    'buffer_key': buffer_key,
                    'gap_duration': (current_time - buffer.end_time).total_seconds(),
                    'instrument': buffer.instrument
                })
                
                # Attempt to replay missing data
                await self._replay_missing_data(buffer_key)
    
    async def _check_primary_broker_health(self) -> None:
        """Check if primary broker is still healthy"""
        if not self.current_primary_broker:
            await self._select_primary_broker()
            return
        
        broker_info = self.broker_manager.brokers.get(self.current_primary_broker)
        if not broker_info or broker_info.status != BrokerStatus.HEALTHY:
            logger.warning(f"Primary broker {self.current_primary_broker} is unhealthy, triggering failover")
            await self._trigger_failover()
    
    async def _select_primary_broker(self) -> None:
        """Select the best available broker as primary"""
        healthy_brokers = self.broker_manager.get_healthy_brokers()
        
        if not healthy_brokers:
            logger.error("No healthy brokers available")
            self.current_primary_broker = None
            return
        
        # Select broker with highest priority and closed circuit
        for broker in healthy_brokers:
            circuit = self.circuit_breakers.get(broker.name)
            if circuit and circuit.state == CircuitState.CLOSED:
                self.current_primary_broker = broker.name
                logger.info(f"Selected {broker.name} as primary broker")
                return
        
        # If no closed circuits, select best available
        self.current_primary_broker = healthy_brokers[0].name
        logger.info(f"Selected {self.current_primary_broker} as primary broker (circuit may be open)")
    
    async def _trigger_failover(self) -> None:
        """Trigger failover to next available broker"""
        start_time = time.time()
        
        try:
            old_primary = self.current_primary_broker
            await self._select_primary_broker()
            
            if self.current_primary_broker != old_primary:
                self.metrics.total_failovers += 1
                self.metrics.last_failover = datetime.now()
                
                # Record failover event
                failover_event = {
                    'timestamp': datetime.now().isoformat(),
                    'old_primary': old_primary,
                    'new_primary': self.current_primary_broker,
                    'reason': 'broker_unhealthy'
                }
                self.failover_history.append(failover_event)
                
                logger.info(f"Failover completed: {old_primary} -> {self.current_primary_broker}")
                
                await self._emit_event(FailoverEvent.FAILOVER_TRIGGERED, failover_event)
                
                # Start buffering data during transition
                if self.buffering_enabled:
                    await self._start_data_buffering()
            
        except Exception as e:
            logger.error(f"Error during failover: {e}")
        finally:
            # Update failover timing metrics
            failover_time_ms = (time.time() - start_time) * 1000
            self._update_failover_timing_metrics(failover_time_ms)
    
    async def _trigger_failback(self, broker_name: str) -> None:
        """Trigger failback to a recovered broker"""
        if broker_name == self.current_primary_broker:
            return
        
        # Check if the broker is now healthy and has closed circuit
        broker_info = self.broker_manager.brokers.get(broker_name)
        circuit = self.circuit_breakers.get(broker_name)
        
        if (broker_info and 
            broker_info.status == BrokerStatus.HEALTHY and 
            circuit and 
            circuit.state == CircuitState.CLOSED):
            
            old_primary = self.current_primary_broker
            self.current_primary_broker = broker_name
            
            self.metrics.total_failbacks += 1
            self.metrics.last_failback = datetime.now()
            
            # Record failback event
            failback_event = {
                'timestamp': datetime.now().isoformat(),
                'old_primary': old_primary,
                'new_primary': broker_name,
                'reason': 'broker_recovered'
            }
            self.failover_history.append(failback_event)
            
            logger.info(f"Failback completed: {old_primary} -> {broker_name}")
            
            await self._emit_event(FailoverEvent.FAILBACK_TRIGGERED, failback_event)
            
            # Stop buffering and replay any buffered data
            if self.buffering_enabled:
                await self._stop_data_buffering()
                await self._replay_buffered_data()
    
    async def execute_with_failover(self, 
                                  operation: str,
                                  operation_func: Callable,
                                  *args, **kwargs) -> Any:
        """Execute an operation with automatic failover"""
        last_error = None
        brokers_tried = set()
        
        # Try primary broker first
        if self.current_primary_broker:
            brokers_tried.add(self.current_primary_broker)
            try:
                return await self._execute_with_circuit_breaker(
                    self.current_primary_broker, operation, operation_func, *args, **kwargs
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Primary broker {self.current_primary_broker} failed: {e}")
        
        # Try other healthy brokers
        healthy_brokers = self.broker_manager.get_healthy_brokers()
        for broker in healthy_brokers:
            if broker.name in brokers_tried:
                continue
            
            brokers_tried.add(broker.name)
            try:
                result = await self._execute_with_circuit_breaker(
                    broker.name, operation, operation_func, *args, **kwargs
                )
                
                # If successful, consider failback
                if broker.name != self.current_primary_broker:
                    await self._trigger_failback(broker.name)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Broker {broker.name} failed: {e}")
        
        # All brokers failed
        raise Exception(f"All brokers failed for operation {operation}. Last error: {last_error}")
    
    async def _execute_with_circuit_breaker(self, 
                                          broker_name: str,
                                          operation: str,
                                          operation_func: Callable,
                                          *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection"""
        circuit = self.circuit_breakers.get(broker_name)
        if not circuit:
            circuit = CircuitBreakerState()
            self.circuit_breakers[broker_name] = circuit
        
        # Check circuit state
        if circuit.state == CircuitState.OPEN:
            if circuit.next_attempt_time and datetime.now() < circuit.next_attempt_time:
                raise Exception(f"Circuit breaker for {broker_name} is OPEN")
            else:
                circuit.state = CircuitState.HALF_OPEN
                circuit.success_count = 0
        
        # Execute operation with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                # Add timeout
                result = await asyncio.wait_for(
                    operation_func(*args, **kwargs),
                    timeout=self.config.timeout
                )
                
                # Success - update circuit breaker
                circuit.failure_count = 0
                circuit.success_count += 1
                circuit.last_success_time = datetime.now()
                
                if circuit.state == CircuitState.HALF_OPEN:
                    if circuit.success_count >= self.config.success_threshold:
                        circuit.state = CircuitState.CLOSED
                        circuit.next_attempt_time = None
                        logger.info(f"Circuit breaker for {broker_name} closed")
                        await self._emit_event(FailoverEvent.CIRCUIT_CLOSED, {
                            'broker': broker_name,
                            'state': 'closed'
                        })
                
                return result
                
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"Operation {operation} on {broker_name} timed out (attempt {attempt + 1})")
            except Exception as e:
                last_error = e
                logger.warning(f"Operation {operation} on {broker_name} failed (attempt {attempt + 1}): {e}")
            
            # Wait before retry with exponential backoff and jitter
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                jitter = random.uniform(0, self.config.jitter * delay)
                await asyncio.sleep(delay + jitter)
        
        # All retries failed - update circuit breaker
        circuit.failure_count += 1
        circuit.last_failure_time = datetime.now()
        
        if circuit.failure_count >= self.config.failure_threshold:
            circuit.state = CircuitState.OPEN
            circuit.next_attempt_time = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
            circuit.success_count = 0
            
            logger.error(f"Circuit breaker for {broker_name} opened after {circuit.failure_count} failures")
            await self._emit_event(FailoverEvent.CIRCUIT_OPENED, {
                'broker': broker_name,
                'failure_count': circuit.failure_count,
                'next_attempt': circuit.next_attempt_time.isoformat()
            })
        
        raise last_error
    
    async def _start_data_buffering(self) -> None:
        """Start buffering data during failover"""
        logger.info("Starting data buffering during failover")
        # Implementation would start buffering incoming data
        # This is a placeholder for the actual buffering logic
    
    async def _stop_data_buffering(self) -> None:
        """Stop data buffering after failover"""
        logger.info("Stopping data buffering after failover")
        # Implementation would stop buffering and process buffered data
        # This is a placeholder for the actual buffering logic
    
    async def _replay_missing_data(self, buffer_key: str) -> None:
        """Replay missing data for a specific buffer"""
        logger.info(f"Replaying missing data for {buffer_key}")
        self.metrics.data_replays += 1
        await self._emit_event(FailoverEvent.DATA_REPLAY_STARTED, {
            'buffer_key': buffer_key
        })
        # Implementation would replay buffered data
        # This is a placeholder for the actual replay logic
    
    async def _replay_buffered_data(self) -> None:
        """Replay all buffered data after failover"""
        logger.info("Replaying all buffered data")
        # Implementation would replay all buffered data
        # This is a placeholder for the actual replay logic
    
    async def _flush_all_buffers(self) -> None:
        """Flush all data buffers"""
        logger.info("Flushing all data buffers")
        # Implementation would flush all buffered data
        # This is a placeholder for the actual flush logic
    
    def _update_failover_timing_metrics(self, failover_time_ms: float) -> None:
        """Update failover timing metrics"""
        self.metrics.avg_failover_time_ms = (
            (self.metrics.avg_failover_time_ms * (self.metrics.total_failovers - 1) + failover_time_ms) /
            self.metrics.total_failovers
        )
        self.metrics.max_failover_time_ms = max(self.metrics.max_failover_time_ms, failover_time_ms)
        self.metrics.min_failover_time_ms = min(self.metrics.min_failover_time_ms, failover_time_ms)
    
    async def _emit_event(self, event_type: FailoverEvent, data: Dict[str, Any]) -> None:
        """Emit a failover event to registered callbacks"""
        event = {
            'type': event_type.value,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        for callback in self.event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in failover event callback: {e}")
    
    def add_event_callback(self, callback: Callable) -> None:
        """Add a callback for failover events"""
        self.event_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get failover metrics"""
        return {
            'total_failovers': self.metrics.total_failovers,
            'total_failbacks': self.metrics.total_failbacks,
            'circuit_opens': self.metrics.circuit_opens,
            'circuit_closes': self.metrics.circuit_closes,
            'data_gaps_detected': self.metrics.data_gaps_detected,
            'data_replays': self.metrics.data_replays,
            'avg_failover_time_ms': round(self.metrics.avg_failover_time_ms, 2),
            'max_failover_time_ms': round(self.metrics.max_failover_time_ms, 2),
            'min_failover_time_ms': round(self.metrics.min_failover_time_ms, 2),
            'current_primary_broker': self.current_primary_broker,
            'circuit_breakers': {
                name: {
                    'state': circuit.state.value,
                    'failure_count': circuit.failure_count,
                    'success_count': circuit.success_count,
                    'last_failure': circuit.last_failure_time.isoformat() if circuit.last_failure_time else None,
                    'last_success': circuit.last_success_time.isoformat() if circuit.last_success_time else None,
                    'next_attempt': circuit.next_attempt_time.isoformat() if circuit.next_attempt_time else None
                }
                for name, circuit in self.circuit_breakers.items()
            },
            'last_failover': self.metrics.last_failover.isoformat() if self.metrics.last_failover else None,
            'last_failback': self.metrics.last_failback.isoformat() if self.metrics.last_failback else None
        }
    
    def get_failover_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent failover history"""
        return list(self.failover_history)[-limit:]
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

# Example usage and testing
async def test_failover_manager():
    """Test the failover manager"""
    from ..brokers.broker_manager import BrokerManager
    
    # Create broker manager
    broker_manager = BrokerManager()
    
    # Create failover manager
    failover_manager = FailoverManager(broker_manager)
    
    # Add event callback
    async def event_callback(event):
        print(f"Failover event: {event}")
    
    failover_manager.add_event_callback(event_callback)
    
    # Start failover manager
    await failover_manager.start()
    
    try:
        # Test metrics
        metrics = failover_manager.get_metrics()
        print(f"Failover metrics: {metrics}")
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Get final metrics
        final_metrics = failover_manager.get_metrics()
        print(f"Final metrics: {final_metrics}")
        
    finally:
        await failover_manager.stop()

if __name__ == "__main__":
    asyncio.run(test_failover_manager())

