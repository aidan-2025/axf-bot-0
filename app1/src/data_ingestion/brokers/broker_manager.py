#!/usr/bin/env python3
"""
Broker Manager
Manages multiple data sources with failover and load balancing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time

from .oanda_client import OANDAClient, CandleData, PriceData, Granularity
from .fxcm_client import FXCMClient, FXCMCandleData, FXCMPriceData, TimeFrame
from .free_forex_client import FreeForexClient, FreeForexCandleData, FreeForexPriceData, FreeForexGranularity

logger = logging.getLogger(__name__)

class BrokerStatus(Enum):
    """Broker connection status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CONNECTING = "connecting"
    FAILED = "failed"

@dataclass
class BrokerInfo:
    """Information about a broker"""
    name: str
    client: Any
    status: BrokerStatus
    last_check: datetime
    error_count: int
    priority: int  # Lower number = higher priority

class BrokerManager:
    """Manages multiple broker connections with failover"""
    
    def __init__(self):
        """Initialize broker manager"""
        self.brokers: Dict[str, BrokerInfo] = {}
        self.health_check_interval = 30  # seconds
        self.max_errors = 5
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def add_broker(self, 
                        name: str, 
                        client: Any, 
                        priority: int = 1) -> None:
        """Add a broker to the manager"""
        broker_info = BrokerInfo(
            name=name,
            client=client,
            status=BrokerStatus.CONNECTING,
            last_check=datetime.now(),
            error_count=0,
            priority=priority
        )
        
        self.brokers[name] = broker_info
        logger.info(f"Added broker: {name} with priority {priority}")
    
    async def remove_broker(self, name: str) -> None:
        """Remove a broker from the manager"""
        if name in self.brokers:
            broker = self.brokers[name]
            if hasattr(broker.client, 'disconnect'):
                await broker.client.disconnect()
            del self.brokers[name]
            logger.info(f"Removed broker: {name}")
    
    async def start_health_monitoring(self) -> None:
        """Start health monitoring for all brokers"""
        if self.health_check_task and not self.health_check_task.done():
            return
        
        self.is_running = True
        self.health_check_task = asyncio.create_task(self._health_monitor())
        logger.info("Started broker health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring"""
        self.is_running = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped broker health monitoring")
    
    async def _health_monitor(self) -> None:
        """Monitor broker health in background"""
        while self.is_running:
            try:
                await self._check_all_brokers()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _check_all_brokers(self) -> None:
        """Check health of all brokers"""
        for name, broker in self.brokers.items():
            try:
                if hasattr(broker.client, 'get_health_status'):
                    health = await broker.client.get_health_status()
                    if health.get('status') == 'healthy':
                        broker.status = BrokerStatus.HEALTHY
                        broker.error_count = 0
                    else:
                        broker.status = BrokerStatus.UNHEALTHY
                        broker.error_count += 1
                else:
                    # Simple connection test
                    broker.status = BrokerStatus.HEALTHY
                    broker.error_count = 0
                
                broker.last_check = datetime.now()
                
            except Exception as e:
                broker.status = BrokerStatus.UNHEALTHY
                broker.error_count += 1
                broker.last_check = datetime.now()
                logger.warning(f"Broker {name} health check failed: {e}")
                
                # Mark as failed if too many errors
                if broker.error_count >= self.max_errors:
                    broker.status = BrokerStatus.FAILED
                    logger.error(f"Broker {name} marked as failed after {broker.error_count} errors")
    
    def get_healthy_brokers(self) -> List[BrokerInfo]:
        """Get list of healthy brokers sorted by priority"""
        healthy = [broker for broker in self.brokers.values() 
                  if broker.status == BrokerStatus.HEALTHY]
        return sorted(healthy, key=lambda x: x.priority)
    
    def get_best_broker(self) -> Optional[BrokerInfo]:
        """Get the best available broker (highest priority, healthy)"""
        healthy = self.get_healthy_brokers()
        return healthy[0] if healthy else None
    
    async def get_candles(self, 
                         instrument: str,
                         granularity: Granularity,
                         count: Optional[int] = None,
                         from_time: Optional[datetime] = None,
                         to_time: Optional[datetime] = None) -> List[CandleData]:
        """Get historical candles with prioritized failover across brokers"""
        brokers_in_order: List[BrokerInfo] = self.get_healthy_brokers()
        # If no healthy brokers, try any available (last resort)
        if not brokers_in_order:
            brokers_in_order = sorted(self.brokers.values(), key=lambda x: x.priority)
        
        last_error: Optional[Exception] = None
        for broker in brokers_in_order:
            # Per-broker small retry loop
            for attempt in range(2):
                try:
                    if isinstance(broker.client, OANDAClient):
                        return await broker.client.get_candles(
                            instrument, granularity, count, from_time, to_time
                        )
                    elif isinstance(broker.client, FXCMClient):
                        fxcm_timeframe = self._convert_granularity_to_fxcm(granularity)
                        fxcm_candles = await broker.client.get_historical_data(
                            instrument, fxcm_timeframe, count, from_time, to_time
                        )
                        return self._convert_fxcm_to_oanda_candles(fxcm_candles)
                    elif isinstance(broker.client, FreeForexClient):
                        base, quote = instrument.split('_') if '_' in instrument else (instrument[:3], instrument[3:])
                        free_forex_candles = await broker.client.get_historical_rates(
                            base, quote,
                            from_time or datetime.now() - timedelta(days=7),
                            to_time or datetime.now()
                        )
                        return self._convert_free_forex_to_oanda_candles(free_forex_candles)
                    else:
                        logger.warning(f"Unsupported broker type: {type(broker.client)} for {broker.name}")
                        break
                except Exception as e:
                    last_error = e
                    broker.error_count += 1
                    # Simple circuit breaker: mark unhealthy on first failure, failed on threshold
                    broker.status = BrokerStatus.UNHEALTHY
                    if broker.error_count >= self.max_errors:
                        broker.status = BrokerStatus.FAILED
                    logger.warning(f"Attempt {attempt+1} failed on {broker.name} for get_candles: {e}")
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
        
        raise Exception(f"No brokers succeeded for get_candles. Last error: {last_error}")
    
    async def get_pricing(self, instruments: List[str]) -> Dict[str, Any]:
        """Get real-time pricing with prioritized failover across brokers"""
        brokers_in_order: List[BrokerInfo] = self.get_healthy_brokers()
        if not brokers_in_order:
            brokers_in_order = sorted(self.brokers.values(), key=lambda x: x.priority)
        
        last_error: Optional[Exception] = None
        for broker in brokers_in_order:
            for attempt in range(2):
                try:
                    if isinstance(broker.client, OANDAClient):
                        return await broker.client.get_pricing(instruments)
                    elif isinstance(broker.client, FXCMClient):
                        fxcm_offers = await broker.client.get_offers(instruments)
                        return self._convert_fxcm_to_oanda_pricing(fxcm_offers)
                    elif isinstance(broker.client, FreeForexClient):
                        free_forex_prices = await broker.client.get_real_time_prices(instruments)
                        return self._convert_free_forex_to_oanda_pricing(free_forex_prices)
                    else:
                        logger.warning(f"Unsupported broker type: {type(broker.client)} for {broker.name}")
                        break
                except Exception as e:
                    last_error = e
                    broker.error_count += 1
                    broker.status = BrokerStatus.UNHEALTHY
                    if broker.error_count >= self.max_errors:
                        broker.status = BrokerStatus.FAILED
                    logger.warning(f"Attempt {attempt+1} failed on {broker.name} for get_pricing: {e}")
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
        
        raise Exception(f"No brokers succeeded for get_pricing. Last error: {last_error}")
    
    async def stream_prices(self, 
                           instruments: List[str],
                           callback: Callable) -> None:
        """Stream real-time prices with failover. If current broker fails, try next."""
        brokers_in_order: List[BrokerInfo] = self.get_healthy_brokers()
        if not brokers_in_order:
            brokers_in_order = sorted(self.brokers.values(), key=lambda x: x.priority)
        
        last_error: Optional[Exception] = None
        for broker in brokers_in_order:
            try:
                if isinstance(broker.client, OANDAClient):
                    await broker.client.stream_prices(instruments, callback)
                    return
                elif isinstance(broker.client, FXCMClient):
                    async def fxcm_callback(fxcm_data: FXCMPriceData):
                        oanda_data = self._convert_fxcm_to_oanda_price(fxcm_data)
                        await callback(oanda_data)
                    await broker.client.stream_offers(instruments, fxcm_callback)
                    return
                elif isinstance(broker.client, FreeForexClient):
                    async def free_forex_callback(free_forex_data: FreeForexPriceData):
                        oanda_data = self._convert_free_forex_to_oanda_price(free_forex_data)
                        await callback(oanda_data)
                    await broker.client.stream_prices(instruments, free_forex_callback)
                    return
                else:
                    logger.warning(f"Unsupported broker type: {type(broker.client)} for {broker.name}")
                    continue
            except Exception as e:
                last_error = e
                broker.error_count += 1
                broker.status = BrokerStatus.UNHEALTHY
                if broker.error_count >= self.max_errors:
                    broker.status = BrokerStatus.FAILED
                logger.warning(f"Streaming failed on {broker.name}, trying next broker: {e}")
                continue
        
        raise Exception(f"No brokers succeeded for stream_prices. Last error: {last_error}")
    
    def _convert_granularity_to_fxcm(self, granularity: Granularity) -> TimeFrame:
        """Convert OANDA granularity to FXCM timeframe"""
        mapping = {
            Granularity.M1: TimeFrame.M1,
            Granularity.M5: TimeFrame.M5,
            Granularity.M15: TimeFrame.M15,
            Granularity.M30: TimeFrame.M30,
            Granularity.H1: TimeFrame.H1,
            Granularity.H2: TimeFrame.H2,
            Granularity.H3: TimeFrame.H3,
            Granularity.H4: TimeFrame.H4,
            Granularity.H6: TimeFrame.H6,
            Granularity.H8: TimeFrame.H8,
            Granularity.H12: TimeFrame.H12,
            Granularity.D: TimeFrame.D1,
            Granularity.W: TimeFrame.W1,
            Granularity.M: TimeFrame.MN1
        }
        return mapping.get(granularity, TimeFrame.M1)
    
    def _convert_fxcm_to_oanda_candles(self, fxcm_candles: List[FXCMCandleData]) -> List[CandleData]:
        """Convert FXCM candles to OANDA format"""
        oanda_candles = []
        for fxcm_candle in fxcm_candles:
            oanda_candle = CandleData(
                time=fxcm_candle.time,
                open=fxcm_candle.open,
                high=fxcm_candle.high,
                low=fxcm_candle.low,
                close=fxcm_candle.close,
                volume=fxcm_candle.volume,
                complete=True
            )
            oanda_candles.append(oanda_candle)
        return oanda_candles
    
    def _convert_fxcm_to_oanda_price(self, fxcm_data: FXCMPriceData) -> PriceData:
        """Convert FXCM price data to OANDA format"""
        return PriceData(
            instrument=fxcm_data.symbol,
            time=fxcm_data.time,
            bid=fxcm_data.bid,
            ask=fxcm_data.ask,
            spread=fxcm_data.spread
        )
    
    def _convert_fxcm_to_oanda_pricing(self, fxcm_offers: Dict[str, Any]) -> Dict[str, Any]:
        """Convert FXCM offers to OANDA pricing format"""
        prices = []
        for offer in fxcm_offers.get('offers', []):
            price_data = {
                'instrument': offer['symbol'],
                'time': offer['time'],
                'bids': [{'price': str(offer['bid'])}],
                'asks': [{'price': str(offer['ask'])}]
            }
            prices.append(price_data)
        
        return {'prices': prices}
    
    def _convert_free_forex_to_oanda_candles(self, free_forex_candles: List[FreeForexCandleData]) -> List[CandleData]:
        """Convert FreeForex candles to OANDA format"""
        oanda_candles = []
        for free_forex_candle in free_forex_candles:
            oanda_candle = CandleData(
                time=free_forex_candle.time,
                open=free_forex_candle.open,
                high=free_forex_candle.high,
                low=free_forex_candle.low,
                close=free_forex_candle.close,
                volume=free_forex_candle.volume,
                complete=True
            )
            oanda_candles.append(oanda_candle)
        return oanda_candles
    
    def _convert_free_forex_to_oanda_price(self, free_forex_data: FreeForexPriceData) -> PriceData:
        """Convert FreeForex price data to OANDA format"""
        return PriceData(
            instrument=free_forex_data.symbol,
            time=free_forex_data.time,
            bid=free_forex_data.bid,
            ask=free_forex_data.ask,
            spread=free_forex_data.spread
        )
    
    def _convert_free_forex_to_oanda_pricing(self, free_forex_prices: List[FreeForexPriceData]) -> Dict[str, Any]:
        """Convert FreeForex prices to OANDA pricing format"""
        prices = []
        for price in free_forex_prices:
            price_data = {
                'instrument': price.symbol,
                'time': price.time.isoformat() + 'Z',
                'bids': [{'price': str(price.bid)}],
                'asks': [{'price': str(price.ask)}]
            }
            prices.append(price_data)
        
        return {'prices': prices}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of all brokers"""
        status = {
            'total_brokers': len(self.brokers),
            'healthy_brokers': len(self.get_healthy_brokers()),
            'brokers': {}
        }
        
        for name, broker in self.brokers.items():
            status['brokers'][name] = {
                'status': broker.status.value,
                'priority': broker.priority,
                'error_count': broker.error_count,
                'last_check': broker.last_check.isoformat()
            }
        
        return status

# Example usage and testing
async def test_broker_manager():
    """Test broker manager functionality"""
    import os
    
    # Create broker manager
    manager = BrokerManager()
    
    # Add OANDA broker
    oanda_api_key = os.getenv("OANDA_API_KEY")
    oanda_account_id = os.getenv("OANDA_ACCOUNT_ID")
    
    if oanda_api_key and oanda_account_id:
        oanda_client = OANDAClient(oanda_api_key, oanda_account_id)
        await manager.add_broker("oanda", oanda_client, priority=1)
    
    # Add FXCM broker
    fxcm_token = os.getenv("FXCM_TOKEN")
    
    if fxcm_token:
        fxcm_client = FXCMClient(fxcm_token)
        await manager.add_broker("fxcm", fxcm_client, priority=2)
    
    # Start health monitoring
    await manager.start_health_monitoring()
    
    try:
        # Test status
        status = await manager.get_status()
        print(f"Broker status: {status}")
        
        # Test getting candles
        try:
            candles = await manager.get_candles("EUR_USD", Granularity.M1, count=5)
            print(f"Retrieved {len(candles)} candles")
        except Exception as e:
            print(f"Failed to get candles: {e}")
        
        # Test getting pricing
        try:
            pricing = await manager.get_pricing(["EUR_USD", "GBP_USD"])
            print(f"Retrieved pricing for {len(pricing.get('prices', []))} instruments")
        except Exception as e:
            print(f"Failed to get pricing: {e}")
        
    finally:
        await manager.stop_health_monitoring()

if __name__ == "__main__":
    asyncio.run(test_broker_manager())
