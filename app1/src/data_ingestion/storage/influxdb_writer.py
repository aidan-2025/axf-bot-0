import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import json

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.rest import ApiException

from ..brokers.broker_manager import PriceData, CandleData
from ..config import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class InfluxDBConfig:
    """Configuration for InfluxDB connection"""
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "axf-bot"
    bucket: str = "forex_data"
    username: str = "admin"
    password: str = "password"
    timeout: int = 10000
    retries: int = 3
    batch_size: int = 1000
    flush_interval: int = 1  # seconds
    
    @classmethod
    def from_env(cls) -> 'InfluxDBConfig':
        """Create configuration from environment variables"""
        import os
        return cls(
            url=os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
            token=os.getenv('INFLUXDB_TOKEN', ''),
            org=os.getenv('INFLUXDB_ORG', 'axf-bot'),
            bucket=os.getenv('INFLUXDB_BUCKET', 'forex_data'),
            username=os.getenv('INFLUXDB_USERNAME', 'admin'),
            password=os.getenv('INFLUXDB_PASSWORD', 'password'),
            timeout=int(os.getenv('INFLUXDB_TIMEOUT', '10000')),
            retries=int(os.getenv('INFLUXDB_RETRIES', '3')),
            batch_size=int(os.getenv('INFLUXDB_BATCH_SIZE', '1000')),
            flush_interval=int(os.getenv('INFLUXDB_FLUSH_INTERVAL', '1'))
        )

class InfluxDBWriter:
    """InfluxDB writer for time series market data"""
    
    def __init__(self, config: InfluxDBConfig = InfluxDBConfig()):
        self.config = config
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None
        self.connected = False
        self.batch_buffer: List[Point] = []
        self.batch_lock = asyncio.Lock()
        logger.info(f"InfluxDBWriter initialized with config: {config}")

    def connect_sync(self) -> bool:
        """Connect to InfluxDB synchronously"""
        try:
            if self.connected:
                return True
            
            # Create client
            self.client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=self.config.timeout
            )
            
            # Test connection
            self.query_api = self.client.query_api()
            self.write_api = self.client.write_api(
                write_options=SYNCHRONOUS
            )
            
            # Test query to verify connection
            try:
                test_query = f'from(bucket: "{self.config.bucket}") |> range(start: -1m) |> limit(n: 1)'
                self.query_api.query(test_query)
                self.connected = True
                logger.info("Successfully connected to InfluxDB")
                return True
            except Exception as e:
                logger.error(f"Failed to test InfluxDB connection: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            return False

    async def connect(self) -> bool:
        """Establishes connection to InfluxDB"""
        try:
            if self.client:
                self.client.close()
            
            # Create client
            self.client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=self.config.timeout
            )
            
            # Test connection
            self.query_api = self.client.query_api()
            self.write_api = self.client.write_api(
                write_options=ASYNCHRONOUS,
                batch_size=self.config.batch_size,
                flush_interval=self.config.flush_interval * 1000  # Convert to milliseconds
            )
            
            # Test query to verify connection
            await self._test_connection()
            
            self.connected = True
            logger.info("Successfully connected to InfluxDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            self.connected = False
            return False

    async def _test_connection(self):
        """Test InfluxDB connection with a simple query"""
        try:
            # Try to get bucket info
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets()
            logger.info(f"Found {len(buckets.buckets)} buckets in InfluxDB")
        except Exception as e:
            logger.warning(f"InfluxDB connection test failed: {e}")

    async def disconnect(self):
        """Closes InfluxDB connection"""
        if self.write_api:
            self.write_api.close()
        if self.client:
            self.client.close()
        self.connected = False
        logger.info("Disconnected from InfluxDB")

    async def is_healthy(self) -> bool:
        """Checks if InfluxDB connection is healthy"""
        if not self.connected or not self.client:
            return False
        try:
            await self._test_connection()
            return True
        except Exception as e:
            logger.warning(f"InfluxDB health check failed: {e}")
            return False

    # Price Data Writing
    async def write_price_data(self, price_data: PriceData, tags: Optional[Dict[str, str]] = None) -> bool:
        """Writes real-time price data to InfluxDB"""
        if not self.connected:
            return False
        
        try:
            point = Point("price_data") \
                .tag("instrument", price_data.instrument) \
                .tag("broker", tags.get("broker", "unknown") if tags else "unknown") \
                .field("bid", price_data.bid) \
                .field("ask", price_data.ask) \
                .field("spread", price_data.spread) \
                .time(price_data.time, WritePrecision.MS)
            
            # Add additional tags if provided
            if tags:
                for key, value in tags.items():
                    if key != "broker":
                        point = point.tag(key, str(value))
            
            await self._write_point(point)
            logger.debug(f"Wrote price data for {price_data.instrument}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write price data for {price_data.instrument}: {e}")
            return False

    async def write_candle_data(self, candle_data: CandleData, instrument: str, 
                               granularity: str, tags: Optional[Dict[str, str]] = None) -> bool:
        """Writes historical candle data to InfluxDB"""
        if not self.connected:
            return False
        
        try:
            point = Point("candle_data") \
                .tag("instrument", instrument) \
                .tag("granularity", granularity) \
                .tag("broker", tags.get("broker", "unknown") if tags else "unknown") \
                .field("open", candle_data.open) \
                .field("high", candle_data.high) \
                .field("low", candle_data.low) \
                .field("close", candle_data.close) \
                .field("volume", candle_data.volume) \
                .field("complete", candle_data.complete) \
                .time(candle_data.time, WritePrecision.MS)
            
            # Add additional tags if provided
            if tags:
                for key, value in tags.items():
                    if key not in ["broker", "instrument", "granularity"]:
                        point = point.tag(key, str(value))
            
            await self._write_point(point)
            logger.debug(f"Wrote candle data for {instrument} ({granularity})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write candle data for {instrument} ({granularity}): {e}")
            return False

    async def write_indicators(self, instrument: str, granularity: str, 
                              indicators: Dict[str, Any], 
                              tags: Optional[Dict[str, str]] = None) -> bool:
        """Writes technical indicators to InfluxDB"""
        if not self.connected:
            return False
        
        try:
            point = Point("indicators") \
                .tag("instrument", instrument) \
                .tag("granularity", granularity) \
                .tag("broker", tags.get("broker", "unknown") if tags else "unknown")
            
            # Add indicator fields
            for indicator_name, value in indicators.items():
                if isinstance(value, (int, float)) and not (value != value):  # Check for NaN
                    point = point.field(indicator_name, value)
            
            # Use current time for indicators
            point = point.time(datetime.now(), WritePrecision.MS)
            
            # Add additional tags if provided
            if tags:
                for key, value in tags.items():
                    if key not in ["broker", "instrument", "granularity"]:
                        point = point.tag(key, str(value))
            
            await self._write_point(point)
            logger.debug(f"Wrote indicators for {instrument} ({granularity})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write indicators for {instrument} ({granularity}): {e}")
            return False

    async def write_market_snapshot(self, snapshot: Dict[str, Any], 
                                   tags: Optional[Dict[str, str]] = None) -> bool:
        """Writes a complete market snapshot to InfluxDB"""
        if not self.connected:
            return False
        
        try:
            point = Point("market_snapshot") \
                .tag("instrument", snapshot.get("instrument", "unknown")) \
                .tag("broker", tags.get("broker", "unknown") if tags else "unknown")
            
            # Add price fields
            if "current_price" in snapshot:
                price = snapshot["current_price"]
                point = point.field("bid", price.get("bid", 0)) \
                           .field("ask", price.get("ask", 0)) \
                           .field("spread", price.get("spread", 0))
            
            # Add OHLCV fields
            for field in ["open_price", "high_price", "low_price", "close_price", "volume"]:
                if field in snapshot and snapshot[field] is not None:
                    point = point.field(field, snapshot[field])
            
            # Add indicator fields
            if "indicators" in snapshot:
                indicators = snapshot["indicators"]
                for indicator_name, value in indicators.items():
                    if isinstance(value, (int, float)) and not (value != value):  # Check for NaN
                        point = point.field(f"indicator_{indicator_name}", value)
            
            # Add sentiment and volatility
            if "sentiment_score" in snapshot and snapshot["sentiment_score"] is not None:
                point = point.field("sentiment_score", snapshot["sentiment_score"])
            if "volatility" in snapshot and snapshot["volatility"] is not None:
                point = point.field("volatility", snapshot["volatility"])
            
            # Use timestamp from snapshot or current time
            timestamp = snapshot.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            point = point.time(timestamp, WritePrecision.MS)
            
            # Add additional tags if provided
            if tags:
                for key, value in tags.items():
                    if key not in ["broker", "instrument"]:
                        point = point.tag(key, str(value))
            
            await self._write_point(point)
            logger.debug(f"Wrote market snapshot for {snapshot.get('instrument', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write market snapshot: {e}")
            return False

    async def _write_point(self, point: Point):
        """Writes a single point to InfluxDB (with batching)"""
        async with self.batch_lock:
            self.batch_buffer.append(point)
            
            # Flush if batch is full
            if len(self.batch_buffer) >= self.config.batch_size:
                await self._flush_batch()

    async def _flush_batch(self):
        """Flushes the current batch to InfluxDB"""
        if not self.batch_buffer:
            return
        
        try:
            points_to_write = self.batch_buffer.copy()
            self.batch_buffer.clear()
            
            # Write points
            self.write_api.write(bucket=self.config.bucket, record=points_to_write)
            logger.debug(f"Flushed {len(points_to_write)} points to InfluxDB")
            
        except Exception as e:
            logger.error(f"Failed to flush batch to InfluxDB: {e}")
            # Re-add points to buffer for retry
            self.batch_buffer.extend(points_to_write)

    async def flush(self):
        """Manually flush any pending points"""
        async with self.batch_lock:
            await self._flush_batch()

    # Data Querying
    async def query_price_data(self, instrument: str, start_time: datetime, 
                              end_time: Optional[datetime] = None, 
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Queries price data from InfluxDB"""
        if not self.connected:
            return []
        
        try:
            end_time = end_time or datetime.now()
            limit_clause = f"|> limit(n: {limit})" if limit else ""
            
            query = f'''
            from(bucket: "{self.config.bucket}")
              |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
              |> filter(fn: (r) => r._measurement == "price_data")
              |> filter(fn: (r) => r.instrument == "{instrument}")
              |> sort(columns: ["_time"])
              {limit_clause}
            '''
            
            result = self.query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        'time': record.get_time().isoformat(),
                        'instrument': record.values.get('instrument'),
                        'bid': record.values.get('_value') if record.get_field() == 'bid' else None,
                        'ask': record.values.get('_value') if record.get_field() == 'ask' else None,
                        'spread': record.values.get('_value') if record.get_field() == 'spread' else None,
                        'field': record.get_field()
                    })
            
            logger.debug(f"Queried {len(data)} price data points for {instrument}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to query price data for {instrument}: {e}")
            return []

    def query_candle_data_sync(self, instrument: str, granularity: str,
                              start_time: datetime, end_time: Optional[datetime] = None,
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Queries candle data from InfluxDB (synchronous version)"""
        if not self.connected:
            return []
        
        try:
            end_time = end_time or datetime.now()
            limit_clause = f"|> limit(n: {limit})" if limit else ""
            
            query = f'''
            from(bucket: "{self.config.bucket}")
              |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
              |> filter(fn: (r) => r._measurement == "candle_data")
              |> filter(fn: (r) => r.instrument == "{instrument}")
              |> filter(fn: (r) => r.granularity == "{granularity}")
              |> sort(columns: ["_time"])
              {limit_clause}
            '''
            
            result = self.query_api.query(query)
            
            # Group by time to reconstruct OHLCV
            candles = {}
            for table in result:
                for record in table.records:
                    time_key = record.get_time().isoformat()
                    if time_key not in candles:
                        candles[time_key] = {
                            'time': time_key,
                            'instrument': record.values.get('instrument'),
                            'granularity': record.values.get('granularity')
                        }
                    
                    field = record.get_field()
                    value = record.values.get('_value')
                    if field in ['open', 'high', 'low', 'close', 'volume', 'complete']:
                        candles[time_key][field] = value
            
            data = list(candles.values())
            data.sort(key=lambda x: x['time'])
            
            logger.debug(f"Queried {len(data)} candle data points for {instrument} ({granularity})")
            return data
            
        except Exception as e:
            logger.error(f"Failed to query candle data for {instrument} ({granularity}): {e}")
            return []

    async def query_candle_data(self, instrument: str, granularity: str,
                               start_time: datetime, end_time: Optional[datetime] = None,
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Queries candle data from InfluxDB"""
        if not self.connected:
            return []
        
        try:
            end_time = end_time or datetime.now()
            limit_clause = f"|> limit(n: {limit})" if limit else ""
            
            query = f'''
            from(bucket: "{self.config.bucket}")
              |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
              |> filter(fn: (r) => r._measurement == "candle_data")
              |> filter(fn: (r) => r.instrument == "{instrument}")
              |> filter(fn: (r) => r.granularity == "{granularity}")
              |> sort(columns: ["_time"])
              {limit_clause}
            '''
            
            result = self.query_api.query(query)
            
            # Group by time to reconstruct OHLCV
            candles = {}
            for table in result:
                for record in table.records:
                    time_key = record.get_time().isoformat()
                    if time_key not in candles:
                        candles[time_key] = {
                            'time': time_key,
                            'instrument': record.values.get('instrument'),
                            'granularity': record.values.get('granularity')
                        }
                    
                    field = record.get_field()
                    value = record.values.get('_value')
                    if field in ['open', 'high', 'low', 'close', 'volume', 'complete']:
                        candles[time_key][field] = value
            
            data = list(candles.values())
            data.sort(key=lambda x: x['time'])
            
            logger.debug(f"Queried {len(data)} candle data points for {instrument} ({granularity})")
            return data
            
        except Exception as e:
            logger.error(f"Failed to query candle data for {instrument} ({granularity}): {e}")
            return []

    async def query_indicators(self, instrument: str, granularity: str,
                              start_time: datetime, end_time: Optional[datetime] = None,
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Queries technical indicators from InfluxDB"""
        if not self.connected:
            return []
        
        try:
            end_time = end_time or datetime.now()
            limit_clause = f"|> limit(n: {limit})" if limit else ""
            
            query = f'''
            from(bucket: "{self.config.bucket}")
              |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
              |> filter(fn: (r) => r._measurement == "indicators")
              |> filter(fn: (r) => r.instrument == "{instrument}")
              |> filter(fn: (r) => r.granularity == "{granularity}")
              |> sort(columns: ["_time"])
              {limit_clause}
            '''
            
            result = self.query_api.query(query)
            
            # Group by time to reconstruct indicators
            indicators = {}
            for table in result:
                for record in table.records:
                    time_key = record.get_time().isoformat()
                    if time_key not in indicators:
                        indicators[time_key] = {
                            'time': time_key,
                            'instrument': record.values.get('instrument'),
                            'granularity': record.values.get('granularity')
                        }
                    
                    field = record.get_field()
                    value = record.values.get('_value')
                    if field and value is not None:
                        indicators[time_key][field] = value
            
            data = list(indicators.values())
            data.sort(key=lambda x: x['time'])
            
            logger.debug(f"Queried {len(data)} indicator data points for {instrument} ({granularity})")
            return data
            
        except Exception as e:
            logger.error(f"Failed to query indicators for {instrument} ({granularity}): {e}")
            return []

    # Utility Methods
    async def get_bucket_info(self) -> Dict[str, Any]:
        """Gets information about the InfluxDB bucket"""
        if not self.connected:
            return {"connected": False}
        
        try:
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets()
            
            bucket_info = None
            for bucket in buckets.buckets:
                if bucket.name == self.config.bucket:
                    bucket_info = {
                        "name": bucket.name,
                        "id": bucket.id,
                        "retention_rules": [rule.duration for rule in bucket.retention_rules],
                        "created": bucket.created_at.isoformat() if bucket.created_at else None
                    }
                    break
            
            return {
                "connected": True,
                "bucket": bucket_info,
                "total_buckets": len(buckets.buckets)
            }
            
        except Exception as e:
            logger.error(f"Failed to get bucket info: {e}")
            return {"connected": False, "error": str(e)}

    async def get_stats(self) -> Dict[str, Any]:
        """Gets InfluxDB statistics"""
        if not self.connected:
            return {"connected": False}
        
        try:
            # Get basic stats
            bucket_info = await self.get_bucket_info()
            
            # Get data point counts (approximate)
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            
            price_count_query = f'''
            from(bucket: "{self.config.bucket}")
              |> range(start: {yesterday.isoformat()}, stop: {now.isoformat()})
              |> filter(fn: (r) => r._measurement == "price_data")
              |> count()
            '''
            
            candle_count_query = f'''
            from(bucket: "{self.config.bucket}")
              |> range(start: {yesterday.isoformat()}, stop: {now.isoformat()})
              |> filter(fn: (r) => r._measurement == "candle_data")
              |> count()
            '''
            
            try:
                price_result = self.query_api.query(price_count_query)
                price_count = sum(1 for table in price_result for _ in table.records)
            except:
                price_count = 0
            
            try:
                candle_result = self.query_api.query(candle_count_query)
                candle_count = sum(1 for table in candle_result for _ in table.records)
            except:
                candle_count = 0
            
            return {
                "connected": True,
                "bucket_info": bucket_info,
                "data_points_last_24h": {
                    "price_data": price_count,
                    "candle_data": candle_count
                },
                "pending_batch_size": len(self.batch_buffer)
            }
            
        except Exception as e:
            logger.error(f"Failed to get InfluxDB stats: {e}")
            return {"connected": False, "error": str(e)}
