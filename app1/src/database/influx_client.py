"""
InfluxDB client for time series data operations
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.query_api import QueryApi

logger = logging.getLogger(__name__)

class InfluxDBClientWrapper:
    """Wrapper for InfluxDB operations"""
    
    def __init__(self):
        self.config = {
            'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
            'token': os.getenv('INFLUXDB_TOKEN', 'admin-token'),
            'org': os.getenv('INFLUXDB_ORG', 'axf-bot-org'),
            'bucket': os.getenv('INFLUXDB_BUCKET', 'axf-bot-data')
        }
        
        self.client = None
        self.write_api = None
        self.query_api = None
        
    def connect(self):
        """Connect to InfluxDB"""
        try:
            self.client = InfluxDBClient(
                url=self.config['url'],
                token=self.config['token'],
                org=self.config['org']
            )
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            # Test connection
            health = self.client.health()
            if health.status == "pass":
                logger.info("Connected to InfluxDB successfully")
                return True
            else:
                logger.error(f"InfluxDB health check failed: {health.status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from InfluxDB"""
        if self.client:
            self.client.close()
            self.client = None
            self.write_api = None
            self.query_api = None
    
    def write_market_data(self, currency_pair: str, timeframe: str, 
                         open_price: float, high_price: float, 
                         low_price: float, close_price: float, 
                         volume: int, timestamp: datetime = None):
        """Write market data point to InfluxDB"""
        try:
            if not self.write_api:
                raise Exception("Not connected to InfluxDB")
            
            point = Point("market_data") \
                .tag("currency_pair", currency_pair) \
                .tag("timeframe", timeframe) \
                .field("open", open_price) \
                .field("high", high_price) \
                .field("low", low_price) \
                .field("close", close_price) \
                .field("volume", volume) \
                .time(timestamp or datetime.utcnow())
            
            self.write_api.write(bucket=self.config['bucket'], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Error writing market data: {e}")
            return False
    
    def write_sentiment_data(self, currency_pair: str, overall_sentiment: float,
                           news_sentiment: float = None, social_sentiment: float = None,
                           technical_sentiment: float = None, confidence: float = None,
                           timestamp: datetime = None):
        """Write sentiment data point to InfluxDB"""
        try:
            if not self.write_api:
                raise Exception("Not connected to InfluxDB")
            
            point = Point("sentiment") \
                .tag("currency_pair", currency_pair) \
                .field("overall", overall_sentiment) \
                .time(timestamp or datetime.utcnow())
            
            if news_sentiment is not None:
                point.field("news", news_sentiment)
            if social_sentiment is not None:
                point.field("social", social_sentiment)
            if technical_sentiment is not None:
                point.field("technical", technical_sentiment)
            if confidence is not None:
                point.field("confidence", confidence)
            
            self.write_api.write(bucket=self.config['bucket'], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Error writing sentiment data: {e}")
            return False
    
    def write_strategy_performance(self, strategy_id: str, profit_loss: float,
                                 trades_count: int = None, win_rate: float = None,
                                 drawdown: float = None, sharpe_ratio: float = None,
                                 timestamp: datetime = None):
        """Write strategy performance data point to InfluxDB"""
        try:
            if not self.write_api:
                raise Exception("Not connected to InfluxDB")
            
            point = Point("strategy_performance") \
                .tag("strategy_id", strategy_id) \
                .field("profit_loss", profit_loss) \
                .time(timestamp or datetime.utcnow())
            
            if trades_count is not None:
                point.field("trades_count", trades_count)
            if win_rate is not None:
                point.field("win_rate", win_rate)
            if drawdown is not None:
                point.field("drawdown", drawdown)
            if sharpe_ratio is not None:
                point.field("sharpe_ratio", sharpe_ratio)
            
            self.write_api.write(bucket=self.config['bucket'], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Error writing strategy performance: {e}")
            return False
    
    def write_system_metrics(self, metric_type: str, metrics: Dict[str, float],
                           timestamp: datetime = None):
        """Write system metrics to InfluxDB"""
        try:
            if not self.write_api:
                raise Exception("Not connected to InfluxDB")
            
            point = Point("system_metrics") \
                .tag("metric_type", metric_type) \
                .time(timestamp or datetime.utcnow())
            
            for field, value in metrics.items():
                point.field(field, value)
            
            self.write_api.write(bucket=self.config['bucket'], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Error writing system metrics: {e}")
            return False
    
    def get_market_data(self, currency_pair: str, timeframe: str, 
                       start_time: datetime, end_time: datetime = None) -> List[Dict]:
        """Get market data from InfluxDB"""
        try:
            if not self.query_api:
                raise Exception("Not connected to InfluxDB")
            
            if end_time is None:
                end_time = datetime.utcnow()
            
            query = f'''
            from(bucket: "{self.config['bucket']}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r._measurement == "market_data")
            |> filter(fn: (r) => r.currency_pair == "{currency_pair}")
            |> filter(fn: (r) => r.timeframe == "{timeframe}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
            '''
            
            result = self.query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        'timestamp': record.get_time(),
                        'open': record.get_value_by_key('open'),
                        'high': record.get_value_by_key('high'),
                        'low': record.get_value_by_key('low'),
                        'close': record.get_value_by_key('close'),
                        'volume': record.get_value_by_key('volume')
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return []
    
    def get_sentiment_data(self, currency_pair: str, start_time: datetime, 
                          end_time: datetime = None) -> List[Dict]:
        """Get sentiment data from InfluxDB"""
        try:
            if not self.query_api:
                raise Exception("Not connected to InfluxDB")
            
            if end_time is None:
                end_time = datetime.utcnow()
            
            query = f'''
            from(bucket: "{self.config['bucket']}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r._measurement == "sentiment")
            |> filter(fn: (r) => r.currency_pair == "{currency_pair}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
            '''
            
            result = self.query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        'timestamp': record.get_time(),
                        'overall': record.get_value_by_key('overall'),
                        'news': record.get_value_by_key('news'),
                        'social': record.get_value_by_key('social'),
                        'technical': record.get_value_by_key('technical'),
                        'confidence': record.get_value_by_key('confidence')
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting sentiment data: {e}")
            return []
    
    def get_strategy_performance(self, strategy_id: str, start_time: datetime,
                               end_time: datetime = None) -> List[Dict]:
        """Get strategy performance data from InfluxDB"""
        try:
            if not self.query_api:
                raise Exception("Not connected to InfluxDB")
            
            if end_time is None:
                end_time = datetime.utcnow()
            
            query = f'''
            from(bucket: "{self.config['bucket']}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r._measurement == "strategy_performance")
            |> filter(fn: (r) => r.strategy_id == "{strategy_id}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
            '''
            
            result = self.query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        'timestamp': record.get_time(),
                        'profit_loss': record.get_value_by_key('profit_loss'),
                        'trades_count': record.get_value_by_key('trades_count'),
                        'win_rate': record.get_value_by_key('win_rate'),
                        'drawdown': record.get_value_by_key('drawdown'),
                        'sharpe_ratio': record.get_value_by_key('sharpe_ratio')
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return []
    
    def get_latest_market_data(self, currency_pair: str, timeframe: str) -> Optional[Dict]:
        """Get latest market data point"""
        try:
            if not self.query_api:
                raise Exception("Not connected to InfluxDB")
            
            query = f'''
            from(bucket: "{self.config['bucket']}")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == "market_data")
            |> filter(fn: (r) => r.currency_pair == "{currency_pair}")
            |> filter(fn: (r) => r.timeframe == "{timeframe}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"], desc: true)
            |> limit(n: 1)
            '''
            
            result = self.query_api.query(query)
            
            for table in result:
                for record in table.records:
                    return {
                        'timestamp': record.get_time(),
                        'open': record.get_value_by_key('open'),
                        'high': record.get_value_by_key('high'),
                        'low': record.get_value_by_key('low'),
                        'close': record.get_value_by_key('close'),
                        'volume': record.get_value_by_key('volume')
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest market data: {e}")
            return None

# Global instance
influx_client = InfluxDBClientWrapper()

def get_influx_client() -> InfluxDBClientWrapper:
    """Get InfluxDB client instance"""
    return influx_client
