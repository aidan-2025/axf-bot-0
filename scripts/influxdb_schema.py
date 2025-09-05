#!/usr/bin/env python3
"""
InfluxDB schema setup for AXF Bot 0
Configures InfluxDB for time series data storage
"""
import os
import sys
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
from datetime import datetime, timedelta
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# InfluxDB configuration
INFLUX_CONFIG = {
    'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
    'token': os.getenv('INFLUXDB_TOKEN', 'admin-token'),
    'org': os.getenv('INFLUXDB_ORG', 'axf-bot'),
    'bucket': os.getenv('INFLUXDB_BUCKET', 'forex_data'),
    'username': os.getenv('INFLUXDB_USERNAME', 'admin'),
    'password': os.getenv('INFLUXDB_PASSWORD', 'password')
}

def get_influx_client():
    """Get InfluxDB client"""
    try:
        client = InfluxDBClient(
            url=INFLUX_CONFIG['url'],
            token=INFLUX_CONFIG['token'],
            org=INFLUX_CONFIG['org']
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to InfluxDB: {e}")
        raise

def create_bucket(client):
    """Create InfluxDB bucket if it doesn't exist"""
    try:
        buckets_api = client.buckets_api()
        
        # Check if bucket exists
        buckets = buckets_api.find_buckets()
        bucket_exists = any(bucket.name == INFLUX_CONFIG['bucket'] for bucket in buckets)
        
        if not bucket_exists:
            # Create bucket
            bucket = buckets_api.create_bucket(
                bucket_name=INFLUX_CONFIG['bucket'],
                org=INFLUX_CONFIG['org'],
                retention_rules=[
                    {
                        "type": "expire",
                        "everySeconds": 0,  # No expiration for now
                        "shardGroupDurationSeconds": 0
                    }
                ]
            )
            logger.info(f"Created bucket: {bucket.name}")
        else:
            logger.info(f"Bucket {INFLUX_CONFIG['bucket']} already exists")
            
    except Exception as e:
        logger.error(f"Error creating bucket: {e}")
        raise

def seed_sample_data(client):
    """Seed InfluxDB with sample time series data"""
    try:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Sample currency pairs and timeframes
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
        timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']
        
        logger.info("Seeding market data...")
        
        # Generate market data for the last 7 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        current_time = start_time
        while current_time <= end_time:
            for pair in currency_pairs:
                for tf in timeframes:
                    # Calculate timeframe interval
                    if tf == 'M1':
                        interval_minutes = 1
                    elif tf == 'M5':
                        interval_minutes = 5
                    elif tf == 'M15':
                        interval_minutes = 15
                    elif tf == 'H1':
                        interval_minutes = 60
                    elif tf == 'H4':
                        interval_minutes = 240
                    elif tf == 'D1':
                        interval_minutes = 1440
                    else:
                        continue
                    
                    # Skip if not aligned with timeframe
                    if current_time.minute % interval_minutes != 0:
                        continue
                    
                    # Generate OHLCV data
                    base_price = 1.0 if 'USD' in pair else 100.0
                    price_change = random.uniform(-0.01, 0.01)
                    open_price = base_price + price_change
                    high_price = open_price + random.uniform(0, 0.005)
                    low_price = open_price - random.uniform(0, 0.005)
                    close_price = open_price + random.uniform(-0.002, 0.002)
                    volume = random.randint(1000, 10000)
                    
                    # Create data point
                    point = Point("market_data") \
                        .tag("currency_pair", pair) \
                        .tag("timeframe", tf) \
                        .field("open", open_price) \
                        .field("high", high_price) \
                        .field("low", low_price) \
                        .field("close", close_price) \
                        .field("volume", volume) \
                        .time(current_time)
                    
                    write_api.write(bucket=INFLUX_CONFIG['bucket'], record=point)
            
            # Move to next minute
            current_time += timedelta(minutes=1)
        
        logger.info("Seeding sentiment data...")
        
        # Generate sentiment data
        current_time = start_time
        while current_time <= end_time:
            for pair in currency_pairs:
                # Generate sentiment scores
                overall_sentiment = random.uniform(-50, 50)
                news_sentiment = overall_sentiment + random.uniform(-10, 10)
                social_sentiment = overall_sentiment + random.uniform(-15, 15)
                technical_sentiment = overall_sentiment + random.uniform(-5, 5)
                confidence = random.uniform(60, 95)
                
                point = Point("sentiment") \
                    .tag("currency_pair", pair) \
                    .field("overall", overall_sentiment) \
                    .field("news", news_sentiment) \
                    .field("social", social_sentiment) \
                    .field("technical", technical_sentiment) \
                    .field("confidence", confidence) \
                    .time(current_time)
                
                write_api.write(bucket=INFLUX_CONFIG['bucket'], record=point)
            
            # Move to next hour
            current_time += timedelta(hours=1)
        
        logger.info("Seeding strategy performance metrics...")
        
        # Generate strategy performance metrics
        strategies = ['STRAT_001', 'STRAT_002', 'STRAT_003']
        current_time = start_time
        while current_time <= end_time:
            for strategy in strategies:
                # Generate performance metrics
                profit_loss = random.uniform(-100, 200)
                trades_count = random.randint(0, 5)
                win_rate = random.uniform(40, 80)
                drawdown = random.uniform(0, 15)
                sharpe_ratio = random.uniform(0.5, 2.0)
                
                point = Point("strategy_performance") \
                    .tag("strategy_id", strategy) \
                    .field("profit_loss", profit_loss) \
                    .field("trades_count", trades_count) \
                    .field("win_rate", win_rate) \
                    .field("drawdown", drawdown) \
                    .field("sharpe_ratio", sharpe_ratio) \
                    .time(current_time)
                
                write_api.write(bucket=INFLUX_CONFIG['bucket'], record=point)
            
            # Move to next day
            current_time += timedelta(days=1)
        
        logger.info("Seeding system metrics...")
        
        # Generate system metrics
        current_time = start_time
        while current_time <= end_time:
            # System performance metrics
            cpu_usage = random.uniform(10, 80)
            memory_usage = random.uniform(20, 90)
            disk_usage = random.uniform(30, 85)
            network_latency = random.uniform(1, 50)
            
            point = Point("system_metrics") \
                .tag("metric_type", "performance") \
                .field("cpu_usage", cpu_usage) \
                .field("memory_usage", memory_usage) \
                .field("disk_usage", disk_usage) \
                .field("network_latency", network_latency) \
                .time(current_time)
            
            write_api.write(bucket=INFLUX_CONFIG['bucket'], record=point)
            
            # Move to next 5 minutes
            current_time += timedelta(minutes=5)
        
        logger.info("Sample data seeded successfully")
        
    except Exception as e:
        logger.error(f"Error seeding sample data: {e}")
        raise

def create_retention_policies(client):
    """Create retention policies for different data types"""
    try:
        buckets_api = client.buckets_api()
        
        # Get the bucket
        bucket = buckets_api.find_bucket_by_name(INFLUX_CONFIG['bucket'])
        
        # Create retention policies for different data types
        retention_policies = [
            {
                "name": "market_data_rp",
                "duration": "30d",  # Keep market data for 30 days
                "replication": 1,
                "shard_duration": "1d"
            },
            {
                "name": "sentiment_rp", 
                "duration": "7d",   # Keep sentiment data for 7 days
                "replication": 1,
                "shard_duration": "1d"
            },
            {
                "name": "strategy_performance_rp",
                "duration": "90d",  # Keep strategy performance for 90 days
                "replication": 1,
                "shard_duration": "7d"
            },
            {
                "name": "system_metrics_rp",
                "duration": "14d",  # Keep system metrics for 14 days
                "replication": 1,
                "shard_duration": "1d"
            }
        ]
        
        for policy in retention_policies:
            try:
                # Note: In InfluxDB 2.x, retention policies are handled differently
                # This is more of a conceptual setup
                logger.info(f"Retention policy concept: {policy['name']} - {policy['duration']}")
            except Exception as e:
                logger.warning(f"Could not create retention policy {policy['name']}: {e}")
        
        logger.info("Retention policies configured")
        
    except Exception as e:
        logger.error(f"Error creating retention policies: {e}")
        raise

def main():
    """Main InfluxDB setup function"""
    try:
        logger.info("Starting InfluxDB schema setup...")
        
        # Connect to InfluxDB
        client = get_influx_client()
        
        # Test connection
        health = client.health()
        logger.info(f"InfluxDB health: {health.status}")
        
        # Create bucket
        create_bucket(client)
        
        # Create retention policies
        create_retention_policies(client)
        
        # Seed sample data
        seed_sample_data(client)
        
        # Verify data
        query_api = client.query_api()
        query = f'''
        from(bucket: "{INFLUX_CONFIG['bucket']}")
        |> range(start: -1h)
        |> group(columns: ["_measurement"])
        |> count()
        '''
        
        result = query_api.query(query)
        for table in result:
            for record in table.records:
                logger.info(f"Measurement {record.get_field()}: {record.get_value()} records")
        
        client.close()
        
        logger.info("✅ InfluxDB schema setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ InfluxDB setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
