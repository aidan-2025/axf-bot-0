#!/usr/bin/env python3
"""
Simple InfluxDB setup using the web interface approach
"""
import os
import sys
import requests
import logging
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_influxdb_via_web():
    """Set up InfluxDB by accessing the web interface"""
    try:
        # Check if InfluxDB is accessible
        health_url = "http://localhost:8086/health"
        response = requests.get(health_url, timeout=10)
        
        if response.status_code != 200:
            logger.error("InfluxDB is not accessible")
            return False
        
        logger.info("InfluxDB is accessible")
        logger.info("Please visit http://localhost:8086 to complete the setup:")
        logger.info("1. Username: admin")
        logger.info("2. Password: password")
        logger.info("3. Organization: axf-bot")
        logger.info("4. Bucket: forex_data")
        logger.info("5. After setup, get the API token from the Data > Tokens section")
        logger.info("6. Set the token as INFLUXDB_TOKEN environment variable")
        
        return True
        
    except Exception as e:
        logger.error(f"Error accessing InfluxDB: {e}")
        return False

def create_sample_data_manually():
    """Create sample data using curl commands"""
    try:
        logger.info("Creating sample data using curl...")
        
        # Sample market data
        sample_data = []
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        timeframes = ['H1', 'H4']
        
        for i in range(10):  # Create 10 sample points
            for pair in currency_pairs:
                for tf in timeframes:
                    timestamp = datetime.now() - timedelta(hours=i)
                    open_price = 1.0 + random.uniform(-0.1, 0.1)
                    high_price = open_price + random.uniform(0, 0.01)
                    low_price = open_price - random.uniform(0, 0.01)
                    close_price = open_price + random.uniform(-0.005, 0.005)
                    volume = random.randint(1000, 10000)
                    
                    # Create line protocol data
                    line = f"market_data,currency_pair={pair},timeframe={tf} open={open_price},high={high_price},low={low_price},close={close_price},volume={volume} {int(timestamp.timestamp() * 1000000000)}"
                    sample_data.append(line)
        
        # Write sample data
        write_url = "http://localhost:8086/api/v2/write"
        headers = {
            "Authorization": "Token YOUR_TOKEN_HERE",
            "Content-Type": "text/plain; charset=utf-8"
        }
        params = {
            "org": "axf-bot",
            "bucket": "forex_data"
        }
        
        data = "\n".join(sample_data)
        
        logger.info("Sample data created (requires valid token):")
        logger.info(f"URL: {write_url}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Params: {params}")
        logger.info(f"Data preview: {data[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Setting up InfluxDB...")
    
    # Check InfluxDB accessibility
    if not setup_influxdb_via_web():
        logger.error("❌ InfluxDB setup failed")
        return False
    
    # Create sample data template
    create_sample_data_manually()
    
    logger.info("✅ InfluxDB setup instructions provided")
    logger.info("Please complete the web setup and update the token")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
