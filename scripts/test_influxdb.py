#!/usr/bin/env python3
"""
Test InfluxDB connectivity and basic operations
"""
import os
import sys
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_influxdb_connection():
    """Test basic InfluxDB connection"""
    try:
        # Test health endpoint
        health_url = "http://localhost:8086/health"
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"InfluxDB health: {response.json()}")
            return True
        else:
            logger.error(f"InfluxDB health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to connect to InfluxDB: {e}")
        return False

def test_influxdb_setup():
    """Test if InfluxDB is properly set up"""
    try:
        # Test setup endpoint
        setup_url = "http://localhost:8086/api/v2/setup"
        response = requests.get(setup_url, timeout=10)
        
        if response.status_code == 200:
            setup_data = response.json()
            logger.info(f"InfluxDB setup status: {setup_data}")
            return setup_data.get('allowed', False)
        else:
            logger.error(f"InfluxDB setup check failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to check InfluxDB setup: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Testing InfluxDB connectivity...")
    
    # Test basic connection
    if not test_influxdb_connection():
        logger.error("❌ InfluxDB connection failed")
        return False
    
    # Test setup status
    if not test_influxdb_setup():
        logger.error("❌ InfluxDB setup check failed")
        return False
    
    logger.info("✅ InfluxDB is running and accessible")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
