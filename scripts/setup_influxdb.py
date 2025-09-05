#!/usr/bin/env python3
"""
InfluxDB Setup Script
Helps users set up InfluxDB for the data ingestion system
"""

import asyncio
import logging
import sys
import os
import requests
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app1.src.data_ingestion.storage.influxdb_writer import InfluxDBWriter, InfluxDBConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def check_influxdb_health():
    """Check if InfluxDB is running and accessible"""
    logger.info("Checking InfluxDB health...")
    
    try:
        # Try to connect to InfluxDB health endpoint
        response = requests.get("http://localhost:8086/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ InfluxDB is running and accessible")
            return True
        else:
            logger.warning(f"⚠️ InfluxDB responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to InfluxDB. Is it running?")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking InfluxDB health: {e}")
        return False

async def setup_influxdb_initial_user():
    """Set up initial InfluxDB user and organization"""
    logger.info("Setting up InfluxDB initial user and organization...")
    
    try:
        # Check if we can access the setup endpoint
        setup_url = "http://localhost:8086/api/v2/setup"
        
        # Try to get setup status
        response = requests.get(setup_url, timeout=5)
        if response.status_code == 200:
            setup_data = response.json()
            if setup_data.get('allowed', False):
                logger.info("✅ InfluxDB setup is available")
                
                # Create initial user and organization
                setup_payload = {
                    "username": "admin",
                    "password": "password",
                    "org": "axf-bot",
                    "bucket": "market_data"
                }
                
                setup_response = requests.post(setup_url, json=setup_payload, timeout=10)
                if setup_response.status_code == 201:
                    logger.info("✅ Initial user and organization created successfully")
                    return True
                else:
                    logger.error(f"❌ Failed to create initial setup: {setup_response.status_code} - {setup_response.text}")
                    return False
            else:
                logger.info("ℹ️ InfluxDB is already set up")
                return True
        else:
            logger.warning(f"⚠️ Setup endpoint returned status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error setting up InfluxDB: {e}")
        return False

async def get_influxdb_token():
    """Get InfluxDB API token"""
    logger.info("Getting InfluxDB API token...")
    
    try:
        # Try to authenticate and get token
        auth_url = "http://localhost:8086/api/v2/signin"
        auth_payload = {
            "username": "admin",
            "password": "password"
        }
        
        response = requests.post(auth_url, json=auth_payload, timeout=10)
        if response.status_code == 204:
            # Get the token from response headers
            token = response.headers.get('Set-Cookie', '').split('=')[1].split(';')[0]
            if token:
                logger.info("✅ Successfully obtained InfluxDB token")
                logger.info(f"🔑 Token: {token}")
                return token
            else:
                logger.error("❌ No token found in response")
                return None
        else:
            logger.error(f"❌ Authentication failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting InfluxDB token: {e}")
        return None

async def test_influxdb_connection(token: str):
    """Test InfluxDB connection with token"""
    logger.info("Testing InfluxDB connection with token...")
    
    try:
        config = InfluxDBConfig(
            url="http://localhost:8086",
            token=token,
            org="axf-bot",
            bucket="market_data"
        )
        
        writer = InfluxDBWriter(config)
        connected = await writer.connect()
        
        if connected:
            logger.info("✅ InfluxDB connection with token successful")
            
            # Test writing a sample data point
            from app1.src.data_ingestion.brokers.broker_manager import PriceData
            test_price = PriceData(
                instrument="EUR_USD",
                time=datetime.now(),
                bid=1.0800,
                ask=1.0802,
                spread=0.0002
            )
            
            success = await writer.write_price_data(test_price, {"broker": "test"})
            if success:
                logger.info("✅ Successfully wrote test data to InfluxDB")
            else:
                logger.warning("⚠️ Failed to write test data")
            
            await writer.disconnect()
            return True
        else:
            logger.error("❌ InfluxDB connection with token failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing InfluxDB connection: {e}")
        return False

def update_env_file(token: str):
    """Update the environment file with the InfluxDB token"""
    logger.info("Updating environment file with InfluxDB token...")
    
    try:
        env_file = "env.development"
        if os.path.exists(env_file):
            # Read current content
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace the token placeholder
            updated_content = content.replace(
                "INFLUXDB_TOKEN=your_influxdb_token_here",
                f"INFLUXDB_TOKEN={token}"
            )
            
            # Write back to file
            with open(env_file, 'w') as f:
                f.write(updated_content)
            
            logger.info("✅ Environment file updated with InfluxDB token")
            return True
        else:
            logger.warning("⚠️ Environment file not found, please update manually")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error updating environment file: {e}")
        return False

async def main():
    """Main setup process"""
    logger.info("🚀 Starting InfluxDB Setup")
    logger.info("=" * 50)
    
    # Step 1: Check InfluxDB health
    if not await check_influxdb_health():
        logger.error("❌ InfluxDB is not accessible. Please start InfluxDB first.")
        logger.info("💡 Run: docker-compose up -d influxdb")
        return False
    
    # Step 2: Set up initial user and organization
    if not await setup_influxdb_initial_user():
        logger.error("❌ Failed to set up initial user and organization")
        return False
    
    # Step 3: Get API token
    token = await get_influxdb_token()
    if not token:
        logger.error("❌ Failed to get InfluxDB token")
        return False
    
    # Step 4: Test connection with token
    if not await test_influxdb_connection(token):
        logger.error("❌ Failed to test InfluxDB connection")
        return False
    
    # Step 5: Update environment file
    update_env_file(token)
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 InfluxDB setup completed successfully!")
    logger.info("=" * 50)
    logger.info("📝 Next steps:")
    logger.info("1. Your InfluxDB token has been saved to env.development")
    logger.info("2. You can now run the data ingestion service with InfluxDB storage")
    logger.info("3. Access InfluxDB UI at: http://localhost:8086")
    logger.info("4. Username: admin, Password: password")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
