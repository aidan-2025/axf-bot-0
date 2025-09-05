#!/usr/bin/env python3
"""
Get InfluxDB Token
Simple script to help users get their InfluxDB token
"""

import requests
import json
import sys

def get_influxdb_token():
    """Get InfluxDB token using the API"""
    print("🔍 Attempting to get InfluxDB token...")
    
    # Try to get the token using the API
    try:
        # First, try to get setup status
        setup_url = "http://localhost:8086/api/v2/setup"
        response = requests.get(setup_url, timeout=5)
        
        if response.status_code == 200:
            setup_data = response.json()
            if setup_data.get('allowed', False):
                print("ℹ️ InfluxDB setup is available. Please complete setup first.")
                print("🌐 Open http://localhost:8086 in your browser")
                print("📝 Create a user with username 'admin' and password 'password'")
                print("🏢 Create an organization named 'axf-bot'")
                print("🪣 Create a bucket named 'market_data'")
                return None
            else:
                print("✅ InfluxDB is already set up")
        
        # Try to authenticate
        auth_url = "http://localhost:8086/api/v2/signin"
        auth_payload = {
            "username": "admin",
            "password": "password"
        }
        
        response = requests.post(auth_url, json=auth_payload, timeout=10)
        
        if response.status_code == 204:
            # Try to get token from cookies
            cookies = response.cookies
            token = None
            
            for cookie in cookies:
                if cookie.name == 'session':
                    token = cookie.value
                    break
            
            if token:
                print(f"✅ Successfully obtained InfluxDB token: {token}")
                return token
            else:
                print("❌ No token found in response cookies")
                return None
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to InfluxDB. Is it running?")
        print("💡 Run: docker-compose up -d influxdb")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def update_env_file(token):
    """Update the environment file with the token"""
    try:
        env_file = "env.development"
        
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
        
        print(f"✅ Updated {env_file} with InfluxDB token")
        return True
        
    except Exception as e:
        print(f"❌ Error updating environment file: {e}")
        return False

def main():
    """Main function"""
    print("🚀 InfluxDB Token Setup")
    print("=" * 40)
    
    # Check if InfluxDB is running
    try:
        response = requests.get("http://localhost:8086/health", timeout=5)
        if response.status_code != 200:
            print("❌ InfluxDB is not running or not accessible")
            return False
    except:
        print("❌ InfluxDB is not running. Please start it first:")
        print("💡 Run: docker-compose up -d influxdb")
        return False
    
    # Get token
    token = get_influxdb_token()
    
    if token:
        # Update environment file
        if update_env_file(token):
            print("\n🎉 Setup completed successfully!")
            print("📝 Your InfluxDB token has been saved to env.development")
            print("🚀 You can now run the data ingestion service with InfluxDB storage")
        else:
            print("⚠️ Token obtained but failed to update environment file")
            print(f"🔑 Please manually set INFLUXDB_TOKEN={token} in env.development")
    else:
        print("\n❌ Failed to get InfluxDB token")
        print("💡 Manual setup required:")
        print("1. Open http://localhost:8086 in your browser")
        print("2. Complete the initial setup")
        print("3. Create a user with username 'admin' and password 'password'")
        print("4. Create an organization named 'axf-bot'")
        print("5. Create a bucket named 'market_data'")
        print("6. Generate an API token")
        print("7. Set INFLUXDB_TOKEN=your_token_here in env.development")
    
    return token is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
