#!/usr/bin/env python3
"""
Data Ingestion Configuration
Centralized configuration for broker APIs and data ingestion settings
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class BrokerConfig:
    """Configuration for a broker"""
    name: str
    enabled: bool
    priority: int
    api_key: Optional[str] = None
    account_id: Optional[str] = None
    token: Optional[str] = None
    environment: str = "practice"
    base_url: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds

@dataclass
class DataIngestionConfig:
    """Main configuration for data ingestion"""
    environment: Environment
    brokers: Dict[str, BrokerConfig]
    health_check_interval: int = 30  # seconds
    max_retries: int = 3
    retry_delay: int = 1  # seconds
    cache_ttl: int = 60  # seconds
    batch_size: int = 100
    max_latency_ms: int = 100

def load_broker_config() -> DataIngestionConfig:
    """Load broker configuration from environment variables"""
    
    # Get environment
    env_str = os.getenv("ENVIRONMENT", "development")
    environment = Environment(env_str)
    
    # OANDA Configuration
    oanda_config = BrokerConfig(
        name="oanda",
        enabled=os.getenv("OANDA_ENABLED", "true").lower() == "true",
        priority=int(os.getenv("OANDA_PRIORITY", "1")),
        api_key=os.getenv("OANDA_API_KEY"),
        account_id=os.getenv("OANDA_ACCOUNT_ID"),
        environment=os.getenv("OANDA_ENVIRONMENT", "practice"),
        base_url=os.getenv("OANDA_BASE_URL"),
        rate_limit=int(os.getenv("OANDA_RATE_LIMIT", "100")),
        timeout=int(os.getenv("OANDA_TIMEOUT", "30"))
    )
    
    # FXCM Configuration
    fxcm_config = BrokerConfig(
        name="fxcm",
        enabled=os.getenv("FXCM_ENABLED", "true").lower() == "true",
        priority=int(os.getenv("FXCM_PRIORITY", "2")),
        token=os.getenv("FXCM_TOKEN"),
        environment=os.getenv("FXCM_ENVIRONMENT", "demo"),
        base_url=os.getenv("FXCM_BASE_URL"),
        rate_limit=int(os.getenv("FXCM_RATE_LIMIT", "1000")),
        timeout=int(os.getenv("FXCM_TIMEOUT", "30"))
    )
    
    # Main configuration
    config = DataIngestionConfig(
        environment=environment,
        brokers={
            "oanda": oanda_config,
            "fxcm": fxcm_config
        },
        health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_delay=int(os.getenv("RETRY_DELAY", "1")),
        cache_ttl=int(os.getenv("CACHE_TTL", "60")),
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        max_latency_ms=int(os.getenv("MAX_LATENCY_MS", "100"))
    )
    
    return config

def get_enabled_brokers(config: DataIngestionConfig) -> Dict[str, BrokerConfig]:
    """Get only enabled brokers"""
    return {name: broker for name, broker in config.brokers.items() if broker.enabled}

def validate_config(config: DataIngestionConfig) -> bool:
    """Validate configuration"""
    errors = []
    
    # Check if at least one broker is enabled
    enabled_brokers = get_enabled_brokers(config)
    if not enabled_brokers:
        errors.append("At least one broker must be enabled")
    
    # Validate OANDA config if enabled
    if config.brokers["oanda"].enabled:
        if not config.brokers["oanda"].api_key:
            errors.append("OANDA_API_KEY is required when OANDA is enabled")
        if not config.brokers["oanda"].account_id:
            errors.append("OANDA_ACCOUNT_ID is required when OANDA is enabled")
    
    # Validate FXCM config if enabled
    if config.brokers["fxcm"].enabled:
        if not config.brokers["fxcm"].token:
            errors.append("FXCM_TOKEN is required when FXCM is enabled")
    
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        return False
    
    return True

# Default configuration
DEFAULT_CONFIG = DataIngestionConfig(
    environment=Environment.DEVELOPMENT,
    brokers={
        "oanda": BrokerConfig(
            name="oanda",
            enabled=True,
            priority=1,
            environment="practice"
        ),
        "fxcm": BrokerConfig(
            name="fxcm",
            enabled=True,
            priority=2,
            environment="demo"
        )
    }
)

# Load configuration
CONFIG = load_broker_config()

# Validate configuration
if not validate_config(CONFIG):
    print("Warning: Configuration validation failed. Using default settings.")
    CONFIG = DEFAULT_CONFIG
