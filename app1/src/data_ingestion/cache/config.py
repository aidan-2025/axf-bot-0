import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class CacheConfig(BaseSettings):
    """Configuration for Redis caching"""
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    db: int = Field(0, env="REDIS_DB")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    retry_on_timeout: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    
    # Cache TTL settings (in seconds)
    price_data_ttl: int = Field(60, env="CACHE_PRICE_DATA_TTL")  # 1 minute for real-time prices
    candle_data_ttl: int = Field(3600, env="CACHE_CANDLE_DATA_TTL")  # 1 hour for historical candles
    indicators_ttl: int = Field(300, env="CACHE_INDICATORS_TTL")  # 5 minutes for calculated indicators
    api_response_ttl: int = Field(1800, env="CACHE_API_RESPONSE_TTL")  # 30 minutes for API responses
    health_check_ttl: int = Field(30, env="CACHE_HEALTH_CHECK_TTL")  # 30 seconds for health checks
    
    # Cache enable/disable flags
    enabled: bool = Field(True, env="CACHE_ENABLED")
    price_caching_enabled: bool = Field(True, env="CACHE_PRICE_ENABLED")
    candle_caching_enabled: bool = Field(True, env="CACHE_CANDLE_ENABLED")
    indicators_caching_enabled: bool = Field(True, env="CACHE_INDICATORS_ENABLED")
    api_response_caching_enabled: bool = Field(True, env="CACHE_API_RESPONSE_ENABLED")
    
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

# Global cache configuration instance
CACHE_CONFIG = CacheConfig()
