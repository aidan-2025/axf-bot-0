"""
Centralized Environment Variable Management
Handles all environment variables across the application
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str
    host: str
    port: int
    name: str
    user: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20

@dataclass
class RedisConfig:
    """Redis configuration settings"""
    url: str
    host: str
    port: int
    password: Optional[str] = None
    db: int = 0

@dataclass
class InfluxDBConfig:
    """InfluxDB configuration settings"""
    url: str
    host: str
    port: int
    token: Optional[str] = None
    org: str = "axf-bot"
    bucket: str = "trading-data"

@dataclass
class APIConfig:
    """API configuration settings"""
    fano_api_key: Optional[str] = None
    fano_base_url: str = "https://api.fano.ai"
    news_api_key: Optional[str] = None
    economic_calendar_api_key: Optional[str] = None

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

@dataclass
class MonitoringConfig:
    """Monitoring configuration settings"""
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    log_level: str = "INFO"
    metrics_interval: int = 30

class EnvironmentManager:
    """Centralized environment variable management"""
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file or ".env"
        self.load_environment()
    
    def load_environment(self):
        """Load environment variables from file"""
        env_path = Path(self.env_file)
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(
            url=os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "axf_bot_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20"))
        )
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        return RedisConfig(
            url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            db=int(os.getenv("REDIS_DB", "0"))
        )
    
    def get_influxdb_config(self) -> InfluxDBConfig:
        """Get InfluxDB configuration"""
        return InfluxDBConfig(
            url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
            host=os.getenv("INFLUXDB_HOST", "localhost"),
            port=int(os.getenv("INFLUXDB_PORT", "8086")),
            token=os.getenv("INFLUXDB_TOKEN"),
            org=os.getenv("INFLUXDB_ORG", "axf-bot"),
            bucket=os.getenv("INFLUXDB_BUCKET", "trading-data")
        )
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        return APIConfig(
            fano_api_key=os.getenv("FANO_API_KEY"),
            fano_base_url=os.getenv("FANO_BASE_URL", "https://api.fano.ai"),
            news_api_key=os.getenv("NEWS_API_KEY"),
            economic_calendar_api_key=os.getenv("ECONOMIC_CALENDAR_API_KEY")
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
            algorithm=os.getenv("ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig(
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            grafana_enabled=os.getenv("GRAFANA_ENABLED", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            metrics_interval=int(os.getenv("METRICS_INTERVAL", "30"))
        )
    
    def get_app_config(self, app_name: str) -> Dict[str, Any]:
        """Get application-specific configuration"""
        return {
            "app_name": app_name,
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "port": int(os.getenv(f"{app_name.upper()}_PORT", "8000" if app_name == "app1" else "8001")),
            "host": os.getenv("HOST", "0.0.0.0"),
            "workers": int(os.getenv("WORKERS", "1")),
            "reload": os.getenv("RELOAD", "false").lower() == "true"
        }
    
    def validate_required_vars(self, required_vars: list) -> bool:
        """Validate that required environment variables are set"""
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        return {
            "database": self.get_database_config().__dict__,
            "redis": self.get_redis_config().__dict__,
            "influxdb": self.get_influxdb_config().__dict__,
            "api": self.get_api_config().__dict__,
            "security": self.get_security_config().__dict__,
            "monitoring": self.get_monitoring_config().__dict__
        }

# Global environment manager instance
env_manager = EnvironmentManager()
