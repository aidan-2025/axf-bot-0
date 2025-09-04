"""
Configuration settings for Application 1: AI-Powered Forex Strategy Generator
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "AXF Bot - Strategy Generator"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    database_url: str = "postgresql://username:password@localhost:5432/axf_bot_db"
    redis_url: str = "redis://localhost:6379/0"
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_org: str = "axf-bot"
    influxdb_bucket: str = "forex_data"
    
    # API Keys
    reuters_api_key: str = ""
    bloomberg_api_key: str = ""
    forex_factory_api_key: str = ""
    alpha_vantage_api_key: str = ""
    
    # Security
    secret_key: str = "your-secret-key-here"
    jwt_secret_key: str = "your-jwt-secret-key-here"
    encryption_key: str = "your-encryption-key-here"
    
    # Market Data
    supported_currency_pairs: List[str] = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
        "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
        "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
        "CADJPY", "CADCHF", "CADNZD",
        "CHFJPY", "NZDJPY", "NZDCHF"
    ]
    
    # Timeframes
    supported_timeframes: List[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
    
    # Strategy Generation
    max_strategies_per_generation: int = 10
    min_backtest_trades: int = 200
    min_profit_factor: float = 1.3
    max_drawdown_threshold: float = 0.20
    min_sharpe_ratio: float = 1.0
    
    # News Processing
    news_sources: List[str] = ["reuters", "bloomberg", "forex_factory"]
    sentiment_update_interval: int = 300  # seconds
    news_cache_ttl: int = 3600  # seconds
    
    # Economic Calendar
    high_impact_events: List[str] = [
        "NFP", "FOMC", "ECB_Rate_Decision", "BOE_Rate_Decision",
        "RBA_Rate_Decision", "BOC_Rate_Decision", "RBNZ_Rate_Decision"
    ]
    event_buffer_minutes: int = 120
    
    # Risk Management
    max_position_size_percent: float = 2.0
    max_daily_drawdown_percent: float = 5.0
    max_correlation_percent: float = 70.0
    
    # File Paths
    data_path: Path = Path("data")
    models_path: Path = Path("models")
    logs_path: Path = Path("logs")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
