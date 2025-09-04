"""
Configuration settings for Application 2: MetaTrader 4 Script Development Application
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "AXF Bot - MT4 EA Generator"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8001
    
    # Database
    database_url: str = "postgresql://username:password@localhost:5432/axf_bot_db"
    
    # MT4 Integration
    mt4_data_path: str = "/path/to/mt4/data"
    mt4_scripts_path: str = "/path/to/mt4/scripts"
    mt4_compiler_path: str = "/path/to/mt4/compiler"
    
    # Code Generation
    template_path: Path = Path("templates")
    output_path: Path = Path("generated")
    backup_path: Path = Path("backups")
    
    # Self-Evaluation
    performance_check_interval: int = 300  # seconds
    min_trades_for_evaluation: int = 10
    performance_degradation_threshold: float = 0.15  # 15% decline
    
    # Fault Detection
    fault_check_interval: int = 60  # seconds
    max_consecutive_losses: int = 3
    max_daily_drawdown_percent: float = 10.0
    
    # Backtesting
    backtest_data_path: Path = Path("backtesting/data")
    backtest_results_path: Path = Path("backtesting/results")
    default_backtest_period_days: int = 365
    
    # Risk Management
    default_risk_per_trade: float = 2.0
    default_max_daily_drawdown: float = 5.0
    default_max_open_positions: int = 3
    
    # Strategy Parameters
    default_fast_ma_period: int = 20
    default_slow_ma_period: int = 50
    default_rsi_period: int = 14
    default_rsi_oversold: float = 30.0
    default_rsi_overbought: float = 70.0
    
    # News Filter
    default_news_buffer_minutes: int = 120
    default_high_impact_events: List[str] = [
        "NFP", "FOMC", "ECB", "BOE", "RBA", "BOC", "RBNZ"
    ]
    
    # Time Filters
    default_trading_hours: str = "08:00-18:00"
    default_trade_on_friday: bool = False
    default_trade_on_monday: bool = True
    
    # File Paths
    logs_path: Path = Path("logs")
    temp_path: Path = Path("temp")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
