"""
Risk Configuration Manager

Provides centralized configuration management for the risk management system,
including dynamic threshold updates, configuration validation, and hot-reloading.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from ..models import RiskLevel, EventImpact, SentimentLevel
from ..core.risk_engine import RiskEngineConfig
from ..event_integration.event_monitor import EventMonitorConfig
from ..event_integration.sentiment_monitor import SentimentMonitorConfig
from ..controls.circuit_breakers import CircuitBreakerConfig
from ..monitoring.risk_dashboard import RiskDashboardConfig
from ..monitoring.alerting import AlertingConfig

logger = logging.getLogger(__name__)


@dataclass
class RiskSystemConfig:
    """Complete risk management system configuration"""
    # Core components
    risk_engine: RiskEngineConfig
    event_monitor: EventMonitorConfig
    sentiment_monitor: SentimentMonitorConfig
    circuit_breaker: CircuitBreakerConfig
    dashboard: RiskDashboardConfig
    alerting: AlertingConfig
    
    # System settings
    config_version: str = "1.0.0"
    last_updated: datetime = None
    auto_reload: bool = True
    config_file_path: str = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


class RiskConfigManager:
    """Manages risk management system configuration"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "risk_config.json"
        self.current_config: Optional[RiskSystemConfig] = None
        self.config_history: List[RiskSystemConfig] = []
        self.logger = logging.getLogger(__name__)
        
        # Load initial configuration
        self.load_config()
    
    def load_config(self) -> RiskSystemConfig:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self.current_config = self._deserialize_config(config_data)
                self.logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                self.current_config = self._create_default_config()
        else:
            self.current_config = self._create_default_config()
            self.save_config()
        
        return self.current_config
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_data = self._serialize_config(self.current_config)
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            # Add to history
            self.config_history.append(self.current_config)
            if len(self.config_history) > 10:  # Keep last 10 configs
                self.config_history.pop(0)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            # Create new config with updates
            current_dict = asdict(self.current_config)
            self._deep_update(current_dict, updates)
            
            # Deserialize to new config object
            new_config = self._deserialize_config(current_dict)
            new_config.last_updated = datetime.utcnow()
            
            # Validate new configuration
            if self._validate_config(new_config):
                self.current_config = new_config
                self.save_config()
                self.logger.info("Configuration updated successfully")
                return True
            else:
                self.logger.error("Configuration validation failed")
                return False
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
            return False
    
    def get_risk_thresholds(self) -> Dict[str, Any]:
        """Get current risk thresholds"""
        return {
            "risk_engine": {
                "var_confidence_level": self.current_config.risk_engine.var_confidence_level,
                "lookback_periods": self.current_config.risk_engine.lookback_periods,
                "high_impact_multiplier": self.current_config.risk_engine.high_impact_multiplier,
                "critical_impact_multiplier": self.current_config.risk_engine.critical_impact_multiplier,
                "sentiment_weight": self.current_config.risk_engine.sentiment_weight,
                "news_weight": self.current_config.risk_engine.news_weight,
                "technical_weight": self.current_config.risk_engine.technical_weight
            },
            "circuit_breakers": {
                "max_daily_loss_threshold": self.current_config.circuit_breaker.max_daily_loss_threshold,
                "max_drawdown_threshold": self.current_config.circuit_breaker.max_drawdown_threshold,
                "max_consecutive_losses": self.current_config.circuit_breaker.max_consecutive_losses,
                "max_position_size_threshold": self.current_config.circuit_breaker.max_position_size_threshold
            },
            "event_impact": {
                "min_impact": self.current_config.event_monitor.min_impact_level.value,
                "high_impact_threshold": self.current_config.event_monitor.high_impact_threshold,
                "critical_impact_threshold": self.current_config.event_monitor.critical_impact_threshold
            },
            "sentiment": {
                "very_bearish": self.current_config.sentiment_monitor.very_bearish_threshold,
                "bearish": self.current_config.sentiment_monitor.bearish_threshold,
                "neutral": self.current_config.sentiment_monitor.neutral_threshold,
                "bullish": self.current_config.sentiment_monitor.bullish_threshold,
                "very_bullish": self.current_config.sentiment_monitor.very_bullish_threshold
            }
        }
    
    def update_risk_thresholds(self, thresholds: Dict[str, Any]) -> bool:
        """Update risk thresholds"""
        updates = {}
        
        # Risk engine settings
        if "risk_engine" in thresholds:
            risk_engine_settings = thresholds["risk_engine"]
            updates["risk_engine"] = {
                "var_confidence_level": risk_engine_settings.get("var_confidence_level"),
                "lookback_periods": risk_engine_settings.get("lookback_periods"),
                "high_impact_multiplier": risk_engine_settings.get("high_impact_multiplier"),
                "critical_impact_multiplier": risk_engine_settings.get("critical_impact_multiplier"),
                "sentiment_weight": risk_engine_settings.get("sentiment_weight"),
                "news_weight": risk_engine_settings.get("news_weight"),
                "technical_weight": risk_engine_settings.get("technical_weight")
            }
        
        # Circuit breaker thresholds
        if "circuit_breakers" in thresholds:
            cb_thresholds = thresholds["circuit_breakers"]
            updates["circuit_breaker"] = {
                "max_daily_loss_threshold": cb_thresholds.get("max_daily_loss_threshold"),
                "max_drawdown_threshold": cb_thresholds.get("max_drawdown_threshold"),
                "max_consecutive_losses": cb_thresholds.get("max_consecutive_losses"),
                "max_position_size_threshold": cb_thresholds.get("max_position_size_threshold")
            }
        
        # Event impact thresholds
        if "event_impact" in thresholds:
            event_thresholds = thresholds["event_impact"]
            updates["event_monitor"] = {
                "min_impact_level": event_thresholds.get("min_impact"),
                "high_impact_threshold": event_thresholds.get("high_impact_threshold"),
                "critical_impact_threshold": event_thresholds.get("critical_impact_threshold")
            }
        
        # Sentiment thresholds
        if "sentiment" in thresholds:
            sentiment_thresholds = thresholds["sentiment"]
            updates["sentiment_monitor"] = {
                "very_bearish_threshold": sentiment_thresholds.get("very_bearish"),
                "bearish_threshold": sentiment_thresholds.get("bearish"),
                "neutral_threshold": sentiment_thresholds.get("neutral"),
                "bullish_threshold": sentiment_thresholds.get("bullish"),
                "very_bullish_threshold": sentiment_thresholds.get("very_bullish")
            }
        
        return self.update_config(updates)
    
    def create_preset_config(self, preset_name: str) -> bool:
        """Create configuration preset"""
        presets = {
            "conservative": self._create_conservative_config(),
            "moderate": self._create_moderate_config(),
            "aggressive": self._create_aggressive_config(),
            "crypto": self._create_crypto_config(),
            "forex": self._create_forex_config()
        }
        
        if preset_name not in presets:
            self.logger.error(f"Unknown preset: {preset_name}")
            return False
        
        self.current_config = presets[preset_name]
        return self.save_config()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "version": self.current_config.config_version,
            "last_updated": self.current_config.last_updated.isoformat(),
            "auto_reload": self.current_config.auto_reload,
            "config_file": self.config_file,
            "history_count": len(self.config_history),
            "thresholds": self.get_risk_thresholds()
        }
    
    def _create_default_config(self) -> RiskSystemConfig:
        """Create default configuration"""
        return RiskSystemConfig(
            risk_engine=RiskEngineConfig(),
            event_monitor=EventMonitorConfig(),
            sentiment_monitor=SentimentMonitorConfig(),
            circuit_breaker=CircuitBreakerConfig(),
            dashboard=RiskDashboardConfig(),
            alerting=AlertingConfig()
        )
    
    def _create_conservative_config(self) -> RiskSystemConfig:
        """Create conservative risk configuration"""
        return RiskSystemConfig(
            risk_engine=RiskEngineConfig(
                var_confidence_level=0.99,
                lookback_periods=500,
                high_impact_multiplier=3.0,
                critical_impact_multiplier=5.0,
                sentiment_weight=0.4,
                news_weight=0.5,
                technical_weight=0.1
            ),
            event_monitor=EventMonitorConfig(
                min_impact_level=EventImpact.LOW,
                high_impact_threshold=1,
                critical_impact_threshold=1
            ),
            sentiment_monitor=SentimentMonitorConfig(
                very_bearish_threshold=-0.6,
                bearish_threshold=-0.3,
                neutral_threshold=-0.1,
                bullish_threshold=0.1,
                very_bullish_threshold=0.6
            ),
            circuit_breaker=CircuitBreakerConfig(
                max_daily_loss_threshold=0.02,
                max_drawdown_threshold=0.05,
                max_consecutive_losses=3,
                max_position_size_threshold=0.1
            ),
            dashboard=RiskDashboardConfig(),
            alerting=AlertingConfig()
        )
    
    def _create_moderate_config(self) -> RiskSystemConfig:
        """Create moderate risk configuration"""
        return RiskSystemConfig(
            risk_engine=RiskEngineConfig(
                var_confidence_level=0.95,
                lookback_periods=252,
                high_impact_multiplier=2.0,
                critical_impact_multiplier=3.0,
                sentiment_weight=0.3,
                news_weight=0.4,
                technical_weight=0.3
            ),
            event_monitor=EventMonitorConfig(
                min_impact_level=EventImpact.MEDIUM,
                high_impact_threshold=1,
                critical_impact_threshold=1
            ),
            sentiment_monitor=SentimentMonitorConfig(
                very_bearish_threshold=-0.8,
                bearish_threshold=-0.5,
                neutral_threshold=-0.2,
                bullish_threshold=0.2,
                very_bullish_threshold=0.8
            ),
            circuit_breaker=CircuitBreakerConfig(
                max_daily_loss_threshold=0.05,
                max_drawdown_threshold=0.10,
                max_consecutive_losses=5,
                max_position_size_threshold=0.2
            ),
            dashboard=RiskDashboardConfig(),
            alerting=AlertingConfig()
        )
    
    def _create_aggressive_config(self) -> RiskSystemConfig:
        """Create aggressive risk configuration"""
        return RiskSystemConfig(
            risk_engine=RiskEngineConfig(
                var_confidence_level=0.90,
                lookback_periods=126,
                high_impact_multiplier=1.5,
                critical_impact_multiplier=2.0,
                sentiment_weight=0.2,
                news_weight=0.3,
                technical_weight=0.5
            ),
            event_monitor=EventMonitorConfig(
                min_impact_level=EventImpact.HIGH,
                high_impact_threshold=2,
                critical_impact_threshold=1
            ),
            sentiment_monitor=SentimentMonitorConfig(
                very_bearish_threshold=-0.9,
                bearish_threshold=-0.7,
                neutral_threshold=-0.3,
                bullish_threshold=0.3,
                very_bullish_threshold=0.9
            ),
            circuit_breaker=CircuitBreakerConfig(
                max_daily_loss_threshold=0.10,
                max_drawdown_threshold=0.20,
                max_consecutive_losses=7,
                max_position_size_threshold=0.5
            ),
            dashboard=RiskDashboardConfig(),
            alerting=AlertingConfig()
        )
    
    def _create_crypto_config(self) -> RiskSystemConfig:
        """Create crypto-specific risk configuration"""
        return RiskSystemConfig(
            risk_engine=RiskEngineConfig(
                var_confidence_level=0.95,
                lookback_periods=365,
                high_impact_multiplier=2.5,
                critical_impact_multiplier=4.0,
                sentiment_weight=0.4,
                news_weight=0.3,
                technical_weight=0.3
            ),
            event_monitor=EventMonitorConfig(
                min_impact_level=EventImpact.MEDIUM,
                high_impact_threshold=1,
                critical_impact_threshold=1
            ),
            sentiment_monitor=SentimentMonitorConfig(
                very_bearish_threshold=-0.8,
                bearish_threshold=-0.5,
                neutral_threshold=-0.2,
                bullish_threshold=0.2,
                very_bullish_threshold=0.8
            ),
            circuit_breaker=CircuitBreakerConfig(
                max_daily_loss_threshold=0.15,
                max_drawdown_threshold=0.30,
                max_consecutive_losses=5,
                max_position_size_threshold=0.3
            ),
            dashboard=RiskDashboardConfig(),
            alerting=AlertingConfig()
        )
    
    def _create_forex_config(self) -> RiskSystemConfig:
        """Create forex-specific risk configuration"""
        return RiskSystemConfig(
            risk_engine=RiskEngineConfig(
                var_confidence_level=0.99,
                lookback_periods=252,
                high_impact_multiplier=3.0,
                critical_impact_multiplier=5.0,
                sentiment_weight=0.2,
                news_weight=0.6,
                technical_weight=0.2
            ),
            event_monitor=EventMonitorConfig(
                min_impact_level=EventImpact.LOW,
                high_impact_threshold=1,
                critical_impact_threshold=1
            ),
            sentiment_monitor=SentimentMonitorConfig(
                very_bearish_threshold=-0.6,
                bearish_threshold=-0.3,
                neutral_threshold=-0.1,
                bullish_threshold=0.1,
                very_bullish_threshold=0.6
            ),
            circuit_breaker=CircuitBreakerConfig(
                max_daily_loss_threshold=0.03,
                max_drawdown_threshold=0.06,
                max_consecutive_losses=4,
                max_position_size_threshold=0.15
            ),
            dashboard=RiskDashboardConfig(),
            alerting=AlertingConfig()
        )
    
    def _serialize_config(self, config: RiskSystemConfig) -> Dict[str, Any]:
        """Serialize configuration to dictionary"""
        config_dict = asdict(config)
        
        # Convert datetime objects to ISO strings
        if config_dict.get("last_updated"):
            config_dict["last_updated"] = config_dict["last_updated"].isoformat()
        
        return config_dict
    
    def _deserialize_config(self, config_dict: Dict[str, Any]) -> RiskSystemConfig:
        """Deserialize configuration from dictionary"""
        # Convert ISO strings back to datetime objects
        if config_dict.get("last_updated"):
            if isinstance(config_dict["last_updated"], str):
                config_dict["last_updated"] = datetime.fromisoformat(config_dict["last_updated"])
        
        # Create individual config objects
        risk_engine = RiskEngineConfig(**config_dict.get("risk_engine", {}))
        event_monitor = EventMonitorConfig(**config_dict.get("event_monitor", {}))
        sentiment_monitor = SentimentMonitorConfig(**config_dict.get("sentiment_monitor", {}))
        circuit_breaker = CircuitBreakerConfig(**config_dict.get("circuit_breaker", {}))
        dashboard = RiskDashboardConfig(**config_dict.get("dashboard", {}))
        alerting = AlertingConfig(**config_dict.get("alerting", {}))
        
        return RiskSystemConfig(
            risk_engine=risk_engine,
            event_monitor=event_monitor,
            sentiment_monitor=sentiment_monitor,
            circuit_breaker=circuit_breaker,
            dashboard=dashboard,
            alerting=alerting,
            config_version=config_dict.get("config_version", "1.0.0"),
            last_updated=config_dict.get("last_updated"),
            auto_reload=config_dict.get("auto_reload", True),
            config_file_path=config_dict.get("config_file_path")
        )
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _validate_config(self, config: RiskSystemConfig) -> bool:
        """Validate configuration"""
        try:
            # Validate risk engine settings
            if config.risk_engine.var_confidence_level <= 0 or config.risk_engine.var_confidence_level > 1:
                self.logger.error("Invalid VaR confidence level")
                return False
            
            if config.risk_engine.lookback_periods <= 0:
                self.logger.error("Invalid lookback periods")
                return False
            
            # Validate circuit breaker limits
            if config.circuit_breaker.max_daily_loss_threshold <= 0 or config.circuit_breaker.max_daily_loss_threshold > 1:
                self.logger.error("Invalid daily loss threshold")
                return False
            
            if config.circuit_breaker.max_drawdown_threshold <= 0 or config.circuit_breaker.max_drawdown_threshold > 1:
                self.logger.error("Invalid max drawdown threshold")
                return False
            
            # Validate sentiment thresholds are in ascending order
            sentiment_thresholds = [
                config.sentiment_monitor.very_bearish_threshold,
                config.sentiment_monitor.bearish_threshold,
                config.sentiment_monitor.neutral_threshold,
                config.sentiment_monitor.bullish_threshold,
                config.sentiment_monitor.very_bullish_threshold
            ]
            if sentiment_thresholds != sorted(sentiment_thresholds):
                self.logger.error("Sentiment thresholds must be in ascending order")
                return False
            
            # Validate sentiment thresholds are within bounds
            for threshold in sentiment_thresholds:
                if threshold < -1 or threshold > 1:
                    self.logger.error(f"Invalid sentiment threshold: {threshold}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
