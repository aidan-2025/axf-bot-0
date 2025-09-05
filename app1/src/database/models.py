"""
SQLAlchemy models for the AXF Bot 0 database
"""
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Date, 
    DECIMAL, JSON, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .connection import Base
from datetime import datetime
from typing import Optional, List, Dict, Any


class CurrencyPair(Base):
    """Currency pairs master table"""
    __tablename__ = "currency_pairs"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    base_currency = Column(String(3), nullable=False)
    quote_currency = Column(String(3), nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    pip_value = Column(DECIMAL(10, 5), default=0.0001)
    min_lot_size = Column(DECIMAL(8, 2), default=0.01)
    max_lot_size = Column(DECIMAL(8, 2), default=100.0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    market_data = relationship("MarketData", back_populates="currency_pair")
    sentiment_scores = relationship("SentimentScore", back_populates="currency_pair")
    strategy_currency_pairs = relationship("StrategyCurrencyPair", back_populates="currency_pair")
    strategy_trades = relationship("StrategyTrade", back_populates="currency_pair")


class Timeframe(Base):
    """Trading timeframes table"""
    __tablename__ = "timeframes"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(10), unique=True, nullable=False, index=True)
    minutes = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    market_data = relationship("MarketData", back_populates="timeframe")
    strategy_timeframes = relationship("StrategyTimeframe", back_populates="timeframe")
    strategy_trades = relationship("StrategyTrade", back_populates="timeframe")


class MarketData(Base):
    """Historical price data table"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    currency_pair_id = Column(Integer, ForeignKey("currency_pairs.id"), nullable=False)
    timeframe_id = Column(Integer, ForeignKey("timeframes.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(DECIMAL(12, 5), nullable=False)
    high_price = Column(DECIMAL(12, 5), nullable=False)
    low_price = Column(DECIMAL(12, 5), nullable=False)
    close_price = Column(DECIMAL(12, 5), nullable=False)
    volume = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    currency_pair = relationship("CurrencyPair", back_populates="market_data")
    timeframe = relationship("Timeframe", back_populates="market_data")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('currency_pair_id', 'timeframe_id', 'timestamp', name='uq_market_data_pair_timeframe_timestamp'),
        Index('idx_market_data_pair_timeframe', 'currency_pair_id', 'timeframe_id'),
        Index('idx_market_data_timestamp', 'timestamp'),
        Index('idx_market_data_pair_timestamp', 'currency_pair_id', 'timestamp'),
    )


class NewsEvent(Base):
    """Economic calendar and news events table"""
    __tablename__ = "news_events"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    currency = Column(String(3), nullable=False, index=True)
    impact_level = Column(String(10), nullable=False, index=True)  # 'low', 'medium', 'high'
    event_time = Column(DateTime, nullable=False, index=True)
    actual_value = Column(DECIMAL(15, 5))
    forecast_value = Column(DECIMAL(15, 5))
    previous_value = Column(DECIMAL(15, 5))
    is_processed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_news_events_currency', 'currency'),
        Index('idx_news_events_time', 'event_time'),
        Index('idx_news_events_impact', 'impact_level'),
        Index('idx_news_events_processed', 'is_processed'),
    )


class SentimentScore(Base):
    """Market sentiment analysis table"""
    __tablename__ = "sentiment_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    currency_pair_id = Column(Integer, ForeignKey("currency_pairs.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    overall_sentiment = Column(DECIMAL(5, 2), nullable=False)  # -100 to +100
    news_sentiment = Column(DECIMAL(5, 2), default=0)
    social_sentiment = Column(DECIMAL(5, 2), default=0)
    technical_sentiment = Column(DECIMAL(5, 2), default=0)
    confidence_score = Column(DECIMAL(5, 2), default=0)  # 0 to 100
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    currency_pair = relationship("CurrencyPair", back_populates="sentiment_scores")
    
    # Indexes
    __table_args__ = (
        Index('idx_sentiment_pair_timestamp', 'currency_pair_id', 'timestamp'),
        Index('idx_sentiment_timestamp', 'timestamp'),
    )


class Strategy(Base):
    """AI-generated trading strategies table"""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50), nullable=False, index=True)
    
    # Strategy configuration (JSON fields for flexibility)
    parameters = Column(JSON, nullable=False)
    risk_management = Column(JSON)
    entry_conditions = Column(JSON)
    exit_conditions = Column(JSON)
    
    # Performance metrics (current)
    profit_factor = Column(DECIMAL(5, 2), default=0)
    win_rate = Column(DECIMAL(5, 2), default=0)
    max_drawdown = Column(DECIMAL(5, 2), default=0)
    sharpe_ratio = Column(DECIMAL(5, 2), default=0)
    sortino_ratio = Column(DECIMAL(5, 2), default=0)
    calmar_ratio = Column(DECIMAL(5, 2), default=0)
    total_trades = Column(Integer, default=0)
    total_profit = Column(DECIMAL(15, 2), default=0)
    current_drawdown = Column(DECIMAL(5, 2), default=0)
    
    # Status and metadata
    status = Column(String(20), default='pending', index=True)  # 'pending', 'active', 'paused', 'archived', 'failed'
    priority = Column(Integer, default=1, index=True)  # 1-10, higher is more important
    is_ai_generated = Column(Boolean, default=True, index=True)
    generation_method = Column(String(50))  # 'genetic_algorithm', 'neural_network', 'rule_based', 'manual'
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), index=True)
    last_trade_at = Column(DateTime)
    last_performance_update = Column(DateTime)
    
    # Relationships
    strategy_currency_pairs = relationship("StrategyCurrencyPair", back_populates="strategy", cascade="all, delete-orphan")
    strategy_timeframes = relationship("StrategyTimeframe", back_populates="strategy", cascade="all, delete-orphan")
    strategy_performance = relationship("StrategyPerformance", back_populates="strategy", cascade="all, delete-orphan")
    strategy_trades = relationship("StrategyTrade", back_populates="strategy", cascade="all, delete-orphan")
    expert_advisors = relationship("ExpertAdvisor", back_populates="strategy", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_strategies_status', 'status'),
        Index('idx_strategies_type', 'strategy_type'),
        Index('idx_strategies_created', 'created_at'),
        Index('idx_strategies_updated', 'updated_at'),
        Index('idx_strategies_ai_generated', 'is_ai_generated'),
        Index('idx_strategies_priority', 'priority'),
    )


class StrategyCurrencyPair(Base):
    """Junction table for strategy-currency pair relationships"""
    __tablename__ = "strategy_currency_pairs"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    currency_pair_id = Column(Integer, ForeignKey("currency_pairs.id", ondelete="CASCADE"), nullable=False)
    weight = Column(DECIMAL(5, 2), default=1.0)  # relative weight for this pair in the strategy
    is_primary = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    strategy = relationship("Strategy", back_populates="strategy_currency_pairs")
    currency_pair = relationship("CurrencyPair", back_populates="strategy_currency_pairs")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('strategy_id', 'currency_pair_id', name='uq_strategy_currency_pair'),
    )


class StrategyTimeframe(Base):
    """Junction table for strategy-timeframe relationships"""
    __tablename__ = "strategy_timeframes"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    timeframe_id = Column(Integer, ForeignKey("timeframes.id", ondelete="CASCADE"), nullable=False)
    weight = Column(DECIMAL(5, 2), default=1.0)  # relative weight for this timeframe in the strategy
    is_primary = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    strategy = relationship("Strategy", back_populates="strategy_timeframes")
    timeframe = relationship("Timeframe", back_populates="strategy_timeframes")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('strategy_id', 'timeframe_id', name='uq_strategy_timeframe'),
    )


class StrategyPerformance(Base):
    """Daily strategy performance tracking table"""
    __tablename__ = "strategy_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    # Daily metrics
    daily_profit = Column(DECIMAL(15, 2), default=0)
    daily_trades = Column(Integer, default=0)
    daily_win_rate = Column(DECIMAL(5, 2), default=0)
    daily_drawdown = Column(DECIMAL(5, 2), default=0)
    
    # Cumulative metrics
    cumulative_profit = Column(DECIMAL(15, 2), default=0)
    cumulative_trades = Column(Integer, default=0)
    cumulative_win_rate = Column(DECIMAL(5, 2), default=0)
    max_drawdown_to_date = Column(DECIMAL(5, 2), default=0)
    
    # Risk metrics
    var_95 = Column(DECIMAL(15, 2), default=0)  # Value at Risk 95%
    expected_shortfall = Column(DECIMAL(15, 2), default=0)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    strategy = relationship("Strategy", back_populates="strategy_performance")
    
    # Unique constraint and indexes
    __table_args__ = (
        UniqueConstraint('strategy_id', 'date', name='uq_strategy_performance_strategy_date'),
        Index('idx_strategy_performance_strategy_date', 'strategy_id', 'date'),
        Index('idx_strategy_performance_date', 'date'),
    )


class StrategyTrade(Base):
    """Individual trade records table"""
    __tablename__ = "strategy_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    currency_pair_id = Column(Integer, ForeignKey("currency_pairs.id"), nullable=True)
    timeframe_id = Column(Integer, ForeignKey("timeframes.id"), nullable=True)
    
    # Trade details
    trade_type = Column(String(10), nullable=False)  # 'buy', 'sell'
    entry_price = Column(DECIMAL(12, 5), nullable=False)
    exit_price = Column(DECIMAL(12, 5))
    lot_size = Column(DECIMAL(8, 2), nullable=False)
    stop_loss = Column(DECIMAL(12, 5))
    take_profit = Column(DECIMAL(12, 5))
    
    # Trade results
    profit_loss = Column(DECIMAL(15, 2), default=0)
    pips = Column(DECIMAL(8, 2), default=0)
    commission = Column(DECIMAL(10, 2), default=0)
    swap = Column(DECIMAL(10, 2), default=0)
    
    # Trade status and timing
    status = Column(String(20), default='open', index=True)  # 'open', 'closed', 'cancelled'
    entry_time = Column(DateTime, nullable=False, index=True)
    exit_time = Column(DateTime)
    duration_minutes = Column(Integer)
    
    # Trade metadata
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    confidence_score = Column(DECIMAL(5, 2), default=0)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    strategy = relationship("Strategy", back_populates="strategy_trades")
    currency_pair = relationship("CurrencyPair", back_populates="strategy_trades")
    timeframe = relationship("Timeframe", back_populates="strategy_trades")
    
    # Indexes
    __table_args__ = (
        Index('idx_strategy_trades_strategy', 'strategy_id'),
        Index('idx_strategy_trades_pair', 'currency_pair_id'),
        Index('idx_strategy_trades_entry_time', 'entry_time'),
        Index('idx_strategy_trades_status', 'status'),
    )


class ExpertAdvisor(Base):
    """Generated MT4 Expert Advisors table"""
    __tablename__ = "expert_advisors"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    ea_name = Column(String(200), nullable=False)
    ea_version = Column(String(20), default='1.0')
    
    # EA configuration
    mql4_code = Column(Text, nullable=False)
    parameters = Column(JSON)
    input_parameters = Column(JSON)
    
    # Deployment info
    status = Column(String(20), default='generated', index=True)  # 'generated', 'tested', 'deployed', 'active', 'paused', 'failed'
    deployment_target = Column(String(100))  # MT4 account or server
    deployment_time = Column(DateTime)
    
    # Performance tracking
    total_runtime_hours = Column(Integer, default=0)
    last_heartbeat = Column(DateTime)
    error_count = Column(Integer, default=0)
    last_error = Column(Text)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    strategy = relationship("Strategy", back_populates="expert_advisors")
    ea_performance = relationship("EAPerformance", back_populates="expert_advisor", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_ea_strategy', 'strategy_id'),
        Index('idx_ea_status', 'status'),
        Index('idx_ea_deployment', 'deployment_target'),
    )


class EAPerformance(Base):
    """EA performance metrics table"""
    __tablename__ = "ea_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    ea_id = Column(Integer, ForeignKey("expert_advisors.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Runtime metrics
    cpu_usage = Column(DECIMAL(5, 2), default=0)
    memory_usage = Column(DECIMAL(10, 2), default=0)
    network_latency = Column(Integer, default=0)  # milliseconds
    
    # Trading metrics
    trades_executed = Column(Integer, default=0)
    trades_successful = Column(Integer, default=0)
    current_drawdown = Column(DECIMAL(5, 2), default=0)
    current_profit = Column(DECIMAL(15, 2), default=0)
    
    # Error tracking
    errors_count = Column(Integer, default=0)
    warnings_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    expert_advisor = relationship("ExpertAdvisor", back_populates="ea_performance")
    
    # Indexes
    __table_args__ = (
        Index('idx_ea_performance_ea_timestamp', 'ea_id', 'timestamp'),
        Index('idx_ea_performance_timestamp', 'timestamp'),
    )


class SystemAlert(Base):
    """System alerts table"""
    __tablename__ = "system_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(50), nullable=False, index=True)  # 'performance', 'system', 'error', 'warning'
    severity = Column(String(20), nullable=False, index=True)   # 'low', 'medium', 'high', 'critical'
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Related entities
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    ea_id = Column(Integer, ForeignKey("expert_advisors.id"), nullable=True)
    
    # Alert status
    status = Column(String(20), default='active', index=True)  # 'active', 'acknowledged', 'resolved', 'dismissed'
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    strategy = relationship("Strategy")
    expert_advisor = relationship("ExpertAdvisor")
    
    # Indexes
    __table_args__ = (
        Index('idx_alerts_type', 'alert_type'),
        Index('idx_alerts_severity', 'severity'),
        Index('idx_alerts_status', 'status'),
        Index('idx_alerts_created', 'created_at'),
    )


class SystemMetric(Base):
    """System metrics table"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(DECIMAL(15, 5), nullable=False)
    metric_unit = Column(String(20))
    tags = Column(JSON)  # additional metadata
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_name', 'metric_name'),
        Index('idx_metrics_timestamp', 'timestamp'),
    )
