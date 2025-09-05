-- AXF Bot 0 - Comprehensive Database Schema
-- Complete schema for AI-powered forex trading system

-- ==============================================
-- CORE ENTITIES
-- ==============================================

-- Currency pairs table - master list of supported pairs
CREATE TABLE currency_pairs (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL, -- e.g., 'EURUSD', 'GBPUSD'
    base_currency VARCHAR(3) NOT NULL,  -- e.g., 'EUR', 'GBP'
    quote_currency VARCHAR(3) NOT NULL, -- e.g., 'USD', 'JPY'
    is_active BOOLEAN DEFAULT true,
    pip_value DECIMAL(10,5) DEFAULT 0.0001,
    min_lot_size DECIMAL(8,2) DEFAULT 0.01,
    max_lot_size DECIMAL(8,2) DEFAULT 100.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Timeframes table - supported trading timeframes
CREATE TABLE timeframes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(10) UNIQUE NOT NULL, -- e.g., 'M1', 'M5', 'M15', 'H1', 'H4', 'D1'
    minutes INTEGER NOT NULL,         -- e.g., 1, 5, 15, 60, 240, 1440
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market data table - historical price data
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    currency_pair_id INTEGER REFERENCES currency_pairs(id),
    timeframe_id INTEGER REFERENCES timeframes(id),
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(12,5) NOT NULL,
    high_price DECIMAL(12,5) NOT NULL,
    low_price DECIMAL(12,5) NOT NULL,
    close_price DECIMAL(12,5) NOT NULL,
    volume BIGINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(currency_pair_id, timeframe_id, timestamp)
);

-- News events table - economic calendar and news
CREATE TABLE news_events (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    currency VARCHAR(3) NOT NULL,
    impact_level VARCHAR(10) NOT NULL, -- 'low', 'medium', 'high'
    event_time TIMESTAMP NOT NULL,
    actual_value DECIMAL(15,5),
    forecast_value DECIMAL(15,5),
    previous_value DECIMAL(15,5),
    is_processed BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment scores table - market sentiment analysis
CREATE TABLE sentiment_scores (
    id SERIAL PRIMARY KEY,
    currency_pair_id INTEGER REFERENCES currency_pairs(id),
    timestamp TIMESTAMP NOT NULL,
    overall_sentiment DECIMAL(5,2) NOT NULL, -- -100 to +100
    news_sentiment DECIMAL(5,2) DEFAULT 0,
    social_sentiment DECIMAL(5,2) DEFAULT 0,
    technical_sentiment DECIMAL(5,2) DEFAULT 0,
    confidence_score DECIMAL(5,2) DEFAULT 0, -- 0 to 100
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================
-- STRATEGY MANAGEMENT
-- ==============================================

-- Strategies table - AI-generated trading strategies
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL, -- 'trend_following', 'range_trading', 'breakout', 'sentiment_based', 'news_trading'
    
    -- Strategy configuration
    parameters JSONB NOT NULL,
    risk_management JSONB,
    entry_conditions JSONB,
    exit_conditions JSONB,
    
    -- Performance metrics (current)
    profit_factor DECIMAL(5,2) DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0,
    max_drawdown DECIMAL(5,2) DEFAULT 0,
    sharpe_ratio DECIMAL(5,2) DEFAULT 0,
    sortino_ratio DECIMAL(5,2) DEFAULT 0,
    calmar_ratio DECIMAL(5,2) DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    total_profit DECIMAL(15,2) DEFAULT 0,
    current_drawdown DECIMAL(5,2) DEFAULT 0,
    
    -- Status and metadata
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'active', 'paused', 'archived', 'failed'
    priority INTEGER DEFAULT 1, -- 1-10, higher is more important
    is_ai_generated BOOLEAN DEFAULT true,
    generation_method VARCHAR(50), -- 'genetic_algorithm', 'neural_network', 'rule_based', 'manual'
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_trade_at TIMESTAMP,
    last_performance_update TIMESTAMP
);

-- Strategy currency pairs junction table
CREATE TABLE strategy_currency_pairs (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    currency_pair_id INTEGER REFERENCES currency_pairs(id) ON DELETE CASCADE,
    weight DECIMAL(5,2) DEFAULT 1.0, -- relative weight for this pair in the strategy
    is_primary BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_id, currency_pair_id)
);

-- Strategy timeframes junction table
CREATE TABLE strategy_timeframes (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    timeframe_id INTEGER REFERENCES timeframes(id) ON DELETE CASCADE,
    weight DECIMAL(5,2) DEFAULT 1.0, -- relative weight for this timeframe in the strategy
    is_primary BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_id, timeframe_id)
);

-- Strategy performance history - daily performance tracking
CREATE TABLE strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    
    -- Daily metrics
    daily_profit DECIMAL(15,2) DEFAULT 0,
    daily_trades INTEGER DEFAULT 0,
    daily_win_rate DECIMAL(5,2) DEFAULT 0,
    daily_drawdown DECIMAL(5,2) DEFAULT 0,
    
    -- Cumulative metrics
    cumulative_profit DECIMAL(15,2) DEFAULT 0,
    cumulative_trades INTEGER DEFAULT 0,
    cumulative_win_rate DECIMAL(5,2) DEFAULT 0,
    max_drawdown_to_date DECIMAL(5,2) DEFAULT 0,
    
    -- Risk metrics
    var_95 DECIMAL(15,2) DEFAULT 0, -- Value at Risk 95%
    expected_shortfall DECIMAL(15,2) DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_id, date)
);

-- Strategy trades table - individual trade records
CREATE TABLE strategy_trades (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    currency_pair_id INTEGER REFERENCES currency_pairs(id),
    timeframe_id INTEGER REFERENCES timeframes(id),
    
    -- Trade details
    trade_type VARCHAR(10) NOT NULL, -- 'buy', 'sell'
    entry_price DECIMAL(12,5) NOT NULL,
    exit_price DECIMAL(12,5),
    lot_size DECIMAL(8,2) NOT NULL,
    stop_loss DECIMAL(12,5),
    take_profit DECIMAL(12,5),
    
    -- Trade results
    profit_loss DECIMAL(15,2) DEFAULT 0,
    pips DECIMAL(8,2) DEFAULT 0,
    commission DECIMAL(10,2) DEFAULT 0,
    swap DECIMAL(10,2) DEFAULT 0,
    
    -- Trade status and timing
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'cancelled'
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    duration_minutes INTEGER,
    
    -- Trade metadata
    entry_reason TEXT,
    exit_reason TEXT,
    confidence_score DECIMAL(5,2) DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================
-- EXPERT ADVISORS (MT4)
-- ==============================================

-- Expert advisors table - generated MT4 EAs
CREATE TABLE expert_advisors (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    ea_name VARCHAR(200) NOT NULL,
    ea_version VARCHAR(20) DEFAULT '1.0',
    
    -- EA configuration
    mql4_code TEXT NOT NULL,
    parameters JSONB,
    input_parameters JSONB,
    
    -- Deployment info
    status VARCHAR(20) DEFAULT 'generated', -- 'generated', 'tested', 'deployed', 'active', 'paused', 'failed'
    deployment_target VARCHAR(100), -- MT4 account or server
    deployment_time TIMESTAMP,
    
    -- Performance tracking
    total_runtime_hours INTEGER DEFAULT 0,
    last_heartbeat TIMESTAMP,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- EA performance metrics
CREATE TABLE ea_performance (
    id SERIAL PRIMARY KEY,
    ea_id INTEGER REFERENCES expert_advisors(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    
    -- Runtime metrics
    cpu_usage DECIMAL(5,2) DEFAULT 0,
    memory_usage DECIMAL(10,2) DEFAULT 0,
    network_latency INTEGER DEFAULT 0, -- milliseconds
    
    -- Trading metrics
    trades_executed INTEGER DEFAULT 0,
    trades_successful INTEGER DEFAULT 0,
    current_drawdown DECIMAL(5,2) DEFAULT 0,
    current_profit DECIMAL(15,2) DEFAULT 0,
    
    -- Error tracking
    errors_count INTEGER DEFAULT 0,
    warnings_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================
-- SYSTEM MONITORING
-- ==============================================

-- System alerts table
CREATE TABLE system_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL, -- 'performance', 'system', 'error', 'warning'
    severity VARCHAR(20) NOT NULL,   -- 'low', 'medium', 'high', 'critical'
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    
    -- Related entities
    strategy_id INTEGER REFERENCES strategies(id),
    ea_id INTEGER REFERENCES expert_advisors(id),
    
    -- Alert status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'acknowledged', 'resolved', 'dismissed'
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,5) NOT NULL,
    metric_unit VARCHAR(20),
    tags JSONB, -- additional metadata
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================
-- INDEXES FOR PERFORMANCE
-- ==============================================

-- Currency pairs indexes
CREATE INDEX idx_currency_pairs_symbol ON currency_pairs(symbol);
CREATE INDEX idx_currency_pairs_active ON currency_pairs(is_active);

-- Timeframes indexes
CREATE INDEX idx_timeframes_name ON timeframes(name);
CREATE INDEX idx_timeframes_active ON timeframes(is_active);

-- Market data indexes
CREATE INDEX idx_market_data_pair_timeframe ON market_data(currency_pair_id, timeframe_id);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_market_data_pair_timestamp ON market_data(currency_pair_id, timestamp);

-- News events indexes
CREATE INDEX idx_news_events_currency ON news_events(currency);
CREATE INDEX idx_news_events_time ON news_events(event_time);
CREATE INDEX idx_news_events_impact ON news_events(impact_level);
CREATE INDEX idx_news_events_processed ON news_events(is_processed);

-- Sentiment scores indexes
CREATE INDEX idx_sentiment_pair_timestamp ON sentiment_scores(currency_pair_id, timestamp);
CREATE INDEX idx_sentiment_timestamp ON sentiment_scores(timestamp);

-- Strategies indexes
CREATE INDEX idx_strategies_status ON strategies(status);
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_strategies_created ON strategies(created_at);
CREATE INDEX idx_strategies_updated ON strategies(updated_at);
CREATE INDEX idx_strategies_ai_generated ON strategies(is_ai_generated);
CREATE INDEX idx_strategies_priority ON strategies(priority);

-- Strategy performance indexes
CREATE INDEX idx_strategy_performance_strategy_date ON strategy_performance(strategy_id, date);
CREATE INDEX idx_strategy_performance_date ON strategy_performance(date);

-- Strategy trades indexes
CREATE INDEX idx_strategy_trades_strategy ON strategy_trades(strategy_id);
CREATE INDEX idx_strategy_trades_pair ON strategy_trades(currency_pair_id);
CREATE INDEX idx_strategy_trades_entry_time ON strategy_trades(entry_time);
CREATE INDEX idx_strategy_trades_status ON strategy_trades(status);

-- Expert advisors indexes
CREATE INDEX idx_ea_strategy ON expert_advisors(strategy_id);
CREATE INDEX idx_ea_status ON expert_advisors(status);
CREATE INDEX idx_ea_deployment ON expert_advisors(deployment_target);

-- EA performance indexes
CREATE INDEX idx_ea_performance_ea_timestamp ON ea_performance(ea_id, timestamp);
CREATE INDEX idx_ea_performance_timestamp ON ea_performance(timestamp);

-- System alerts indexes
CREATE INDEX idx_alerts_type ON system_alerts(alert_type);
CREATE INDEX idx_alerts_severity ON system_alerts(severity);
CREATE INDEX idx_alerts_status ON system_alerts(status);
CREATE INDEX idx_alerts_created ON system_alerts(created_at);

-- System metrics indexes
CREATE INDEX idx_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_metrics_timestamp ON system_metrics(timestamp);

-- ==============================================
-- FUNCTIONS AND TRIGGERS
-- ==============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $func$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$func$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_strategies_updated_at 
    BEFORE UPDATE ON strategies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_expert_advisors_updated_at 
    BEFORE UPDATE ON expert_advisors 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate trade duration
CREATE OR REPLACE FUNCTION calculate_trade_duration()
RETURNS TRIGGER AS $func$
BEGIN
    IF NEW.exit_time IS NOT NULL AND NEW.entry_time IS NOT NULL THEN
        NEW.duration_minutes = EXTRACT(EPOCH FROM (NEW.exit_time - NEW.entry_time)) / 60;
    END IF;
    RETURN NEW;
END;
$func$ language 'plpgsql';

-- Trigger for trade duration calculation
CREATE TRIGGER calculate_trade_duration_trigger
    BEFORE INSERT OR UPDATE ON strategy_trades
    FOR EACH ROW EXECUTE FUNCTION calculate_trade_duration();

-- Function to update strategy performance when trades are closed
CREATE OR REPLACE FUNCTION update_strategy_performance()
RETURNS TRIGGER AS $func$
DECLARE
    strategy_id_val INTEGER;
    trade_date DATE;
    daily_profit DECIMAL(15,2);
    daily_trades INTEGER;
BEGIN
    -- Only process when trade is closed
    IF NEW.status = 'closed' AND (OLD.status IS NULL OR OLD.status != 'closed') THEN
        strategy_id_val := NEW.strategy_id;
        trade_date := DATE(NEW.exit_time);
        
        -- Calculate daily totals
        SELECT 
            COALESCE(SUM(profit_loss), 0),
            COUNT(*)
        INTO daily_profit, daily_trades
        FROM strategy_trades 
        WHERE strategy_id = strategy_id_val 
        AND DATE(exit_time) = trade_date 
        AND status = 'closed';
        
        -- Insert or update daily performance
        INSERT INTO strategy_performance (
            strategy_id, date, daily_profit, daily_trades
        ) VALUES (
            strategy_id_val, trade_date, daily_profit, daily_trades
        )
        ON CONFLICT (strategy_id, date) 
        DO UPDATE SET 
            daily_profit = EXCLUDED.daily_profit,
            daily_trades = EXCLUDED.daily_trades;
    END IF;
    
    RETURN NEW;
END;
$func$ language 'plpgsql';

-- Trigger for strategy performance updates
CREATE TRIGGER update_strategy_performance_trigger
    AFTER INSERT OR UPDATE ON strategy_trades
    FOR EACH ROW EXECUTE FUNCTION update_strategy_performance();
