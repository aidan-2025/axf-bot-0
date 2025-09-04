-- AXF Bot 0 - Database Initialization Script
-- This script creates the initial database schema for the axf-bot-0 project

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS axf_bot_db;

-- Use the database
\c axf_bot_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Application 1 Tables (Strategy Generator)

-- Currency pairs table
CREATE TABLE currency_pairs (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    base_currency VARCHAR(3) NOT NULL,
    quote_currency VARCHAR(3) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market data table
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    currency_pair_id INTEGER REFERENCES currency_pairs(id),
    timeframe VARCHAR(5) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(currency_pair_id, timeframe, timestamp)
);

-- News articles table
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    url VARCHAR(500),
    published_at TIMESTAMP NOT NULL,
    sentiment_score DECIMAL(5,2),
    sentiment_confidence DECIMAL(5,2),
    currency_pairs TEXT[], -- Array of affected currency pairs
    impact_level VARCHAR(20), -- low, medium, high
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Economic events table
CREATE TABLE economic_events (
    id SERIAL PRIMARY KEY,
    event_name VARCHAR(200) NOT NULL,
    country VARCHAR(50) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    event_time TIMESTAMP NOT NULL,
    impact_level VARCHAR(20) NOT NULL, -- low, medium, high
    previous_value VARCHAR(100),
    forecast_value VARCHAR(100),
    actual_value VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategies table
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL,
    currency_pairs TEXT[] NOT NULL,
    timeframes TEXT[] NOT NULL,
    entry_conditions JSONB NOT NULL,
    exit_conditions JSONB NOT NULL,
    risk_management JSONB NOT NULL,
    backtesting_results JSONB,
    expected_performance JSONB,
    status VARCHAR(20) DEFAULT 'pending', -- pending, active, paused, archived
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategy performance table
CREATE TABLE strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    date DATE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_profit DECIMAL(20,8) DEFAULT 0,
    total_loss DECIMAL(20,8) DEFAULT 0,
    max_drawdown DECIMAL(5,2) DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0,
    profit_factor DECIMAL(5,2) DEFAULT 0,
    sharpe_ratio DECIMAL(5,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Application 2 Tables (MT4 EA Generator)

-- Expert Advisors table
CREATE TABLE expert_advisors (
    id SERIAL PRIMARY KEY,
    ea_id VARCHAR(50) UNIQUE NOT NULL,
    strategy_id INTEGER REFERENCES strategies(id),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    mql4_code TEXT NOT NULL,
    parameters JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'generated', -- generated, compiled, deployed, active, paused, archived
    compilation_errors TEXT,
    deployment_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- EA performance table
CREATE TABLE ea_performance (
    id SERIAL PRIMARY KEY,
    ea_id INTEGER REFERENCES expert_advisors(id),
    timestamp TIMESTAMP NOT NULL,
    account_balance DECIMAL(20,8) NOT NULL,
    equity DECIMAL(20,8) NOT NULL,
    margin DECIMAL(20,8) NOT NULL,
    free_margin DECIMAL(20,8) NOT NULL,
    open_positions INTEGER DEFAULT 0,
    total_profit DECIMAL(20,8) DEFAULT 0,
    daily_profit DECIMAL(20,8) DEFAULT 0,
    max_drawdown DECIMAL(5,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- EA trades table
CREATE TABLE ea_trades (
    id SERIAL PRIMARY KEY,
    ea_id INTEGER REFERENCES expert_advisors(id),
    trade_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_type VARCHAR(10) NOT NULL, -- buy, sell
    volume DECIMAL(20,8) NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8),
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    profit DECIMAL(20,8) DEFAULT 0,
    commission DECIMAL(20,8) DEFAULT 0,
    swap DECIMAL(20,8) DEFAULT 0,
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'open', -- open, closed, cancelled
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System monitoring table
CREATE TABLE system_monitoring (
    id SERIAL PRIMARY KEY,
    component VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    unit VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for better performance
CREATE INDEX idx_market_data_pair_timeframe ON market_data(currency_pair_id, timeframe, timestamp);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_news_articles_published ON news_articles(published_at);
CREATE INDEX idx_news_articles_sentiment ON news_articles(sentiment_score);
CREATE INDEX idx_economic_events_time ON economic_events(event_time);
CREATE INDEX idx_economic_events_currency ON economic_events(currency);
CREATE INDEX idx_strategies_status ON strategies(status);
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_ea_performance_timestamp ON ea_performance(timestamp);
CREATE INDEX idx_ea_trades_ea_id ON ea_trades(ea_id);
CREATE INDEX idx_ea_trades_symbol ON ea_trades(symbol);
CREATE INDEX idx_system_monitoring_component ON system_monitoring(component, timestamp);

-- Insert initial currency pairs
INSERT INTO currency_pairs (symbol, base_currency, quote_currency) VALUES
('EURUSD', 'EUR', 'USD'),
('GBPUSD', 'GBP', 'USD'),
('USDJPY', 'USD', 'JPY'),
('USDCHF', 'USD', 'CHF'),
('AUDUSD', 'AUD', 'USD'),
('USDCAD', 'USD', 'CAD'),
('NZDUSD', 'NZD', 'USD'),
('EURGBP', 'EUR', 'GBP'),
('EURJPY', 'EUR', 'JPY'),
('EURCHF', 'EUR', 'CHF'),
('EURAUD', 'EUR', 'AUD'),
('EURCAD', 'EUR', 'CAD'),
('EURNZD', 'EUR', 'NZD'),
('GBPJPY', 'GBP', 'JPY'),
('GBPCHF', 'GBP', 'CHF'),
('GBPAUD', 'GBP', 'AUD'),
('GBPCAD', 'GBP', 'CAD'),
('GBPNZD', 'GBP', 'NZD'),
('AUDJPY', 'AUD', 'JPY'),
('AUDCHF', 'AUD', 'CHF'),
('AUDCAD', 'AUD', 'CAD'),
('AUDNZD', 'AUD', 'NZD'),
('CADJPY', 'CAD', 'JPY'),
('CADCHF', 'CAD', 'CHF'),
('CADNZD', 'CAD', 'NZD'),
('CHFJPY', 'CHF', 'JPY'),
('NZDJPY', 'NZD', 'JPY'),
('NZDCHF', 'NZD', 'CHF');

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_currency_pairs_updated_at BEFORE UPDATE ON currency_pairs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_expert_advisors_updated_at BEFORE UPDATE ON expert_advisors FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW active_strategies AS
SELECT s.*, cp.symbol as currency_pair_symbol
FROM strategies s
JOIN currency_pairs cp ON cp.symbol = ANY(s.currency_pairs)
WHERE s.status = 'active';

CREATE VIEW ea_performance_summary AS
SELECT 
    ea.ea_id,
    ea.name,
    ea.status,
    COUNT(ep.id) as performance_records,
    AVG(ep.total_profit) as avg_total_profit,
    MAX(ep.max_drawdown) as max_drawdown,
    COUNT(et.id) as total_trades,
    COUNT(CASE WHEN et.profit > 0 THEN 1 END) as winning_trades
FROM expert_advisors ea
LEFT JOIN ea_performance ep ON ea.id = ep.ea_id
LEFT JOIN ea_trades et ON ea.id = et.ea_id
GROUP BY ea.id, ea.ea_id, ea.name, ea.status;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO axf_bot_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO axf_bot_user;
