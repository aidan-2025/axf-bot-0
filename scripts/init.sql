-- AXF Bot 0 - Simple Database Schema
-- Minimal database for storing strategies and their performance

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS axf_bot_db;

-- Use the database
\c axf_bot_db;

-- Main strategies table - stores all generated strategies
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL,
    currency_pairs TEXT[] NOT NULL,
    timeframes TEXT[] NOT NULL,
    
    -- Strategy parameters (stored as JSON for flexibility)
    parameters JSONB NOT NULL,
    
    -- Performance metrics
    profit_factor DECIMAL(5,2) DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0,
    max_drawdown DECIMAL(5,2) DEFAULT 0,
    sharpe_ratio DECIMAL(5,2) DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    total_profit DECIMAL(15,2) DEFAULT 0,
    
    -- Status and timestamps
    status VARCHAR(20) DEFAULT 'pending', -- pending, active, paused, archived
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategy performance history - tracks daily performance
CREATE TABLE strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    date DATE NOT NULL,
    daily_profit DECIMAL(15,2) DEFAULT 0,
    daily_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2) DEFAULT 0,
    max_drawdown DECIMAL(5,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Generated EAs table - stores MT4 Expert Advisors
CREATE TABLE expert_advisors (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    ea_name VARCHAR(200) NOT NULL,
    mql4_code TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'generated', -- generated, deployed, active, paused
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Simple indexes for performance
CREATE INDEX idx_strategies_status ON strategies(status);
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_strategies_created ON strategies(created_at);
CREATE INDEX idx_performance_strategy_date ON strategy_performance(strategy_id, date);
CREATE INDEX idx_ea_strategy ON expert_advisors(strategy_id);

-- Insert some sample strategies for testing
INSERT INTO strategies (strategy_id, name, description, strategy_type, currency_pairs, timeframes, parameters, profit_factor, win_rate, max_drawdown, sharpe_ratio, total_trades, total_profit, status) VALUES
('STRAT_001', 'EUR/USD Trend Following', 'Simple moving average crossover strategy for EUR/USD', 'trend_following', ARRAY['EURUSD'], ARRAY['H1', 'H4'], '{"ma_fast": 20, "ma_slow": 50, "stop_loss": 50, "take_profit": 100}', 1.45, 62.5, 8.2, 1.23, 156, 2340.50, 'active'),
('STRAT_002', 'GBP/USD Range Trading', 'RSI-based range trading strategy for GBP/USD', 'range_trading', ARRAY['GBPUSD'], ARRAY['M15', 'H1'], '{"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "stop_loss": 30, "take_profit": 60}', 1.32, 58.3, 12.1, 1.15, 89, 1875.25, 'active'),
('STRAT_003', 'USD/JPY Breakout', 'Breakout strategy for USD/JPY with volume confirmation', 'breakout', ARRAY['USDJPY'], ARRAY['H4', 'D1'], '{"breakout_period": 20, "volume_threshold": 1.5, "stop_loss": 40, "take_profit": 80}', 1.67, 71.2, 6.8, 1.45, 203, 4567.80, 'paused');

-- Create a simple function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER update_strategies_updated_at 
    BEFORE UPDATE ON strategies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
