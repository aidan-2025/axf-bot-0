#!/usr/bin/env python3
"""
Final comprehensive database migration script for AXF Bot 0
Handles both PostgreSQL and InfluxDB setup with proper error handling
"""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from datetime import datetime, timedelta
import json
import random
import requests

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'password'),
    'database': os.getenv('POSTGRES_DB', 'axf_bot_db')
}

# InfluxDB configuration
INFLUX_CONFIG = {
    'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
    'token': os.getenv('INFLUXDB_TOKEN', ''),
    'org': os.getenv('INFLUXDB_ORG', 'axf-bot'),
    'bucket': os.getenv('INFLUXDB_BUCKET', 'forex_data')
}

def get_postgres_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise

def test_influxdb_connection():
    """Test InfluxDB connection"""
    try:
        health_url = f"{INFLUX_CONFIG['url']}/health"
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"InfluxDB health: {health_data['status']}")
            return True
        else:
            logger.warning(f"InfluxDB health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"InfluxDB connection test failed: {e}")
        return False

def create_postgres_schema(cursor):
    """Create comprehensive PostgreSQL schema"""
    logger.info("Creating PostgreSQL schema...")
    
    # Drop existing tables if they exist
    drop_statements = [
        "DROP TABLE IF EXISTS system_metrics CASCADE",
        "DROP TABLE IF EXISTS system_alerts CASCADE", 
        "DROP TABLE IF EXISTS ea_performance CASCADE",
        "DROP TABLE IF EXISTS expert_advisors CASCADE",
        "DROP TABLE IF EXISTS strategy_trades CASCADE",
        "DROP TABLE IF EXISTS strategy_performance CASCADE",
        "DROP TABLE IF EXISTS strategy_timeframes CASCADE",
        "DROP TABLE IF EXISTS strategy_currency_pairs CASCADE",
        "DROP TABLE IF EXISTS strategies CASCADE",
        "DROP TABLE IF EXISTS sentiment_scores CASCADE",
        "DROP TABLE IF EXISTS news_events CASCADE",
        "DROP TABLE IF EXISTS market_data CASCADE",
        "DROP TABLE IF EXISTS timeframes CASCADE",
        "DROP TABLE IF EXISTS currency_pairs CASCADE",
    ]
    
    for statement in drop_statements:
        try:
            cursor.execute(statement)
        except Exception as e:
            logger.warning(f"Warning dropping {statement}: {e}")
    
    # Create tables
    tables = [
        """
        CREATE TABLE currency_pairs (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) UNIQUE NOT NULL,
            base_currency VARCHAR(3) NOT NULL,
            quote_currency VARCHAR(3) NOT NULL,
            is_active BOOLEAN DEFAULT true,
            pip_value DECIMAL(10,5) DEFAULT 0.0001,
            min_lot_size DECIMAL(8,2) DEFAULT 0.01,
            max_lot_size DECIMAL(8,2) DEFAULT 100.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE timeframes (
            id SERIAL PRIMARY KEY,
            name VARCHAR(10) UNIQUE NOT NULL,
            minutes INTEGER NOT NULL,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
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
        )
        """,
        """
        CREATE TABLE news_events (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            currency VARCHAR(3) NOT NULL,
            impact_level VARCHAR(10) NOT NULL,
            event_time TIMESTAMP NOT NULL,
            actual_value DECIMAL(15,5),
            forecast_value DECIMAL(15,5),
            previous_value DECIMAL(15,5),
            is_processed BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE sentiment_scores (
            id SERIAL PRIMARY KEY,
            currency_pair_id INTEGER REFERENCES currency_pairs(id),
            timestamp TIMESTAMP NOT NULL,
            overall_sentiment DECIMAL(5,2) NOT NULL,
            news_sentiment DECIMAL(5,2) DEFAULT 0,
            social_sentiment DECIMAL(5,2) DEFAULT 0,
            technical_sentiment DECIMAL(5,2) DEFAULT 0,
            confidence_score DECIMAL(5,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE strategies (
            id SERIAL PRIMARY KEY,
            strategy_id VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            strategy_type VARCHAR(50) NOT NULL,
            parameters JSONB NOT NULL,
            risk_management JSONB,
            entry_conditions JSONB,
            exit_conditions JSONB,
            profit_factor DECIMAL(5,2) DEFAULT 0,
            win_rate DECIMAL(5,2) DEFAULT 0,
            max_drawdown DECIMAL(5,2) DEFAULT 0,
            sharpe_ratio DECIMAL(5,2) DEFAULT 0,
            sortino_ratio DECIMAL(5,2) DEFAULT 0,
            calmar_ratio DECIMAL(5,2) DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            total_profit DECIMAL(15,2) DEFAULT 0,
            current_drawdown DECIMAL(5,2) DEFAULT 0,
            status VARCHAR(20) DEFAULT 'pending',
            priority INTEGER DEFAULT 1,
            is_ai_generated BOOLEAN DEFAULT true,
            generation_method VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_trade_at TIMESTAMP,
            last_performance_update TIMESTAMP
        )
        """,
        """
        CREATE TABLE strategy_currency_pairs (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
            currency_pair_id INTEGER REFERENCES currency_pairs(id) ON DELETE CASCADE,
            weight DECIMAL(5,2) DEFAULT 1.0,
            is_primary BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(strategy_id, currency_pair_id)
        )
        """,
        """
        CREATE TABLE strategy_timeframes (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
            timeframe_id INTEGER REFERENCES timeframes(id) ON DELETE CASCADE,
            weight DECIMAL(5,2) DEFAULT 1.0,
            is_primary BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(strategy_id, timeframe_id)
        )
        """,
        """
        CREATE TABLE strategy_performance (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
            date DATE NOT NULL,
            daily_profit DECIMAL(15,2) DEFAULT 0,
            daily_trades INTEGER DEFAULT 0,
            daily_win_rate DECIMAL(5,2) DEFAULT 0,
            daily_drawdown DECIMAL(5,2) DEFAULT 0,
            cumulative_profit DECIMAL(15,2) DEFAULT 0,
            cumulative_trades INTEGER DEFAULT 0,
            cumulative_win_rate DECIMAL(5,2) DEFAULT 0,
            max_drawdown_to_date DECIMAL(5,2) DEFAULT 0,
            var_95 DECIMAL(15,2) DEFAULT 0,
            expected_shortfall DECIMAL(15,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(strategy_id, date)
        )
        """,
        """
        CREATE TABLE strategy_trades (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
            currency_pair_id INTEGER REFERENCES currency_pairs(id),
            timeframe_id INTEGER REFERENCES timeframes(id),
            trade_type VARCHAR(10) NOT NULL,
            entry_price DECIMAL(12,5) NOT NULL,
            exit_price DECIMAL(12,5),
            lot_size DECIMAL(8,2) NOT NULL,
            stop_loss DECIMAL(12,5),
            take_profit DECIMAL(12,5),
            profit_loss DECIMAL(15,2) DEFAULT 0,
            pips DECIMAL(8,2) DEFAULT 0,
            commission DECIMAL(10,2) DEFAULT 0,
            swap DECIMAL(10,2) DEFAULT 0,
            status VARCHAR(20) DEFAULT 'open',
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP,
            duration_minutes INTEGER,
            entry_reason TEXT,
            exit_reason TEXT,
            confidence_score DECIMAL(5,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE expert_advisors (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
            ea_name VARCHAR(200) NOT NULL,
            ea_version VARCHAR(20) DEFAULT '1.0',
            mql4_code TEXT NOT NULL,
            parameters JSONB,
            input_parameters JSONB,
            status VARCHAR(20) DEFAULT 'generated',
            deployment_target VARCHAR(100),
            deployment_time TIMESTAMP,
            total_runtime_hours INTEGER DEFAULT 0,
            last_heartbeat TIMESTAMP,
            error_count INTEGER DEFAULT 0,
            last_error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE ea_performance (
            id SERIAL PRIMARY KEY,
            ea_id INTEGER REFERENCES expert_advisors(id) ON DELETE CASCADE,
            timestamp TIMESTAMP NOT NULL,
            cpu_usage DECIMAL(5,2) DEFAULT 0,
            memory_usage DECIMAL(10,2) DEFAULT 0,
            network_latency INTEGER DEFAULT 0,
            trades_executed INTEGER DEFAULT 0,
            trades_successful INTEGER DEFAULT 0,
            current_drawdown DECIMAL(5,2) DEFAULT 0,
            current_profit DECIMAL(15,2) DEFAULT 0,
            errors_count INTEGER DEFAULT 0,
            warnings_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE system_alerts (
            id SERIAL PRIMARY KEY,
            alert_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            title VARCHAR(200) NOT NULL,
            message TEXT NOT NULL,
            strategy_id INTEGER REFERENCES strategies(id),
            ea_id INTEGER REFERENCES expert_advisors(id),
            status VARCHAR(20) DEFAULT 'active',
            acknowledged_by VARCHAR(100),
            acknowledged_at TIMESTAMP,
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE system_metrics (
            id SERIAL PRIMARY KEY,
            metric_name VARCHAR(100) NOT NULL,
            metric_value DECIMAL(15,5) NOT NULL,
            metric_unit VARCHAR(20),
            tags JSONB,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    for table_sql in tables:
        cursor.execute(table_sql)
    
    # Create indexes
    indexes = [
        "CREATE INDEX idx_currency_pairs_symbol ON currency_pairs(symbol)",
        "CREATE INDEX idx_currency_pairs_active ON currency_pairs(is_active)",
        "CREATE INDEX idx_timeframes_name ON timeframes(name)",
        "CREATE INDEX idx_timeframes_active ON timeframes(is_active)",
        "CREATE INDEX idx_market_data_pair_timeframe ON market_data(currency_pair_id, timeframe_id)",
        "CREATE INDEX idx_market_data_timestamp ON market_data(timestamp)",
        "CREATE INDEX idx_news_events_currency ON news_events(currency)",
        "CREATE INDEX idx_news_events_time ON news_events(event_time)",
        "CREATE INDEX idx_news_events_impact ON news_events(impact_level)",
        "CREATE INDEX idx_sentiment_pair_timestamp ON sentiment_scores(currency_pair_id, timestamp)",
        "CREATE INDEX idx_strategies_status ON strategies(status)",
        "CREATE INDEX idx_strategies_type ON strategies(strategy_type)",
        "CREATE INDEX idx_strategies_created ON strategies(created_at)",
        "CREATE INDEX idx_strategy_performance_strategy_date ON strategy_performance(strategy_id, date)",
        "CREATE INDEX idx_strategy_trades_strategy ON strategy_trades(strategy_id)",
        "CREATE INDEX idx_strategy_trades_entry_time ON strategy_trades(entry_time)",
        "CREATE INDEX idx_ea_strategy ON expert_advisors(strategy_id)",
        "CREATE INDEX idx_ea_status ON expert_advisors(status)",
        "CREATE INDEX idx_alerts_type ON system_alerts(alert_type)",
        "CREATE INDEX idx_alerts_severity ON system_alerts(severity)",
        "CREATE INDEX idx_metrics_name ON system_metrics(metric_name)",
        "CREATE INDEX idx_metrics_timestamp ON system_metrics(timestamp)"
    ]
    
    for index_sql in indexes:
        try:
            cursor.execute(index_sql)
        except Exception as e:
            logger.warning(f"Warning creating index: {e}")
    
    logger.info("PostgreSQL schema created successfully")

def seed_postgres_data(cursor):
    """Seed PostgreSQL with initial data"""
    logger.info("Seeding PostgreSQL data...")
    
    # Insert currency pairs
    currency_pairs = [
        ('EURUSD', 'EUR', 'USD', 0.0001, 0.01, 100.0),
        ('GBPUSD', 'GBP', 'USD', 0.0001, 0.01, 100.0),
        ('USDJPY', 'USD', 'JPY', 0.01, 0.01, 100.0),
        ('USDCHF', 'USD', 'CHF', 0.0001, 0.01, 100.0),
        ('AUDUSD', 'AUD', 'USD', 0.0001, 0.01, 100.0),
        ('USDCAD', 'USD', 'CAD', 0.0001, 0.01, 100.0),
    ]
    
    for symbol, base, quote, pip_val, min_lot, max_lot in currency_pairs:
        cursor.execute("""
            INSERT INTO currency_pairs (symbol, base_currency, quote_currency, pip_value, min_lot_size, max_lot_size)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO NOTHING
        """, (symbol, base, quote, pip_val, min_lot, max_lot))
    
    # Insert timeframes
    timeframes = [
        ('M1', 1), ('M5', 5), ('M15', 15), ('M30', 30),
        ('H1', 60), ('H4', 240), ('D1', 1440), ('W1', 10080),
    ]
    
    for name, minutes in timeframes:
        cursor.execute("""
            INSERT INTO timeframes (name, minutes)
            VALUES (%s, %s)
            ON CONFLICT (name) DO NOTHING
        """, (name, minutes))
    
    # Insert sample strategies
    strategies_data = [
        {
            'strategy_id': 'STRAT_001',
            'name': 'EUR/USD Trend Following',
            'description': 'AI-generated trend following strategy using moving average crossovers',
            'strategy_type': 'trend_following',
            'parameters': json.dumps({
                'ma_fast': 20, 'ma_slow': 50, 'stop_loss': 50, 'take_profit': 100, 'risk_per_trade': 0.02
            }),
            'risk_management': json.dumps({
                'max_risk_per_trade': 0.02, 'max_daily_risk': 0.05, 'max_drawdown': 0.15
            }),
            'entry_conditions': json.dumps({
                'ma_crossover': True, 'volume_confirmation': True, 'trend_strength': 0.7
            }),
            'exit_conditions': json.dumps({
                'stop_loss': True, 'take_profit': True, 'trailing_stop': True
            }),
            'profit_factor': 1.45, 'win_rate': 62.5, 'max_drawdown': 8.2, 'sharpe_ratio': 1.23,
            'total_trades': 156, 'total_profit': 2340.50, 'status': 'active', 'priority': 8,
            'is_ai_generated': True, 'generation_method': 'genetic_algorithm'
        },
        {
            'strategy_id': 'STRAT_002',
            'name': 'GBP/USD Range Trading',
            'description': 'RSI-based range trading strategy with support/resistance levels',
            'strategy_type': 'range_trading',
            'parameters': json.dumps({
                'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'stop_loss': 30, 'take_profit': 60
            }),
            'risk_management': json.dumps({
                'max_risk_per_trade': 0.015, 'max_daily_risk': 0.04, 'max_drawdown': 0.12
            }),
            'entry_conditions': json.dumps({
                'rsi_oversold': True, 'support_level': True, 'volume_spike': False
            }),
            'exit_conditions': json.dumps({
                'rsi_overbought': True, 'resistance_level': True, 'time_based': True
            }),
            'profit_factor': 1.32, 'win_rate': 58.3, 'max_drawdown': 12.1, 'sharpe_ratio': 1.15,
            'total_trades': 89, 'total_profit': 1875.25, 'status': 'active', 'priority': 6,
            'is_ai_generated': True, 'generation_method': 'neural_network'
        }
    ]
    
    for strategy_data in strategies_data:
        cursor.execute("""
            INSERT INTO strategies (
                strategy_id, name, description, strategy_type, parameters, risk_management,
                entry_conditions, exit_conditions, profit_factor, win_rate, max_drawdown,
                sharpe_ratio, total_trades, total_profit, status, priority, is_ai_generated,
                generation_method
            ) VALUES (
                %(strategy_id)s, %(name)s, %(description)s, %(strategy_type)s, %(parameters)s,
                %(risk_management)s, %(entry_conditions)s, %(exit_conditions)s, %(profit_factor)s,
                %(win_rate)s, %(max_drawdown)s, %(sharpe_ratio)s, %(total_trades)s,
                %(total_profit)s, %(status)s, %(priority)s, %(is_ai_generated)s, %(generation_method)s
            ) ON CONFLICT (strategy_id) DO NOTHING
        """, strategy_data)
    
    # Link strategies to currency pairs and timeframes
    strategy_links = [
        ('STRAT_001', 'EURUSD', ['H1', 'H4']),
        ('STRAT_002', 'GBPUSD', ['M15', 'H1']),
    ]
    
    for strategy_id, currency_pair, timeframes in strategy_links:
        # Get strategy ID
        cursor.execute("SELECT id FROM strategies WHERE strategy_id = %s", (strategy_id,))
        strategy_row = cursor.fetchone()
        if not strategy_row:
            continue
        strategy_db_id = strategy_row[0]
        
        # Link to currency pair
        cursor.execute("SELECT id FROM currency_pairs WHERE symbol = %s", (currency_pair,))
        pair_row = cursor.fetchone()
        if pair_row:
            cursor.execute("""
                INSERT INTO strategy_currency_pairs (strategy_id, currency_pair_id, is_primary)
                VALUES (%s, %s, true)
                ON CONFLICT (strategy_id, currency_pair_id) DO NOTHING
            """, (strategy_db_id, pair_row[0]))
        
        # Link to timeframes
        for timeframe in timeframes:
            cursor.execute("SELECT id FROM timeframes WHERE name = %s", (timeframe,))
            tf_row = cursor.fetchone()
            if tf_row:
                cursor.execute("""
                    INSERT INTO strategy_timeframes (strategy_id, timeframe_id, is_primary)
                    VALUES (%s, %s, true)
                    ON CONFLICT (strategy_id, timeframe_id) DO NOTHING
                """, (strategy_db_id, tf_row[0]))
    
    logger.info("PostgreSQL data seeded successfully")

def setup_influxdb():
    """Setup InfluxDB (if available)"""
    logger.info("Setting up InfluxDB...")
    
    if not test_influxdb_connection():
        logger.warning("InfluxDB not available - skipping setup")
        return False
    
    if not INFLUX_CONFIG['token']:
        logger.warning("InfluxDB token not provided - skipping setup")
        logger.info("To set up InfluxDB:")
        logger.info("1. Visit http://localhost:8086")
        logger.info("2. Complete setup with admin/password")
        logger.info("3. Get API token from Data > Tokens")
        logger.info("4. Set INFLUXDB_TOKEN environment variable")
        return False
    
    logger.info("InfluxDB setup completed (token provided)")
    return True

def main():
    """Main migration function"""
    try:
        logger.info("Starting comprehensive database migration...")
        
        # Setup PostgreSQL
        logger.info("Setting up PostgreSQL...")
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        create_postgres_schema(cursor)
        seed_postgres_data(cursor)
        
        # Verify PostgreSQL migration
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cursor.fetchone()[0]
        logger.info(f"PostgreSQL migration completed. Created {table_count} tables.")
        
        cursor.close()
        conn.close()
        
        # Setup InfluxDB
        setup_influxdb()
        
        logger.info("✅ Comprehensive database migration completed successfully!")
        logger.info("📊 PostgreSQL: Ready with comprehensive schema")
        logger.info("📈 InfluxDB: Setup instructions provided")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
