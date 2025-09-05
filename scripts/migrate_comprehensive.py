#!/usr/bin/env python3
"""
Comprehensive database migration script for AXF Bot 0
Applies the complete database schema and seeds initial data
"""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from datetime import datetime, timedelta
import json
import random

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

def get_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def execute_sql_file(cursor, file_path):
    """Execute SQL file"""
    try:
        with open(file_path, 'r') as f:
            sql_content = f.read()
        
        # Split by semicolon and execute each statement
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        for statement in statements:
            if statement:
                cursor.execute(statement)
        
        logger.info(f"Successfully executed SQL file: {file_path}")
    except Exception as e:
        logger.error(f"Error executing SQL file {file_path}: {e}")
        raise

def seed_initial_data(cursor):
    """Seed initial data for development"""
    logger.info("Seeding initial data...")
    
    # Insert currency pairs
    currency_pairs = [
        ('EURUSD', 'EUR', 'USD', 0.0001, 0.01, 100.0),
        ('GBPUSD', 'GBP', 'USD', 0.0001, 0.01, 100.0),
        ('USDJPY', 'USD', 'JPY', 0.01, 0.01, 100.0),
        ('USDCHF', 'USD', 'CHF', 0.0001, 0.01, 100.0),
        ('AUDUSD', 'AUD', 'USD', 0.0001, 0.01, 100.0),
        ('USDCAD', 'USD', 'CAD', 0.0001, 0.01, 100.0),
        ('NZDUSD', 'NZD', 'USD', 0.0001, 0.01, 100.0),
        ('EURJPY', 'EUR', 'JPY', 0.01, 0.01, 100.0),
        ('GBPJPY', 'GBP', 'JPY', 0.01, 0.01, 100.0),
        ('EURGBP', 'EUR', 'GBP', 0.0001, 0.01, 100.0),
    ]
    
    for symbol, base, quote, pip_val, min_lot, max_lot in currency_pairs:
        cursor.execute("""
            INSERT INTO currency_pairs (symbol, base_currency, quote_currency, pip_value, min_lot_size, max_lot_size)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO NOTHING
        """, (symbol, base, quote, pip_val, min_lot, max_lot))
    
    # Insert timeframes
    timeframes = [
        ('M1', 1),
        ('M5', 5),
        ('M15', 15),
        ('M30', 30),
        ('H1', 60),
        ('H4', 240),
        ('D1', 1440),
        ('W1', 10080),
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
            'parameters': {
                'ma_fast': 20,
                'ma_slow': 50,
                'stop_loss': 50,
                'take_profit': 100,
                'risk_per_trade': 0.02
            },
            'risk_management': {
                'max_risk_per_trade': 0.02,
                'max_daily_risk': 0.05,
                'max_drawdown': 0.15
            },
            'entry_conditions': {
                'ma_crossover': True,
                'volume_confirmation': True,
                'trend_strength': 0.7
            },
            'exit_conditions': {
                'stop_loss': True,
                'take_profit': True,
                'trailing_stop': True
            },
            'profit_factor': 1.45,
            'win_rate': 62.5,
            'max_drawdown': 8.2,
            'sharpe_ratio': 1.23,
            'total_trades': 156,
            'total_profit': 2340.50,
            'status': 'active',
            'priority': 8,
            'is_ai_generated': True,
            'generation_method': 'genetic_algorithm'
        },
        {
            'strategy_id': 'STRAT_002',
            'name': 'GBP/USD Range Trading',
            'description': 'RSI-based range trading strategy with support/resistance levels',
            'strategy_type': 'range_trading',
            'parameters': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stop_loss': 30,
                'take_profit': 60
            },
            'risk_management': {
                'max_risk_per_trade': 0.015,
                'max_daily_risk': 0.04,
                'max_drawdown': 0.12
            },
            'entry_conditions': {
                'rsi_oversold': True,
                'support_level': True,
                'volume_spike': False
            },
            'exit_conditions': {
                'rsi_overbought': True,
                'resistance_level': True,
                'time_based': True
            },
            'profit_factor': 1.32,
            'win_rate': 58.3,
            'max_drawdown': 12.1,
            'sharpe_ratio': 1.15,
            'total_trades': 89,
            'total_profit': 1875.25,
            'status': 'active',
            'priority': 6,
            'is_ai_generated': True,
            'generation_method': 'neural_network'
        },
        {
            'strategy_id': 'STRAT_003',
            'name': 'USD/JPY Breakout',
            'description': 'Breakout strategy with volume confirmation and volatility filters',
            'strategy_type': 'breakout',
            'parameters': {
                'breakout_period': 20,
                'volume_threshold': 1.5,
                'stop_loss': 40,
                'take_profit': 80,
                'volatility_filter': True
            },
            'risk_management': {
                'max_risk_per_trade': 0.025,
                'max_daily_risk': 0.06,
                'max_drawdown': 0.10
            },
            'entry_conditions': {
                'price_breakout': True,
                'volume_confirmation': True,
                'volatility_expansion': True
            },
            'exit_conditions': {
                'stop_loss': True,
                'take_profit': True,
                'momentum_loss': True
            },
            'profit_factor': 1.67,
            'win_rate': 71.2,
            'max_drawdown': 6.8,
            'sharpe_ratio': 1.45,
            'total_trades': 203,
            'total_profit': 4567.80,
            'status': 'paused',
            'priority': 9,
            'is_ai_generated': True,
            'generation_method': 'genetic_algorithm'
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
        ('STRAT_003', 'USDJPY', ['H4', 'D1']),
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
    
    # Insert sample news events
    news_events = [
        {
            'title': 'Non-Farm Payrolls',
            'description': 'Change in the number of employed people during the previous month',
            'currency': 'USD',
            'impact_level': 'high',
            'event_time': datetime.now() + timedelta(days=5),
            'forecast_value': 180000,
            'previous_value': 175000
        },
        {
            'title': 'ECB Interest Rate Decision',
            'description': 'European Central Bank interest rate announcement',
            'currency': 'EUR',
            'impact_level': 'high',
            'event_time': datetime.now() + timedelta(days=12),
            'forecast_value': 4.25,
            'previous_value': 4.00
        },
        {
            'title': 'BoE Interest Rate Decision',
            'description': 'Bank of England interest rate announcement',
            'currency': 'GBP',
            'impact_level': 'high',
            'event_time': datetime.now() + timedelta(days=19),
            'forecast_value': 5.50,
            'previous_value': 5.25
        }
    ]
    
    for event in news_events:
        cursor.execute("""
            INSERT INTO news_events (title, description, currency, impact_level, event_time, forecast_value, previous_value)
            VALUES (%(title)s, %(description)s, %(currency)s, %(impact_level)s, %(event_time)s, %(forecast_value)s, %(previous_value)s)
        """, event)
    
    # Insert sample sentiment scores
    currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    for i in range(24):  # Last 24 hours
        for pair in currency_pairs:
            cursor.execute("SELECT id FROM currency_pairs WHERE symbol = %s", (pair,))
            pair_row = cursor.fetchone()
            if pair_row:
                timestamp = datetime.now() - timedelta(hours=i)
                sentiment = random.uniform(-50, 50)
                
                cursor.execute("""
                    INSERT INTO sentiment_scores (
                        currency_pair_id, timestamp, overall_sentiment, news_sentiment,
                        social_sentiment, technical_sentiment, confidence_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    pair_row[0], timestamp, sentiment,
                    sentiment + random.uniform(-10, 10),
                    sentiment + random.uniform(-15, 15),
                    sentiment + random.uniform(-5, 5),
                    random.uniform(60, 95)
                ))
    
    logger.info("Initial data seeded successfully")

def main():
    """Main migration function"""
    try:
        logger.info("Starting comprehensive database migration...")
        
        # Connect to database
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if we should drop existing tables
        if len(sys.argv) > 1 and sys.argv[1] == '--drop':
            logger.warning("Dropping existing tables...")
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            cursor.execute("GRANT ALL ON SCHEMA public TO postgres")
            cursor.execute("GRANT ALL ON SCHEMA public TO public")
        
        # Apply comprehensive schema
        schema_file = os.path.join(os.path.dirname(__file__), 'schema_comprehensive.sql')
        if os.path.exists(schema_file):
            execute_sql_file(cursor, schema_file)
        else:
            logger.error(f"Schema file not found: {schema_file}")
            return False
        
        # Seed initial data
        seed_initial_data(cursor)
        
        # Verify migration
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cursor.fetchone()[0]
        logger.info(f"Migration completed. Created {table_count} tables.")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Comprehensive database migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
