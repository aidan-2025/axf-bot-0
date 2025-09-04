#!/usr/bin/env python3
"""
Database Migration Script for AXF Bot 0
Handles database schema migrations and data seeding for local development
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DatabaseMigrator:
    def __init__(self, database_url=None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/axf_bot_db')
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to the database"""
        try:
            self.conn = psycopg2.connect(self.database_url)
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.conn.cursor()
            print("✅ Connected to database successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the database"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✅ Disconnected from database")
    
    def create_migrations_table(self):
        """Create migrations tracking table"""
        create_migrations_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            version VARCHAR(50) UNIQUE NOT NULL,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cursor.execute(create_migrations_table_sql)
        print("✅ Created migrations tracking table")
    
    def get_applied_migrations(self):
        """Get list of applied migrations"""
        self.cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [row[0] for row in self.cursor.fetchall()]
    
    def mark_migration_applied(self, version, description):
        """Mark a migration as applied"""
        self.cursor.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (%s, %s)",
            (version, description)
        )
    
    def run_migration(self, version, description, sql_commands):
        """Run a single migration"""
        print(f"🔄 Running migration {version}: {description}")
        
        try:
            # Execute all SQL commands in the migration
            for sql in sql_commands:
                if sql.strip():
                    self.cursor.execute(sql)
            
            # Mark migration as applied
            self.mark_migration_applied(version, description)
            print(f"✅ Migration {version} completed successfully")
            return True
        except Exception as e:
            print(f"❌ Migration {version} failed: {e}")
            return False
    
    def run_all_migrations(self):
        """Run all pending migrations"""
        print("🚀 Starting database migrations...")
        
        # Create migrations table if it doesn't exist
        self.create_migrations_table()
        
        # Get applied migrations
        applied_migrations = self.get_applied_migrations()
        
        # Define migrations in order
        migrations = [
            {
                "version": "001_initial_schema",
                "description": "Create initial database schema",
                "sql": self.get_initial_schema_sql()
            },
            {
                "version": "002_add_indexes",
                "description": "Add database indexes for performance",
                "sql": self.get_indexes_sql()
            },
            {
                "version": "003_seed_data",
                "description": "Seed initial data for development",
                "sql": self.get_seed_data_sql()
            }
        ]
        
        # Run pending migrations
        for migration in migrations:
            if migration["version"] not in applied_migrations:
                success = self.run_migration(
                    migration["version"],
                    migration["description"],
                    migration["sql"]
                )
                if not success:
                    print(f"❌ Migration failed, stopping")
                    return False
            else:
                print(f"⏭️  Migration {migration['version']} already applied, skipping")
        
        print("✅ All migrations completed successfully")
        return True
    
    def get_initial_schema_sql(self):
        """Get initial schema SQL commands"""
        return [
            """
            -- Create strategies table
            CREATE TABLE IF NOT EXISTS strategies (
                id SERIAL PRIMARY KEY,
                strategy_id VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                strategy_type VARCHAR(50) NOT NULL,
                currency_pairs TEXT[] NOT NULL,
                timeframes TEXT[] NOT NULL,
                parameters JSONB NOT NULL,
                profit_factor DECIMAL(5,2) DEFAULT 0,
                win_rate DECIMAL(5,2) DEFAULT 0,
                max_drawdown DECIMAL(5,2) DEFAULT 0,
                sharpe_ratio DECIMAL(5,2) DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                total_profit DECIMAL(15,2) DEFAULT 0,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            -- Create strategy_performance table
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id SERIAL PRIMARY KEY,
                strategy_id VARCHAR(50) NOT NULL REFERENCES strategies(strategy_id),
                performance_date DATE NOT NULL,
                profit_loss DECIMAL(15,2) DEFAULT 0,
                trades_count INTEGER DEFAULT 0,
                win_rate DECIMAL(5,2) DEFAULT 0,
                max_drawdown DECIMAL(5,2) DEFAULT 0,
                sharpe_ratio DECIMAL(5,2) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            -- Create expert_advisors table
            CREATE TABLE IF NOT EXISTS expert_advisors (
                id SERIAL PRIMARY KEY,
                ea_id VARCHAR(50) UNIQUE NOT NULL,
                strategy_id VARCHAR(50) NOT NULL REFERENCES strategies(strategy_id),
                name VARCHAR(200) NOT NULL,
                description TEXT,
                mql4_code TEXT NOT NULL,
                parameters JSONB NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            -- Create updated_at trigger function
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            """,
            """
            -- Create triggers for updated_at
            DROP TRIGGER IF EXISTS update_strategies_updated_at ON strategies;
            CREATE TRIGGER update_strategies_updated_at
                BEFORE UPDATE ON strategies
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """,
            """
            DROP TRIGGER IF EXISTS update_expert_advisors_updated_at ON expert_advisors;
            CREATE TRIGGER update_expert_advisors_updated_at
                BEFORE UPDATE ON expert_advisors
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """
        ]
    
    def get_indexes_sql(self):
        """Get database indexes SQL commands"""
        return [
            """
            -- Create indexes for better performance
            CREATE INDEX IF NOT EXISTS idx_strategies_strategy_id ON strategies(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status);
            CREATE INDEX IF NOT EXISTS idx_strategies_created_at ON strategies(created_at);
            CREATE INDEX IF NOT EXISTS idx_strategies_profit_factor ON strategies(profit_factor);
            CREATE INDEX IF NOT EXISTS idx_strategies_win_rate ON strategies(win_rate);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_id ON strategy_performance(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_strategy_performance_date ON strategy_performance(performance_date);
            CREATE INDEX IF NOT EXISTS idx_strategy_performance_profit_loss ON strategy_performance(profit_loss);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_expert_advisors_ea_id ON expert_advisors(ea_id);
            CREATE INDEX IF NOT EXISTS idx_expert_advisors_strategy_id ON expert_advisors(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_expert_advisors_status ON expert_advisors(status);
            """,
            """
            -- Create GIN indexes for JSONB columns
            CREATE INDEX IF NOT EXISTS idx_strategies_parameters ON strategies USING GIN(parameters);
            CREATE INDEX IF NOT EXISTS idx_expert_advisors_parameters ON expert_advisors USING GIN(parameters);
            """
        ]
    
    def get_seed_data_sql(self):
        """Get seed data SQL commands"""
        return [
            """
            -- Insert sample strategies
            INSERT INTO strategies (strategy_id, name, description, strategy_type, currency_pairs, timeframes, parameters, status)
            VALUES 
            (
                'SMA_CROSS_001',
                'Simple Moving Average Crossover',
                'Basic SMA crossover strategy for EUR/USD',
                'trend_following',
                ARRAY['EUR/USD'],
                ARRAY['H1', 'H4'],
                '{"fast_sma": 10, "slow_sma": 20, "stop_loss": 50, "take_profit": 100}'::jsonb,
                'active'
            ),
            (
                'RSI_OVERSOLD_001',
                'RSI Oversold Strategy',
                'RSI-based strategy for oversold conditions',
                'mean_reversion',
                ARRAY['GBP/USD', 'USD/JPY'],
                ARRAY['M15', 'H1'],
                '{"rsi_period": 14, "oversold_level": 30, "overbought_level": 70, "stop_loss": 30, "take_profit": 60}'::jsonb,
                'active'
            ),
            (
                'BOLLINGER_BANDS_001',
                'Bollinger Bands Strategy',
                'Bollinger Bands mean reversion strategy',
                'mean_reversion',
                ARRAY['AUD/USD', 'USD/CAD'],
                ARRAY['H1', 'H4'],
                '{"bb_period": 20, "bb_std": 2, "stop_loss": 40, "take_profit": 80}'::jsonb,
                'pending'
            )
            ON CONFLICT (strategy_id) DO NOTHING;
            """,
            """
            -- Insert sample performance data
            INSERT INTO strategy_performance (strategy_id, performance_date, profit_loss, trades_count, win_rate, max_drawdown, sharpe_ratio)
            VALUES 
            ('SMA_CROSS_001', CURRENT_DATE - INTERVAL '1 day', 125.50, 5, 80.0, 15.2, 1.8),
            ('SMA_CROSS_001', CURRENT_DATE - INTERVAL '2 days', -25.30, 3, 33.3, 20.1, 0.5),
            ('RSI_OVERSOLD_001', CURRENT_DATE - INTERVAL '1 day', 89.75, 4, 75.0, 12.8, 2.1),
            ('RSI_OVERSOLD_001', CURRENT_DATE - INTERVAL '2 days', 156.20, 6, 83.3, 8.5, 2.8)
            ON CONFLICT DO NOTHING;
            """,
            """
            -- Insert sample expert advisors
            INSERT INTO expert_advisors (ea_id, strategy_id, name, description, mql4_code, parameters, status)
            VALUES 
            (
                'EA_SMA_CROSS_001',
                'SMA_CROSS_001',
                'SMA Cross EA v1.0',
                'Expert Advisor implementing SMA crossover strategy',
                '// SMA Cross EA Code\n// Implementation details...',
                '{"lot_size": 0.1, "max_trades": 1, "magic_number": 12345}'::jsonb,
                'active'
            ),
            (
                'EA_RSI_OVERSOLD_001',
                'RSI_OVERSOLD_001',
                'RSI Oversold EA v1.0',
                'Expert Advisor implementing RSI oversold strategy',
                '// RSI Oversold EA Code\n// Implementation details...',
                '{"lot_size": 0.1, "max_trades": 2, "magic_number": 12346}'::jsonb,
                'active'
            )
            ON CONFLICT (ea_id) DO NOTHING;
            """
        ]
    
    def reset_database(self):
        """Reset database (WARNING: This will delete all data)"""
        print("⚠️  WARNING: This will delete all data in the database!")
        confirm = input("Are you sure you want to continue? (yes/no): ")
        
        if confirm.lower() != 'yes':
            print("❌ Database reset cancelled")
            return False
        
        try:
            # Drop all tables
            self.cursor.execute("DROP SCHEMA public CASCADE;")
            self.cursor.execute("CREATE SCHEMA public;")
            self.cursor.execute("GRANT ALL ON SCHEMA public TO postgres;")
            self.cursor.execute("GRANT ALL ON SCHEMA public TO public;")
            
            print("✅ Database reset completed")
            return True
        except Exception as e:
            print(f"❌ Database reset failed: {e}")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AXF Bot 0 Database Migrator')
    parser.add_argument('--database-url', help='Database URL')
    parser.add_argument('--reset', action='store_true', help='Reset database (WARNING: deletes all data)')
    parser.add_argument('--migrate', action='store_true', help='Run migrations')
    
    args = parser.parse_args()
    
    # Default to migrate if no action specified
    if not args.reset and not args.migrate:
        args.migrate = True
    
    migrator = DatabaseMigrator(args.database_url)
    
    if not migrator.connect():
        sys.exit(1)
    
    try:
        if args.reset:
            success = migrator.reset_database()
        else:
            success = migrator.run_all_migrations()
        
        if success:
            print("✅ Operation completed successfully")
            sys.exit(0)
        else:
            print("❌ Operation failed")
            sys.exit(1)
    finally:
        migrator.disconnect()

if __name__ == "__main__":
    main()
