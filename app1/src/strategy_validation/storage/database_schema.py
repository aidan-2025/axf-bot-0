#!/usr/bin/env python3
"""
Database Schema

Defines the database schema for strategy validation storage.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime


@dataclass
class DatabaseSchema:
    """Database schema definitions"""
    
    @staticmethod
    def get_validation_results_schema() -> str:
        """Get validation results table schema"""
        return """
        CREATE TABLE IF NOT EXISTS validation_results (
            id SERIAL PRIMARY KEY,
            strategy_id VARCHAR(255) NOT NULL,
            strategy_name VARCHAR(255) NOT NULL,
            strategy_type VARCHAR(100) NOT NULL,
            validation_timestamp TIMESTAMP NOT NULL,
            validation_passed BOOLEAN NOT NULL,
            validation_score DECIMAL(5,4) NOT NULL,
            critical_violations JSONB,
            warnings JSONB,
            performance_metrics JSONB NOT NULL,
            scoring_metrics JSONB NOT NULL,
            backtest_config JSONB NOT NULL,
            validation_duration_seconds DECIMAL(10,3),
            backtest_duration_days INTEGER,
            total_trades INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    
    @staticmethod
    def get_indexes() -> List[str]:
        """Get database indexes"""
        return [
            "CREATE INDEX IF NOT EXISTS idx_validation_results_strategy_id ON validation_results(strategy_id)",
            "CREATE INDEX IF NOT EXISTS idx_validation_results_timestamp ON validation_results(validation_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_validation_results_passed ON validation_results(validation_passed)",
            "CREATE INDEX IF NOT EXISTS idx_validation_results_score ON validation_results(validation_score)",
            "CREATE INDEX IF NOT EXISTS idx_validation_results_strategy_type ON validation_results(strategy_type)"
        ]
    
    @staticmethod
    def get_views() -> List[str]:
        """Get database views"""
        return [
            """
            CREATE OR REPLACE VIEW strategy_performance_summary AS
            SELECT 
                strategy_id,
                strategy_name,
                strategy_type,
                MAX(validation_timestamp) as last_validated,
                COUNT(*) as total_validations,
                AVG(validation_score) as avg_score,
                MAX(validation_score) as best_score,
                MIN(validation_score) as worst_score,
                AVG((performance_metrics->>'total_return')::DECIMAL) as avg_return,
                AVG((performance_metrics->>'annualized_return')::DECIMAL) as avg_annualized_return,
                AVG((performance_metrics->>'sharpe_ratio')::DECIMAL) as avg_sharpe,
                AVG((performance_metrics->>'sortino_ratio')::DECIMAL) as avg_sortino,
                AVG((performance_metrics->>'calmar_ratio')::DECIMAL) as avg_calmar,
                AVG((performance_metrics->>'max_drawdown')::DECIMAL) as avg_drawdown,
                AVG((performance_metrics->>'win_rate')::DECIMAL) as avg_win_rate,
                AVG((performance_metrics->>'profit_factor')::DECIMAL) as avg_profit_factor,
                AVG((performance_metrics->>'total_trades')::INTEGER) as avg_trades,
                AVG((performance_metrics->>'volatility')::DECIMAL) as avg_volatility,
                AVG((performance_metrics->>'consistency_score')::DECIMAL) as avg_consistency,
                AVG((performance_metrics->>'stability_score')::DECIMAL) as avg_stability
            FROM validation_results
            GROUP BY strategy_id, strategy_name, strategy_type
            """,
            
            """
            CREATE OR REPLACE VIEW validation_statistics AS
            SELECT 
                DATE(validation_timestamp) as validation_date,
                COUNT(*) as total_validations,
                COUNT(CASE WHEN validation_passed = true THEN 1 END) as passed_validations,
                AVG(validation_score) as avg_score,
                AVG((performance_metrics->>'total_return')::DECIMAL) as avg_return,
                AVG((performance_metrics->>'sharpe_ratio')::DECIMAL) as avg_sharpe,
                AVG((performance_metrics->>'max_drawdown')::DECIMAL) as avg_drawdown
            FROM validation_results
            GROUP BY DATE(validation_timestamp)
            ORDER BY validation_date DESC
            """,
            
            """
            CREATE OR REPLACE VIEW top_strategies AS
            SELECT 
                strategy_id,
                strategy_name,
                strategy_type,
                avg_score,
                avg_return,
                avg_sharpe,
                avg_drawdown,
                total_validations,
                last_validated
            FROM strategy_performance_summary
            WHERE avg_score >= 0.7
            ORDER BY avg_score DESC
            """
        ]
    
    @staticmethod
    def get_functions() -> List[str]:
        """Get database functions"""
        return [
            """
            CREATE OR REPLACE FUNCTION update_validation_timestamp()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """,
            
            """
            CREATE TRIGGER update_validation_results_timestamp
                BEFORE UPDATE ON validation_results
                FOR EACH ROW
                EXECUTE FUNCTION update_validation_timestamp();
            """,
            
            """
            CREATE OR REPLACE FUNCTION get_strategy_performance_history(
                p_strategy_id VARCHAR(255),
                p_days INTEGER DEFAULT 30
            )
            RETURNS TABLE (
                validation_timestamp TIMESTAMP,
                validation_score DECIMAL(5,4),
                total_return DECIMAL(10,4),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(8,4),
                win_rate DECIMAL(5,4),
                profit_factor DECIMAL(8,4)
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    vr.validation_timestamp,
                    vr.validation_score,
                    (vr.performance_metrics->>'total_return')::DECIMAL as total_return,
                    (vr.performance_metrics->>'sharpe_ratio')::DECIMAL as sharpe_ratio,
                    (vr.performance_metrics->>'max_drawdown')::DECIMAL as max_drawdown,
                    (vr.performance_metrics->>'win_rate')::DECIMAL as win_rate,
                    (vr.performance_metrics->>'profit_factor')::DECIMAL as profit_factor
                FROM validation_results vr
                WHERE vr.strategy_id = p_strategy_id
                AND vr.validation_timestamp >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY vr.validation_timestamp DESC;
            END;
            $$ LANGUAGE plpgsql;
            """.replace('%s', 'p_days')
        ]
    
    @staticmethod
    def get_full_schema() -> str:
        """Get complete database schema"""
        schema_parts = [
            DatabaseSchema.get_validation_results_schema(),
            *DatabaseSchema.get_indexes(),
            *DatabaseSchema.get_views(),
            *DatabaseSchema.get_functions()
        ]
        
        return '\n\n'.join(schema_parts)
    
    @staticmethod
    def get_sample_queries() -> Dict[str, str]:
        """Get sample queries for common operations"""
        return {
            'get_recent_validations': """
                SELECT strategy_id, strategy_name, validation_score, validation_passed
                FROM validation_results
                WHERE validation_timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY validation_timestamp DESC
                LIMIT 10
            """,
            
            'get_failed_validations': """
                SELECT strategy_id, strategy_name, critical_violations, warnings
                FROM validation_results
                WHERE validation_passed = false
                ORDER BY validation_timestamp DESC
                LIMIT 20
            """,
            
            'get_strategy_trends': """
                SELECT 
                    strategy_id,
                    DATE(validation_timestamp) as date,
                    AVG(validation_score) as avg_score,
                    COUNT(*) as validation_count
                FROM validation_results
                WHERE validation_timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY strategy_id, DATE(validation_timestamp)
                ORDER BY strategy_id, date
            """,
            
            'get_performance_rankings': """
                SELECT 
                    strategy_id,
                    strategy_name,
                    avg_score,
                    avg_return,
                    avg_sharpe,
                    avg_drawdown,
                    RANK() OVER (ORDER BY avg_score DESC) as score_rank,
                    RANK() OVER (ORDER BY avg_return DESC) as return_rank,
                    RANK() OVER (ORDER BY avg_sharpe DESC) as sharpe_rank
                FROM strategy_performance_summary
                WHERE total_validations >= 3
                ORDER BY avg_score DESC
            """,
            
            'get_validation_quality_metrics': """
                SELECT 
                    COUNT(*) as total_validations,
                    COUNT(CASE WHEN validation_passed = true THEN 1 END) as passed_count,
                    ROUND(COUNT(CASE WHEN validation_passed = true THEN 1 END) * 100.0 / COUNT(*), 2) as pass_rate,
                    ROUND(AVG(validation_score), 4) as avg_score,
                    ROUND(STDDEV(validation_score), 4) as score_stddev,
                    MIN(validation_timestamp) as first_validation,
                    MAX(validation_timestamp) as last_validation
                FROM validation_results
            """
        }

