#!/usr/bin/env python3
"""
Validation Storage Service

PostgreSQL storage service for strategy validation results using our new validation framework.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import asyncpg
from sqlalchemy.orm import Session
from sqlalchemy import text

from ...database.connection import get_db_session
from ...database.models import Strategy

logger = logging.getLogger(__name__)


@dataclass
class ValidationResultRecord:
    """Validation result record for database storage"""
    
    # Basic info
    strategy_id: str
    strategy_name: str
    strategy_type: str
    validation_timestamp: datetime
    
    # Validation results
    validation_passed: bool
    validation_score: float
    critical_violations: List[str]
    warnings: List[str]
    
    # Performance metrics (JSON)
    performance_metrics: Dict[str, Any]
    
    # Scoring metrics (JSON)
    scoring_metrics: Dict[str, Any]
    
    # Backtest configuration (JSON)
    backtest_config: Dict[str, Any]
    
    # Additional metadata
    validation_duration_seconds: float
    backtest_duration_days: int
    total_trades: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'validation_passed': self.validation_passed,
            'validation_score': self.validation_score,
            'critical_violations': self.critical_violations,
            'warnings': self.warnings,
            'performance_metrics': self.performance_metrics,
            'scoring_metrics': self.scoring_metrics,
            'backtest_config': self.backtest_config,
            'validation_duration_seconds': self.validation_duration_seconds,
            'backtest_duration_days': self.backtest_duration_days,
            'total_trades': self.total_trades
        }


class ValidationStorageService:
    """PostgreSQL storage service for validation results"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
        
        # Initialize database schema
        self._init_database()
        
        self.logger.info("ValidationStorageService initialized")
    
    def _init_database(self):
        """Initialize database schema for validation results"""
        try:
            with get_db_session() as session:
                # Create validation_results table
                session.execute(text("""
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
                """))
                
                # Create indexes
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_validation_results_strategy_id 
                    ON validation_results(strategy_id)
                """))
                
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_validation_results_timestamp 
                    ON validation_results(validation_timestamp)
                """))
                
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_validation_results_passed 
                    ON validation_results(validation_passed)
                """))
                
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_validation_results_score 
                    ON validation_results(validation_score)
                """))
                
                # Create strategy_performance_summary view
                session.execute(text("""
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
                        AVG((performance_metrics->>'sharpe_ratio')::DECIMAL) as avg_sharpe,
                        AVG((performance_metrics->>'max_drawdown')::DECIMAL) as avg_drawdown,
                        AVG((performance_metrics->>'win_rate')::DECIMAL) as avg_win_rate,
                        AVG((performance_metrics->>'profit_factor')::DECIMAL) as avg_profit_factor
                    FROM validation_results
                    GROUP BY strategy_id, strategy_name, strategy_type
                """))
                
                session.commit()
                self.logger.info("Database schema initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {e}")
            raise
    
    async def store_validation_result(self, result: ValidationResultRecord) -> bool:
        """Store validation result in database"""
        try:
            conn = await asyncpg.connect(self.connection_string)
            try:
                await conn.execute("""
                    INSERT INTO validation_results (
                        strategy_id, strategy_name, strategy_type, validation_timestamp,
                        validation_passed, validation_score, critical_violations, warnings,
                        performance_metrics, scoring_metrics, backtest_config,
                        validation_duration_seconds, backtest_duration_days, total_trades
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                    )
                """, 
                result.strategy_id,
                result.strategy_name,
                result.strategy_type,
                result.validation_timestamp,
                result.validation_passed,
                result.validation_score,
                json.dumps(result.critical_violations),
                json.dumps(result.warnings),
                json.dumps(result.performance_metrics),
                json.dumps(result.scoring_metrics),
                json.dumps(result.backtest_config),
                result.validation_duration_seconds,
                result.backtest_duration_days,
                result.total_trades
                )
                
                self.logger.info(f"Stored validation result for strategy {result.strategy_id}")
                return True
            finally:
                await conn.close()
                
        except Exception as e:
            self.logger.error(f"Failed to store validation result: {e}")
            return False
    
    async def get_validation_results(self, strategy_id: Optional[str] = None,
                                   limit: int = 100,
                                   offset: int = 0) -> List[Dict[str, Any]]:
        """Get validation results from database"""
        try:
            conn = await asyncpg.connect(self.connection_string)
            try:
                if strategy_id:
                    query = """
                        SELECT * FROM validation_results 
                        WHERE strategy_id = $1 
                        ORDER BY validation_timestamp DESC 
                        LIMIT $2 OFFSET $3
                    """
                    rows = await conn.fetch(query, strategy_id, limit, offset)
                else:
                    query = """
                        SELECT * FROM validation_results 
                        ORDER BY validation_timestamp DESC 
                        LIMIT $1 OFFSET $2
                    """
                    rows = await conn.fetch(query, limit, offset)
                
                results = []
                for row in rows:
                    result = {
                        'id': row['id'],
                        'strategy_id': row['strategy_id'],
                        'strategy_name': row['strategy_name'],
                        'strategy_type': row['strategy_type'],
                        'validation_timestamp': row['validation_timestamp'].isoformat(),
                        'validation_passed': row['validation_passed'],
                        'validation_score': float(row['validation_score']),
                        'critical_violations': json.loads(row['critical_violations'] or '[]'),
                        'warnings': json.loads(row['warnings'] or '[]'),
                        'performance_metrics': json.loads(row['performance_metrics']),
                        'scoring_metrics': json.loads(row['scoring_metrics']),
                        'backtest_config': json.loads(row['backtest_config']),
                        'validation_duration_seconds': float(row['validation_duration_seconds'] or 0),
                        'backtest_duration_days': row['backtest_duration_days'] or 0,
                        'total_trades': row['total_trades'] or 0,
                        'created_at': row['created_at'].isoformat(),
                        'updated_at': row['updated_at'].isoformat()
                    }
                    results.append(result)
                
                return results
            finally:
                await conn.close()
                
        except Exception as e:
            self.logger.error(f"Failed to get validation results: {e}")
            return []
    
    async def get_strategy_performance_summary(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get strategy performance summary"""
        try:
            conn = await asyncpg.connect(self.connection_string)
            try:
                if strategy_id:
                    query = """
                        SELECT * FROM strategy_performance_summary 
                        WHERE strategy_id = $1
                    """
                    rows = await conn.fetch(query, strategy_id)
                else:
                    query = "SELECT * FROM strategy_performance_summary ORDER BY avg_score DESC"
                    rows = await conn.fetch(query)
                
                return [dict(row) for row in rows]
            finally:
                await conn.close()
                
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return []
    
    async def get_top_strategies(self, limit: int = 10, 
                               min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Get top performing strategies"""
        try:
            async with asyncpg.connect(self.connection_string) as conn:
                query = """
                    SELECT * FROM strategy_performance_summary 
                    WHERE avg_score >= $1
                    ORDER BY avg_score DESC 
                    LIMIT $2
                """
                rows = await conn.fetch(query, min_score, limit)
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get top strategies: {e}")
            return []
    
    async def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        try:
            async with asyncpg.connect(self.connection_string) as conn:
                # Total validations
                total_validations = await conn.fetchval("SELECT COUNT(*) FROM validation_results")
                
                # Passed validations
                passed_validations = await conn.fetchval(
                    "SELECT COUNT(*) FROM validation_results WHERE validation_passed = true"
                )
                
                # Average score
                avg_score = await conn.fetchval(
                    "SELECT AVG(validation_score) FROM validation_results"
                )
                
                # Score distribution
                score_distribution = await conn.fetch("""
                    SELECT 
                        CASE 
                            WHEN validation_score >= 0.9 THEN 'Excellent (0.9+)'
                            WHEN validation_score >= 0.8 THEN 'Good (0.8-0.9)'
                            WHEN validation_score >= 0.7 THEN 'Fair (0.7-0.8)'
                            WHEN validation_score >= 0.6 THEN 'Poor (0.6-0.7)'
                            ELSE 'Very Poor (<0.6)'
                        END as score_range,
                        COUNT(*) as count
                    FROM validation_results
                    GROUP BY score_range
                    ORDER BY score_range
                """)
                
                # Recent validations (last 24 hours)
                recent_validations = await conn.fetchval("""
                    SELECT COUNT(*) FROM validation_results 
                    WHERE validation_timestamp >= NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    'total_validations': total_validations,
                    'passed_validations': passed_validations,
                    'pass_rate': passed_validations / total_validations if total_validations > 0 else 0,
                    'average_score': float(avg_score) if avg_score else 0,
                    'score_distribution': [dict(row) for row in score_distribution],
                    'recent_validations_24h': recent_validations
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get validation statistics: {e}")
            return {}
    
    async def delete_validation_results(self, strategy_id: str) -> bool:
        """Delete validation results for a strategy"""
        try:
            async with asyncpg.connect(self.connection_string) as conn:
                result = await conn.execute(
                    "DELETE FROM validation_results WHERE strategy_id = $1",
                    strategy_id
                )
                
                deleted_count = int(result.split()[-1])
                self.logger.info(f"Deleted {deleted_count} validation results for strategy {strategy_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete validation results: {e}")
            return False
    
    async def cleanup_old_results(self, days_to_keep: int = 30) -> int:
        """Clean up old validation results"""
        try:
            async with asyncpg.connect(self.connection_string) as conn:
                result = await conn.execute("""
                    DELETE FROM validation_results 
                    WHERE validation_timestamp < NOW() - INTERVAL '%s days'
                """ % days_to_keep)
                
                deleted_count = int(result.split()[-1])
                self.logger.info(f"Cleaned up {deleted_count} old validation results")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old results: {e}")
            return 0
