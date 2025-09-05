#!/usr/bin/env python3
"""
Validation Integration

Integrates the validation framework with PostgreSQL storage.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..filtering.filtering_pipeline import FilteringPipeline, PipelineResult
from ..evaluation.strategy_evaluator import EvaluationResult
from ..scoring.scoring_engine import ScoringResult
from .validation_storage_service import ValidationStorageService, ValidationResultRecord

logger = logging.getLogger(__name__)


class ValidationIntegration:
    """Integrates validation framework with PostgreSQL storage"""
    
    def __init__(self, connection_string: str):
        self.storage_service = ValidationStorageService(connection_string)
        self.logger = logging.getLogger(__name__)
    
    async def store_pipeline_result(self, pipeline_result: PipelineResult) -> bool:
        """Store pipeline result in database"""
        try:
            # Extract validation results from pipeline
            validation_records = []
            
            for strategy_result in pipeline_result.passing_strategies:
                # Create validation record
                record = ValidationResultRecord(
                    strategy_id=strategy_result.get('strategy_id', 'unknown'),
                    strategy_name=strategy_result.get('strategy_name', 'Unknown Strategy'),
                    strategy_type=strategy_result.get('strategy_type', 'unknown'),
                    validation_timestamp=datetime.now(),
                    validation_passed=True,  # These are passing strategies
                    validation_score=float(strategy_result.get('total_score', 0.0)),
                    critical_violations=strategy_result.get('critical_violations', []),
                    warnings=strategy_result.get('warnings', []),
                    performance_metrics=strategy_result.get('performance_metrics', {}),
                    scoring_metrics=strategy_result.get('scoring_metrics', {}),
                    backtest_config=strategy_result.get('backtest_config', {}),
                    validation_duration_seconds=pipeline_result.duration,
                    backtest_duration_days=30,  # Default
                    total_trades=strategy_result.get('total_trades', 0)
                )
                validation_records.append(record)
            
            # Store each validation result
            success_count = 0
            for record in validation_records:
                if await self.storage_service.store_validation_result(record):
                    success_count += 1
            
            self.logger.info(f"Stored {success_count}/{len(validation_records)} validation results")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to store pipeline result: {e}")
            return False
    
    async def store_evaluation_result(self, evaluation_result: EvaluationResult) -> bool:
        """Store evaluation result in database"""
        try:
            # Create validation record from evaluation result
            record = ValidationResultRecord(
                strategy_id=evaluation_result.strategy_id,
                strategy_name=evaluation_result.strategy_name,
                strategy_type=evaluation_result.strategy_type,
                validation_timestamp=datetime.now(),
                validation_passed=evaluation_result.status == 'passed',
                validation_score=float(evaluation_result.metrics.overall_score),
                critical_violations=evaluation_result.metrics.critical_violations,
                warnings=evaluation_result.metrics.warnings,
                performance_metrics=evaluation_result.metrics.to_dict(),
                scoring_metrics={},  # Will be filled by scoring
                backtest_config=evaluation_result.backtest_config,
                validation_duration_seconds=evaluation_result.duration,
                backtest_duration_days=30,  # Default
                total_trades=evaluation_result.metrics.total_trades
            )
            
            return await self.storage_service.store_validation_result(record)
            
        except Exception as e:
            self.logger.error(f"Failed to store evaluation result: {e}")
            return False
    
    async def store_scoring_result(self, scoring_result: ScoringResult) -> bool:
        """Store scoring result in database"""
        try:
            # Create validation record from scoring result
            record = ValidationResultRecord(
                strategy_id=scoring_result.strategy_id,
                strategy_name=scoring_result.strategy_name,
                strategy_type=scoring_result.strategy_type,
                validation_timestamp=datetime.now(),
                validation_passed=scoring_result.total_score >= 0.5,  # Threshold
                validation_score=float(scoring_result.total_score),
                critical_violations=[],
                warnings=[],
                performance_metrics={},  # Will be filled by evaluation
                scoring_metrics=scoring_result.to_dict(),
                backtest_config={},
                validation_duration_seconds=0.0,
                backtest_duration_days=0,
                total_trades=0
            )
            
            return await self.storage_service.store_validation_result(record)
            
        except Exception as e:
            self.logger.error(f"Failed to store scoring result: {e}")
            return False
    
    async def get_validation_results(self, strategy_id: Optional[str] = None,
                                   limit: int = 100,
                                   offset: int = 0) -> List[Dict[str, Any]]:
        """Get validation results from database"""
        return await self.storage_service.get_validation_results(
            strategy_id, limit, offset
        )
    
    async def get_strategy_performance_summary(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get strategy performance summary"""
        return await self.storage_service.get_strategy_performance_summary(strategy_id)
    
    async def get_top_strategies(self, limit: int = 10, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Get top performing strategies"""
        return await self.storage_service.get_top_strategies(limit, min_score)
    
    async def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return await self.storage_service.get_validation_statistics()
    
    async def delete_validation_results(self, strategy_id: str) -> bool:
        """Delete validation results for a strategy"""
        return await self.storage_service.delete_validation_results(strategy_id)
    
    async def cleanup_old_results(self, days_to_keep: int = 30) -> int:
        """Clean up old validation results"""
        return await self.storage_service.cleanup_old_results(days_to_keep)

