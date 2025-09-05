#!/usr/bin/env python3
"""
Validation Workflow Orchestrator

Orchestrates the entire validation process from strategy generation through validation, scoring, and storage.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import traceback

from ...strategy_generation.strategy_generator import StrategyGenerator, StrategyGenerationRequest
from ..filtering.filtering_pipeline import FilteringPipeline, PipelineConfig, PipelineResult
from ..storage.simple_validation_storage import SimpleValidationStorage, ValidationResultRecord

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStage(Enum):
    """Workflow execution stages"""
    INITIALIZATION = "initialization"
    STRATEGY_GENERATION = "strategy_generation"
    VALIDATION = "validation"
    SCORING = "scoring"
    STORAGE = "storage"
    COMPLETION = "completion"


@dataclass
class WorkflowConfig:
    """Configuration for validation workflow"""
    
    # Strategy generation settings
    strategy_count: int = 10
    strategy_types: List[str] = field(default_factory=lambda: ["trend_following", "mean_reversion", "breakout"])
    currency_pairs: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    
    # Validation settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 5
    timeout_seconds: float = 300.0
    
    # Storage settings
    store_results: bool = True
    cleanup_old_results: bool = True
    days_to_keep: int = 30
    
    # Monitoring settings
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True


@dataclass
class WorkflowMetrics:
    """Workflow execution metrics"""
    
    # Timing metrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    
    # Stage timings
    stage_timings: Dict[str, float] = field(default_factory=dict)
    
    # Count metrics
    strategies_generated: int = 0
    strategies_validated: int = 0
    strategies_passed: int = 0
    strategies_failed: int = 0
    strategies_stored: int = 0
    
    # Performance metrics
    strategies_per_second: float = 0.0
    average_validation_time: float = 0.0
    average_storage_time: float = 0.0
    
    # Error metrics
    total_errors: int = 0
    stage_errors: Dict[str, int] = field(default_factory=dict)
    retry_count: int = 0


@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    
    # Basic info
    workflow_id: str
    status: WorkflowStatus
    message: str
    
    # Results
    generated_strategies: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Optional[PipelineResult] = None
    stored_results: List[ValidationResultRecord] = field(default_factory=list)
    
    # Metrics
    metrics: WorkflowMetrics = field(default_factory=WorkflowMetrics)
    
    # Error info
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'message': self.message,
            'generated_strategies': self.generated_strategies,
            'validation_results': self.validation_results.to_dict() if self.validation_results else None,
            'stored_results': [r.to_dict() for r in self.stored_results],
            'metrics': {
                'start_time': self.metrics.start_time.isoformat() if self.metrics.start_time else None,
                'end_time': self.metrics.end_time.isoformat() if self.metrics.end_time else None,
                'duration': self.metrics.duration,
                'stage_timings': self.metrics.stage_timings,
                'strategies_generated': self.metrics.strategies_generated,
                'strategies_validated': self.metrics.strategies_validated,
                'strategies_passed': self.metrics.strategies_passed,
                'strategies_failed': self.metrics.strategies_failed,
                'strategies_stored': self.metrics.strategies_stored,
                'strategies_per_second': self.metrics.strategies_per_second,
                'average_validation_time': self.metrics.average_validation_time,
                'average_storage_time': self.metrics.average_storage_time,
                'total_errors': self.metrics.total_errors,
                'stage_errors': self.metrics.stage_errors,
                'retry_count': self.metrics.retry_count
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class ValidationWorkflow:
    """End-to-end validation workflow orchestrator"""
    
    def __init__(self, config: WorkflowConfig, database_url: str):
        self.config = config
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.strategy_generator = StrategyGenerator()
        self.validation_pipeline = None
        self.storage_service = SimpleValidationStorage(database_url)
        
        # Workflow state
        self.active_workflows: Dict[str, WorkflowResult] = {}
        
        # Setup logging
        if config.enable_logging:
            self._setup_logging()
        
        self.logger.info("ValidationWorkflow initialized")
    
    def _setup_logging(self):
        """Setup workflow-specific logging"""
        workflow_logger = logging.getLogger(f"{__name__}.workflow")
        workflow_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Add file handler for workflow logs
        handler = logging.FileHandler(f"logs/validation_workflow_{datetime.now().strftime('%Y%m%d')}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        workflow_logger.addHandler(handler)
    
    async def run_workflow(self, workflow_id: Optional[str] = None) -> WorkflowResult:
        """Run the complete validation workflow"""
        
        # Generate workflow ID if not provided
        if not workflow_id:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create workflow result
        result = WorkflowResult(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            message="Workflow initialized"
        )
        
        # Add to active workflows
        self.active_workflows[workflow_id] = result
        
        try:
            self.logger.info(f"Starting workflow {workflow_id}")
            result.status = WorkflowStatus.RUNNING
            result.metrics.start_time = datetime.now()
            
            # Stage 1: Strategy Generation
            await self._run_stage(
                result, 
                WorkflowStage.STRATEGY_GENERATION,
                self._generate_strategies
            )
            
            # Stage 2: Validation
            await self._run_stage(
                result,
                WorkflowStage.VALIDATION,
                self._validate_strategies
            )
            
            # Stage 3: Storage
            if self.config.store_results:
                await self._run_stage(
                    result,
                    WorkflowStage.STORAGE,
                    self._store_results
                )
            
            # Stage 4: Cleanup
            if self.config.cleanup_old_results:
                await self._run_stage(
                    result,
                    WorkflowStage.COMPLETION,
                    self._cleanup_old_results
                )
            
            # Complete workflow
            result.status = WorkflowStatus.COMPLETED
            result.message = "Workflow completed successfully"
            result.metrics.end_time = datetime.now()
            result.metrics.duration = (result.metrics.end_time - result.metrics.start_time).total_seconds()
            
            # Calculate performance metrics
            self._calculate_metrics(result)
            
            self.logger.info(f"Workflow {workflow_id} completed successfully")
            
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.message = f"Workflow failed: {str(e)}"
            result.errors.append(str(e))
            result.metrics.total_errors += 1
            
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            self.logger.error(traceback.format_exc())
            
        finally:
            result.updated_at = datetime.now()
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        return result
    
    async def _run_stage(self, result: WorkflowResult, stage: WorkflowStage, stage_func: Callable):
        """Run a workflow stage with error handling and metrics"""
        
        stage_start = datetime.now()
        self.logger.info(f"Starting stage: {stage.value}")
        
        try:
            await stage_func(result)
            stage_duration = (datetime.now() - stage_start).total_seconds()
            result.metrics.stage_timings[stage.value] = stage_duration
            
            self.logger.info(f"Stage {stage.value} completed in {stage_duration:.2f}s")
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            result.metrics.stage_timings[stage.value] = stage_duration
            result.metrics.stage_errors[stage.value] = result.metrics.stage_errors.get(stage.value, 0) + 1
            result.metrics.total_errors += 1
            result.errors.append(f"Stage {stage.value} failed: {str(e)}")
            
            self.logger.error(f"Stage {stage.value} failed: {e}")
            raise
    
    async def _generate_strategies(self, result: WorkflowResult):
        """Generate strategies using the strategy generator"""
        
        self.logger.info(f"Generating {self.config.strategy_count} strategies")
        
        # Create strategy generation request
        request = StrategyGenerationRequest(
            count=self.config.strategy_count,
            strategy_types=self.config.strategy_types,
            symbols=self.config.currency_pairs,
            timeframes=self.config.timeframes
        )
        
        # Generate strategies
        strategies = self.strategy_generator.generate_strategies(request)
        result.generated_strategies = [s.to_dict() for s in strategies]
        result.metrics.strategies_generated = len(strategies)
        
        self.logger.info(f"Generated {len(strategies)} strategies")
    
    async def _validate_strategies(self, result: WorkflowResult):
        """Validate generated strategies using the validation pipeline"""
        
        if not result.generated_strategies:
            raise ValueError("No strategies to validate")
        
        self.logger.info(f"Validating {len(result.generated_strategies)} strategies")
        
        # Create validation pipeline configuration
        pipeline_config = PipelineConfig(
            enable_filtering=True,
            enable_scoring=True,
            enable_evaluation=True,
            enable_ranking=True,
            enable_parallel_processing=self.config.enable_parallel_processing,
            max_workers=self.config.max_workers,
            batch_size=self.config.batch_size,
            timeout_seconds=self.config.timeout_seconds
        )
        
        # Create and run validation pipeline
        self.validation_pipeline = FilteringPipeline(pipeline_config)
        validation_result = await self.validation_pipeline.run_pipeline(
            strategies=result.generated_strategies
        )
        
        result.validation_results = validation_result
        result.metrics.strategies_validated = validation_result.input_strategies
        result.metrics.strategies_passed = validation_result.passing_strategies
        result.metrics.strategies_failed = validation_result.input_strategies - validation_result.passing_strategies
        
        self.logger.info(f"Validation completed: {validation_result.passing_strategies}/{validation_result.input_strategies} strategies passed")
    
    async def _store_results(self, result: WorkflowResult):
        """Store validation results in database"""
        
        if not result.validation_results or not result.validation_results.passing_strategies:
            self.logger.warning("No validation results to store")
            return
        
        self.logger.info(f"Storing {result.validation_results.passing_strategies} validation results")
        
        stored_count = 0
        for strategy_result in result.validation_results.top_strategies:
            try:
                # Create validation record
                validation_record = ValidationResultRecord(
                    strategy_id=strategy_result.get('strategy_id', f"strategy_{uuid.uuid4().hex[:8]}"),
                    strategy_name=strategy_result.get('strategy_name', 'Unknown Strategy'),
                    strategy_type=strategy_result.get('strategy_type', 'unknown'),
                    validation_timestamp=datetime.now(),
                    validation_passed=True,
                    validation_score=float(strategy_result.get('total_score', 0.0)),
                    critical_violations=strategy_result.get('critical_violations', []),
                    warnings=strategy_result.get('warnings', []),
                    performance_metrics=strategy_result.get('performance_metrics', {}),
                    scoring_metrics=strategy_result.get('scoring_metrics', {}),
                    backtest_config=strategy_result.get('backtest_config', {}),
                    validation_duration_seconds=result.validation_results.duration,
                    backtest_duration_days=30,  # Default
                    total_trades=strategy_result.get('total_trades', 0)
                )
                
                # Store in database
                success = await self.storage_service.store_validation_result(validation_record)
                if success:
                    result.stored_results.append(validation_record)
                    stored_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to store validation result: {e}")
                result.errors.append(f"Storage error: {str(e)}")
        
        result.metrics.strategies_stored = stored_count
        self.logger.info(f"Stored {stored_count} validation results")
    
    async def _cleanup_old_results(self, result: WorkflowResult):
        """Clean up old validation results"""
        
        self.logger.info(f"Cleaning up results older than {self.config.days_to_keep} days")
        
        try:
            deleted_count = await self.storage_service.cleanup_old_results(self.config.days_to_keep)
            self.logger.info(f"Cleaned up {deleted_count} old results")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            result.warnings.append(f"Cleanup warning: {str(e)}")
    
    def _calculate_metrics(self, result: WorkflowResult):
        """Calculate performance metrics"""
        
        if result.metrics.duration > 0:
            result.metrics.strategies_per_second = result.metrics.strategies_generated / result.metrics.duration
        
        if result.metrics.strategies_validated > 0:
            result.metrics.average_validation_time = result.metrics.stage_timings.get('validation', 0) / result.metrics.strategies_validated
        
        if result.metrics.strategies_stored > 0:
            result.metrics.average_storage_time = result.metrics.stage_timings.get('storage', 0) / result.metrics.strategies_stored
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get the status of a workflow"""
        return self.active_workflows.get(workflow_id)
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id in self.active_workflows:
            result = self.active_workflows[workflow_id]
            result.status = WorkflowStatus.CANCELLED
            result.message = "Workflow cancelled by user"
            result.updated_at = datetime.now()
            return True
        return False
    
    async def get_workflow_history(self, limit: int = 10) -> List[WorkflowResult]:
        """Get workflow execution history"""
        # In a real implementation, this would query the database
        # For now, return active workflows
        return list(self.active_workflows.values())[:limit]
