"""
Strategy Validation API Routes

Provides REST API endpoints for strategy validation, filtering, and evaluation.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ...strategy_validation.filtering.filtering_pipeline import (
    FilteringPipeline, PipelineConfig, PipelineResult, PipelineStatus
)
from ...strategy_validation.filtering.strategy_filter import FilterConfig
from ...strategy_validation.scoring.scoring_engine import ScoringConfig
from ...strategy_validation.evaluation.strategy_evaluator import EvaluationConfig
from ...strategy_validation.storage.validation_integration import ValidationIntegration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/strategy-validation", tags=["strategy-validation"])

# In-memory storage for pipeline results (in production, use Redis or database)
pipeline_results: Dict[str, PipelineResult] = {}

# Initialize validation integration
import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
validation_integration = ValidationIntegration(DATABASE_URL)


class ValidationRequest(BaseModel):
    """Request model for strategy validation"""
    
    strategies: List[Dict[str, Any]] = Field(..., description="List of strategies to validate")
    backtest_results: Optional[List[Dict[str, Any]]] = Field(None, description="Backtest results for strategies")
    benchmark_results: Optional[Dict[str, Any]] = Field(None, description="Benchmark results for comparison")
    
    # Pipeline configuration
    enable_filtering: bool = Field(True, description="Enable strategy filtering")
    enable_scoring: bool = Field(True, description="Enable strategy scoring")
    enable_evaluation: bool = Field(True, description="Enable strategy evaluation")
    enable_ranking: bool = Field(True, description="Enable strategy ranking")
    
    # Performance settings
    enable_parallel_processing: bool = Field(True, description="Enable parallel processing")
    max_workers: int = Field(4, description="Maximum number of parallel workers")
    batch_size: int = Field(10, description="Batch size for parallel processing")
    
    # Output settings
    top_n_strategies: int = Field(10, description="Number of top strategies to return")
    min_passing_score: float = Field(70.0, description="Minimum score to pass validation")
    generate_reports: bool = Field(True, description="Generate validation reports")
    
    # Timeout settings
    timeout_seconds: float = Field(300.0, description="Pipeline timeout in seconds")


class ValidationResponse(BaseModel):
    """Response model for strategy validation"""
    
    success: bool
    message: str
    pipeline_id: str
    status: str
    duration: float
    
    # Results summary
    input_strategies: int
    filtered_strategies: int
    evaluated_strategies: int
    passing_strategies: int
    
    # Top strategies
    top_strategies: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance metrics
    strategies_per_second: float
    average_evaluation_time: float
    
    # Pipeline metadata
    stages_completed: List[str]
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status"""
    
    pipeline_id: str
    status: str
    progress: float
    current_stage: str
    duration: float
    
    # Results summary
    input_strategies: int
    filtered_strategies: int
    evaluated_strategies: int
    passing_strategies: int
    
    # Performance metrics
    strategies_per_second: float
    
    # Pipeline metadata
    stages_completed: List[str]
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ValidationConfigRequest(BaseModel):
    """Request model for validation configuration"""
    
    filter_config: Optional[Dict[str, Any]] = Field(None, description="Filter configuration")
    scoring_config: Optional[Dict[str, Any]] = Field(None, description="Scoring configuration")
    evaluation_config: Optional[Dict[str, Any]] = Field(None, description="Evaluation configuration")
    
    # Pipeline settings
    enable_parallel_processing: bool = Field(True, description="Enable parallel processing")
    max_workers: int = Field(4, description="Maximum number of parallel workers")
    batch_size: int = Field(10, description="Batch size for parallel processing")
    
    # Output settings
    top_n_strategies: int = Field(10, description="Number of top strategies to return")
    min_passing_score: float = Field(70.0, description="Minimum score to pass validation")
    generate_reports: bool = Field(True, description="Generate validation reports")
    
    # Timeout settings
    timeout_seconds: float = Field(300.0, description="Pipeline timeout in seconds")


@router.post("/validate", response_model=ValidationResponse)
async def validate_strategies(request: ValidationRequest) -> ValidationResponse:
    """Validate a list of strategies using the complete pipeline"""
    
    try:
        logger.info(f"Starting validation for {len(request.strategies)} strategies")
        
        # Create pipeline configuration
        config = PipelineConfig(
            enable_filtering=request.enable_filtering,
            enable_scoring=request.enable_scoring,
            enable_evaluation=request.enable_evaluation,
            enable_ranking=request.enable_ranking,
            enable_parallel_processing=request.enable_parallel_processing,
            max_workers=request.max_workers,
            batch_size=request.batch_size,
            top_n_strategies=request.top_n_strategies,
            min_passing_score=request.min_passing_score,
            generate_reports=request.generate_reports,
            timeout_seconds=request.timeout_seconds
        )
        
        # Create and run pipeline
        pipeline = FilteringPipeline(config)
        result = await pipeline.run_pipeline(
            strategies=request.strategies,
            backtest_results=request.backtest_results,
            benchmark_results=request.benchmark_results
        )
        
        # Store result
        pipeline_results[result.pipeline_id] = result
        
        # Convert to response
        response = ValidationResponse(
            success=result.status == PipelineStatus.COMPLETED,
            message=f"Validation completed with status: {result.status.value}",
            pipeline_id=result.pipeline_id,
            status=result.status.value,
            duration=result.duration,
            input_strategies=result.input_strategies,
            filtered_strategies=result.filtered_strategies,
            evaluated_strategies=result.evaluated_strategies,
            passing_strategies=result.passing_strategies,
            top_strategies=[r.to_dict() for r in result.top_strategies],
            strategies_per_second=result.strategies_per_second,
            average_evaluation_time=result.average_evaluation_time,
            stages_completed=[stage.value for stage in result.stages_completed],
            errors=result.errors,
            warnings=result.warnings
        )
        
        logger.info(f"Validation completed: {result.passing_strategies}/{result.input_strategies} strategies passed")
        
        return response
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/validate-async", response_model=Dict[str, str])
async def validate_strategies_async(request: ValidationRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Start validation asynchronously and return pipeline ID"""
    
    try:
        logger.info(f"Starting async validation for {len(request.strategies)} strategies")
        
        # Create pipeline configuration
        config = PipelineConfig(
            enable_filtering=request.enable_filtering,
            enable_scoring=request.enable_scoring,
            enable_evaluation=request.enable_evaluation,
            enable_ranking=request.enable_ranking,
            enable_parallel_processing=request.enable_parallel_processing,
            max_workers=request.max_workers,
            batch_size=request.batch_size,
            top_n_strategies=request.top_n_strategies,
            min_passing_score=request.min_passing_score,
            generate_reports=request.generate_reports,
            timeout_seconds=request.timeout_seconds
        )
        
        # Create pipeline
        pipeline = FilteringPipeline(config)
        
        # Generate pipeline ID
        pipeline_id = f"async_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add background task
        background_tasks.add_task(
            _run_async_validation,
            pipeline,
            request.strategies,
            request.backtest_results,
            request.benchmark_results,
            pipeline_id
        )
        
        logger.info(f"Async validation started with pipeline ID: {pipeline_id}")
        
        return {
            "success": True,
            "message": "Async validation started",
            "pipeline_id": pipeline_id,
            "status_url": f"/api/v1/strategy-validation/status/{pipeline_id}"
        }
        
    except Exception as e:
        logger.error(f"Async validation failed to start: {e}")
        raise HTTPException(status_code=500, detail=f"Async validation failed to start: {str(e)}")


async def _run_async_validation(
    pipeline: FilteringPipeline,
    strategies: List[Dict[str, Any]],
    backtest_results: Optional[List[Dict[str, Any]]],
    benchmark_results: Optional[Dict[str, Any]],
    pipeline_id: str
) -> None:
    """Run validation asynchronously in background"""
    
    try:
        logger.info(f"Running async validation for pipeline {pipeline_id}")
        
        result = await pipeline.run_pipeline(
            strategies=strategies,
            backtest_results=backtest_results,
            benchmark_results=benchmark_results,
            pipeline_id=pipeline_id
        )
        
        # Store result
        pipeline_results[pipeline_id] = result
        
        logger.info(f"Async validation completed for pipeline {pipeline_id}")
        
    except Exception as e:
        logger.error(f"Async validation failed for pipeline {pipeline_id}: {e}")
        
        # Store error result
        error_result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            errors=[str(e)]
        )
        pipeline_results[pipeline_id] = error_result


@router.get("/status/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(pipeline_id: str) -> PipelineStatusResponse:
    """Get the status of a validation pipeline"""
    
    if pipeline_id not in pipeline_results:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    
    result = pipeline_results[pipeline_id]
    
    # Calculate progress
    total_stages = 6  # initialization, filtering, scoring, evaluation, ranking, reporting
    progress = len(result.stages_completed) / total_stages * 100
    
    # Get current stage
    current_stage = "completed"
    if result.status == PipelineStatus.RUNNING:
        if not result.stages_completed:
            current_stage = "initialization"
        else:
            last_stage = result.stages_completed[-1]
            if last_stage.value == "initialization":
                current_stage = "filtering"
            elif last_stage.value == "filtering":
                current_stage = "scoring"
            elif last_stage.value == "scoring":
                current_stage = "evaluation"
            elif last_stage.value == "evaluation":
                current_stage = "ranking"
            elif last_stage.value == "ranking":
                current_stage = "reporting"
    
    response = PipelineStatusResponse(
        pipeline_id=result.pipeline_id,
        status=result.status.value,
        progress=progress,
        current_stage=current_stage,
        duration=result.duration,
        input_strategies=result.input_strategies,
        filtered_strategies=result.filtered_strategies,
        evaluated_strategies=result.evaluated_strategies,
        passing_strategies=result.passing_strategies,
        strategies_per_second=result.strategies_per_second,
        stages_completed=[stage.value for stage in result.stages_completed],
        errors=result.errors,
        warnings=result.warnings
    )
    
    return response


@router.get("/result/{pipeline_id}", response_model=ValidationResponse)
async def get_pipeline_result(pipeline_id: str) -> ValidationResponse:
    """Get the complete result of a validation pipeline"""
    
    if pipeline_id not in pipeline_results:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    
    result = pipeline_results[pipeline_id]
    
    # Convert to response
    response = ValidationResponse(
        success=result.status == PipelineStatus.COMPLETED,
        message=f"Pipeline status: {result.status.value}",
        pipeline_id=result.pipeline_id,
        status=result.status.value,
        duration=result.duration,
        input_strategies=result.input_strategies,
        filtered_strategies=result.filtered_strategies,
        evaluated_strategies=result.evaluated_strategies,
        passing_strategies=result.passing_strategies,
        top_strategies=[r.to_dict() for r in result.top_strategies],
        strategies_per_second=result.strategies_per_second,
        average_evaluation_time=result.average_evaluation_time,
        stages_completed=[stage.value for stage in result.stages_completed],
        errors=result.errors,
        warnings=result.warnings
    )
    
    return response


@router.get("/pipelines", response_model=List[Dict[str, Any]])
async def list_pipelines(
    status: Optional[str] = Query(None, description="Filter by pipeline status"),
    limit: int = Query(10, description="Maximum number of pipelines to return")
) -> List[Dict[str, Any]]:
    """List all validation pipelines"""
    
    pipelines = list(pipeline_results.values())
    
    # Filter by status if provided
    if status:
        pipelines = [p for p in pipelines if p.status.value == status]
    
    # Sort by start time (newest first)
    pipelines.sort(key=lambda x: x.start_time, reverse=True)
    
    # Limit results
    pipelines = pipelines[:limit]
    
    # Convert to response format
    response = []
    for pipeline in pipelines:
        response.append({
            "pipeline_id": pipeline.pipeline_id,
            "status": pipeline.status.value,
            "start_time": pipeline.start_time.isoformat(),
            "duration": pipeline.duration,
            "input_strategies": pipeline.input_strategies,
            "passing_strategies": pipeline.passing_strategies,
            "stages_completed": [stage.value for stage in pipeline.stages_completed],
            "has_errors": len(pipeline.errors) > 0,
            "has_warnings": len(pipeline.warnings) > 0
        })
    
    return response


@router.delete("/pipeline/{pipeline_id}")
async def delete_pipeline(pipeline_id: str) -> Dict[str, str]:
    """Delete a validation pipeline and its results"""
    
    if pipeline_id not in pipeline_results:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    
    del pipeline_results[pipeline_id]
    
    return {
        "success": True,
        "message": f"Pipeline {pipeline_id} deleted successfully"
    }


@router.get("/config/default", response_model=ValidationConfigRequest)
async def get_default_config() -> ValidationConfigRequest:
    """Get the default validation configuration"""
    
    return ValidationConfigRequest(
        enable_parallel_processing=True,
        max_workers=4,
        batch_size=10,
        top_n_strategies=10,
        min_passing_score=70.0,
        generate_reports=True,
        timeout_seconds=300.0
    )


@router.post("/config/validate", response_model=Dict[str, str])
async def validate_config(request: ValidationConfigRequest) -> Dict[str, str]:
    """Validate a configuration without running the pipeline"""
    
    try:
        # Create pipeline configuration
        config = PipelineConfig(
            enable_parallel_processing=request.enable_parallel_processing,
            max_workers=request.max_workers,
            batch_size=request.batch_size,
            top_n_strategies=request.top_n_strategies,
            min_passing_score=request.min_passing_score,
            generate_reports=request.generate_reports,
            timeout_seconds=request.timeout_seconds
        )
        
        # Validate configuration
        if config.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        if config.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if config.top_n_strategies < 1:
            raise ValueError("top_n_strategies must be at least 1")
        
        if config.min_passing_score < 0 or config.min_passing_score > 100:
            raise ValueError("min_passing_score must be between 0 and 100")
        
        if config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        
        return {
            "success": True,
            "message": "Configuration is valid"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for strategy validation service"""
    
    return {
        "status": "healthy",
        "service": "strategy-validation",
        "timestamp": datetime.now().isoformat(),
        "active_pipelines": len(pipeline_results)
    }


# Storage endpoints
@router.get("/storage/results", response_model=List[Dict[str, Any]])
async def get_validation_results(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    limit: int = Query(100, description="Maximum number of results to return"),
    offset: int = Query(0, description="Number of results to skip")
) -> List[Dict[str, Any]]:
    """Get validation results from database storage"""
    
    try:
        results = await validation_integration.get_validation_results(
            strategy_id, limit, offset
        )
        return results
        
    except Exception as e:
        logger.error(f"Failed to get validation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation results: {str(e)}")


@router.get("/storage/performance-summary", response_model=List[Dict[str, Any]])
async def get_strategy_performance_summary(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID")
) -> List[Dict[str, Any]]:
    """Get strategy performance summary from database storage"""
    
    try:
        summary = await validation_integration.get_strategy_performance_summary(strategy_id)
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")


@router.get("/storage/top-strategies", response_model=List[Dict[str, Any]])
async def get_top_strategies(
    limit: int = Query(10, description="Maximum number of strategies to return"),
    min_score: float = Query(0.0, description="Minimum score threshold")
) -> List[Dict[str, Any]]:
    """Get top performing strategies from database storage"""
    
    try:
        strategies = await validation_integration.get_top_strategies(limit, min_score)
        return strategies
        
    except Exception as e:
        logger.error(f"Failed to get top strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get top strategies: {str(e)}")


@router.get("/storage/statistics", response_model=Dict[str, Any])
async def get_validation_statistics() -> Dict[str, Any]:
    """Get validation statistics from database storage"""
    
    try:
        stats = await validation_integration.get_validation_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get validation statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get validation statistics: {str(e)}")


@router.delete("/storage/results/{strategy_id}")
async def delete_validation_results(strategy_id: str) -> Dict[str, str]:
    """Delete validation results for a specific strategy"""
    
    try:
        success = await validation_integration.delete_validation_results(strategy_id)
        
        if success:
            return {
                "success": True,
                "message": f"Validation results for strategy {strategy_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete validation results")
            
    except Exception as e:
        logger.error(f"Failed to delete validation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete validation results: {str(e)}")


@router.post("/storage/cleanup")
async def cleanup_old_results(
    days_to_keep: int = Query(30, description="Number of days to keep results")
) -> Dict[str, Any]:
    """Clean up old validation results"""
    
    try:
        deleted_count = await validation_integration.cleanup_old_results(days_to_keep)
        
        return {
            "success": True,
            "message": f"Cleaned up {deleted_count} old validation results",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old results: {str(e)}")
