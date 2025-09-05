"""
Workflow API Routes

Provides REST API endpoints for the validation workflow orchestrator.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ...strategy_validation.workflow.validation_workflow import (
    ValidationWorkflow, WorkflowConfig, WorkflowResult, WorkflowStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workflow", tags=["workflow"])

# Global workflow instance
workflow_instance: Optional[ValidationWorkflow] = None


def get_workflow() -> ValidationWorkflow:
    """Get or create workflow instance"""
    global workflow_instance
    if workflow_instance is None:
        import os
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/axf_bot_db")
        workflow_instance = ValidationWorkflow(
            config=WorkflowConfig(),
            database_url=database_url
        )
    return workflow_instance


class WorkflowRequest(BaseModel):
    """Request model for starting a workflow"""
    
    # Strategy generation settings
    strategy_count: int = Field(10, description="Number of strategies to generate")
    strategy_types: List[str] = Field(
        default=["trend_following", "mean_reversion", "breakout"],
        description="Types of strategies to generate"
    )
    currency_pairs: List[str] = Field(
        default=["EURUSD", "GBPUSD", "USDJPY"],
        description="Currency pairs to use"
    )
    timeframes: List[str] = Field(
        default=["H1", "H4", "D1"],
        description="Timeframes to use"
    )
    
    # Validation settings
    enable_parallel_processing: bool = Field(True, description="Enable parallel processing")
    max_workers: int = Field(4, description="Maximum number of parallel workers")
    batch_size: int = Field(5, description="Batch size for parallel processing")
    timeout_seconds: float = Field(300.0, description="Pipeline timeout in seconds")
    
    # Storage settings
    store_results: bool = Field(True, description="Store results in database")
    cleanup_old_results: bool = Field(True, description="Clean up old results")
    days_to_keep: int = Field(30, description="Days to keep old results")
    
    # Monitoring settings
    enable_logging: bool = Field(True, description="Enable detailed logging")
    log_level: str = Field("INFO", description="Log level")
    enable_metrics: bool = Field(True, description="Enable metrics collection")


class WorkflowResponse(BaseModel):
    """Response model for workflow operations"""
    
    success: bool
    message: str
    workflow_id: str
    status: str
    
    # Optional result data
    result: Optional[Dict[str, Any]] = None


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    
    workflow_id: str
    status: str
    message: str
    
    # Progress info
    current_stage: Optional[str] = None
    progress_percentage: float = 0.0
    
    # Metrics
    strategies_generated: int = 0
    strategies_validated: int = 0
    strategies_passed: int = 0
    strategies_stored: int = 0
    
    # Timing
    start_time: Optional[str] = None
    duration: float = 0.0
    
    # Errors
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


@router.post("/start", response_model=WorkflowResponse)
async def start_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks
) -> WorkflowResponse:
    """Start a new validation workflow"""
    
    try:
        logger.info("Starting new validation workflow")
        
        # Create workflow configuration
        config = WorkflowConfig(
            strategy_count=request.strategy_count,
            strategy_types=request.strategy_types,
            currency_pairs=request.currency_pairs,
            timeframes=request.timeframes,
            enable_parallel_processing=request.enable_parallel_processing,
            max_workers=request.max_workers,
            batch_size=request.batch_size,
            timeout_seconds=request.timeout_seconds,
            store_results=request.store_results,
            cleanup_old_results=request.cleanup_old_results,
            days_to_keep=request.days_to_keep,
            enable_logging=request.enable_logging,
            log_level=request.log_level,
            enable_metrics=request.enable_metrics
        )
        
        # Create workflow instance
        workflow = ValidationWorkflow(config, get_workflow().database_url)
        
        # Generate workflow ID
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start workflow in background
        background_tasks.add_task(workflow.run_workflow, workflow_id)
        
        logger.info(f"Workflow {workflow_id} started in background")
        
        return WorkflowResponse(
            success=True,
            message="Workflow started successfully",
            workflow_id=workflow_id,
            status="running"
        )
        
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


@router.post("/start-sync", response_model=WorkflowResponse)
async def start_workflow_sync(request: WorkflowRequest) -> WorkflowResponse:
    """Start a validation workflow and wait for completion"""
    
    try:
        logger.info("Starting synchronous validation workflow")
        
        # Create workflow configuration
        config = WorkflowConfig(
            strategy_count=request.strategy_count,
            strategy_types=request.strategy_types,
            currency_pairs=request.currency_pairs,
            timeframes=request.timeframes,
            enable_parallel_processing=request.enable_parallel_processing,
            max_workers=request.max_workers,
            batch_size=request.batch_size,
            timeout_seconds=request.timeout_seconds,
            store_results=request.store_results,
            cleanup_old_results=request.cleanup_old_results,
            days_to_keep=request.days_to_keep,
            enable_logging=request.enable_logging,
            log_level=request.log_level,
            enable_metrics=request.enable_metrics
        )
        
        # Create workflow instance
        workflow = ValidationWorkflow(config, get_workflow().database_url)
        
        # Run workflow synchronously
        result = await workflow.run_workflow()
        
        return WorkflowResponse(
            success=result.status == WorkflowStatus.COMPLETED,
            message=result.message,
            workflow_id=result.workflow_id,
            status=result.status.value,
            result=result.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Failed to run synchronous workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run workflow: {str(e)}")


@router.get("/status/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str) -> WorkflowStatusResponse:
    """Get the status of a workflow"""
    
    try:
        workflow = get_workflow()
        result = await workflow.get_workflow_status(workflow_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        # Calculate progress
        total_stages = 5  # initialization, generation, validation, storage, completion
        completed_stages = len(result.metrics.stage_timings)
        progress_percentage = (completed_stages / total_stages) * 100
        
        # Get current stage
        current_stage = None
        if result.status == WorkflowStatus.RUNNING:
            if not result.metrics.stage_timings:
                current_stage = "initialization"
            else:
                # Find the last completed stage
                stage_order = ["initialization", "strategy_generation", "validation", "storage", "completion"]
                for stage in stage_order:
                    if stage in result.metrics.stage_timings:
                        current_stage = stage
                    else:
                        break
        
        return WorkflowStatusResponse(
            workflow_id=result.workflow_id,
            status=result.status.value,
            message=result.message,
            current_stage=current_stage,
            progress_percentage=progress_percentage,
            strategies_generated=result.metrics.strategies_generated,
            strategies_validated=result.metrics.strategies_validated,
            strategies_passed=result.metrics.strategies_passed,
            strategies_stored=result.metrics.strategies_stored,
            start_time=result.metrics.start_time.isoformat() if result.metrics.start_time else None,
            duration=result.metrics.duration,
            errors=result.errors,
            warnings=result.warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.get("/result/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow_result(workflow_id: str) -> WorkflowResponse:
    """Get the complete result of a workflow"""
    
    try:
        workflow = get_workflow()
        result = await workflow.get_workflow_status(workflow_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return WorkflowResponse(
            success=result.status == WorkflowStatus.COMPLETED,
            message=result.message,
            workflow_id=result.workflow_id,
            status=result.status.value,
            result=result.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow result: {str(e)}")


@router.post("/cancel/{workflow_id}", response_model=WorkflowResponse)
async def cancel_workflow(workflow_id: str) -> WorkflowResponse:
    """Cancel a running workflow"""
    
    try:
        workflow = get_workflow()
        success = await workflow.cancel_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found or not running")
        
        return WorkflowResponse(
            success=True,
            message=f"Workflow {workflow_id} cancelled successfully",
            workflow_id=workflow_id,
            status="cancelled"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {str(e)}")


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_workflow_history(
    limit: int = Query(10, description="Maximum number of workflows to return")
) -> List[Dict[str, Any]]:
    """Get workflow execution history"""
    
    try:
        workflow = get_workflow()
        history = await workflow.get_workflow_history(limit)
        
        return [result.to_dict() for result in history]
        
    except Exception as e:
        logger.error(f"Failed to get workflow history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow history: {str(e)}")


@router.get("/active", response_model=List[Dict[str, Any]])
async def get_active_workflows() -> List[Dict[str, Any]]:
    """Get currently active workflows"""
    
    try:
        workflow = get_workflow()
        active_workflows = list(workflow.active_workflows.values())
        
        return [result.to_dict() for result in active_workflows]
        
    except Exception as e:
        logger.error(f"Failed to get active workflows: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active workflows: {str(e)}")


@router.get("/config/default", response_model=WorkflowRequest)
async def get_default_config() -> WorkflowRequest:
    """Get the default workflow configuration"""
    
    return WorkflowRequest()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for workflow service"""
    
    try:
        workflow = get_workflow()
        active_count = len(workflow.active_workflows)
        
        return {
            "status": "healthy",
            "service": "validation-workflow",
            "timestamp": datetime.now().isoformat(),
            "active_workflows": active_count
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "validation-workflow",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

