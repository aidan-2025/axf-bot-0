"""
Filtering Pipeline

Orchestrates the complete strategy validation and filtering pipeline,
combining filtering, scoring, and evaluation stages.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .strategy_filter import StrategyFilter, FilterConfig
from .scoring_engine import ScoringEngine, ScoringConfig
from ..evaluation.strategy_evaluator import StrategyEvaluator, EvaluationConfig, EvaluationResult

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Stages of the filtering pipeline"""
    INITIALIZATION = "initialization"
    FILTERING = "filtering"
    SCORING = "scoring"
    EVALUATION = "evaluation"
    RANKING = "ranking"
    REPORTING = "reporting"
    COMPLETED = "completed"


class PipelineStatus(Enum):
    """Status of the pipeline execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    
    # Input/Output counts
    input_strategies: int = 0
    filtered_strategies: int = 0
    evaluated_strategies: int = 0
    passing_strategies: int = 0
    
    # Results
    evaluation_results: List[EvaluationResult] = field(default_factory=list)
    top_strategies: List[EvaluationResult] = field(default_factory=list)
    
    # Pipeline metadata
    stages_completed: List[PipelineStage] = field(default_factory=list)
    stage_durations: Dict[PipelineStage, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_evaluation_time: float = 0.0
    average_evaluation_time: float = 0.0
    strategies_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'pipeline_id': self.pipeline_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'input_strategies': self.input_strategies,
            'filtered_strategies': self.filtered_strategies,
            'evaluated_strategies': self.evaluated_strategies,
            'passing_strategies': self.passing_strategies,
            'evaluation_results': [result.to_dict() for result in self.evaluation_results],
            'top_strategies': [result.to_dict() for result in self.top_strategies],
            'stages_completed': [stage.value for stage in self.stages_completed],
            'stage_durations': {stage.value: duration for stage, duration in self.stage_durations.items()},
            'errors': self.errors,
            'warnings': self.warnings,
            'total_evaluation_time': self.total_evaluation_time,
            'average_evaluation_time': self.average_evaluation_time,
            'strategies_per_second': self.strategies_per_second
        }


@dataclass
class PipelineConfig:
    """Configuration for the filtering pipeline"""
    
    # Component configurations
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    scoring_config: ScoringConfig = field(default_factory=ScoringConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Pipeline settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 10
    
    # Filtering settings
    enable_filtering: bool = True
    enable_scoring: bool = True
    enable_evaluation: bool = True
    enable_ranking: bool = True
    
    # Output settings
    top_n_strategies: int = 10
    min_passing_score: float = 70.0
    generate_reports: bool = True
    
    # Performance settings
    timeout_seconds: float = 300.0  # 5 minutes
    enable_progress_tracking: bool = True


class FilteringPipeline:
    """Complete strategy filtering and evaluation pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.filter = StrategyFilter(config.filter_config)
        self.scorer = ScoringEngine(config.scoring_config)
        self.evaluator = StrategyEvaluator(config.evaluation_config)
    
    async def run_pipeline(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: Optional[List[Dict[str, Any]]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None,
        pipeline_id: Optional[str] = None
    ) -> PipelineResult:
        """Run the complete filtering pipeline"""
        
        if pipeline_id is None:
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting pipeline {pipeline_id} with {len(strategies)} strategies")
        
        # Initialize result
        result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(),
            input_strategies=len(strategies)
        )
        
        try:
            # Stage 1: Initialization
            await self._run_stage(result, PipelineStage.INITIALIZATION, self._initialize_pipeline, strategies)
            
            # Stage 2: Filtering
            if self.config.enable_filtering:
                await self._run_stage(result, PipelineStage.FILTERING, self._run_filtering_stage, strategies, backtest_results)
            
            # Stage 3: Scoring
            if self.config.enable_scoring and backtest_results:
                await self._run_stage(result, PipelineStage.SCORING, self._run_scoring_stage, strategies, backtest_results, benchmark_results)
            
            # Stage 4: Evaluation
            if self.config.enable_evaluation:
                await self._run_stage(result, PipelineStage.EVALUATION, self._run_evaluation_stage, strategies, backtest_results, benchmark_results)
            
            # Stage 5: Ranking
            if self.config.enable_ranking:
                await self._run_stage(result, PipelineStage.RANKING, self._run_ranking_stage, result.evaluation_results)
            
            # Stage 6: Reporting
            if self.config.generate_reports:
                await self._run_stage(result, PipelineStage.REPORTING, self._run_reporting_stage, result)
            
            # Complete pipeline
            result.status = PipelineStatus.COMPLETED
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.stages_completed.append(PipelineStage.COMPLETED)
            
            # Calculate performance metrics
            self._calculate_performance_metrics(result)
            
            self.logger.info(f"Pipeline {pipeline_id} completed successfully in {result.duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Pipeline {pipeline_id} failed: {e}")
            result.status = PipelineStatus.FAILED
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.errors.append(str(e))
        
        return result
    
    def run_pipeline_sync(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: Optional[List[Dict[str, Any]]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None,
        pipeline_id: Optional[str] = None
    ) -> PipelineResult:
        """Run the pipeline synchronously"""
        
        return asyncio.run(self.run_pipeline(strategies, backtest_results, benchmark_results, pipeline_id))
    
    async def _run_stage(self, result: PipelineResult, stage: PipelineStage, 
                        stage_func, *args, **kwargs) -> None:
        """Run a pipeline stage with timing and error handling"""
        
        stage_start = datetime.now()
        self.logger.info(f"Starting stage: {stage.value}")
        
        try:
            await stage_func(*args, **kwargs)
            result.stages_completed.append(stage)
            self.logger.info(f"Completed stage: {stage.value}")
        except Exception as e:
            self.logger.error(f"Stage {stage.value} failed: {e}")
            result.errors.append(f"Stage {stage.value}: {str(e)}")
            raise
        
        finally:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            result.stage_durations[stage] = stage_duration
    
    async def _initialize_pipeline(self, strategies: List[Dict[str, Any]]) -> None:
        """Initialize the pipeline"""
        
        self.logger.info(f"Initializing pipeline with {len(strategies)} strategies")
        
        # Validate input
        if not strategies:
            raise ValueError("No strategies provided")
        
        # Check for required fields
        for i, strategy in enumerate(strategies):
            if 'strategy_id' not in strategy:
                strategy['strategy_id'] = f'strategy_{i}'
        
        self.logger.info("Pipeline initialization completed")
    
    async def _run_filtering_stage(self, strategies: List[Dict[str, Any]], 
                                 backtest_results: Optional[List[Dict[str, Any]]]) -> None:
        """Run the filtering stage"""
        
        self.logger.info("Running filtering stage")
        
        # Filter strategies
        if self.config.enable_parallel_processing and len(strategies) > self.config.batch_size:
            # Parallel filtering
            filtered_strategies = await self._parallel_filter_strategies(strategies, backtest_results)
        else:
            # Sequential filtering
            filtered_strategies = self.filter.get_passing_strategies(strategies, backtest_results)
        
        self.logger.info(f"Filtering completed: {len(filtered_strategies)}/{len(strategies)} strategies passed")
    
    async def _run_scoring_stage(self, strategies: List[Dict[str, Any]], 
                               backtest_results: List[Dict[str, Any]], 
                               benchmark_results: Optional[Dict[str, Any]]) -> None:
        """Run the scoring stage"""
        
        self.logger.info("Running scoring stage")
        
        # Score strategies
        if self.config.enable_parallel_processing and len(strategies) > self.config.batch_size:
            # Parallel scoring
            scoring_results = await self._parallel_score_strategies(strategies, backtest_results, benchmark_results)
        else:
            # Sequential scoring
            scoring_results = self.scorer.score_strategies(strategies, backtest_results, benchmark_results)
        
        self.logger.info(f"Scoring completed: {len(scoring_results)} strategies scored")
    
    async def _run_evaluation_stage(self, strategies: List[Dict[str, Any]], 
                                  backtest_results: Optional[List[Dict[str, Any]]], 
                                  benchmark_results: Optional[Dict[str, Any]]) -> None:
        """Run the evaluation stage"""
        
        self.logger.info("Running evaluation stage")
        
        # Evaluate strategies
        if self.config.enable_parallel_processing and len(strategies) > self.config.batch_size:
            # Parallel evaluation
            evaluation_results = await self._parallel_evaluate_strategies(strategies, backtest_results, benchmark_results)
        else:
            # Sequential evaluation
            evaluation_results = self.evaluator.evaluate_strategies(strategies, backtest_results, benchmark_results)
        
        self.logger.info(f"Evaluation completed: {len(evaluation_results)} strategies evaluated")
    
    async def _run_ranking_stage(self, evaluation_results: List[EvaluationResult]) -> None:
        """Run the ranking stage"""
        
        self.logger.info("Running ranking stage")
        
        # Sort by normalized score
        evaluation_results.sort(key=lambda x: x.evaluation_metrics.normalized_score, reverse=True)
        
        # Get top strategies
        top_strategies = evaluation_results[:self.config.top_n_strategies]
        
        # Count passing strategies
        passing_strategies = [r for r in evaluation_results if r.evaluation_metrics.normalized_score >= self.config.min_passing_score]
        
        self.logger.info(f"Ranking completed: {len(passing_strategies)} strategies passed, {len(top_strategies)} top strategies selected")
    
    async def _run_reporting_stage(self, result: PipelineResult) -> None:
        """Run the reporting stage"""
        
        self.logger.info("Running reporting stage")
        
        # Generate summary statistics
        if result.evaluation_results:
            scores = [r.evaluation_metrics.normalized_score for r in result.evaluation_results]
            result.warnings.append(f"Score distribution: min={min(scores):.1f}, max={max(scores):.1f}, avg={np.mean(scores):.1f}")
        
        self.logger.info("Reporting stage completed")
    
    async def _parallel_filter_strategies(self, strategies: List[Dict[str, Any]], 
                                        backtest_results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Filter strategies in parallel"""
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Split strategies into batches
            batches = [strategies[i:i + self.config.batch_size] for i in range(0, len(strategies), self.config.batch_size)]
            bt_batches = [backtest_results[i:i + self.config.batch_size] for i in range(0, len(backtest_results), self.config.batch_size)] if backtest_results else [None] * len(batches)
            
            # Submit batches
            futures = []
            for i, batch in enumerate(batches):
                bt_batch = bt_batches[i] if i < len(bt_batches) else None
                future = executor.submit(self.filter.get_passing_strategies, batch, bt_batch)
                futures.append(future)
            
            # Collect results
            filtered_strategies = []
            for future in as_completed(futures):
                try:
                    batch_result = future.result()
                    filtered_strategies.extend(batch_result)
                except Exception as e:
                    self.logger.warning(f"Batch filtering failed: {e}")
        
        return filtered_strategies
    
    async def _parallel_score_strategies(self, strategies: List[Dict[str, Any]], 
                                       backtest_results: List[Dict[str, Any]], 
                                       benchmark_results: Optional[Dict[str, Any]]) -> List[Any]:
        """Score strategies in parallel"""
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Split strategies into batches
            batches = [strategies[i:i + self.config.batch_size] for i in range(0, len(strategies), self.config.batch_size)]
            bt_batches = [backtest_results[i:i + self.config.batch_size] for i in range(0, len(backtest_results), self.config.batch_size)]
            
            # Submit batches
            futures = []
            for i, batch in enumerate(batches):
                bt_batch = bt_batches[i] if i < len(bt_batches) else []
                future = executor.submit(self.scorer.score_strategies, batch, bt_batch, benchmark_results)
                futures.append(future)
            
            # Collect results
            scoring_results = []
            for future in as_completed(futures):
                try:
                    batch_result = future.result()
                    scoring_results.extend(batch_result)
                except Exception as e:
                    self.logger.warning(f"Batch scoring failed: {e}")
        
        return scoring_results
    
    async def _parallel_evaluate_strategies(self, strategies: List[Dict[str, Any]], 
                                          backtest_results: Optional[List[Dict[str, Any]]], 
                                          benchmark_results: Optional[Dict[str, Any]]) -> List[EvaluationResult]:
        """Evaluate strategies in parallel"""
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Split strategies into batches
            batches = [strategies[i:i + self.config.batch_size] for i in range(0, len(strategies), self.config.batch_size)]
            bt_batches = [backtest_results[i:i + self.config.batch_size] for i in range(0, len(backtest_results), self.config.batch_size)] if backtest_results else [None] * len(batches)
            
            # Submit batches
            futures = []
            for i, batch in enumerate(batches):
                bt_batch = bt_batches[i] if i < len(bt_batches) else None
                future = executor.submit(self.evaluator.evaluate_strategies, batch, bt_batch, benchmark_results)
                futures.append(future)
            
            # Collect results
            evaluation_results = []
            for future in as_completed(futures):
                try:
                    batch_result = future.result()
                    evaluation_results.extend(batch_result)
                except Exception as e:
                    self.logger.warning(f"Batch evaluation failed: {e}")
        
        return evaluation_results
    
    def _calculate_performance_metrics(self, result: PipelineResult) -> None:
        """Calculate performance metrics for the pipeline result"""
        
        if result.evaluation_results:
            result.evaluated_strategies = len(result.evaluation_results)
            result.passing_strategies = len([r for r in result.evaluation_results if r.evaluation_metrics.normalized_score >= self.config.min_passing_score])
            result.top_strategies = result.evaluation_results[:self.config.top_n_strategies]
            
            # Calculate timing metrics
            total_eval_time = sum(r.evaluation_metrics.evaluation_duration for r in result.evaluation_results)
            result.total_evaluation_time = total_eval_time
            result.average_evaluation_time = total_eval_time / len(result.evaluation_results) if result.evaluation_results else 0.0
            
            if result.duration > 0:
                result.strategies_per_second = len(result.evaluation_results) / result.duration
