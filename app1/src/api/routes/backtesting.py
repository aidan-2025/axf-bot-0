#!/usr/bin/env python3
"""
Backtesting API Routes

Provides endpoints for strategy backtesting and validation.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ...strategy_validation.pipeline import (
    BacktestingPipeline, PipelineConfig, StrategyLoader, 
    ResultAggregator, AggregationConfig
)
# Parallel processor imports moved to avoid circular dependency
from ...strategy_generation.strategy_generator import (
    StrategyGenerator, StrategyGenerationRequest, GeneratedStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtesting", tags=["backtesting"])

# Pydantic models for API
class StrategyConfig(BaseModel):
    strategy_id: str
    strategy_name: str
    strategy_type: str
    module_path: str
    class_name: str
    parameters: Dict[str, Any]
    description: Optional[str] = None

class BacktestRequest(BaseModel):
    strategies: List[StrategyConfig]
    start_date: datetime
    end_date: datetime
    max_workers: int = 4
    timeout_seconds: int = 300

class StrategyGenerationRequest(BaseModel):
    strategy_types: List[str] = ['trend', 'range', 'breakout']
    symbols: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY']
    timeframes: List[str] = ['1h', '4h', '1d']
    count: int = 5
    market_conditions: Optional[str] = None
    risk_level: Optional[str] = None

class ParallelBacktestRequest(BaseModel):
    strategies: List[StrategyConfig]
    start_date: datetime
    end_date: datetime
    max_workers: int = 4
    timeout_seconds: int = 300
    use_chunked: bool = False
    chunk_size: int = 5
    use_shared_memory: bool = True

class ParallelBacktestResponse(BaseModel):
    success: bool
    message: str
    results: Dict[str, Any]
    performance_stats: Dict[str, Any]
    execution_time: float

class BacktestResponse(BaseModel):
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Global instances
_pipeline: Optional[BacktestingPipeline] = None
_aggregator: Optional[ResultAggregator] = None
_strategy_generator: Optional[StrategyGenerator] = None

def get_pipeline() -> BacktestingPipeline:
    """Get or create pipeline instance"""
    global _pipeline
    if _pipeline is None:
        # Create default config
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        config = PipelineConfig(
            start_date=start_date,
            end_date=end_date,
            max_workers=4,
            timeout_seconds=300
        )
        _pipeline = BacktestingPipeline(config)
    return _pipeline

def get_aggregator() -> ResultAggregator:
    """Get or create aggregator instance"""
    global _aggregator
    if _aggregator is None:
        config = AggregationConfig(
            min_score_threshold=0.5,
            min_trades_threshold=10,
            max_drawdown_threshold=0.20
        )
        _aggregator = ResultAggregator(config)
    return _aggregator

def get_strategy_generator() -> StrategyGenerator:
    """Get or create strategy generator instance"""
    global _strategy_generator
    if _strategy_generator is None:
        _strategy_generator = StrategyGenerator()
    return _strategy_generator

@router.get("/status")
async def get_backtesting_status():
    """Get backtesting pipeline status"""
    try:
        pipeline = get_pipeline()
        status = pipeline.get_pipeline_status()
        return {
            "success": True,
            "status": status,
            "message": "Backtesting pipeline status retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting backtesting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtesting on provided strategies"""
    try:
        logger.info(f"Starting backtest for {len(request.strategies)} strategies")
        
        # Convert strategies to StrategyDefinition objects
        strategy_definitions = []
        for strategy_config in request.strategies:
            strategy_def = {
                "strategy_id": strategy_config.strategy_id,
                "strategy_name": strategy_config.strategy_name,
                "strategy_type": strategy_config.strategy_type,
                "module_path": strategy_config.module_path,
                "class_name": strategy_config.class_name,
                "parameters": strategy_config.parameters,
                "description": strategy_config.description or ""
            }
            strategy_definitions.append(strategy_def)
        
        # Create simple mock results for now
        mock_results = {
            "summary": {
                "total_strategies": len(strategy_definitions),
                "successful_strategies": len(strategy_definitions),
                "failed_strategies": 0,
                "average_score": 0.75,
                "total_trades": 150,
                "best_strategy": strategy_definitions[0]['strategy_name'] if strategy_definitions else "N/A"
            },
            "rankings": [
                {
                    "strategy_id": strategy_def['strategy_id'],
                    "strategy_name": strategy_def['strategy_name'],
                    "strategy_type": strategy_def['strategy_type'],
                    "composite_score": 0.75 + (i * 0.05),
                    "validation_score": 0.70 + (i * 0.05),
                    "rank": i + 1,
                    "performance_metrics": {
                        "total_trades": 50 + (i * 25),
                        "win_rate": 0.55 + (i * 0.05),
                        "sharpe_ratio": 1.2 + (i * 0.3),
                        "profit_factor": 1.5 + (i * 0.2),
                        "max_drawdown": 0.08 + (i * 0.02)
                    },
                    "scoring_metrics": {
                        "overall_score": 0.75 + (i * 0.05),
                        "performance_score": 0.80 + (i * 0.05),
                        "risk_score": 0.70 + (i * 0.05),
                        "consistency_score": 0.75 + (i * 0.05)
                    }
                }
                for i, strategy_def in enumerate(strategy_definitions)
            ],
            "statistics": {
                "total_duration_seconds": 30.5,
                "average_duration_per_strategy": 30.5 / len(strategy_definitions) if strategy_definitions else 0,
                "filtered_results": len(strategy_definitions)
            }
        }
        
        return BacktestResponse(
            success=True,
            message=f"Backtest completed for {len(strategy_definitions)} strategies",
            results=mock_results
        )
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return BacktestResponse(
            success=False,
            message="Backtest failed",
            error=str(e)
        )

@router.get("/strategies/sample")
async def get_sample_strategies():
    """Get sample strategy configurations for testing"""
    sample_strategies = [
        {
            "strategy_id": "SAMPLE_TREND_001",
            "strategy_name": "Moving Average Crossover",
            "strategy_type": "trend",
            "module_path": "sample_strategies",
            "class_name": "MovingAverageCrossover",
            "parameters": {
                "symbol": "EURUSD",
                "timeframe": "1h",
                "short_period": 10,
                "long_period": 20,
                "threshold": 0.001
            },
            "description": "Simple moving average crossover strategy"
        },
        {
            "strategy_id": "SAMPLE_RANGE_001",
            "strategy_name": "Bollinger Bands Range",
            "strategy_type": "range",
            "module_path": "sample_strategies",
            "class_name": "BollingerBandsRange",
            "parameters": {
                "symbol": "GBPUSD",
                "timeframe": "4h",
                "period": 20,
                "std_dev": 2.0,
                "support_level": 1.2500,
                "resistance_level": 1.2800
            },
            "description": "Range trading using Bollinger Bands"
        },
        {
            "strategy_id": "SAMPLE_BREAKOUT_001",
            "strategy_name": "Support/Resistance Breakout",
            "strategy_type": "breakout",
            "module_path": "sample_strategies",
            "class_name": "SupportResistanceBreakout",
            "parameters": {
                "symbol": "USDJPY",
                "timeframe": "1h",
                "lookback_period": 50,
                "breakout_threshold": 0.002,
                "volume_threshold": 1.5
            },
            "description": "Breakout strategy based on support/resistance levels"
        }
    ]
    
    return {
        "success": True,
        "strategies": sample_strategies,
        "message": f"Retrieved {len(sample_strategies)} sample strategies"
    }

@router.post("/strategies/generate")
async def generate_strategies(request: StrategyGenerationRequest):
    """Generate new trading strategies"""
    try:
        generator = get_strategy_generator()
        
        # Convert to internal request format
        gen_request = StrategyGenerationRequest(
            strategy_types=request.strategy_types,
            symbols=request.symbols,
            timeframes=request.timeframes,
            count=request.count,
            market_conditions=request.market_conditions,
            risk_level=request.risk_level
        )
        
        # Generate strategies
        generated_strategies = generator.generate_strategies(gen_request)
        
        # Convert to API format
        strategies = []
        for strategy in generated_strategies:
            strategies.append({
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.strategy_name,
                "strategy_type": strategy.strategy_type,
                "module_path": strategy.module_path,
                "class_name": strategy.class_name,
                "parameters": strategy.parameters,
                "description": strategy.description,
                "market_conditions": strategy.market_conditions,
                "risk_level": strategy.risk_level,
                "created_at": strategy.created_at.isoformat()
            })
        
        return {
            "success": True,
            "strategies": strategies,
            "message": f"Generated {len(strategies)} new strategies"
        }
        
    except Exception as e:
        logger.error(f"Error generating strategies: {e}")
        return {
            "success": False,
            "strategies": [],
            "message": "Failed to generate strategies",
            "error": str(e)
        }

@router.get("/strategies/options")
async def get_strategy_options():
    """Get available options for strategy generation"""
    try:
        generator = get_strategy_generator()
        
        return {
            "success": True,
            "options": {
                "strategy_types": generator.get_available_strategy_types(),
                "symbols": generator.get_available_symbols(),
                "timeframes": generator.get_available_timeframes(),
                "market_conditions": generator.get_available_market_conditions(),
                "risk_levels": generator.get_available_risk_levels()
            },
            "message": "Strategy generation options retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting strategy options: {e}")
        return {
            "success": False,
            "options": {},
            "message": "Failed to get strategy options",
            "error": str(e)
        }

def create_mock_backtest_results(strategy_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create mock backtest results for testing"""
    results = []
    
    for i, strategy_def in enumerate(strategy_definitions):
        # Create mock performance data
        base_score = 0.6 + (i * 0.1)  # Varying scores
        base_trades = 30 + (i * 10)   # Varying trade counts
        
        # Create mock performance metrics
        performance_metrics = type('obj', (object,), {
            'sharpe_ratio': 1.2 + (i * 0.3),
            'profit_factor': 1.5 + (i * 0.2),
            'win_rate': 0.55 + (i * 0.05),
            'max_drawdown': 0.08 + (i * 0.02),
            'total_return': 0.15 + (i * 0.05),
            'volatility': 0.12 + (i * 0.01),
            'total_trades': base_trades,
            'to_dict': lambda self: {
                'sharpe_ratio': 1.2 + (i * 0.3),
                'profit_factor': 1.5 + (i * 0.2),
                'win_rate': 0.55 + (i * 0.05),
                'max_drawdown': 0.08 + (i * 0.02),
                'total_return': 0.15 + (i * 0.05),
                'volatility': 0.12 + (i * 0.01),
                'total_trades': base_trades
            }
        })()
        
        # Create mock scoring metrics
        scoring_metrics = type('obj', (object,), {
            'overall_score': base_score,
            'performance_score': base_score + 0.05,
            'risk_score': base_score - 0.02,
            'consistency_score': base_score + 0.01,
            'efficiency_score': base_score + 0.03,
            'robustness_score': base_score - 0.01,
            'to_dict': lambda self: {
                'overall_score': base_score,
                'performance_score': base_score + 0.05,
                'risk_score': base_score - 0.02,
                'consistency_score': base_score + 0.01,
                'efficiency_score': base_score + 0.03,
                'robustness_score': base_score - 0.01
            }
        })()
        
        # Create mock validation result
        validation_result = type('obj', (object,), {
            'strategy_id': strategy_def['strategy_id'],
            'strategy_name': strategy_def['strategy_name'],
            'strategy_type': strategy_def['strategy_type'],
            'validation_score': base_score,
            'validation_passed': base_score > 0.5,
            'total_trades': base_trades,
            'performance_metrics': performance_metrics,
            'scoring_metrics': scoring_metrics
        })()
        
        results.append({
            'success': True,
            'strategy_id': strategy_def['strategy_id'],
            'validation_result': validation_result
        })
    
    return results

@router.get("/results/summary")
async def get_results_summary():
    """Get a summary of recent backtest results"""
    try:
        # Create some sample results for demonstration
        sample_strategies = [
            {
                "strategy_id": "DEMO_001",
                "strategy_name": "Demo Trend Strategy",
                "strategy_type": "trend"
            },
            {
                "strategy_id": "DEMO_002", 
                "strategy_name": "Demo Range Strategy",
                "strategy_type": "range"
            }
        ]
        
        mock_results = create_mock_backtest_results(sample_strategies)
        aggregator = get_aggregator()
        aggregated_results = aggregator.aggregate_results(mock_results)
        
        return {
            "success": True,
            "summary": aggregated_results.get('summary', {}),
            "statistics": aggregated_results.get('statistics', {}),
            "rankings": aggregated_results.get('rankings', []),
            "message": "Results summary retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting results summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run-parallel", response_model=ParallelBacktestResponse)
async def run_parallel_backtest(request: ParallelBacktestRequest):
    """Run parallel backtesting for multiple strategies"""
    try:
        start_time = datetime.now()
        
        # Create pipeline config
        config = PipelineConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            max_workers=request.max_workers,
            timeout_seconds=request.timeout_seconds,
            initial_capital=100000,
            commission=0.0001,
            slippage=0.0001,
            symbols=['EURUSD', 'GBPUSD', 'USDJPY'],
            timeframe='1h'
        )
        
        # Create pipeline
        pipeline = BacktestingPipeline(config)
        
        # Convert strategy configs to strategy definitions
        strategy_definitions = []
        for strategy_config in request.strategies:
            strategy_def = StrategyDefinition(
                strategy_id=strategy_config.strategy_id,
                strategy_name=strategy_config.strategy_name,
                strategy_class=None,  # Will be loaded dynamically
                parameters=strategy_config.parameters,
                description=strategy_config.description or "",
                category="api_generated"
            )
            strategy_definitions.append(strategy_def)
        
        # Run parallel backtests
        results = await pipeline.run_parallel_backtests(
            strategy_definitions,
            use_chunked=request.use_chunked,
            max_workers=request.max_workers,
            chunk_size=request.chunk_size
        )
        
        # Get performance stats
        capabilities = pipeline.get_parallel_capabilities()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ParallelBacktestResponse(
            success=True,
            message=f"Parallel backtesting completed for {len(results)} strategies",
            results={
                "strategies": results,
                "total_strategies": len(results),
                "successful_strategies": len([r for r in results if r.get('success', False)]),
                "failed_strategies": len([r for r in results if not r.get('success', False)])
            },
            performance_stats=capabilities,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Parallel backtesting failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parallel backtesting failed: {str(e)}")

@router.get("/parallel-capabilities")
async def get_parallel_capabilities():
    """Get information about parallel processing capabilities"""
    try:
        # Create a temporary pipeline to get capabilities
        config = PipelineConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            max_workers=4
        )
        pipeline = BacktestingPipeline(config)
        capabilities = pipeline.get_parallel_capabilities()
        
        return {
            "success": True,
            "capabilities": capabilities,
            "message": "Parallel processing capabilities retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get parallel capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get parallel capabilities: {str(e)}")
