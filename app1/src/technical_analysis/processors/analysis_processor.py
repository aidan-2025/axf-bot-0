"""
Analysis processor for multi-timeframe technical analysis
"""

import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import time

from ..models import (
    Timeframe, MarketAnalysis, AnalysisResult, 
    AggregatedData, TechnicalIndicator, IndicatorConfig
)
from ..indicators.technical_indicators import TechnicalIndicatorCalculator
from ..timeframes.timeframe_aggregator import TimeframeAggregator

logger = logging.getLogger(__name__)


class AnalysisProcessor:
    """Processes multi-timeframe technical analysis"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.timeframe_aggregator = TimeframeAggregator()
        
        # Default analysis configuration
        self.default_timeframes = [
            Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4, Timeframe.D1
        ]
        
        self.default_indicators = {
            'sma': {'period': 20},
            'ema': {'period': 20},
            'rsi': {'period': 14},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2}
        }
    
    async def analyze_symbol(
        self,
        symbol: str,
        data: Dict[Timeframe, AggregatedData],
        timeframes: Optional[List[Timeframe]] = None,
        indicators: Optional[Dict[str, Dict]] = None
    ) -> AnalysisResult:
        """Analyze a symbol across multiple timeframes"""
        start_time = time.time()
        
        try:
            if timeframes is None:
                timeframes = self.default_timeframes
            
            if indicators is None:
                indicators = self.default_indicators
            
            # Filter available timeframes
            available_timeframes = [tf for tf in timeframes if tf in data]
            if not available_timeframes:
                logger.error(f"No data available for any requested timeframes for {symbol}")
                return AnalysisResult(
                    symbol=symbol,
                    timeframes=timeframes,
                    analyses={},
                    calculated_at=datetime.now(),
                    processing_time_ms=0,
                    success=False,
                    error_message="No data available for requested timeframes"
                )
            
            # Process each timeframe
            analyses = {}
            
            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_timeframe = {
                    executor.submit(
                        self._analyze_timeframe,
                        symbol,
                        data[tf],
                        tf,
                        indicators
                    ): tf for tf in available_timeframes
                }
                
                for future in concurrent.futures.as_completed(future_to_timeframe):
                    timeframe = future_to_timeframe[future]
                    try:
                        analysis = future.result()
                        if analysis:
                            analyses[timeframe] = analysis
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol} at {timeframe.value}: {str(e)}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return AnalysisResult(
                symbol=symbol,
                timeframes=timeframes,
                analyses=analyses,
                calculated_at=datetime.now(),
                processing_time_ms=processing_time,
                success=len(analyses) > 0,
                error_message=None if len(analyses) > 0 else "No successful analyses"
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            
            return AnalysisResult(
                symbol=symbol,
                timeframes=timeframes or [],
                analyses={},
                calculated_at=datetime.now(),
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _analyze_timeframe(
        self,
        symbol: str,
        aggregated_data: AggregatedData,
        timeframe: Timeframe,
        indicators: Dict[str, Dict]
    ) -> Optional[MarketAnalysis]:
        """Analyze a single timeframe"""
        try:
            # Validate data
            if not self.timeframe_aggregator.validate_timeframe_data(
                aggregated_data.data, timeframe, min_data_points=10
            ):
                logger.warning(f"Insufficient data for {symbol} at {timeframe.value}")
                return None
            
            # Calculate indicators
            indicator_configs = [
                {'name': name, 'parameters': params}
                for name, params in indicators.items()
            ]
            
            calculated_indicators = self.indicator_calculator.calculate_multiple_indicators(
                aggregated_data.data,
                symbol,
                timeframe,
                indicator_configs
            )
            
            if not calculated_indicators:
                logger.warning(f"No indicators calculated for {symbol} at {timeframe.value}")
                return None
            
            # Create analysis metadata
            analysis_metadata = {
                'timeframe_minutes': self.timeframe_aggregator.timeframe_minutes.get(timeframe, 0),
                'data_points': len(aggregated_data.data),
                'time_span': {
                    'start': aggregated_data.start_time.isoformat(),
                    'end': aggregated_data.end_time.isoformat()
                },
                'indicators_calculated': list(calculated_indicators.keys()),
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            return MarketAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                indicators=calculated_indicators,
                price_data=aggregated_data.data,
                analysis_metadata=analysis_metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} at {timeframe.value}: {str(e)}")
            return None
    
    async def analyze_multiple_symbols(
        self,
        symbols_data: Dict[str, Dict[Timeframe, AggregatedData]],
        timeframes: Optional[List[Timeframe]] = None,
        indicators: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, AnalysisResult]:
        """Analyze multiple symbols in parallel"""
        try:
            # Create tasks for all symbols
            tasks = [
                self.analyze_symbol(symbol, data, timeframes, indicators)
                for symbol, data in symbols_data.items()
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            analysis_results = {}
            for i, (symbol, result) in enumerate(zip(symbols_data.keys(), results)):
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing {symbol}: {str(result)}")
                    analysis_results[symbol] = AnalysisResult(
                        symbol=symbol,
                        timeframes=timeframes or [],
                        analyses={},
                        calculated_at=datetime.now(),
                        processing_time_ms=0,
                        success=False,
                        error_message=str(result)
                    )
                else:
                    analysis_results[symbol] = result
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in multi-symbol analysis: {str(e)}")
            return {}
    
    def get_analysis_summary(self, analysis_result: AnalysisResult) -> Dict[str, any]:
        """Get a summary of analysis results"""
        try:
            summary = {
                'symbol': analysis_result.symbol,
                'success': analysis_result.success,
                'processing_time_ms': analysis_result.processing_time_ms,
                'timeframes_analyzed': len(analysis_result.analyses),
                'total_indicators': 0,
                'timeframe_details': {}
            }
            
            for timeframe, analysis in analysis_result.analyses.items():
                timeframe_summary = {
                    'indicators_count': len(analysis.indicators),
                    'data_points': len(analysis.price_data),
                    'indicators': list(analysis.indicators.keys())
                }
                
                summary['timeframe_details'][timeframe.value] = timeframe_summary
                summary['total_indicators'] += len(analysis.indicators)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating analysis summary: {str(e)}")
            return {}
    
    def validate_analysis_config(
        self,
        timeframes: List[Timeframe],
        indicators: Dict[str, Dict]
    ) -> Tuple[bool, List[str]]:
        """Validate analysis configuration"""
        errors = []
        
        try:
            # Validate timeframes
            if not timeframes:
                errors.append("No timeframes specified")
            
            # Validate indicators
            if not indicators:
                errors.append("No indicators specified")
            
            # Check if indicators are supported
            supported_indicators = self.indicator_calculator.supported_indicators.keys()
            for indicator_name in indicators.keys():
                if indicator_name not in supported_indicators:
                    errors.append(f"Unsupported indicator: {indicator_name}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating analysis config: {str(e)}")
            return False, [str(e)]
    
    def get_supported_indicators(self) -> List[str]:
        """Get list of supported indicators"""
        return list(self.indicator_calculator.supported_indicators.keys())
    
    def get_supported_timeframes(self) -> List[Timeframe]:
        """Get list of supported timeframes"""
        return list(Timeframe)
    
    def estimate_processing_time(
        self,
        symbols_count: int,
        timeframes_count: int,
        indicators_count: int
    ) -> float:
        """Estimate processing time in milliseconds"""
        # Rough estimation based on typical processing times
        base_time_per_indicator = 50  # ms
        base_time_per_timeframe = 100  # ms
        base_time_per_symbol = 200  # ms
        
        estimated_time = (
            symbols_count * base_time_per_symbol +
            symbols_count * timeframes_count * base_time_per_timeframe +
            symbols_count * timeframes_count * indicators_count * base_time_per_indicator
        )
        
        return estimated_time

