#!/usr/bin/env python3
"""
Result Aggregator

Aggregates and analyzes backtesting results from multiple strategies.
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for result aggregation"""
    
    # Filtering settings
    min_score_threshold: float = 0.0
    min_trades_threshold: int = 0
    max_drawdown_threshold: float = 1.0
    
    # Ranking settings
    ranking_metrics: List[str] = None
    ranking_weights: Dict[str, float] = None
    
    # Analysis settings
    include_failed_strategies: bool = False
    group_by_type: bool = True
    calculate_statistics: bool = True
    
    def __post_init__(self):
        if self.ranking_metrics is None:
            self.ranking_metrics = ['validation_score', 'sharpe_ratio', 'profit_factor', 'win_rate']
        
        if self.ranking_weights is None:
            self.ranking_weights = {
                'validation_score': 0.4,
                'sharpe_ratio': 0.3,
                'profit_factor': 0.2,
                'win_rate': 0.1
            }


class ResultAggregator:
    """Aggregates and analyzes backtesting results"""
    
    def __init__(self, config: AggregationConfig = None):
        self.config = config or AggregationConfig()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("ResultAggregator initialized")
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple backtests"""
        self.logger.info(f"Aggregating {len(results)} results")
        
        # Filter results
        filtered_results = self._filter_results(results)
        
        # Extract metrics
        metrics_data = self._extract_metrics(filtered_results)
        
        # Calculate statistics
        statistics = self._calculate_statistics(metrics_data)
        
        # Rank strategies
        rankings = self._rank_strategies(filtered_results)
        
        # Group by type
        grouped_results = self._group_by_type(filtered_results) if self.config.group_by_type else {}
        
        # Create summary
        summary = self._create_summary(filtered_results, statistics, rankings)
        
        return {
            'summary': summary,
            'statistics': statistics,
            'rankings': rankings,
            'grouped_results': grouped_results,
            'raw_results': filtered_results,
            'total_results': len(results),
            'filtered_results': len(filtered_results),
            'aggregation_timestamp': datetime.now().isoformat()
        }
    
    def _filter_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results based on configuration"""
        filtered = []
        
        for result in results:
            # Skip failed strategies if configured
            if not self.config.include_failed_strategies and not result.get('success', False):
                continue
            
            # Apply filters
            if self._passes_filters(result):
                filtered.append(result)
        
        self.logger.info(f"Filtered {len(filtered)} results from {len(results)} total")
        return filtered
    
    def _passes_filters(self, result: Dict[str, Any]) -> bool:
        """Check if result passes all filters"""
        if not result.get('success', False):
            return False
        
        validation_result = result.get('validation_result')
        if not validation_result:
            return False
        
        # Check score threshold
        if hasattr(validation_result, 'validation_score'):
            if validation_result.validation_score < self.config.min_score_threshold:
                return False
        
        # Check trades threshold
        if hasattr(validation_result, 'total_trades'):
            if validation_result.total_trades < self.config.min_trades_threshold:
                return False
        
        # Check drawdown threshold
        if hasattr(validation_result, 'performance_metrics'):
            performance_metrics = validation_result.performance_metrics
            if hasattr(performance_metrics, 'max_drawdown'):
                if performance_metrics.max_drawdown > self.config.max_drawdown_threshold:
                    return False
        
        return True
    
    def _extract_metrics(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract metrics from results into DataFrame"""
        data = []
        
        for result in results:
            if not result.get('success', False):
                continue
            
            validation_result = result.get('validation_result')
            if not validation_result:
                continue
            
            # Extract basic info
            row = {
                'strategy_id': validation_result.strategy_id,
                'strategy_name': validation_result.strategy_name,
                'strategy_type': validation_result.strategy_type,
                'validation_passed': validation_result.validation_passed,
                'validation_score': validation_result.validation_score,
                'duration_seconds': result.get('duration_seconds', 0)
            }
            
            # Extract performance metrics
            perf_metrics = validation_result.performance_metrics
            if perf_metrics:
                row.update({
                    'total_return': getattr(perf_metrics, 'total_return', 0),
                    'annualized_return': getattr(perf_metrics, 'annualized_return', 0),
                    'sharpe_ratio': getattr(perf_metrics, 'sharpe_ratio', 0),
                    'sortino_ratio': getattr(perf_metrics, 'sortino_ratio', 0),
                    'calmar_ratio': getattr(perf_metrics, 'calmar_ratio', 0),
                    'max_drawdown': getattr(perf_metrics, 'max_drawdown', 0),
                    'volatility': getattr(perf_metrics, 'volatility', 0),
                    'win_rate': getattr(perf_metrics, 'win_rate', 0),
                    'profit_factor': getattr(perf_metrics, 'profit_factor', 0),
                    'total_trades': getattr(perf_metrics, 'total_trades', 0),
                    'consistency_score': getattr(perf_metrics, 'consistency_score', 0),
                    'stability_score': getattr(perf_metrics, 'stability_score', 0)
                })
            
            # Extract scoring metrics
            scoring_metrics = validation_result.scoring_metrics
            if scoring_metrics:
                row.update({
                    'overall_score': getattr(scoring_metrics, 'overall_score', 0),
                    'performance_score': getattr(scoring_metrics, 'performance_score', 0),
                    'risk_score': getattr(scoring_metrics, 'risk_score', 0),
                    'consistency_score': getattr(scoring_metrics, 'consistency_score', 0),
                    'efficiency_score': getattr(scoring_metrics, 'efficiency_score', 0),
                    'robustness_score': getattr(scoring_metrics, 'robustness_score', 0)
                })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _calculate_statistics(self, metrics_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if metrics_data.empty:
            return {}
        
        statistics = {}
        
        # Basic counts
        statistics['total_strategies'] = len(metrics_data)
        statistics['passed_strategies'] = len(metrics_data[metrics_data['validation_passed'] == True])
        statistics['failed_strategies'] = len(metrics_data[metrics_data['validation_passed'] == False])
        statistics['pass_rate'] = statistics['passed_strategies'] / statistics['total_strategies'] if statistics['total_strategies'] > 0 else 0
        
        # Strategy type distribution
        statistics['strategy_types'] = metrics_data['strategy_type'].value_counts().to_dict()
        
        # Performance statistics
        numeric_columns = metrics_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['strategy_id']:  # Skip non-metric columns
                statistics[f'{col}_mean'] = metrics_data[col].mean()
                statistics[f'{col}_median'] = metrics_data[col].median()
                statistics[f'{col}_std'] = metrics_data[col].std()
                statistics[f'{col}_min'] = metrics_data[col].min()
                statistics[f'{col}_max'] = metrics_data[col].max()
                statistics[f'{col}_q25'] = metrics_data[col].quantile(0.25)
                statistics[f'{col}_q75'] = metrics_data[col].quantile(0.75)
        
        # Correlation analysis
        if len(numeric_columns) > 1:
            correlation_matrix = metrics_data[numeric_columns].corr()
            statistics['correlations'] = correlation_matrix.to_dict()
        
        # Top performers
        if 'validation_score' in metrics_data.columns:
            top_strategies = metrics_data.nlargest(10, 'validation_score')
            statistics['top_10_strategies'] = top_strategies[['strategy_id', 'strategy_name', 'validation_score']].to_dict('records')
        
        return statistics
    
    def _rank_strategies(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank strategies based on multiple metrics"""
        if not results:
            return []
        
        # Calculate composite scores
        ranked_results = []
        
        for result in results:
            if not result.get('success', False):
                continue
            
            validation_result = result.get('validation_result')
            if not validation_result:
                continue
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(validation_result)
            
            # Handle performance_metrics
            perf_metrics = {}
            if hasattr(validation_result, 'performance_metrics') and validation_result.performance_metrics:
                if hasattr(validation_result.performance_metrics, 'to_dict'):
                    perf_metrics = validation_result.performance_metrics.to_dict()
                else:
                    # Convert to dict manually for mock objects
                    perf_metrics = {attr: getattr(validation_result.performance_metrics, attr) 
                                  for attr in dir(validation_result.performance_metrics) 
                                  if not attr.startswith('_')}
            
            # Handle scoring_metrics
            scoring_metrics = {}
            if hasattr(validation_result, 'scoring_metrics') and validation_result.scoring_metrics:
                if hasattr(validation_result.scoring_metrics, 'to_dict'):
                    scoring_metrics = validation_result.scoring_metrics.to_dict()
                else:
                    # Convert to dict manually for mock objects
                    scoring_metrics = {attr: getattr(validation_result.scoring_metrics, attr) 
                                     for attr in dir(validation_result.scoring_metrics) 
                                     if not attr.startswith('_')}
            
            ranked_results.append({
                'strategy_id': getattr(validation_result, 'strategy_id', 'unknown'),
                'strategy_name': getattr(validation_result, 'strategy_name', 'unknown'),
                'strategy_type': getattr(validation_result, 'strategy_type', 'unknown'),
                'composite_score': composite_score,
                'validation_score': getattr(validation_result, 'validation_score', 0),
                'performance_metrics': perf_metrics,
                'scoring_metrics': scoring_metrics
            })
        
        # Sort by composite score
        ranked_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add ranking position
        for i, result in enumerate(ranked_results):
            result['rank'] = i + 1
        
        return ranked_results
    
    def _calculate_composite_score(self, validation_result) -> float:
        """Calculate composite score for ranking"""
        score = 0.0
        total_weight = 0.0
        
        # Get weights
        weights = self.config.ranking_weights
        
        # Validation score
        if 'validation_score' in weights and hasattr(validation_result, 'validation_score'):
            score += validation_result.validation_score * weights['validation_score']
            total_weight += weights['validation_score']
        
        # Performance metrics
        perf_metrics = getattr(validation_result, 'performance_metrics', None)
        if perf_metrics:
            if 'sharpe_ratio' in weights:
                sharpe = getattr(perf_metrics, 'sharpe_ratio', 0)
                # Normalize Sharpe ratio (0-3 range to 0-1)
                normalized_sharpe = min(1.0, max(0.0, sharpe / 3.0))
                score += normalized_sharpe * weights['sharpe_ratio']
                total_weight += weights['sharpe_ratio']
            
            if 'profit_factor' in weights:
                pf = getattr(perf_metrics, 'profit_factor', 0)
                # Normalize profit factor (1-5 range to 0-1)
                normalized_pf = min(1.0, max(0.0, (pf - 1) / 4.0))
                score += normalized_pf * weights['profit_factor']
                total_weight += weights['profit_factor']
            
            if 'win_rate' in weights:
                wr = getattr(perf_metrics, 'win_rate', 0)
                score += wr * weights['win_rate']
                total_weight += weights['win_rate']
        
        # Normalize by total weight
        if total_weight > 0:
            score = score / total_weight
        
        return score
    
    def _group_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by strategy type"""
        grouped = defaultdict(list)
        
        for result in results:
            if not result.get('success', False):
                continue
            
            validation_result = result.get('validation_result')
            if not validation_result:
                continue
            
            strategy_type = validation_result.strategy_type
            grouped[strategy_type].append(result)
        
        return dict(grouped)
    
    def _create_summary(self, results: List[Dict[str, Any]], 
                       statistics: Dict[str, Any], 
                       rankings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create executive summary"""
        summary = {
            'overview': {
                'total_strategies_tested': len(results),
                'successful_backtests': len([r for r in results if r.get('success', False)]),
                'failed_backtests': len([r for r in results if not r.get('success', False)]),
                'pass_rate': statistics.get('pass_rate', 0),
                'average_validation_score': statistics.get('validation_score_mean', 0),
                'best_validation_score': statistics.get('validation_score_max', 0)
            },
            'performance_highlights': {
                'best_sharpe_ratio': statistics.get('sharpe_ratio_max', 0),
                'best_profit_factor': statistics.get('profit_factor_max', 0),
                'best_win_rate': statistics.get('win_rate_max', 0),
                'lowest_drawdown': statistics.get('max_drawdown_min', 0)
            },
            'top_strategies': rankings[:5] if rankings else [],
            'strategy_type_breakdown': statistics.get('strategy_types', {}),
            'recommendations': self._generate_recommendations(statistics, rankings)
        }
        
        return summary
    
    def _generate_recommendations(self, statistics: Dict[str, Any], 
                                rankings: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Check pass rate
        pass_rate = statistics.get('pass_rate', 0)
        if pass_rate < 0.5:
            recommendations.append("Low pass rate detected. Consider reviewing validation criteria or strategy quality.")
        elif pass_rate > 0.8:
            recommendations.append("High pass rate achieved. Consider tightening validation criteria for better quality control.")
        
        # Check average performance
        avg_sharpe = statistics.get('sharpe_ratio_mean', 0)
        if avg_sharpe < 0.5:
            recommendations.append("Low average Sharpe ratio. Focus on risk-adjusted returns in strategy development.")
        
        avg_drawdown = statistics.get('max_drawdown_mean', 0)
        if avg_drawdown > 0.15:
            recommendations.append("High average drawdown detected. Implement better risk management in strategies.")
        
        # Check strategy diversity
        strategy_types = statistics.get('strategy_types', {})
        if len(strategy_types) < 3:
            recommendations.append("Limited strategy diversity. Consider developing strategies across different types.")
        
        # Top performer analysis
        if rankings:
            top_strategy = rankings[0]
            if top_strategy['composite_score'] > 0.8:
                recommendations.append(f"Excellent performance from {top_strategy['strategy_name']}. Consider this as a template for future strategies.")
        
        return recommendations
    
    def export_results(self, aggregation_results: Dict[str, Any], 
                      output_path: str, format: str = 'json') -> None:
        """Export aggregation results to file"""
        try:
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(aggregation_results, f, indent=2, default=str)
            elif format.lower() == 'csv':
                # Export rankings as CSV
                if 'rankings' in aggregation_results:
                    df = pd.DataFrame(aggregation_results['rankings'])
                    df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Results exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            raise
    
    def get_performance_insights(self, aggregation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance insights from aggregated results"""
        insights = {
            'overall_performance': {},
            'strategy_type_analysis': {},
            'risk_analysis': {},
            'efficiency_analysis': {}
        }
        
        # Overall performance insights
        summary = aggregation_results.get('summary', {})
        insights['overall_performance'] = {
            'success_rate': summary.get('overview', {}).get('pass_rate', 0),
            'average_score': summary.get('overview', {}).get('average_validation_score', 0),
            'best_performer': summary.get('top_strategies', [{}])[0].get('strategy_name', 'N/A') if summary.get('top_strategies') else 'N/A'
        }
        
        # Strategy type analysis
        type_breakdown = summary.get('strategy_type_breakdown', {})
        if type_breakdown:
            insights['strategy_type_analysis'] = {
                'most_common_type': max(type_breakdown, key=type_breakdown.get),
                'type_distribution': type_breakdown,
                'diversity_score': len(type_breakdown) / 10.0  # Normalize to 0-1
            }
        
        # Risk analysis
        highlights = summary.get('performance_highlights', {})
        insights['risk_analysis'] = {
            'best_risk_adjusted_return': highlights.get('best_sharpe_ratio', 0),
            'lowest_drawdown': highlights.get('lowest_drawdown', 0),
            'risk_level': 'Low' if highlights.get('lowest_drawdown', 0) < 0.05 else 'Medium' if highlights.get('lowest_drawdown', 0) < 0.15 else 'High'
        }
        
        # Efficiency analysis
        insights['efficiency_analysis'] = {
            'best_profit_factor': highlights.get('best_profit_factor', 0),
            'best_win_rate': highlights.get('best_win_rate', 0),
            'efficiency_score': (highlights.get('best_profit_factor', 0) + highlights.get('best_win_rate', 0)) / 2
        }
        
        return insights
