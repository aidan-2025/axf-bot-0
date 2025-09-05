"""
Strategy Filter

Core filtering logic for trading strategies based on various criteria
including completeness, logical consistency, feasibility, and performance.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


class FilterStatus(Enum):
    """Status of filter evaluation"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"


class FilterCriteria(Enum):
    """Types of filtering criteria"""
    COMPLETENESS = "completeness"
    LOGICAL_CONSISTENCY = "logical_consistency"
    FEASIBILITY = "feasibility"
    PARAMETER_BOUNDS = "parameter_bounds"
    DATA_SUFFICIENCY = "data_sufficiency"
    NO_LOOKAHEAD_BIAS = "no_lookahead_bias"
    NO_SURVIVORSHIP_BIAS = "no_survivorship_bias"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    RISK_THRESHOLD = "risk_threshold"
    TRADE_FREQUENCY = "trade_frequency"


@dataclass
class FilterResult:
    """Result of a single filter evaluation"""
    
    criteria: FilterCriteria
    status: FilterStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    threshold: float = 0.0
    passed: bool = False
    
    def __post_init__(self):
        """Calculate passed status"""
        self.passed = self.status in [FilterStatus.PASS, FilterStatus.WARNING]


@dataclass
class FilterConfig:
    """Configuration for strategy filtering"""
    
    # Completeness criteria
    required_fields: List[str] = field(default_factory=lambda: [
        'strategy_id', 'name', 'description', 'entry_rules', 'exit_rules',
        'risk_management', 'asset_universe', 'timeframe', 'parameters'
    ])
    
    # Logical consistency criteria
    check_contradictions: bool = True
    check_parameter_consistency: bool = True
    check_rule_consistency: bool = True
    
    # Feasibility criteria
    supported_indicators: List[str] = field(default_factory=lambda: [
        'SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'STOCH', 'CCI', 'WILLR'
    ])
    supported_timeframes: List[str] = field(default_factory=lambda: [
        '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'
    ])
    max_parameters: int = 20
    max_indicators: int = 10
    
    # Parameter bounds
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'stop_loss': (0.001, 0.1),  # 0.1% to 10%
        'take_profit': (0.001, 0.2),  # 0.1% to 20%
        'position_size': (0.01, 1.0),  # 1% to 100%
        'risk_per_trade': (0.001, 0.05),  # 0.1% to 5%
        'max_drawdown': (0.01, 0.5),  # 1% to 50%
        'lookback_period': (1, 1000),  # 1 to 1000 periods
        'threshold': (0.0, 1.0)  # 0% to 100%
    })
    
    # Data sufficiency criteria
    min_data_points: int = 1000
    min_trades: int = 100
    min_backtest_days: int = 90
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.2  # 20%
    min_win_rate: float = 0.4  # 40%
    min_profit_factor: float = 1.1
    min_total_return: float = 0.05  # 5%
    
    # Risk thresholds
    max_volatility: float = 0.3  # 30%
    max_var_95: float = 0.05  # 5%
    max_correlation: float = 0.8  # 80%
    
    # Trade frequency criteria
    min_trades_per_month: float = 2.0
    max_trades_per_day: float = 10.0
    min_trade_duration_hours: float = 0.5
    max_trade_duration_days: float = 30.0


class StrategyFilter:
    """Core strategy filtering engine"""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def filter_strategy(
        self, 
        strategy: Dict[str, Any], 
        backtest_results: Optional[Dict[str, Any]] = None
    ) -> List[FilterResult]:
        """Filter a strategy against all criteria"""
        
        results = []
        
        # Apply all filters
        results.extend(self._check_completeness(strategy))
        results.extend(self._check_logical_consistency(strategy))
        results.extend(self._check_feasibility(strategy))
        results.extend(self._check_parameter_bounds(strategy))
        results.extend(self._check_data_sufficiency(strategy, backtest_results))
        results.extend(self._check_lookahead_bias(strategy))
        results.extend(self._check_survivorship_bias(strategy))
        
        if backtest_results:
            results.extend(self._check_performance_thresholds(backtest_results))
            results.extend(self._check_risk_thresholds(backtest_results))
            results.extend(self._check_trade_frequency(backtest_results))
        
        return results
    
    def filter_strategies(
        self, 
        strategies: List[Dict[str, Any]], 
        backtest_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[FilterResult]]:
        """Filter multiple strategies"""
        
        results = {}
        
        for i, strategy in enumerate(strategies):
            strategy_id = strategy.get('strategy_id', f'strategy_{i}')
            bt_results = backtest_results[i] if backtest_results and i < len(backtest_results) else None
            
            results[strategy_id] = self.filter_strategy(strategy, bt_results)
        
        return results
    
    def get_passing_strategies(
        self, 
        strategies: List[Dict[str, Any]], 
        backtest_results: Optional[List[Dict[str, Any]]] = None,
        require_all_pass: bool = True
    ) -> List[Dict[str, Any]]:
        """Get strategies that pass all or most filters"""
        
        filter_results = self.filter_strategies(strategies, backtest_results)
        passing_strategies = []
        
        for i, strategy in enumerate(strategies):
            strategy_id = strategy.get('strategy_id', f'strategy_{i}')
            results = filter_results[strategy_id]
            
            if require_all_pass:
                # All filters must pass
                if all(result.passed for result in results):
                    passing_strategies.append(strategy)
            else:
                # Most filters must pass (at least 80%)
                pass_rate = sum(1 for result in results if result.passed) / len(results)
                if pass_rate >= 0.8:
                    passing_strategies.append(strategy)
        
        return passing_strategies
    
    def _check_completeness(self, strategy: Dict[str, Any]) -> List[FilterResult]:
        """Check if strategy has all required fields"""
        
        results = []
        missing_fields = []
        
        for field in self.config.required_fields:
            if field not in strategy or strategy[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            results.append(FilterResult(
                criteria=FilterCriteria.COMPLETENESS,
                status=FilterStatus.FAIL,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                details={'missing_fields': missing_fields},
                score=0.0,
                threshold=1.0,
                passed=False
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.COMPLETENESS,
                status=FilterStatus.PASS,
                message="All required fields present",
                details={'present_fields': self.config.required_fields},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_logical_consistency(self, strategy: Dict[str, Any]) -> List[FilterResult]:
        """Check for logical consistency in strategy rules"""
        
        results = []
        issues = []
        
        # Check for contradictory rules
        if self.config.check_contradictions:
            issues.extend(self._check_contradictory_rules(strategy))
        
        # Check parameter consistency
        if self.config.check_parameter_consistency:
            issues.extend(self._check_parameter_consistency(strategy))
        
        # Check rule consistency
        if self.config.check_rule_consistency:
            issues.extend(self._check_rule_consistency(strategy))
        
        if issues:
            results.append(FilterResult(
                criteria=FilterCriteria.LOGICAL_CONSISTENCY,
                status=FilterStatus.FAIL,
                message=f"Logical consistency issues found: {len(issues)}",
                details={'issues': issues},
                score=0.0,
                threshold=1.0,
                passed=False
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.LOGICAL_CONSISTENCY,
                status=FilterStatus.PASS,
                message="No logical consistency issues found",
                details={'checked_aspects': ['contradictions', 'parameters', 'rules']},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_feasibility(self, strategy: Dict[str, Any]) -> List[FilterResult]:
        """Check if strategy is feasible to implement"""
        
        results = []
        issues = []
        
        # Check indicators
        if 'indicators' in strategy:
            for indicator in strategy['indicators']:
                if indicator not in self.config.supported_indicators:
                    issues.append(f"Unsupported indicator: {indicator}")
        
        # Check timeframe
        if 'timeframe' in strategy:
            if strategy['timeframe'] not in self.config.supported_timeframes:
                issues.append(f"Unsupported timeframe: {strategy['timeframe']}")
        
        # Check parameter count
        if 'parameters' in strategy:
            param_count = len(strategy['parameters'])
            if param_count > self.config.max_parameters:
                issues.append(f"Too many parameters: {param_count} > {self.config.max_parameters}")
        
        # Check indicator count
        if 'indicators' in strategy:
            indicator_count = len(strategy['indicators'])
            if indicator_count > self.config.max_indicators:
                issues.append(f"Too many indicators: {indicator_count} > {self.config.max_indicators}")
        
        if issues:
            results.append(FilterResult(
                criteria=FilterCriteria.FEASIBILITY,
                status=FilterStatus.FAIL,
                message=f"Feasibility issues found: {len(issues)}",
                details={'issues': issues},
                score=0.0,
                threshold=1.0,
                passed=False
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.FEASIBILITY,
                status=FilterStatus.PASS,
                message="Strategy is feasible to implement",
                details={'checked_aspects': ['indicators', 'timeframe', 'parameters']},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_parameter_bounds(self, strategy: Dict[str, Any]) -> List[FilterResult]:
        """Check if parameters are within acceptable bounds"""
        
        results = []
        issues = []
        
        if 'parameters' in strategy:
            for param_name, param_value in strategy['parameters'].items():
                if param_name in self.config.parameter_bounds:
                    min_val, max_val = self.config.parameter_bounds[param_name]
                    
                    try:
                        value = float(param_value)
                        if value < min_val or value > max_val:
                            issues.append(f"Parameter {param_name} out of bounds: {value} not in [{min_val}, {max_val}]")
                    except (ValueError, TypeError):
                        issues.append(f"Parameter {param_name} has invalid value: {param_value}")
        
        if issues:
            results.append(FilterResult(
                criteria=FilterCriteria.PARAMETER_BOUNDS,
                status=FilterStatus.FAIL,
                message=f"Parameter bound violations: {len(issues)}",
                details={'issues': issues},
                score=0.0,
                threshold=1.0,
                passed=False
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.PARAMETER_BOUNDS,
                status=FilterStatus.PASS,
                message="All parameters within acceptable bounds",
                details={'checked_parameters': list(self.config.parameter_bounds.keys())},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_data_sufficiency(self, strategy: Dict[str, Any], backtest_results: Optional[Dict[str, Any]]) -> List[FilterResult]:
        """Check if there's sufficient data for backtesting"""
        
        results = []
        
        if not backtest_results:
            results.append(FilterResult(
                criteria=FilterCriteria.DATA_SUFFICIENCY,
                status=FilterStatus.WARNING,
                message="No backtest results available for data sufficiency check",
                details={},
                score=0.5,
                threshold=1.0,
                passed=True
            ))
            return results
        
        issues = []
        
        # Check data points
        if 'data_points' in backtest_results:
            if backtest_results['data_points'] < self.config.min_data_points:
                issues.append(f"Insufficient data points: {backtest_results['data_points']} < {self.config.min_data_points}")
        
        # Check trade count
        if 'total_trades' in backtest_results:
            if backtest_results['total_trades'] < self.config.min_trades:
                issues.append(f"Insufficient trades: {backtest_results['total_trades']} < {self.config.min_trades}")
        
        # Check backtest duration
        if 'backtest_days' in backtest_results:
            if backtest_results['backtest_days'] < self.config.min_backtest_days:
                issues.append(f"Insufficient backtest duration: {backtest_results['backtest_days']} < {self.config.min_backtest_days}")
        
        if issues:
            results.append(FilterResult(
                criteria=FilterCriteria.DATA_SUFFICIENCY,
                status=FilterStatus.FAIL,
                message=f"Data sufficiency issues: {len(issues)}",
                details={'issues': issues},
                score=0.0,
                threshold=1.0,
                passed=False
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.DATA_SUFFICIENCY,
                status=FilterStatus.PASS,
                message="Sufficient data for backtesting",
                details={'checked_metrics': ['data_points', 'total_trades', 'backtest_days']},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_lookahead_bias(self, strategy: Dict[str, Any]) -> List[FilterResult]:
        """Check for lookahead bias in strategy rules"""
        
        results = []
        issues = []
        
        # Check entry/exit rules for future data references
        for rule_type in ['entry_rules', 'exit_rules']:
            if rule_type in strategy:
                rules = strategy[rule_type]
                if isinstance(rules, str):
                    # Simple text-based check for future references
                    future_keywords = ['future', 'next', 'tomorrow', 'ahead', 'forward']
                    for keyword in future_keywords:
                        if keyword.lower() in rules.lower():
                            issues.append(f"Potential lookahead bias in {rule_type}: '{keyword}'")
                elif isinstance(rules, list):
                    for rule in rules:
                        if isinstance(rule, str):
                            for keyword in future_keywords:
                                if keyword.lower() in rule.lower():
                                    issues.append(f"Potential lookahead bias in {rule_type}: '{keyword}'")
        
        if issues:
            results.append(FilterResult(
                criteria=FilterCriteria.NO_LOOKAHEAD_BIAS,
                status=FilterStatus.WARNING,
                message=f"Potential lookahead bias detected: {len(issues)}",
                details={'issues': issues},
                score=0.5,
                threshold=1.0,
                passed=True  # Warning, not failure
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.NO_LOOKAHEAD_BIAS,
                status=FilterStatus.PASS,
                message="No lookahead bias detected",
                details={'checked_rules': ['entry_rules', 'exit_rules']},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_survivorship_bias(self, strategy: Dict[str, Any]) -> List[FilterResult]:
        """Check for survivorship bias in asset universe"""
        
        results = []
        
        # This is a placeholder for more sophisticated survivorship bias detection
        # In practice, this would require historical asset data and delisting information
        
        results.append(FilterResult(
            criteria=FilterCriteria.NO_SURVIVORSHIP_BIAS,
            status=FilterStatus.PASS,
            message="Survivorship bias check not implemented (requires historical data)",
            details={'note': 'This check requires historical asset delisting data'},
            score=1.0,
            threshold=1.0,
            passed=True
        ))
        
        return results
    
    def _check_performance_thresholds(self, backtest_results: Dict[str, Any]) -> List[FilterResult]:
        """Check if strategy meets performance thresholds"""
        
        results = []
        issues = []
        
        # Check Sharpe ratio
        if 'sharpe_ratio' in backtest_results:
            sharpe = backtest_results['sharpe_ratio']
            if sharpe < self.config.min_sharpe_ratio:
                issues.append(f"Sharpe ratio too low: {sharpe:.3f} < {self.config.min_sharpe_ratio}")
        
        # Check max drawdown
        if 'max_drawdown' in backtest_results:
            max_dd = backtest_results['max_drawdown']
            if max_dd > self.config.max_drawdown_threshold:
                issues.append(f"Max drawdown too high: {max_dd:.3f} > {self.config.max_drawdown_threshold}")
        
        # Check win rate
        if 'win_rate' in backtest_results:
            win_rate = backtest_results['win_rate']
            if win_rate < self.config.min_win_rate:
                issues.append(f"Win rate too low: {win_rate:.3f} < {self.config.min_win_rate}")
        
        # Check profit factor
        if 'profit_factor' in backtest_results:
            pf = backtest_results['profit_factor']
            if pf < self.config.min_profit_factor:
                issues.append(f"Profit factor too low: {pf:.3f} < {self.config.min_profit_factor}")
        
        # Check total return
        if 'total_return' in backtest_results:
            total_ret = backtest_results['total_return']
            if total_ret < self.config.min_total_return:
                issues.append(f"Total return too low: {total_ret:.3f} < {self.config.min_total_return}")
        
        if issues:
            results.append(FilterResult(
                criteria=FilterCriteria.PERFORMANCE_THRESHOLD,
                status=FilterStatus.FAIL,
                message=f"Performance threshold violations: {len(issues)}",
                details={'issues': issues},
                score=0.0,
                threshold=1.0,
                passed=False
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.PERFORMANCE_THRESHOLD,
                status=FilterStatus.PASS,
                message="All performance thresholds met",
                details={'checked_metrics': ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'total_return']},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_risk_thresholds(self, backtest_results: Dict[str, Any]) -> List[FilterResult]:
        """Check if strategy meets risk thresholds"""
        
        results = []
        issues = []
        
        # Check volatility
        if 'volatility' in backtest_results:
            vol = backtest_results['volatility']
            if vol > self.config.max_volatility:
                issues.append(f"Volatility too high: {vol:.3f} > {self.config.max_volatility}")
        
        # Check VaR
        if 'var_95' in backtest_results:
            var = backtest_results['var_95']
            if var > self.config.max_var_95:
                issues.append(f"VaR 95% too high: {var:.3f} > {self.config.max_var_95}")
        
        if issues:
            results.append(FilterResult(
                criteria=FilterCriteria.RISK_THRESHOLD,
                status=FilterStatus.FAIL,
                message=f"Risk threshold violations: {len(issues)}",
                details={'issues': issues},
                score=0.0,
                threshold=1.0,
                passed=False
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.RISK_THRESHOLD,
                status=FilterStatus.PASS,
                message="All risk thresholds met",
                details={'checked_metrics': ['volatility', 'var_95']},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_trade_frequency(self, backtest_results: Dict[str, Any]) -> List[FilterResult]:
        """Check if trade frequency is within acceptable bounds"""
        
        results = []
        issues = []
        
        if 'total_trades' in backtest_results and 'backtest_days' in backtest_results:
            trades = backtest_results['total_trades']
            days = backtest_results['backtest_days']
            
            if days > 0:
                trades_per_month = (trades / days) * 30
                trades_per_day = trades / days
                
                if trades_per_month < self.config.min_trades_per_month:
                    issues.append(f"Trade frequency too low: {trades_per_month:.2f} trades/month < {self.config.min_trades_per_month}")
                
                if trades_per_day > self.config.max_trades_per_day:
                    issues.append(f"Trade frequency too high: {trades_per_day:.2f} trades/day > {self.config.max_trades_per_day}")
        
        if issues:
            results.append(FilterResult(
                criteria=FilterCriteria.TRADE_FREQUENCY,
                status=FilterStatus.FAIL,
                message=f"Trade frequency issues: {len(issues)}",
                details={'issues': issues},
                score=0.0,
                threshold=1.0,
                passed=False
            ))
        else:
            results.append(FilterResult(
                criteria=FilterCriteria.TRADE_FREQUENCY,
                status=FilterStatus.PASS,
                message="Trade frequency within acceptable bounds",
                details={'checked_metrics': ['trades_per_month', 'trades_per_day']},
                score=1.0,
                threshold=1.0,
                passed=True
            ))
        
        return results
    
    def _check_contradictory_rules(self, strategy: Dict[str, Any]) -> List[str]:
        """Check for contradictory rules in strategy"""
        
        issues = []
        
        # Check for simultaneous long/short on same asset
        if 'entry_rules' in strategy and 'exit_rules' in strategy:
            entry_rules = strategy['entry_rules']
            exit_rules = strategy['exit_rules']
            
            # Simple check for contradictory signals
            if isinstance(entry_rules, str) and isinstance(exit_rules, str):
                if 'long' in entry_rules.lower() and 'short' in exit_rules.lower():
                    issues.append("Potential contradiction: long entry with short exit")
                if 'short' in entry_rules.lower() and 'long' in exit_rules.lower():
                    issues.append("Potential contradiction: short entry with long exit")
        
        return issues
    
    def _check_parameter_consistency(self, strategy: Dict[str, Any]) -> List[str]:
        """Check for parameter consistency"""
        
        issues = []
        
        if 'parameters' in strategy:
            params = strategy['parameters']
            
            # Check stop loss vs take profit
            if 'stop_loss' in params and 'take_profit' in params:
                try:
                    stop_loss = float(params['stop_loss'])
                    take_profit = float(params['take_profit'])
                    
                    if stop_loss >= take_profit:
                        issues.append("Stop loss should be less than take profit")
                except (ValueError, TypeError):
                    issues.append("Invalid stop loss or take profit values")
            
            # Check position size vs risk per trade
            if 'position_size' in params and 'risk_per_trade' in params:
                try:
                    pos_size = float(params['position_size'])
                    risk = float(params['risk_per_trade'])
                    
                    if pos_size < risk:
                        issues.append("Position size should be greater than or equal to risk per trade")
                except (ValueError, TypeError):
                    issues.append("Invalid position size or risk per trade values")
        
        return issues
    
    def _check_rule_consistency(self, strategy: Dict[str, Any]) -> List[str]:
        """Check for rule consistency"""
        
        issues = []
        
        # Check if entry and exit rules are compatible
        if 'entry_rules' in strategy and 'exit_rules' in strategy:
            entry_rules = strategy['entry_rules']
            exit_rules = strategy['exit_rules']
            
            # Check for empty or missing rules
            if not entry_rules or entry_rules.strip() == '':
                issues.append("Entry rules are empty or missing")
            
            if not exit_rules or exit_rules.strip() == '':
                issues.append("Exit rules are empty or missing")
        
        return issues

