#!/usr/bin/env python3
"""
Risk Engine

Core risk assessment and decision engine that evaluates market conditions,
economic events, sentiment data, and portfolio metrics to determine
appropriate risk management actions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..models import (
    RiskLevel, RiskEvent, RiskThreshold, RiskAction, RiskMetrics,
    PortfolioData, EconomicEventData, SentimentData, RiskConfig,
    RiskState, TradingState, EventImpact, SentimentLevel
)

logger = logging.getLogger(__name__)


@dataclass
class RiskEngineConfig:
    """Configuration for the risk engine"""
    # Risk calculation settings
    var_confidence_level: float = 0.95
    lookback_periods: int = 252  # 1 year of trading days
    min_data_points: int = 30
    
    # Event impact scoring
    high_impact_multiplier: float = 2.0
    critical_impact_multiplier: float = 3.0
    event_decay_hours: int = 24
    
    # Sentiment scoring
    sentiment_weight: float = 0.3
    news_weight: float = 0.4
    technical_weight: float = 0.3
    
    # Portfolio risk calculation
    correlation_lookback: int = 60
    volatility_lookback: int = 20
    max_correlation: float = 0.8
    
    # Update intervals
    metrics_update_interval: int = 300  # 5 minutes
    event_check_interval: int = 60  # 1 minute
    alert_check_interval: int = 30  # 30 seconds


class RiskEngine:
    """
    Core risk assessment and decision engine.
    
    Evaluates multiple risk factors including:
    - Portfolio metrics (drawdown, VaR, volatility)
    - Economic events (impact, timing, relevance)
    - Market sentiment (news, social media, technical)
    - Position sizing and concentration
    - Market conditions and volatility
    """
    
    def __init__(self, config: RiskEngineConfig = None):
        """Initialize risk engine"""
        self.config = config or RiskEngineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Risk thresholds
        self.risk_thresholds: List[RiskThreshold] = []
        self._initialize_default_thresholds()
        
        # State tracking
        self.current_state = RiskState(
            trading_state=TradingState.ACTIVE,
            current_risk_level=RiskLevel.LOW
        )
        
        # Data storage
        self.portfolio_history: List[PortfolioData] = []
        self.economic_events: List[EconomicEventData] = []
        self.sentiment_data: List[SentimentData] = []
        self.risk_events: List[RiskEvent] = []
        
        # Performance tracking
        self.risk_calculations_count = 0
        self.last_metrics_update = datetime.utcnow()
        
        self.logger.info("RiskEngine initialized")
    
    def _initialize_default_thresholds(self):
        """Initialize default risk thresholds"""
        self.risk_thresholds = [
            RiskThreshold(
                name="max_drawdown",
                threshold_value=0.15,  # 15%
                risk_level=RiskLevel.HIGH,
                action=RiskAction.REDUCE_POSITION,
                cooldown_minutes=60
            ),
            RiskThreshold(
                name="critical_drawdown",
                threshold_value=0.20,  # 20%
                risk_level=RiskLevel.CRITICAL,
                action=RiskAction.EMERGENCY_STOP,
                cooldown_minutes=0
            ),
            RiskThreshold(
                name="daily_loss",
                threshold_value=0.05,  # 5%
                risk_level=RiskLevel.HIGH,
                action=RiskAction.SUSPEND_NEW_TRADES,
                cooldown_minutes=120
            ),
            RiskThreshold(
                name="consecutive_losses",
                threshold_value=5,
                risk_level=RiskLevel.MEDIUM,
                action=RiskAction.REDUCE_POSITION,
                cooldown_minutes=30
            ),
            RiskThreshold(
                name="portfolio_risk",
                threshold_value=0.10,  # 10%
                risk_level=RiskLevel.HIGH,
                action=RiskAction.REDUCE_POSITION,
                cooldown_minutes=60
            )
        ]
    
    async def assess_risk(self, portfolio_data: PortfolioData, 
                         economic_events: List[EconomicEventData] = None,
                         sentiment_data: List[SentimentData] = None) -> RiskState:
        """
        Comprehensive risk assessment.
        
        Args:
            portfolio_data: Current portfolio state
            economic_events: Recent economic events
            sentiment_data: Current sentiment data
            
        Returns:
            Updated risk state with recommendations
        """
        try:
            self.logger.debug("Starting comprehensive risk assessment")
            
            # Update data
            self._update_portfolio_data(portfolio_data)
            if economic_events:
                self.economic_events = economic_events
            if sentiment_data:
                self.sentiment_data = sentiment_data
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(portfolio_data)
            
            # Assess event risk
            event_risk = await self._assess_event_risk()
            
            # Assess sentiment risk
            sentiment_risk = await self._assess_sentiment_risk()
            
            # Assess portfolio risk
            portfolio_risk = await self._assess_portfolio_risk(portfolio_data, risk_metrics)
            
            # Determine overall risk level
            overall_risk = self._determine_overall_risk(
                event_risk, sentiment_risk, portfolio_risk
            )
            
            # Determine required actions
            actions = self._determine_actions(overall_risk, portfolio_data)
            
            # Update state
            self._update_risk_state(overall_risk, actions, risk_metrics, portfolio_data)
            
            self.risk_calculations_count += 1
            self.logger.debug(f"Risk assessment completed. Risk level: {overall_risk}")
            
            return self.current_state
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            # Return safe state in case of error
            return RiskState(
                trading_state=TradingState.SUSPENDED,
                current_risk_level=RiskLevel.CRITICAL
            )
    
    def _update_portfolio_data(self, portfolio_data: PortfolioData):
        """Update portfolio data and maintain history"""
        self.portfolio_history.append(portfolio_data)
        
        # Keep only recent history
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-500:]
    
    async def _calculate_risk_metrics(self, portfolio_data: PortfolioData) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            if len(self.portfolio_history) < self.config.min_data_points:
                # Return basic metrics if insufficient data
                return RiskMetrics(
                    portfolio_value=portfolio_data.total_value,
                    total_risk=0.0,
                    risk_per_trade=0.0,
                    max_drawdown=portfolio_data.max_drawdown,
                    current_drawdown=portfolio_data.current_drawdown,
                    var_95=0.0,
                    var_99=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    calmar_ratio=0.0,
                    max_consecutive_losses=0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    recovery_factor=0.0
                )
            
            # Calculate returns
            returns = self._calculate_returns()
            
            # Calculate VaR
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            
            # Calculate drawdown metrics
            max_dd, current_dd = self._calculate_drawdowns(returns)
            
            # Calculate performance ratios
            sharpe = self._calculate_sharpe_ratio(returns)
            sortino = self._calculate_sortino_ratio(returns)
            calmar = self._calculate_calmar_ratio(returns, max_dd)
            
            # Calculate other metrics
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)
            recovery_factor = self._calculate_recovery_factor(returns)
            consecutive_losses = self._calculate_consecutive_losses(returns)
            
            # Calculate total risk
            total_risk = self._calculate_total_risk(portfolio_data)
            
            return RiskMetrics(
                portfolio_value=portfolio_data.total_value,
                total_risk=total_risk,
                risk_per_trade=total_risk / max(len(portfolio_data.positions), 1),
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                var_95=var_95,
                var_99=var_99,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                max_consecutive_losses=consecutive_losses,
                win_rate=win_rate,
                profit_factor=profit_factor,
                recovery_factor=recovery_factor
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(portfolio_value=portfolio_data.total_value)
    
    def _calculate_returns(self) -> List[float]:
        """Calculate portfolio returns from history"""
        if len(self.portfolio_history) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1].total_value
            curr_value = self.portfolio_history[i].total_value
            
            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                returns.append(ret)
        
        return returns
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0.0
        
        import numpy as np
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0
    
    def _calculate_drawdowns(self, returns: List[float]) -> Tuple[float, float]:
        """Calculate maximum and current drawdowns"""
        if not returns:
            return 0.0, 0.0
        
        import numpy as np
        
        # Calculate cumulative returns
        cumulative = np.cumprod([1 + r for r in returns])
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdowns
        drawdowns = (cumulative - running_max) / running_max
        
        max_drawdown = abs(np.min(drawdowns))
        current_drawdown = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0.0
        
        return max_drawdown, current_drawdown
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        import numpy as np
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        import numpy as np
        
        mean_return = np.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        return mean_return / downside_std * np.sqrt(252)  # Annualized
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if not returns or max_drawdown == 0:
            return 0.0
        
        import numpy as np
        
        mean_return = np.mean(returns)
        annual_return = mean_return * 252
        
        return annual_return / max_drawdown
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """Calculate win rate"""
        if not returns:
            return 0.0
        
        winning_trades = sum(1 for r in returns if r > 0)
        return winning_trades / len(returns)
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor"""
        if not returns:
            return 0.0
        
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_recovery_factor(self, returns: List[float]) -> float:
        """Calculate recovery factor"""
        if not returns:
            return 0.0
        
        total_return = sum(returns)
        max_dd, _ = self._calculate_drawdowns(returns)
        
        if max_dd == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / max_dd
    
    def _calculate_consecutive_losses(self, returns: List[float]) -> int:
        """Calculate maximum consecutive losses"""
        if not returns:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_total_risk(self, portfolio_data: PortfolioData) -> float:
        """Calculate total portfolio risk"""
        if not portfolio_data.positions:
            return 0.0
        
        # Simple risk calculation based on position sizes and volatility
        total_risk = 0.0
        
        for position in portfolio_data.positions:
            # Risk = position value * estimated volatility
            position_value = abs(position.size * position.current_price)
            estimated_volatility = 0.15  # 15% default volatility
            position_risk = position_value * estimated_volatility
            total_risk += position_risk
        
        return total_risk / portfolio_data.total_value if portfolio_data.total_value > 0 else 0.0
    
    async def _assess_event_risk(self) -> RiskLevel:
        """Assess risk from economic events"""
        if not self.economic_events:
            return RiskLevel.LOW
        
        current_time = datetime.utcnow()
        high_impact_events = 0
        critical_events = 0
        
        for event in self.economic_events:
            # Check if event is within lookahead period
            time_diff = (event.event_time - current_time).total_seconds() / 3600  # hours
            
            if 0 <= time_diff <= self.config.event_lookahead_hours:
                if event.impact == EventImpact.HIGH:
                    high_impact_events += 1
                elif event.impact == EventImpact.CRITICAL:
                    critical_events += 1
        
        # Determine risk level based on events
        if critical_events > 0:
            return RiskLevel.CRITICAL
        elif high_impact_events >= 2:
            return RiskLevel.HIGH
        elif high_impact_events == 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _assess_sentiment_risk(self) -> RiskLevel:
        """Assess risk from market sentiment"""
        if not self.sentiment_data:
            return RiskLevel.LOW
        
        # Calculate average sentiment
        sentiment_scores = [s.sentiment_score for s in self.sentiment_data]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Determine risk level based on sentiment
        if avg_sentiment <= -0.8:
            return RiskLevel.CRITICAL
        elif avg_sentiment <= -0.5:
            return RiskLevel.HIGH
        elif avg_sentiment <= -0.2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _assess_portfolio_risk(self, portfolio_data: PortfolioData, 
                                   risk_metrics: RiskMetrics) -> RiskLevel:
        """Assess risk from portfolio metrics"""
        # Check various risk thresholds
        if risk_metrics.current_drawdown >= 0.20:
            return RiskLevel.CRITICAL
        elif risk_metrics.current_drawdown >= 0.15:
            return RiskLevel.HIGH
        elif risk_metrics.current_drawdown >= 0.10:
            return RiskLevel.MEDIUM
        
        if risk_metrics.var_95 >= 0.05:  # 5% VaR
            return RiskLevel.HIGH
        elif risk_metrics.var_95 >= 0.03:  # 3% VaR
            return RiskLevel.MEDIUM
        
        if risk_metrics.max_consecutive_losses >= 5:
            return RiskLevel.HIGH
        elif risk_metrics.max_consecutive_losses >= 3:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _determine_overall_risk(self, event_risk: RiskLevel, 
                              sentiment_risk: RiskLevel, 
                              portfolio_risk: RiskLevel) -> RiskLevel:
        """Determine overall risk level"""
        # Take the highest risk level
        risk_levels = [event_risk, sentiment_risk, portfolio_risk]
        
        if RiskLevel.CRITICAL in risk_levels:
            return RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            return RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _determine_actions(self, risk_level: RiskLevel, 
                          portfolio_data: PortfolioData) -> List[RiskAction]:
        """Determine required risk management actions"""
        actions = []
        
        # Check threshold triggers
        for threshold in self.risk_thresholds:
            if threshold.name == "max_drawdown" and threshold.is_triggered(portfolio_data.current_drawdown):
                actions.append(threshold.action)
                threshold.trigger()
            elif threshold.name == "critical_drawdown" and threshold.is_triggered(portfolio_data.current_drawdown):
                actions.append(threshold.action)
                threshold.trigger()
            elif threshold.name == "daily_loss" and threshold.is_triggered(abs(portfolio_data.total_pnl / portfolio_data.total_value)):
                actions.append(threshold.action)
                threshold.trigger()
            elif threshold.name == "consecutive_losses" and threshold.is_triggered(portfolio_data.max_drawdown):
                actions.append(threshold.action)
                threshold.trigger()
            elif threshold.name == "portfolio_risk" and threshold.is_triggered(portfolio_data.total_pnl / portfolio_data.total_value):
                actions.append(threshold.action)
                threshold.trigger()
        
        # Add actions based on risk level
        if risk_level == RiskLevel.CRITICAL:
            actions.append(RiskAction.EMERGENCY_STOP)
        elif risk_level == RiskLevel.HIGH:
            actions.append(RiskAction.REDUCE_POSITION)
        elif risk_level == RiskLevel.MEDIUM:
            actions.append(RiskAction.SUSPEND_NEW_TRADES)
        
        return list(set(actions))  # Remove duplicates
    
    def _update_risk_state(self, risk_level: RiskLevel, actions: List[RiskAction],
                          risk_metrics: RiskMetrics, portfolio_data: PortfolioData):
        """Update current risk state"""
        self.current_state.current_risk_level = risk_level
        self.current_state.risk_metrics = risk_metrics
        self.current_state.portfolio_data = portfolio_data
        self.current_state.last_update = datetime.utcnow()
        
        # Update trading state based on actions
        if RiskAction.EMERGENCY_STOP in actions:
            self.current_state.trading_state = TradingState.EMERGENCY_STOP
        elif RiskAction.CLOSE_ALL_POSITIONS in actions:
            self.current_state.trading_state = TradingState.SUSPENDED
        elif RiskAction.SUSPEND_NEW_TRADES in actions:
            self.current_state.trading_state = TradingState.SUSPENDED
        elif RiskAction.REDUCE_POSITION in actions:
            self.current_state.trading_state = TradingState.REDUCED
        else:
            self.current_state.trading_state = TradingState.ACTIVE
        
        # Create risk events for significant changes
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            event = RiskEvent(
                event_id=f"risk_{datetime.utcnow().timestamp()}",
                event_type="risk_assessment",
                risk_level=risk_level,
                description=f"Risk level elevated to {risk_level.value}",
                timestamp=datetime.utcnow(),
                source="risk_engine",
                data={
                    "actions": [action.value for action in actions],
                    "metrics": risk_metrics.to_dict()
                }
            )
            self.current_state.active_events.append(event)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary"""
        return {
            "current_risk_level": self.current_state.current_risk_level.value,
            "trading_state": self.current_state.trading_state.value,
            "active_events_count": len(self.current_state.active_events),
            "active_alerts_count": len(self.current_state.active_alerts),
            "last_update": self.current_state.last_update.isoformat(),
            "risk_calculations_count": self.risk_calculations_count,
            "portfolio_value": self.current_state.portfolio_data.total_value if self.current_state.portfolio_data else 0.0,
            "current_drawdown": self.current_state.risk_metrics.current_drawdown if self.current_state.risk_metrics else 0.0
        }

