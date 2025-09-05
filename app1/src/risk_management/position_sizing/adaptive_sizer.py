"""
Adaptive Position Sizing System

Implements sophisticated position sizing logic that adapts to risk conditions,
market volatility, and economic events.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from ..models import RiskLevel, RiskEvent, PortfolioData, RiskMetrics, EventImpact, SentimentLevel


class SizingMode(Enum):
    """Position sizing modes"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class SizingFactor(Enum):
    """Factors that influence position sizing"""
    RISK_LEVEL = "risk_level"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    SENTIMENT = "sentiment"
    EVENTS = "events"
    DRAWDOWN = "drawdown"
    LIQUIDITY = "liquidity"


@dataclass
class SizingRule:
    """Individual position sizing rule"""
    factor: SizingFactor
    weight: float
    min_multiplier: float = 0.1
    max_multiplier: float = 2.0
    enabled: bool = True
    lookback_periods: int = 20
    threshold: float = 0.5


@dataclass
class SizingConfig:
    """Configuration for adaptive position sizing"""
    # Base sizing parameters
    base_position_size: float = 0.02  # 2% of portfolio
    max_position_size: float = 0.10  # 10% of portfolio
    min_position_size: float = 0.001  # 0.1% of portfolio
    
    # Risk-based adjustments
    risk_level_multipliers: Dict[RiskLevel, float] = field(default_factory=lambda: {
        RiskLevel.LOW: 1.5,
        RiskLevel.MEDIUM: 1.0,
        RiskLevel.HIGH: 0.5,
        RiskLevel.CRITICAL: 0.1
    })
    
    # Volatility adjustments
    volatility_lookback: int = 20
    high_volatility_threshold: float = 0.3
    volatility_reduction_factor: float = 0.7
    
    # Correlation adjustments
    max_correlation: float = 0.7
    correlation_penalty: float = 0.5
    
    # Event-based adjustments
    event_impact_multipliers: Dict[EventImpact, float] = field(default_factory=lambda: {
        EventImpact.LOW: 1.0,
        EventImpact.MEDIUM: 0.8,
        EventImpact.HIGH: 0.5,
        EventImpact.CRITICAL: 0.2
    })
    
    # Sentiment adjustments
    sentiment_multipliers: Dict[SentimentLevel, float] = field(default_factory=lambda: {
        SentimentLevel.VERY_BEARISH: 0.3,
        SentimentLevel.BEARISH: 0.6,
        SentimentLevel.NEUTRAL: 1.0,
        SentimentLevel.BULLISH: 1.2,
        SentimentLevel.VERY_BULLISH: 1.5
    })
    
    # Drawdown adjustments
    drawdown_reduction_start: float = 0.05  # Start reducing at 5% drawdown
    drawdown_reduction_factor: float = 0.8  # Reduce by 20% for each 5% drawdown
    
    # Liquidity adjustments
    min_liquidity_threshold: float = 0.5
    liquidity_multiplier: float = 0.8
    
    # Sizing rules
    sizing_rules: List[SizingRule] = field(default_factory=lambda: [
        SizingRule(SizingFactor.RISK_LEVEL, 0.3),
        SizingRule(SizingFactor.VOLATILITY, 0.2),
        SizingRule(SizingFactor.CORRELATION, 0.15),
        SizingRule(SizingFactor.SENTIMENT, 0.1),
        SizingRule(SizingFactor.EVENTS, 0.15),
        SizingRule(SizingFactor.DRAWDOWN, 0.1)
    ])
    
    # Update intervals
    update_interval_seconds: int = 60
    rebalance_threshold: float = 0.05  # Rebalance if position size changes by 5%


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    symbol: str
    recommended_size: float
    current_size: float
    size_change: float
    sizing_factors: Dict[str, float]
    risk_adjustment: float
    volatility_adjustment: float
    correlation_adjustment: float
    sentiment_adjustment: float
    event_adjustment: float
    drawdown_adjustment: float
    final_multiplier: float
    confidence: float
    timestamp: datetime
    warnings: List[str] = field(default_factory=list)


class AdaptivePositionSizer:
    """Adaptive position sizing system"""
    
    def __init__(self, config: SizingConfig = None):
        self.config = config or SizingConfig()
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.current_positions: Dict[str, float] = {}
        self.position_history: List[Dict[str, Any]] = []
        self.market_data: Dict[str, Any] = {}
        self.last_update: datetime = datetime.utcnow()
        
        # Performance tracking
        self.sizing_performance: Dict[str, Any] = {}
        self.adaptive_learning_enabled = True
    
    def calculate_position_size(self, symbol: str, 
                              portfolio_data: PortfolioData,
                              risk_metrics: RiskMetrics,
                              market_data: Dict[str, Any] = None,
                              active_events: List[RiskEvent] = None) -> PositionSizingResult:
        """Calculate optimal position size for a symbol"""
        market_data = market_data or {}
        active_events = active_events or []
        
        # Update market data
        self.market_data.update(market_data)
        
        # Get current position size
        current_size = self.current_positions.get(symbol, 0.0)
        
        # Calculate individual adjustments
        risk_adjustment = self._calculate_risk_adjustment(risk_metrics)
        volatility_adjustment = self._calculate_volatility_adjustment(symbol, market_data)
        correlation_adjustment = self._calculate_correlation_adjustment(symbol, portfolio_data, market_data)
        sentiment_adjustment = self._calculate_sentiment_adjustment(symbol, market_data)
        event_adjustment = self._calculate_event_adjustment(active_events)
        drawdown_adjustment = self._calculate_drawdown_adjustment(risk_metrics)
        
        # Calculate final multiplier
        sizing_factors = {
            "risk_level": risk_adjustment,
            "volatility": volatility_adjustment,
            "correlation": correlation_adjustment,
            "sentiment": sentiment_adjustment,
            "events": event_adjustment,
            "drawdown": drawdown_adjustment
        }
        
        final_multiplier = self._calculate_final_multiplier(sizing_factors)
        
        # Calculate recommended size
        recommended_size = self.config.base_position_size * final_multiplier
        
        # Apply bounds
        recommended_size = max(
            self.config.min_position_size,
            min(self.config.max_position_size, recommended_size)
        )
        
        # Calculate size change
        size_change = recommended_size - current_size
        
        # Calculate confidence
        confidence = self._calculate_confidence(sizing_factors, market_data)
        
        # Generate warnings
        warnings = self._generate_warnings(sizing_factors, recommended_size, current_size)
        
        result = PositionSizingResult(
            symbol=symbol,
            recommended_size=recommended_size,
            current_size=current_size,
            size_change=size_change,
            sizing_factors=sizing_factors,
            risk_adjustment=risk_adjustment,
            volatility_adjustment=volatility_adjustment,
            correlation_adjustment=correlation_adjustment,
            sentiment_adjustment=sentiment_adjustment,
            event_adjustment=event_adjustment,
            drawdown_adjustment=drawdown_adjustment,
            final_multiplier=final_multiplier,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            warnings=warnings
        )
        
        # Update state
        self.current_positions[symbol] = recommended_size
        self._record_sizing_decision(result)
        
        return result
    
    def _calculate_risk_adjustment(self, risk_metrics: RiskMetrics) -> float:
        """Calculate risk-based position size adjustment"""
        # Get current risk level (simplified)
        if risk_metrics.current_drawdown > 0.15:
            risk_level = RiskLevel.CRITICAL
        elif risk_metrics.current_drawdown > 0.10:
            risk_level = RiskLevel.HIGH
        elif risk_metrics.current_drawdown > 0.05:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return self.config.risk_level_multipliers.get(risk_level, 1.0)
    
    def _calculate_volatility_adjustment(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-based position size adjustment"""
        volatility = market_data.get(f"{symbol}_volatility", 0.0)
        
        if volatility > self.config.high_volatility_threshold:
            return self.config.volatility_reduction_factor
        elif volatility > 0:
            # Linear reduction based on volatility
            reduction = (volatility / self.config.high_volatility_threshold) * (1 - self.config.volatility_reduction_factor)
            return 1.0 - reduction
        
        return 1.0
    
    def _calculate_correlation_adjustment(self, symbol: str, portfolio_data: PortfolioData, market_data: Dict[str, Any]) -> float:
        """Calculate correlation-based position size adjustment"""
        if not portfolio_data.positions:
            return 1.0
        
        # Get correlation with existing positions
        correlations = market_data.get(f"{symbol}_correlations", {})
        max_correlation = 0.0
        
        for pos in portfolio_data.positions:
            pos_symbol = pos.get("symbol", "")
            if pos_symbol in correlations:
                max_correlation = max(max_correlation, abs(correlations[pos_symbol]))
        
        if max_correlation > self.config.max_correlation:
            return self.config.correlation_penalty
        
        return 1.0
    
    def _calculate_sentiment_adjustment(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate sentiment-based position size adjustment"""
        sentiment = market_data.get(f"{symbol}_sentiment", 0.0)
        
        # Convert sentiment score to level
        if sentiment <= -0.8:
            level = SentimentLevel.VERY_BEARISH
        elif sentiment <= -0.5:
            level = SentimentLevel.BEARISH
        elif sentiment <= 0.2:
            level = SentimentLevel.NEUTRAL
        elif sentiment <= 0.8:
            level = SentimentLevel.BULLISH
        else:
            level = SentimentLevel.VERY_BULLISH
        
        return self.config.sentiment_multipliers.get(level, 1.0)
    
    def _calculate_event_adjustment(self, active_events: List[RiskEvent]) -> float:
        """Calculate event-based position size adjustment"""
        if not active_events:
            return 1.0
        
        # Find highest risk level event
        max_risk_level = RiskLevel.LOW
        for event in active_events:
            # Compare risk levels by their enum order
            risk_order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3, RiskLevel.CRITICAL: 4}
            if risk_order[event.risk_level] > risk_order[max_risk_level]:
                max_risk_level = event.risk_level
        
        # Map risk level to impact level for multiplier lookup
        risk_to_impact = {
            RiskLevel.LOW: EventImpact.LOW,
            RiskLevel.MEDIUM: EventImpact.MEDIUM,
            RiskLevel.HIGH: EventImpact.HIGH,
            RiskLevel.CRITICAL: EventImpact.CRITICAL
        }
        
        impact_level = risk_to_impact.get(max_risk_level, EventImpact.LOW)
        return self.config.event_impact_multipliers.get(impact_level, 1.0)
    
    def _calculate_drawdown_adjustment(self, risk_metrics: RiskMetrics) -> float:
        """Calculate drawdown-based position size adjustment"""
        drawdown = risk_metrics.current_drawdown
        
        if drawdown <= self.config.drawdown_reduction_start:
            return 1.0
        
        # Calculate reduction factor
        excess_drawdown = drawdown - self.config.drawdown_reduction_start
        reduction_steps = excess_drawdown / 0.05  # Each 5% drawdown
        reduction_factor = self.config.drawdown_reduction_factor ** reduction_steps
        
        return max(0.1, reduction_factor)
    
    def _calculate_final_multiplier(self, sizing_factors: Dict[str, float]) -> float:
        """Calculate final position size multiplier"""
        # Weighted average of all factors
        total_weight = 0.0
        weighted_sum = 0.0
        
        for rule in self.config.sizing_rules:
            if not rule.enabled:
                continue
            
            factor_value = sizing_factors.get(rule.factor.value, 1.0)
            # Apply rule bounds
            factor_value = max(rule.min_multiplier, min(rule.max_multiplier, factor_value))
            
            weighted_sum += factor_value * rule.weight
            total_weight += rule.weight
        
        if total_weight == 0:
            return 1.0
        
        final_multiplier = weighted_sum / total_weight
        
        # Apply global bounds
        return max(0.1, min(2.0, final_multiplier))
    
    def _calculate_confidence(self, sizing_factors: Dict[str, float], market_data: Dict[str, Any]) -> float:
        """Calculate confidence in sizing decision"""
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence for extreme adjustments
        for factor, value in sizing_factors.items():
            if value < 0.3 or value > 1.7:
                confidence -= 0.1
        
        # Reduce confidence for missing market data
        required_data = ["volatility", "correlation", "sentiment"]
        missing_data = sum(1 for data in required_data if f"_{data}" not in str(market_data))
        confidence -= missing_data * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_warnings(self, sizing_factors: Dict[str, float], recommended_size: float, current_size: float) -> List[str]:
        """Generate warnings for position sizing"""
        warnings = []
        
        # Check for extreme adjustments
        for factor, value in sizing_factors.items():
            if value < 0.3:
                warnings.append(f"Large reduction due to {factor}: {value:.2f}")
            elif value > 1.7:
                warnings.append(f"Large increase due to {factor}: {value:.2f}")
        
        # Check for large size changes
        if abs(recommended_size - current_size) > self.config.rebalance_threshold:
            warnings.append(f"Large position size change: {abs(recommended_size - current_size):.3f}")
        
        # Check for minimum position size
        if recommended_size <= self.config.min_position_size:
            warnings.append("Position size at minimum threshold")
        
        # Check for maximum position size
        if recommended_size >= self.config.max_position_size:
            warnings.append("Position size at maximum threshold")
        
        return warnings
    
    def _record_sizing_decision(self, result: PositionSizingResult) -> None:
        """Record sizing decision for analysis"""
        decision_record = {
            "timestamp": result.timestamp,
            "symbol": result.symbol,
            "recommended_size": result.recommended_size,
            "current_size": result.current_size,
            "size_change": result.size_change,
            "sizing_factors": result.sizing_factors,
            "confidence": result.confidence,
            "warnings": result.warnings
        }
        
        self.position_history.append(decision_record)
        
        # Keep only last 1000 records
        if len(self.position_history) > 1000:
            self.position_history.pop(0)
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing activity"""
        if not self.position_history:
            return {"message": "No sizing decisions recorded"}
        
        recent_decisions = self.position_history[-10:]  # Last 10 decisions
        
        return {
            "total_decisions": len(self.position_history),
            "recent_decisions": len(recent_decisions),
            "average_confidence": np.mean([d["confidence"] for d in recent_decisions]),
            "average_size_change": np.mean([abs(d["size_change"]) for d in recent_decisions]),
            "current_positions": len(self.current_positions),
            "last_update": self.last_update.isoformat()
        }
    
    def update_config(self, new_config: SizingConfig) -> None:
        """Update sizing configuration"""
        self.config = new_config
        self.logger.info("Position sizing configuration updated")
    
    def reset_positions(self) -> None:
        """Reset all position sizes"""
        self.current_positions.clear()
        self.logger.info("Position sizes reset")
    
    def get_position_recommendations(self, symbols: List[str],
                                   portfolio_data: PortfolioData,
                                   risk_metrics: RiskMetrics,
                                   market_data: Dict[str, Any] = None,
                                   active_events: List[RiskEvent] = None) -> Dict[str, PositionSizingResult]:
        """Get position size recommendations for multiple symbols"""
        recommendations = {}
        
        for symbol in symbols:
            try:
                result = self.calculate_position_size(
                    symbol, portfolio_data, risk_metrics, market_data, active_events
                )
                recommendations[symbol] = result
            except Exception as e:
                self.logger.error(f"Error calculating position size for {symbol}: {e}")
                # Create error result
                recommendations[symbol] = PositionSizingResult(
                    symbol=symbol,
                    recommended_size=0.0,
                    current_size=0.0,
                    size_change=0.0,
                    sizing_factors={},
                    risk_adjustment=0.0,
                    volatility_adjustment=0.0,
                    correlation_adjustment=0.0,
                    sentiment_adjustment=0.0,
                    event_adjustment=0.0,
                    drawdown_adjustment=0.0,
                    final_multiplier=0.0,
                    confidence=0.0,
                    timestamp=datetime.utcnow(),
                    warnings=[f"Error: {str(e)}"]
                )
        
        return recommendations
