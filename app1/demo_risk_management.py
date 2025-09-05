#!/usr/bin/env python3
"""
Risk Management System Demonstration

Comprehensive demonstration of the risk management and event avoidance system,
including economic calendar integration, sentiment monitoring, circuit breakers,
and real-time risk assessment.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import risk management components
from src.risk_management.models import (
    RiskConfig, RiskLevel, RiskEvent, RiskAction, TradingState,
    PortfolioData, PositionData, EconomicEventData, SentimentData,
    EventImpact, SentimentLevel
)
from src.risk_management.core.risk_manager import RiskManager, RiskManagerConfig
from src.risk_management.core.risk_engine import RiskEngine, RiskEngineConfig
from src.risk_management.event_integration.event_monitor import EventMonitor, EventMonitorConfig
from src.risk_management.event_integration.sentiment_monitor import SentimentMonitor, SentimentMonitorConfig
from src.risk_management.controls.circuit_breakers import CircuitBreaker, CircuitBreakerConfig
from src.risk_management.monitoring.risk_dashboard import RiskDashboard, RiskDashboardConfig
from src.risk_management.monitoring.alerting import RiskAlerting, AlertingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskManagementDemo:
    """Comprehensive risk management system demonstration"""
    
    def __init__(self):
        """Initialize the demonstration"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.risk_manager = None
        self.event_monitor = None
        self.sentiment_monitor = None
        self.circuit_breaker = None
        self.dashboard = None
        self.alerting = None
        
        # Demo data
        self.demo_portfolio = self._create_demo_portfolio()
        self.demo_events = self._create_demo_events()
        self.demo_sentiment = self._create_demo_sentiment()
        
        self.logger.info("Risk Management Demo initialized")
    
    def _create_demo_portfolio(self) -> PortfolioData:
        """Create demo portfolio data"""
        positions = [
            PositionData(
                currency_pair="EUR/USD",
                size=10000,
                entry_price=1.0850,
                current_price=1.0820,
                unrealized_pnl=-300,
                realized_pnl=0,
                timestamp=datetime.utcnow()
            ),
            PositionData(
                currency_pair="GBP/USD",
                size=5000,
                entry_price=1.2650,
                current_price=1.2620,
                unrealized_pnl=-150,
                realized_pnl=0,
                timestamp=datetime.utcnow()
            ),
            PositionData(
                currency_pair="USD/JPY",
                size=8000,
                entry_price=150.50,
                current_price=150.20,
                unrealized_pnl=-240,
                realized_pnl=0,
                timestamp=datetime.utcnow()
            )
        ]
        
        total_value = 100000
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in positions)
        
        return PortfolioData(
            total_value=total_value,
            available_margin=total_value * 0.5,
            used_margin=total_value * 0.1,
            total_pnl=total_pnl,
            unrealized_pnl=sum(p.unrealized_pnl for p in positions),
            realized_pnl=sum(p.realized_pnl for p in positions),
            max_drawdown=0.05,
            current_drawdown=abs(total_pnl) / total_value,
            positions=positions
        )
    
    def _create_demo_events(self) -> List[EconomicEventData]:
        """Create demo economic events"""
        current_time = datetime.utcnow()
        
        return [
            EconomicEventData(
                event_id="demo_1",
                title="US Non-Farm Payrolls",
                event_time=current_time + timedelta(hours=2),
                impact=EventImpact.HIGH,
                currency="USD",
                currency_pairs=["EUR/USD", "GBP/USD", "USD/JPY"],
                actual=None,
                forecast=200000,
                previous=195000,
                country="US",
                category="Employment",
                relevance_score=0.9
            ),
            EconomicEventData(
                event_id="demo_2",
                title="ECB Interest Rate Decision",
                event_time=current_time + timedelta(hours=6),
                impact=EventImpact.CRITICAL,
                currency="EUR",
                currency_pairs=["EUR/USD", "EUR/GBP", "EUR/JPY"],
                actual=None,
                forecast=4.25,
                previous=4.25,
                country="EU",
                category="Central Bank",
                relevance_score=0.95
            ),
            EconomicEventData(
                event_id="demo_3",
                title="UK GDP Growth Rate",
                event_time=current_time + timedelta(hours=12),
                impact=EventImpact.MEDIUM,
                currency="GBP",
                currency_pairs=["GBP/USD", "EUR/GBP"],
                actual=None,
                forecast=0.3,
                previous=0.2,
                country="UK",
                category="GDP",
                relevance_score=0.7
            )
        ]
    
    def _create_demo_sentiment(self) -> List[SentimentData]:
        """Create demo sentiment data"""
        current_time = datetime.utcnow()
        
        return [
            SentimentData(
                currency_pair="EUR/USD",
                sentiment_level=SentimentLevel.BEARISH,
                sentiment_score=-0.6,
                confidence=0.8,
                timestamp=current_time,
                source="news_analysis",
                factors={
                    "news_sentiment": -0.7,
                    "social_sentiment": -0.5,
                    "technical_sentiment": -0.6
                }
            ),
            SentimentData(
                currency_pair="GBP/USD",
                sentiment_level=SentimentLevel.VERY_BEARISH,
                sentiment_score=-0.9,
                confidence=0.9,
                timestamp=current_time,
                source="news_analysis",
                factors={
                    "news_sentiment": -0.9,
                    "social_sentiment": -0.8,
                    "technical_sentiment": -0.9
                }
            ),
            SentimentData(
                currency_pair="USD/JPY",
                sentiment_level=SentimentLevel.NEUTRAL,
                sentiment_score=0.1,
                confidence=0.6,
                timestamp=current_time,
                source="news_analysis",
                factors={
                    "news_sentiment": 0.2,
                    "social_sentiment": 0.0,
                    "technical_sentiment": 0.1
                }
            )
        ]
    
    async def demonstrate_risk_engine(self):
        """Demonstrate risk engine functionality"""
        print("\n" + "="*60)
        print("RISK ENGINE DEMONSTRATION")
        print("="*60)
        
        # Initialize risk engine
        config = RiskEngineConfig()
        risk_engine = RiskEngine(config)
        
        print(f"✓ Risk Engine initialized with {len(risk_engine.risk_thresholds)} thresholds")
        
        # Perform risk assessment
        print("\nPerforming risk assessment...")
        risk_state = await risk_engine.assess_risk(
            self.demo_portfolio,
            self.demo_events,
            self.demo_sentiment
        )
        
        print(f"✓ Risk Assessment Complete:")
        print(f"  - Risk Level: {risk_state.current_risk_level.value}")
        print(f"  - Trading State: {risk_state.trading_state.value}")
        print(f"  - Active Events: {len(risk_state.active_events)}")
        
        if risk_state.risk_metrics:
            metrics = risk_state.risk_metrics
            print(f"  - Portfolio Value: ${metrics.portfolio_value:,.2f}")
            print(f"  - Current Drawdown: {metrics.current_drawdown:.2%}")
            print(f"  - Max Drawdown: {metrics.max_drawdown:.2%}")
            print(f"  - VaR (95%): {metrics.var_95:.2%}")
            print(f"  - Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"  - Win Rate: {metrics.win_rate:.1%}")
        
        return risk_state
    
    async def demonstrate_event_monitoring(self):
        """Demonstrate event monitoring functionality"""
        print("\n" + "="*60)
        print("EVENT MONITORING DEMONSTRATION")
        print("="*60)
        
        # Initialize event monitor
        config = EventMonitorConfig()
        event_monitor = EventMonitor(config)
        
        print(f"✓ Event Monitor initialized")
        
        # Process demo events
        print("\nProcessing economic events...")
        await event_monitor._process_events(self.demo_events)
        
        # Get active events
        active_events = event_monitor.get_active_events()
        print(f"✓ Active Events: {len(active_events)}")
        
        for event in active_events:
            print(f"  - {event.title} ({event.impact.value}) at {event.event_time.strftime('%H:%M')}")
        
        # Get event risk summary
        summary = event_monitor.get_event_risk_summary()
        print(f"\nEvent Risk Summary:")
        print(f"  - High Impact Events: {summary['high_impact_events']}")
        print(f"  - Critical Impact Events: {summary['critical_impact_events']}")
        print(f"  - Risk Events Generated: {summary['risk_events_generated']}")
        
        return event_monitor
    
    async def demonstrate_sentiment_monitoring(self):
        """Demonstrate sentiment monitoring functionality"""
        print("\n" + "="*60)
        print("SENTIMENT MONITORING DEMONSTRATION")
        print("="*60)
        
        # Initialize sentiment monitor
        config = SentimentMonitorConfig()
        sentiment_monitor = SentimentMonitor(config)
        
        print(f"✓ Sentiment Monitor initialized")
        
        # Process demo sentiment
        print("\nProcessing sentiment data...")
        await sentiment_monitor._process_sentiment_data(self.demo_sentiment)
        
        # Get current sentiment
        current_sentiment = sentiment_monitor.get_current_sentiment()
        print(f"✓ Current Sentiment: {len(current_sentiment)} currency pairs")
        
        for pair, sentiment in current_sentiment.items():
            print(f"  - {pair}: {sentiment.sentiment_level.value} ({sentiment.sentiment_score:.2f})")
        
        # Get sentiment summary
        summary = sentiment_monitor.get_sentiment_summary()
        print(f"\nSentiment Summary:")
        print(f"  - Average Sentiment: {summary['average_sentiment']:.2f}")
        print(f"  - Bearish Pairs: {summary['bearish_pairs']}")
        print(f"  - Bullish Pairs: {summary['bullish_pairs']}")
        print(f"  - Risk Events Generated: {summary['risk_events_generated']}")
        
        return sentiment_monitor
    
    async def demonstrate_circuit_breakers(self):
        """Demonstrate circuit breaker functionality"""
        print("\n" + "="*60)
        print("CIRCUIT BREAKERS DEMONSTRATION")
        print("="*60)
        
        # Initialize circuit breaker
        config = CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker(config)
        
        print(f"✓ Circuit Breaker initialized")
        
        # Test with normal portfolio
        print("\nTesting with normal portfolio...")
        normal_portfolio = self.demo_portfolio
        
        from src.risk_management.models import RiskMetrics
        normal_metrics = RiskMetrics(
            portfolio_value=normal_portfolio.total_value,
            total_risk=0.0,
            risk_per_trade=0.0,
            max_drawdown=0.05,
            current_drawdown=0.05,
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
        
        result = circuit_breaker.check_circuit_breakers(normal_portfolio, normal_metrics)
        print(f"✓ Normal Portfolio Result: {result['overall_state']}")
        
        # Test with high-risk portfolio
        print("\nTesting with high-risk portfolio...")
        high_risk_portfolio = PortfolioData(
            total_value=100000,
            available_margin=50000,
            used_margin=10000,
            total_pnl=-20000,  # 20% loss
            unrealized_pnl=-20000,
            realized_pnl=0,
            max_drawdown=0.20,
            current_drawdown=0.20,
            positions=[]
        )
        
        high_risk_metrics = RiskMetrics(
            portfolio_value=high_risk_portfolio.total_value,
            total_risk=0.0,
            risk_per_trade=0.0,
            max_drawdown=0.20,
            current_drawdown=0.20,
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
        
        result = circuit_breaker.check_circuit_breakers(high_risk_portfolio, high_risk_metrics)
        print(f"✓ High-Risk Portfolio Result: {result['overall_state']}")
        
        # Show breaker details
        for breaker_name, breaker_result in result["breakers"].items():
            status = breaker_result["state"]
            triggered = breaker_result["triggered"]
            print(f"  - {breaker_name}: {status} {'(TRIGGERED)' if triggered else ''}")
        
        return circuit_breaker
    
    async def demonstrate_risk_dashboard(self):
        """Demonstrate risk dashboard functionality"""
        print("\n" + "="*60)
        print("RISK DASHBOARD DEMONSTRATION")
        print("="*60)
        
        # Initialize dashboard
        config = RiskDashboardConfig()
        dashboard = RiskDashboard(config)
        
        print(f"✓ Risk Dashboard initialized")
        
        # Create mock risk state
        from src.risk_management.models import RiskState, RiskMetrics
        
        risk_metrics = RiskMetrics(
            portfolio_value=100000,
            total_risk=0.0,
            risk_per_trade=0.0,
            max_drawdown=0.05,
            current_drawdown=0.05,
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
        
        risk_state = RiskState(
            trading_state=TradingState.ACTIVE,
            current_risk_level=RiskLevel.MEDIUM,
            risk_metrics=risk_metrics,
            portfolio_data=self.demo_portfolio
        )
        
        # Update dashboard
        print("\nUpdating dashboard...")
        await dashboard.update_dashboard(risk_state)
        
        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()
        print(f"✓ Dashboard Updated:")
        print(f"  - Risk Level: {dashboard_data['risk_level']}")
        print(f"  - Trading State: {dashboard_data['trading_state']}")
        
        if "risk_metrics" in dashboard_data:
            metrics = dashboard_data["risk_metrics"]
            print(f"  - Portfolio Value: {metrics['portfolio_value']['formatted']}")
            print(f"  - Current Drawdown: {metrics['current_drawdown']['formatted']}")
            print(f"  - VaR (95%): {metrics['var_95']['formatted']}")
            print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']['formatted']}")
        
        # Get summary
        summary = dashboard.get_summary()
        print(f"\nDashboard Summary:")
        print(f"  - Updates Count: {summary['updates_count']}")
        print(f"  - Data Points Collected: {summary['data_points_collected']}")
        print(f"  - Metrics History: {summary['metrics_history_count']}")
        
        return dashboard
    
    async def demonstrate_alerting_system(self):
        """Demonstrate alerting system functionality"""
        print("\n" + "="*60)
        print("ALERTING SYSTEM DEMONSTRATION")
        print("="*60)
        
        # Initialize alerting system
        config = AlertingConfig()
        alerting = RiskAlerting(config)
        
        print(f"✓ Alerting System initialized")
        
        # Create various alerts
        print("\nCreating alerts...")
        
        # High risk alert
        alert1_id = await alerting.create_alert(
            alert_type="risk_event",
            severity=RiskLevel.HIGH,
            message="High drawdown detected: 15%",
            data={"drawdown": 0.15, "threshold": 0.15}
        )
        print(f"✓ Created high risk alert: {alert1_id}")
        
        # Critical risk alert
        alert2_id = await alerting.create_alert(
            alert_type="circuit_breaker",
            severity=RiskLevel.CRITICAL,
            message="Circuit breaker triggered: Emergency stop",
            data={"breaker": "drawdown", "action": "emergency_stop"}
        )
        print(f"✓ Created critical risk alert: {alert2_id}")
        
        # Trading state change alert
        alert3_id = await alerting.create_alert(
            alert_type="trading_state_change",
            severity=RiskLevel.MEDIUM,
            message="Trading suspended due to risk conditions",
            data={"old_state": "active", "new_state": "suspended"}
        )
        print(f"✓ Created trading state alert: {alert3_id}")
        
        # Acknowledge an alert
        print("\nAcknowledging alert...")
        await alerting.acknowledge_alert(alert1_id, "demo_user")
        print(f"✓ Alert {alert1_id} acknowledged by demo_user")
        
        # Get active alerts
        active_alerts = await alerting.get_active_alerts()
        print(f"\nActive Alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  - {alert.alert_type}: {alert.message} ({alert.severity.value})")
        
        # Get alerting summary
        summary = alerting.get_alerting_summary()
        print(f"\nAlerting Summary:")
        print(f"  - Alerts Created: {summary['total_alerts_created']}")
        print(f"  - Alerts Delivered: {summary['alerts_delivered']}")
        print(f"  - Alerts Acknowledged: {summary['alerts_acknowledged']}")
        print(f"  - Active Alerts: {summary['active_alerts_count']}")
        
        return alerting
    
    async def demonstrate_integrated_system(self):
        """Demonstrate integrated risk management system"""
        print("\n" + "="*60)
        print("INTEGRATED RISK MANAGEMENT SYSTEM DEMONSTRATION")
        print("="*60)
        
        # Initialize integrated system
        config = RiskManagerConfig(
            enable_event_monitoring=True,
            enable_sentiment_monitoring=True,
            enable_circuit_breakers=True,
            enable_dashboard=True,
            enable_alerting=True
        )
        
        risk_manager = RiskManager(config)
        print(f"✓ Integrated Risk Management System initialized")
        
        # Perform comprehensive risk assessment
        print("\nPerforming comprehensive risk assessment...")
        risk_state = await risk_manager.force_risk_assessment()
        
        print(f"✓ Comprehensive Risk Assessment Complete:")
        print(f"  - Risk Level: {risk_state.current_risk_level.value}")
        print(f"  - Trading State: {risk_state.trading_state.value}")
        print(f"  - Active Events: {len(risk_state.active_events)}")
        print(f"  - Active Alerts: {len(risk_state.active_alerts)}")
        
        # Get risk summary
        summary = await risk_manager.get_risk_summary()
        print(f"\nRisk Summary:")
        print(f"  - Risk Assessments: {summary['performance']['risk_assessments_count']}")
        print(f"  - Alerts Generated: {summary['performance']['alerts_generated']}")
        print(f"  - Actions Taken: {summary['performance']['actions_taken']}")
        
        # Show component status
        if "event_monitor" in summary:
            event_summary = summary["event_monitor"]
            print(f"\nEvent Monitor:")
            print(f"  - Active Events: {event_summary['active_events_count']}")
            print(f"  - High Impact Events: {event_summary['high_impact_events']}")
        
        if "sentiment_monitor" in summary:
            sentiment_summary = summary["sentiment_monitor"]
            print(f"\nSentiment Monitor:")
            print(f"  - Currency Pairs: {sentiment_summary['currency_pairs']}")
            print(f"  - Average Sentiment: {sentiment_summary['average_sentiment']:.2f}")
            print(f"  - Bearish Pairs: {sentiment_summary['bearish_pairs']}")
        
        if "circuit_breakers" in summary:
            breaker_summary = summary["circuit_breakers"]
            print(f"\nCircuit Breakers:")
            print(f"  - Overall State: {breaker_summary['overall_state']}")
            print(f"  - Consecutive Losses: {breaker_summary['consecutive_losses']}")
        
        return risk_manager
    
    async def run_comprehensive_demo(self):
        """Run comprehensive risk management demonstration"""
        print("🚀 STARTING COMPREHENSIVE RISK MANAGEMENT DEMONSTRATION")
        print("="*80)
        
        try:
            # Demonstrate individual components
            await self.demonstrate_risk_engine()
            await self.demonstrate_event_monitoring()
            await self.demonstrate_sentiment_monitoring()
            await self.demonstrate_circuit_breakers()
            await self.demonstrate_risk_dashboard()
            await self.demonstrate_alerting_system()
            
            # Demonstrate integrated system
            await self.demonstrate_integrated_system()
            
            print("\n" + "="*80)
            print("✅ COMPREHENSIVE RISK MANAGEMENT DEMONSTRATION COMPLETED")
            print("="*80)
            
            print("\n🎯 KEY FEATURES DEMONSTRATED:")
            print("  ✓ Real-time risk assessment and monitoring")
            print("  ✓ Economic calendar event integration")
            print("  ✓ Market sentiment analysis and monitoring")
            print("  ✓ Circuit breaker protection mechanisms")
            print("  ✓ Real-time risk dashboard and visualization")
            print("  ✓ Comprehensive alerting and notification system")
            print("  ✓ Integrated risk management orchestration")
            
            print("\n🔧 TECHNICAL CAPABILITIES:")
            print("  ✓ Multi-factor risk assessment (portfolio, events, sentiment)")
            print("  ✓ Configurable risk thresholds and circuit breakers")
            print("  ✓ Real-time monitoring and alerting")
            print("  ✓ Comprehensive risk metrics calculation")
            print("  ✓ Event-driven risk management actions")
            print("  ✓ Scalable and modular architecture")
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive demo: {e}")
            print(f"\n❌ Demo failed with error: {e}")


async def main():
    """Main demonstration function"""
    demo = RiskManagementDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
