#!/usr/bin/env python3
"""
Lightweight Orchestrator

Plans and executes a multi-step strategy generation workflow:
1) Research (Perplexity Sonar-Pro)
2) Synthesis (OpenAI)
3) Strategy generation (OpenAI)

Extensible tool interface for: news/market/sentiment/indicators/DB.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.services.ai_service import AIService
from src.services.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """Agentic workflow orchestrator (lightweight)."""

    def __init__(self) -> None:
        self.ai = AIService()
        self.tools = ToolRegistry()

    def plan(self, instruments: List[str], timeframes: List[str], risk: str = "medium") -> Dict[str, Any]:
        """Produce a concrete plan of steps with rationale."""
        steps = [
            {"id": 1, "name": "research_news", "desc": "Fetch recent macro/news and summarize with citations"},
            {"id": 2, "name": "market_context", "desc": "Summarize volatility/sentiment/key levels"},
            {"id": 3, "name": "synthesis", "desc": "Identify themes and constraints for trading"},
            {"id": 4, "name": "strategy_candidates", "desc": "Draft 2–3 strategies with rules and params"},
            {"id": 5, "name": "feasibility", "desc": "Sanity checks and constraints validation"},
            {"id": 6, "name": "evaluation_plan", "desc": "Backtest setup and metrics thresholds"},
        ]
        return {
            "created_at": datetime.utcnow().isoformat(),
            "instruments": instruments,
            "timeframes": timeframes,
            "risk": risk,
            "steps": steps,
        }

    def run(self, instruments: List[str], timeframes: List[str], risk: str = "medium") -> Dict[str, Any]:
        """Execute the workflow and return structured outputs with citations."""
        run_log: List[Dict[str, Any]] = []

        # Step 1: Research news/themes using tools
        research_query = (
            f"Recent macro and FX news impacting {', '.join(instruments)} in timeframes {', '.join(timeframes)}. "
            f"Summarize drivers and tradable themes for a {risk} risk appetite. Provide citations."
        )
        research = self.ai.research(research_query)
        run_log.append({"step": "research_news", "output": research})

        # Step 1.5: Get additional tool data
        tool_analysis = self.tools.get_comprehensive_analysis(instruments)
        run_log.append({"step": "tool_analysis", "output": tool_analysis})

        # Step 2: Synthesis with OpenAI (enhanced with tool data)
        synthesis_prompt = (
            "You are an expert FX strategist. Using the following research summary with citations and tool analysis, "
            "identify 2-3 tradable themes and constraints (risk, liquidity, sessions).\n\n"
            f"RESEARCH:\n{research.get('content','')}\n\n"
            f"TOOL ANALYSIS:\n{tool_analysis}\n\n"
            "Return JSON with keys: themes[], constraints[], notes."
        )
        synthesis = self.ai.generate_strategy(synthesis_prompt)
        run_log.append({"step": "synthesis", "output": synthesis})

        # Step 3: Strategy candidates with rules/params
        gen_prompt = (
            "Propose 2-3 FX strategies as JSON with fields: name, description, instruments, timeframes, "
            "entry_rules[], exit_rules[], filters[], risk_model{risk_per_trade, stop, target}, "
            "parameters{name, default, range}, evaluation{lookback_days, metrics{profit_factor, win_rate, max_drawdown, min_trades}}.\n\n"
            f"CONTEXT THEMES:\n{synthesis.get('content','')}\n\n"
            f"TOOL DATA:\n{tool_analysis}"
        )
        strategies = self.ai.generate_strategy(gen_prompt)
        run_log.append({"step": "strategy_candidates", "output": strategies})

        return {
            "created_at": datetime.utcnow().isoformat(),
            "instruments": instruments,
            "timeframes": timeframes,
            "risk": risk,
            "results": {
                "research": research,
                "tool_analysis": tool_analysis,
                "synthesis": synthesis,
                "strategies": strategies,
            },
            "log": run_log,
        }


