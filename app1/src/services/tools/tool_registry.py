#!/usr/bin/env python3
"""
Tool Registry
Manages and coordinates all tool adapters for the orchestrator.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .news_adapter import NewsAdapter
from .sentiment_adapter import SentimentAdapter
from .indicators_adapter import IndicatorsAdapter

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing all tool adapters."""
    
    def __init__(self):
        self.tools = {
            "news": NewsAdapter(),
            "sentiment": SentimentAdapter(),
            "indicators": IndicatorsAdapter()
        }
        logger.info("Tool registry initialized with adapters: %s", list(self.tools.keys()))
    
    def get_tool(self, tool_name: str):
        """Get a specific tool adapter."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())
    
    def execute_tool(self, tool_name: str, method: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a method on a specific tool.
        
        Args:
            tool_name: Name of the tool
            method: Method to execute
            **kwargs: Arguments for the method
            
        Returns:
            Dict containing the result
        """
        try:
            tool = self.get_tool(tool_name)
            if not tool:
                return {"error": f"Tool '{tool_name}' not found", "result": None}
            
            if not hasattr(tool, method):
                return {"error": f"Method '{method}' not found on tool '{tool_name}'", "result": None}
            
            method_func = getattr(tool, method)
            result = method_func(**kwargs)
            
            return {
                "tool": tool_name,
                "method": method,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}.{method}: {e}")
            return {
                "tool": tool_name,
                "method": method,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def get_news_data(self, instruments: List[str], hours_back: int = 24) -> Dict[str, Any]:
        """Get news data for specified instruments."""
        return self.execute_tool("news", "get_forex_news", instruments=instruments, hours_back=hours_back)
    
    def get_economic_calendar(self, days_ahead: int = 7) -> Dict[str, Any]:
        """Get economic calendar events."""
        return self.execute_tool("news", "get_economic_calendar", days_ahead=days_ahead)
    
    def get_market_sentiment(self, instruments: List[str]) -> Dict[str, Any]:
        """Get market sentiment for specified instruments."""
        return self.execute_tool("news", "get_market_sentiment", instruments=instruments)
    
    def analyze_news_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of news articles."""
        return self.execute_tool("sentiment", "analyze_news_sentiment", news_articles=news_articles)
    
    def analyze_social_sentiment(self, instruments: List[str]) -> Dict[str, Any]:
        """Analyze social media sentiment."""
        return self.execute_tool("sentiment", "analyze_social_sentiment", instruments=instruments)
    
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get fear and greed index."""
        return self.execute_tool("sentiment", "get_fear_greed_index")
    
    def analyze_central_bank_sentiment(self) -> Dict[str, Any]:
        """Analyze central bank sentiment."""
        return self.execute_tool("sentiment", "analyze_central_bank_sentiment")
    
    def calculate_technical_indicators(self, price_data: List[Dict[str, Any]], 
                                     indicators: List[str] = None) -> Dict[str, Any]:
        """Calculate technical indicators."""
        return self.execute_tool("indicators", "calculate_technical_indicators", 
                               price_data=price_data, indicators=indicators)
    
    def get_support_resistance_levels(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get support and resistance levels."""
        return self.execute_tool("indicators", "get_support_resistance_levels", price_data=price_data)
    
    def get_comprehensive_analysis(self, instruments: List[str], 
                                 price_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Get comprehensive analysis using all available tools.
        
        Args:
            instruments: List of currency pairs
            price_data: Optional price data for technical analysis
            
        Returns:
            Dict containing comprehensive analysis
        """
        try:
            analysis = {
                "instruments": instruments,
                "timestamp": datetime.now().isoformat(),
                "tools_used": [],
                "results": {}
            }
            
            # Get news data
            news_result = self.get_news_data(instruments, hours_back=24)
            analysis["tools_used"].append("news")
            analysis["results"]["news"] = news_result
            
            # Analyze news sentiment
            if news_result.get("success") and news_result.get("result", {}).get("articles"):
                sentiment_result = self.analyze_news_sentiment(
                    news_result["result"]["articles"]
                )
                analysis["tools_used"].append("sentiment")
                analysis["results"]["news_sentiment"] = sentiment_result
            
            # Get market sentiment
            market_sentiment = self.get_market_sentiment(instruments)
            analysis["tools_used"].append("market_sentiment")
            analysis["results"]["market_sentiment"] = market_sentiment
            
            # Get social sentiment
            social_sentiment = self.analyze_social_sentiment(instruments)
            analysis["tools_used"].append("social_sentiment")
            analysis["results"]["social_sentiment"] = social_sentiment
            
            # Get fear/greed index
            fear_greed = self.get_fear_greed_index()
            analysis["tools_used"].append("fear_greed")
            analysis["results"]["fear_greed"] = fear_greed
            
            # Get central bank sentiment
            cb_sentiment = self.analyze_central_bank_sentiment()
            analysis["tools_used"].append("central_bank_sentiment")
            analysis["results"]["central_bank_sentiment"] = cb_sentiment
            
            # Technical analysis if price data provided
            if price_data:
                tech_indicators = self.calculate_technical_indicators(price_data)
                analysis["tools_used"].append("technical_indicators")
                analysis["results"]["technical_indicators"] = tech_indicators
                
                support_resistance = self.get_support_resistance_levels(price_data)
                analysis["tools_used"].append("support_resistance")
                analysis["results"]["support_resistance"] = support_resistance
            
            # Get economic calendar
            economic_calendar = self.get_economic_calendar(days_ahead=7)
            analysis["tools_used"].append("economic_calendar")
            analysis["results"]["economic_calendar"] = economic_calendar
            
            analysis["success"] = True
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "error": str(e),
                "instruments": instruments,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

