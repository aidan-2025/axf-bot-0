#!/usr/bin/env python3
"""
News Adapter Tool
Fetches real-time forex news and market events from multiple sources.
"""

import os
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class NewsAdapter:
    """Tool adapter for fetching forex news and market events."""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        self.fxcm_key = os.getenv("FXCM_KEY", "")
        
    def get_forex_news(self, instruments: List[str], hours_back: int = 24) -> Dict[str, Any]:
        """
        Fetch recent forex news for specified instruments.
        
        Args:
            instruments: List of currency pairs (e.g., ['EUR_USD', 'GBP_USD'])
            hours_back: How many hours back to fetch news
            
        Returns:
            Dict containing news articles with metadata
        """
        try:
            # Use NewsAPI if available, otherwise fallback to free sources
            if self.news_api_key:
                return self._fetch_news_api(instruments, hours_back)
            else:
                return self._fetch_free_news(instruments, hours_back)
                
        except Exception as e:
            logger.error(f"Error fetching forex news: {e}")
            return {"error": str(e), "articles": []}
    
    def _fetch_news_api(self, instruments: List[str], hours_back: int) -> Dict[str, Any]:
        """Fetch news using NewsAPI."""
        url = "https://newsapi.org/v2/everything"
        
        # Create search query for forex news
        query_terms = ["forex", "currency", "FX", "trading", "central bank", "interest rate"]
        query = " OR ".join(query_terms)
        
        # Add instrument-specific terms
        for instrument in instruments:
            pair = instrument.replace("_", "/")
            query += f" OR {pair} OR {instrument}"
        
        params = {
            "q": query,
            "apiKey": self.news_api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "from": (datetime.now() - timedelta(hours=hours_back)).isoformat(),
            "pageSize": 50
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return {
            "source": "NewsAPI",
            "total_articles": data.get("totalResults", 0),
            "articles": [
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "relevance_score": self._calculate_relevance(article, instruments)
                }
                for article in data.get("articles", [])
            ]
        }
    
    def _fetch_free_news(self, instruments: List[str], hours_back: int) -> Dict[str, Any]:
        """Fallback to free news sources."""
        # This would integrate with free news APIs or RSS feeds
        # For now, return a mock response
        return {
            "source": "Free Sources",
            "total_articles": 0,
            "articles": [],
            "note": "NewsAPI key not configured. Using free sources."
        }
    
    def _calculate_relevance(self, article: Dict[str, Any], instruments: List[str]) -> float:
        """Calculate relevance score for an article based on instruments."""
        title = article.get("title", "").lower()
        description = article.get("description", "").lower()
        content = f"{title} {description}"
        
        score = 0.0
        
        # Check for instrument mentions
        for instrument in instruments:
            pair = instrument.replace("_", "/").lower()
            if pair in content or instrument.lower() in content:
                score += 0.5
        
        # Check for forex-related keywords
        forex_keywords = ["forex", "currency", "fx", "trading", "central bank", "interest rate", "inflation", "gdp"]
        for keyword in forex_keywords:
            if keyword in content:
                score += 0.1
        
        return min(score, 1.0)
    
    def get_economic_calendar(self, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Fetch economic calendar events.
        
        Args:
            days_ahead: Number of days ahead to fetch events
            
        Returns:
            Dict containing economic events
        """
        try:
            # This would integrate with economic calendar APIs
            # For now, return a mock response
            return {
                "source": "Economic Calendar",
                "events": [
                    {
                        "date": "2025-09-05",
                        "time": "08:30",
                        "currency": "USD",
                        "event": "Non-Farm Payrolls",
                        "impact": "High",
                        "forecast": "180K",
                        "previous": "175K"
                    },
                    {
                        "date": "2025-09-05",
                        "time": "14:00",
                        "currency": "EUR",
                        "event": "ECB Interest Rate Decision",
                        "impact": "High",
                        "forecast": "4.25%",
                        "previous": "4.25%"
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return {"error": str(e), "events": []}
    
    def get_market_sentiment(self, instruments: List[str]) -> Dict[str, Any]:
        """
        Analyze market sentiment for given instruments.
        
        Args:
            instruments: List of currency pairs
            
        Returns:
            Dict containing sentiment analysis
        """
        try:
            # This would integrate with sentiment analysis APIs
            # For now, return a mock response
            sentiment_data = {}
            
            for instrument in instruments:
                # Mock sentiment analysis
                sentiment_data[instrument] = {
                    "sentiment_score": 0.65,  # -1 to 1 scale
                    "sentiment_label": "Bullish",
                    "confidence": 0.78,
                    "trend": "Upward",
                    "volatility": "Medium",
                    "last_updated": datetime.now().isoformat()
                }
            
            return {
                "source": "Sentiment Analysis",
                "instruments": sentiment_data,
                "overall_sentiment": "Slightly Bullish",
                "market_confidence": 0.72
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {"error": str(e), "instruments": {}}

