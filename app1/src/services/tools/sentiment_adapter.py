#!/usr/bin/env python3
"""
Sentiment Analysis Adapter Tool
Analyzes market sentiment from news, social media, and market data.
"""

import os
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SentimentAdapter:
    """Tool adapter for market sentiment analysis."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        
    def analyze_news_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles.
        
        Args:
            news_articles: List of news articles with title and description
            
        Returns:
            Dict containing sentiment analysis results
        """
        try:
            if not self.openai_api_key:
                return self._mock_sentiment_analysis(news_articles)
            
            # Use OpenAI for sentiment analysis
            return self._analyze_with_openai(news_articles)
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {"error": str(e), "sentiment": "neutral"}
    
    def _analyze_with_openai(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment using OpenAI."""
        # Prepare articles for analysis
        articles_text = []
        for article in news_articles[:10]:  # Limit to 10 articles
            articles_text.append(f"Title: {article.get('title', '')}\nDescription: {article.get('description', '')}")
        
        combined_text = "\n\n".join(articles_text)
        
        # Create prompt for sentiment analysis
        prompt = f"""
        Analyze the sentiment of these forex news articles. Consider:
        1. Overall market sentiment (bullish, bearish, neutral)
        2. Risk appetite (high, medium, low)
        3. Key themes and concerns
        4. Currency-specific sentiment
        
        Articles:
        {combined_text}
        
        Return JSON with:
        - overall_sentiment: "bullish", "bearish", or "neutral"
        - sentiment_score: -1 to 1
        - risk_appetite: "high", "medium", or "low"
        - key_themes: list of main themes
        - currency_sentiment: dict with currency-specific sentiment
        """
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert forex market sentiment analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        try:
            sentiment_data = json.loads(content)
        except json.JSONDecodeError:
            sentiment_data = self._mock_sentiment_analysis(news_articles)
        
        return {
            "source": "OpenAI Sentiment Analysis",
            "analysis": sentiment_data,
            "articles_analyzed": len(news_articles),
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_sentiment_analysis(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback mock sentiment analysis."""
        return {
            "source": "Mock Sentiment Analysis",
            "analysis": {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.1,
                "risk_appetite": "medium",
                "key_themes": ["central bank policy", "economic data", "market volatility"],
                "currency_sentiment": {
                    "EUR": "slightly bullish",
                    "USD": "neutral",
                    "GBP": "bearish",
                    "JPY": "neutral"
                }
            },
            "articles_analyzed": len(news_articles),
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_social_sentiment(self, instruments: List[str]) -> Dict[str, Any]:
        """
        Analyze social media sentiment for given instruments.
        
        Args:
            instruments: List of currency pairs
            
        Returns:
            Dict containing social sentiment analysis
        """
        try:
            # This would integrate with Twitter API or other social media APIs
            # For now, return a mock response
            social_sentiment = {}
            
            for instrument in instruments:
                social_sentiment[instrument] = {
                    "twitter_sentiment": 0.3,  # -1 to 1 scale
                    "reddit_sentiment": 0.1,
                    "overall_social_sentiment": 0.2,
                    "mentions_count": 150,
                    "trending": False,
                    "key_topics": ["inflation", "rate cuts", "economic growth"]
                }
            
            return {
                "source": "Social Media Analysis",
                "instruments": social_sentiment,
                "overall_social_sentiment": "Slightly Bullish",
                "trending_topics": ["Fed rate cuts", "EUR strength", "GBP weakness"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return {"error": str(e), "instruments": {}}
    
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """
        Get market fear and greed index.
        
        Returns:
            Dict containing fear/greed metrics
        """
        try:
            # This would integrate with fear/greed index APIs
            # For now, return a mock response
            return {
                "source": "Fear & Greed Index",
                "current_value": 45,  # 0-100 scale
                "label": "Fear",
                "previous_value": 52,
                "change": -7,
                "components": {
                    "volatility": 25,
                    "market_momentum": 30,
                    "junk_bond_demand": 40,
                    "safe_haven_demand": 60,
                    "put_call_options": 35,
                    "market_breadth": 50
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching fear/greed index: {e}")
            return {"error": str(e), "current_value": 50}
    
    def analyze_central_bank_sentiment(self) -> Dict[str, Any]:
        """
        Analyze central bank sentiment and policy stance.
        
        Returns:
            Dict containing central bank sentiment analysis
        """
        try:
            # This would analyze central bank statements and speeches
            # For now, return a mock response
            return {
                "source": "Central Bank Sentiment Analysis",
                "banks": {
                    "Fed": {
                        "sentiment": "dovish",
                        "policy_stance": "rate_cuts_expected",
                        "confidence": 0.8,
                        "key_concerns": ["inflation", "employment"]
                    },
                    "ECB": {
                        "sentiment": "neutral",
                        "policy_stance": "hold_rates",
                        "confidence": 0.7,
                        "key_concerns": ["growth", "inflation"]
                    },
                    "BoE": {
                        "sentiment": "hawkish",
                        "policy_stance": "rate_hikes_possible",
                        "confidence": 0.6,
                        "key_concerns": ["inflation", "fiscal_policy"]
                    }
                },
                "overall_policy_sentiment": "Mixed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing central bank sentiment: {e}")
            return {"error": str(e), "banks": {}}

