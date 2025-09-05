"""
Data API routes
Simple endpoints for market data
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
import random
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/market")
async def get_market_data():
    """Get current market data"""
    try:
        # Mock market data - return array of MarketData objects
        market_data = [
            {
                "symbol": "EUR/USD", 
                "price": 1.08542, 
                "change24h": 0.00123, 
                "changePercent24h": 0.11, 
                "volume": 1234567,
                "high24h": 1.08750,
                "low24h": 1.08320,
                "timestamp": datetime.now().isoformat()
            },
            {
                "symbol": "GBP/USD", 
                "price": 1.26478, 
                "change24h": -0.00234, 
                "changePercent24h": -0.18, 
                "volume": 987654,
                "high24h": 1.26890,
                "low24h": 1.26210,
                "timestamp": datetime.now().isoformat()
            },
            {
                "symbol": "USD/JPY", 
                "price": 149.123, 
                "change24h": 0.456, 
                "changePercent24h": 0.31, 
                "volume": 2345678,
                "high24h": 149.890,
                "low24h": 148.750,
                "timestamp": datetime.now().isoformat()
            },
            {
                "symbol": "USD/CHF", 
                "price": 0.8750, 
                "change24h": -0.0010, 
                "changePercent24h": -0.11, 
                "volume": 750000,
                "high24h": 0.8765,
                "low24h": 0.8735,
                "timestamp": datetime.now().isoformat()
            },
            {
                "symbol": "AUD/USD", 
                "price": 0.6520, 
                "change24h": 0.0015, 
                "changePercent24h": 0.23, 
                "volume": 650000,
                "high24h": 0.6545,
                "low24h": 0.6495,
                "timestamp": datetime.now().isoformat()
            },
            {
                "symbol": "USD/CAD", 
                "price": 1.3650, 
                "change24h": -0.0020, 
                "changePercent24h": -0.15, 
                "volume": 580000,
                "high24h": 1.3680,
                "low24h": 1.3620,
                "timestamp": datetime.now().isoformat()
            },
        ]
        
        # Generate price history for the last 24 hours
        price_history = []
        now = datetime.now()
        for i in range(24):
            time = now - timedelta(hours=i)
            price_history.append({
                "time": time.isoformat(),
                "eurusd": 1.0850 + random.uniform(-0.01, 0.01),
                "gbpusd": 1.2650 + random.uniform(-0.01, 0.01),
                "usdjpy": 149.85 + random.uniform(-2, 2)
            })
        price_history.reverse()
        
        return market_data
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market data")

@router.get("/analyze/{pair}")
async def analyze_pair(pair: str):
    """Analyze a specific currency pair with technical indicators"""
    try:
        # Mock analysis data based on the pair
        base_data = {
            "EUR/USD": {
                "price": 1.08542,
                "change24h": 0.00123,
                "changePercent24h": 0.11,
                "volume": 1234567,
                "high24h": 1.08750,
                "low24h": 1.08320,
                "rsi": 65.2,
                "macd": 0.0012,
                "bollingerUpper": 1.08900,
                "bollingerLower": 1.08200,
                "support": 1.08300,
                "resistance": 1.08800,
                "trend": "bullish",
                "sentiment": "positive"
            },
            "GBP/USD": {
                "price": 1.26478,
                "change24h": -0.00234,
                "changePercent24h": -0.18,
                "volume": 987654,
                "high24h": 1.26890,
                "low24h": 1.26210,
                "rsi": 58.7,
                "macd": -0.0008,
                "bollingerUpper": 1.27100,
                "bollingerLower": 1.25800,
                "support": 1.26000,
                "resistance": 1.26800,
                "trend": "bearish",
                "sentiment": "negative"
            },
            "USD/JPY": {
                "price": 149.123,
                "change24h": 0.456,
                "changePercent24h": 0.31,
                "volume": 2345678,
                "high24h": 149.890,
                "low24h": 148.750,
                "rsi": 72.1,
                "macd": 0.0025,
                "bollingerUpper": 150.200,
                "bollingerLower": 148.100,
                "support": 148.500,
                "resistance": 149.500,
                "trend": "bullish",
                "sentiment": "positive"
            }
        }
        
        # Get data for the specific pair or use default
        pair_data = base_data.get(pair, {
            "price": 1.0000,
            "change24h": 0.0000,
            "changePercent24h": 0.00,
            "volume": 1000000,
            "high24h": 1.0100,
            "low24h": 0.9900,
            "rsi": 50.0,
            "macd": 0.0000,
            "bollingerUpper": 1.0200,
            "bollingerLower": 0.9800,
            "support": 0.9950,
            "resistance": 1.0050,
            "trend": "neutral",
            "sentiment": "neutral"
        })
        
        # Generate recommendations based on analysis
        recommendations = []
        if pair_data["trend"] == "bullish":
            recommendations.append("Strong upward momentum detected. Consider long positions with proper risk management.")
            if pair_data["rsi"] > 70:
                recommendations.append("RSI indicates overbought conditions. Monitor for potential reversal signals.")
        elif pair_data["trend"] == "bearish":
            recommendations.append("Downward pressure observed. Consider short positions or wait for reversal signals.")
            if pair_data["rsi"] < 30:
                recommendations.append("RSI indicates oversold conditions. Potential reversal opportunity.")
        else:
            recommendations.append("Market showing sideways movement. Consider range trading strategies.")
        
        if pair_data["macd"] > 0:
            recommendations.append("MACD shows bullish divergence. Trend continuation likely.")
        else:
            recommendations.append("MACD indicates bearish momentum. Exercise caution with long positions.")
        
        # Add the recommendations to the response
        pair_data["recommendations"] = recommendations
        pair_data["symbol"] = pair
        pair_data["timestamp"] = datetime.now().isoformat()
        
        return pair_data
        
    except Exception as e:
        logger.error(f"Error analyzing pair {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze pair {pair}")
