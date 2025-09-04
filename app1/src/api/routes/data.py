"""
Data API routes
Simple endpoints for market data
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import random
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/market")
async def get_market_data():
    """Get current market data"""
    try:
        # Mock market data
        currency_pairs = [
            {"symbol": "EURUSD", "currentPrice": 1.0850, "change24h": 0.0025, "changePercent24h": 0.23, "volume": 1250000},
            {"symbol": "GBPUSD", "currentPrice": 1.2650, "change24h": -0.0015, "changePercent24h": -0.12, "volume": 980000},
            {"symbol": "USDJPY", "currentPrice": 149.85, "change24h": 0.35, "changePercent24h": 0.23, "volume": 1100000},
            {"symbol": "USDCHF", "currentPrice": 0.8750, "change24h": -0.0010, "changePercent24h": -0.11, "volume": 750000},
            {"symbol": "AUDUSD", "currentPrice": 0.6520, "change24h": 0.0015, "changePercent24h": 0.23, "volume": 650000},
            {"symbol": "USDCAD", "currentPrice": 1.3650, "change24h": -0.0020, "changePercent24h": -0.15, "volume": 580000},
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
        
        return {
            "currencyPairs": currency_pairs,
            "sentiment": {
                "overall": 15.5,
                "news": 12.3,
                "social": 18.7,
                "technical": 15.5
            },
            "economicEvents": [
                {"name": "Non-Farm Payrolls", "time": "2024-01-05T13:30:00Z", "impact": "high", "currency": "USD"},
                {"name": "ECB Interest Rate Decision", "time": "2024-01-11T12:45:00Z", "impact": "high", "currency": "EUR"},
                {"name": "BoE Interest Rate Decision", "time": "2024-01-18T12:00:00Z", "impact": "high", "currency": "GBP"}
            ],
            "priceHistory": price_history
        }
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market data")
