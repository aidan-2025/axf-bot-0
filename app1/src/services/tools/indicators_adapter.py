#!/usr/bin/env python3
"""
Technical Indicators Adapter Tool
Calculates and analyzes technical indicators for forex pairs.
"""

import os
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np

logger = logging.getLogger(__name__)


class IndicatorsAdapter:
    """Tool adapter for technical indicators calculation and analysis."""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        self.fxcm_key = os.getenv("FXCM_KEY", "")
        
    def calculate_technical_indicators(self, price_data: List[Dict[str, Any]], 
                                     indicators: List[str] = None) -> Dict[str, Any]:
        """
        Calculate technical indicators for given price data.
        
        Args:
            price_data: List of price data with OHLCV
            indicators: List of indicators to calculate
            
        Returns:
            Dict containing calculated indicators
        """
        try:
            if not price_data:
                return {"error": "No price data provided", "indicators": {}}
            
            if indicators is None:
                indicators = ["sma_20", "ema_12", "ema_26", "rsi", "macd", "bollinger_bands", "stochastic"]
            
            # Extract price arrays
            closes = [float(candle["close"]) for candle in price_data]
            highs = [float(candle["high"]) for candle in price_data]
            lows = [float(candle["low"]) for candle in price_data]
            volumes = [float(candle.get("volume", 0)) for candle in price_data]
            
            calculated_indicators = {}
            
            for indicator in indicators:
                if indicator == "sma_20":
                    calculated_indicators["sma_20"] = self._calculate_sma(closes, 20)
                elif indicator == "ema_12":
                    calculated_indicators["ema_12"] = self._calculate_ema(closes, 12)
                elif indicator == "ema_26":
                    calculated_indicators["ema_26"] = self._calculate_ema(closes, 26)
                elif indicator == "rsi":
                    calculated_indicators["rsi"] = self._calculate_rsi(closes, 14)
                elif indicator == "macd":
                    calculated_indicators["macd"] = self._calculate_macd(closes)
                elif indicator == "bollinger_bands":
                    calculated_indicators["bollinger_bands"] = self._calculate_bollinger_bands(closes, 20, 2)
                elif indicator == "stochastic":
                    calculated_indicators["stochastic"] = self._calculate_stochastic(highs, lows, closes, 14)
                elif indicator == "atr":
                    calculated_indicators["atr"] = self._calculate_atr(highs, lows, closes, 14)
                elif indicator == "adx":
                    calculated_indicators["adx"] = self._calculate_adx(highs, lows, closes, 14)
            
            return {
                "source": "Technical Indicators Calculator",
                "indicators": calculated_indicators,
                "data_points": len(price_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {"error": str(e), "indicators": {}}
    
    def _calculate_sma(self, prices: List[float], period: int) -> Dict[str, Any]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return {"values": [], "current": None, "signal": "insufficient_data"}
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)
        
        current_sma = sma_values[-1] if sma_values else None
        current_price = prices[-1]
        
        signal = "bullish" if current_price > current_sma else "bearish" if current_price < current_sma else "neutral"
        
        return {
            "values": sma_values,
            "current": current_sma,
            "signal": signal,
            "period": period
        }
    
    def _calculate_ema(self, prices: List[float], period: int) -> Dict[str, Any]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return {"values": [], "current": None, "signal": "insufficient_data"}
        
        multiplier = 2 / (period + 1)
        ema_values = [prices[0]]  # Start with first price
        
        for i in range(1, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        current_ema = ema_values[-1]
        current_price = prices[-1]
        
        signal = "bullish" if current_price > current_ema else "bearish" if current_price < current_ema else "neutral"
        
        return {
            "values": ema_values,
            "current": current_ema,
            "signal": signal,
            "period": period
        }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Dict[str, Any]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return {"values": [], "current": None, "signal": "insufficient_data"}
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        rsi_values = []
        for i in range(period - 1, len(gains)):
            avg_gain = sum(gains[i - period + 1:i + 1]) / period
            avg_loss = sum(losses[i - period + 1:i + 1]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        current_rsi = rsi_values[-1] if rsi_values else None
        
        if current_rsi is None:
            signal = "insufficient_data"
        elif current_rsi > 70:
            signal = "overbought"
        elif current_rsi < 30:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return {
            "values": rsi_values,
            "current": current_rsi,
            "signal": signal,
            "period": period
        }
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(prices) < slow:
            return {"macd": [], "signal": [], "histogram": [], "current": None, "trend": "insufficient_data"}
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        if not ema_fast["values"] or not ema_slow["values"]:
            return {"macd": [], "signal": [], "histogram": [], "current": None, "trend": "insufficient_data"}
        
        # Align the arrays
        min_len = min(len(ema_fast["values"]), len(ema_slow["values"]))
        macd_line = [ema_fast["values"][i] - ema_slow["values"][i] for i in range(min_len)]
        
        # Calculate signal line (EMA of MACD)
        signal_line = self._calculate_ema(macd_line, signal)
        
        # Calculate histogram
        histogram = []
        for i in range(min(len(macd_line), len(signal_line["values"]))):
            histogram.append(macd_line[i] - signal_line["values"][i])
        
        current_macd = macd_line[-1] if macd_line else None
        current_signal = signal_line["values"][-1] if signal_line["values"] else None
        
        if current_macd is None or current_signal is None:
            trend = "insufficient_data"
        elif current_macd > current_signal:
            trend = "bullish"
        else:
            trend = "bearish"
        
        return {
            "macd": macd_line,
            "signal": signal_line["values"],
            "histogram": histogram,
            "current": current_macd,
            "trend": trend
        }
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {"upper": [], "middle": [], "lower": [], "current": None, "signal": "insufficient_data"}
        
        sma = self._calculate_sma(prices, period)
        if not sma["values"]:
            return {"upper": [], "middle": [], "lower": [], "current": None, "signal": "insufficient_data"}
        
        upper_band = []
        lower_band = []
        
        for i in range(period - 1, len(prices)):
            price_slice = prices[i - period + 1:i + 1]
            std = np.std(price_slice)
            sma_value = sma["values"][i - period + 1]
            
            upper_band.append(sma_value + (std_dev * std))
            lower_band.append(sma_value - (std_dev * std))
        
        current_price = prices[-1]
        current_upper = upper_band[-1] if upper_band else None
        current_lower = lower_band[-1] if lower_band else None
        
        if current_upper is None or current_lower is None:
            signal = "insufficient_data"
        elif current_price > current_upper:
            signal = "overbought"
        elif current_price < current_lower:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return {
            "upper": upper_band,
            "middle": sma["values"],
            "lower": lower_band,
            "current": {
                "upper": current_upper,
                "middle": sma["current"],
                "lower": current_lower
            },
            "signal": signal,
            "period": period,
            "std_dev": std_dev
        }
    
    def _calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator."""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return {"k": [], "d": [], "current": None, "signal": "insufficient_data"}
        
        k_values = []
        for i in range(period - 1, len(closes)):
            high_slice = highs[i - period + 1:i + 1]
            low_slice = lows[i - period + 1:i + 1]
            close = closes[i]
            
            highest_high = max(high_slice)
            lowest_low = min(low_slice)
            
            k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            k_values.append(k)
        
        # Calculate %D (3-period SMA of %K)
        d_values = self._calculate_sma(k_values, 3)["values"]
        
        current_k = k_values[-1] if k_values else None
        current_d = d_values[-1] if d_values else None
        
        if current_k is None or current_d is None:
            signal = "insufficient_data"
        elif current_k > 80 and current_d > 80:
            signal = "overbought"
        elif current_k < 20 and current_d < 20:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return {
            "k": k_values,
            "d": d_values,
            "current": {"k": current_k, "d": current_d},
            "signal": signal,
            "period": period
        }
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, Any]:
        """Calculate Average True Range."""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return {"values": [], "current": None, "signal": "insufficient_data"}
        
        true_ranges = []
        for i in range(1, len(highs)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        atr_values = self._calculate_sma(true_ranges, period)["values"]
        current_atr = atr_values[-1] if atr_values else None
        
        return {
            "values": atr_values,
            "current": current_atr,
            "signal": "volatility_measure",
            "period": period
        }
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, Any]:
        """Calculate Average Directional Index."""
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return {"values": [], "current": None, "signal": "insufficient_data"}
        
        # Calculate directional movement
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i - 1]
            low_diff = lows[i - 1] - lows[i]
            
            plus_dm.append(max(high_diff, 0) if high_diff > low_diff else 0)
            minus_dm.append(max(low_diff, 0) if low_diff > high_diff else 0)
        
        # Calculate True Range
        atr_data = self._calculate_atr(highs, lows, closes, period)
        if not atr_data["values"]:
            return {"values": [], "current": None, "signal": "insufficient_data"}
        
        # Calculate DI+ and DI-
        di_plus = []
        di_minus = []
        
        for i in range(len(plus_dm)):
            if i < len(atr_data["values"]):
                di_plus.append((plus_dm[i] / atr_data["values"][i]) * 100)
                di_minus.append((minus_dm[i] / atr_data["values"][i]) * 100)
        
        # Calculate ADX
        adx_values = []
        for i in range(period - 1, len(di_plus)):
            dx = abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i]) * 100
            adx_values.append(dx)
        
        current_adx = adx_values[-1] if adx_values else None
        
        if current_adx is None:
            signal = "insufficient_data"
        elif current_adx > 25:
            signal = "strong_trend"
        elif current_adx > 20:
            signal = "moderate_trend"
        else:
            signal = "weak_trend"
        
        return {
            "values": adx_values,
            "current": current_adx,
            "signal": signal,
            "period": period
        }
    
    def get_support_resistance_levels(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify support and resistance levels.
        
        Args:
            price_data: List of price data with OHLCV
            
        Returns:
            Dict containing support and resistance levels
        """
        try:
            if not price_data:
                return {"error": "No price data provided", "levels": {}}
            
            highs = [float(candle["high"]) for candle in price_data]
            lows = [float(candle["low"]) for candle in price_data]
            closes = [float(candle["close"]) for candle in price_data]
            
            # Simple pivot point calculation
            current_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            current_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)
            current_close = closes[-1]
            
            # Pivot points
            pivot = (current_high + current_low + current_close) / 3
            r1 = 2 * pivot - current_low
            r2 = pivot + (current_high - current_low)
            s1 = 2 * pivot - current_high
            s2 = pivot - (current_high - current_low)
            
            return {
                "source": "Support/Resistance Calculator",
                "levels": {
                    "resistance_2": r2,
                    "resistance_1": r1,
                    "pivot_point": pivot,
                    "support_1": s1,
                    "support_2": s2
                },
                "current_price": current_close,
                "trend": "bullish" if current_close > pivot else "bearish" if current_close < pivot else "neutral",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance levels: {e}")
            return {"error": str(e), "levels": {}}

