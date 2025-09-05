#!/usr/bin/env python3
"""
Data Processor
Processes and transforms ingested market data
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from ..brokers.oanda_client import PriceData, CandleData
from ..config import CONFIG

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Data processing status"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"

@dataclass
class TechnicalIndicator:
    """Technical indicator data"""
    name: str
    value: float
    timestamp: datetime
    parameters: Dict[str, Any]

@dataclass
class MarketSnapshot:
    """Complete market snapshot"""
    timestamp: datetime
    prices: Dict[str, PriceData]
    indicators: Dict[str, List[TechnicalIndicator]]
    sentiment_score: Optional[float] = None
    volatility: Optional[float] = None

class DataProcessor:
    """Processes and transforms market data"""
    
    def __init__(self):
        """Initialize data processor"""
        self.status = ProcessingStatus.IDLE
        self.price_history: Dict[str, List[PriceData]] = {}
        self.candle_history: Dict[str, List[CandleData]] = {}
        self.max_history_size = 1000
        
        # Processing callbacks
        self.snapshot_callbacks: List[Callable] = []
        self.indicator_callbacks: List[Callable] = []
        
        # Technical indicators
        self.enabled_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'bollinger_bands',
            'atr_14', 'stochastic', 'williams_r'
        ]
    
    async def process_price_data(self, price_data: PriceData) -> None:
        """Process incoming price data"""
        try:
            self.status = ProcessingStatus.PROCESSING
            
            # Store price data
            instrument = price_data.instrument
            if instrument not in self.price_history:
                self.price_history[instrument] = []
            
            self.price_history[instrument].append(price_data)
            
            # Maintain history size
            if len(self.price_history[instrument]) > self.max_history_size:
                self.price_history[instrument] = self.price_history[instrument][-self.max_history_size:]
            
            # Calculate technical indicators
            indicators = await self._calculate_indicators(instrument)
            
            # Create market snapshot
            snapshot = await self._create_market_snapshot(price_data, indicators)
            
            # Call callbacks
            for callback in self.snapshot_callbacks:
                try:
                    await callback(snapshot)
                except Exception as e:
                    logger.error(f"Error in snapshot callback: {e}")
            
            self.status = ProcessingStatus.IDLE
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error(f"Error processing price data: {e}")
    
    async def process_candle_data(self, candle_data: CandleData) -> None:
        """Process incoming candle data"""
        try:
            self.status = ProcessingStatus.PROCESSING
            
            # Store candle data
            instrument = candle_data.time.strftime('%Y%m%d%H%M')
            if instrument not in self.candle_history:
                self.candle_history[instrument] = []
            
            self.candle_history[instrument].append(candle_data)
            
            # Maintain history size
            if len(self.candle_history[instrument]) > self.max_history_size:
                self.candle_history[instrument] = self.candle_history[instrument][-self.max_history_size:]
            
            self.status = ProcessingStatus.IDLE
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error(f"Error processing candle data: {e}")
    
    async def _calculate_indicators(self, instrument: str) -> Dict[str, List[TechnicalIndicator]]:
        """Calculate technical indicators for an instrument"""
        indicators = {}
        
        if instrument not in self.price_history:
            return indicators
        
        price_data = self.price_history[instrument]
        if len(price_data) < 20:  # Need minimum data for indicators
            return indicators
        
        # Convert to pandas DataFrame
        df = self._price_data_to_dataframe(price_data)
        
        # Calculate each indicator
        for indicator_name in self.enabled_indicators:
            try:
                indicator_values = await self._calculate_indicator(df, indicator_name)
                indicators[indicator_name] = indicator_values
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator_name}: {e}")
        
        return indicators
    
    def _price_data_to_dataframe(self, price_data: List[PriceData]) -> pd.DataFrame:
        """Convert price data to pandas DataFrame"""
        data = []
        for price in price_data:
            data.append({
                'timestamp': price.time,
                'bid': price.bid,
                'ask': price.ask,
                'spread': price.spread,
                'mid': (price.bid + price.ask) / 2
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    async def _calculate_indicator(self, df: pd.DataFrame, indicator_name: str) -> List[TechnicalIndicator]:
        """Calculate a specific technical indicator"""
        indicators = []
        
        if indicator_name == 'sma_20':
            values = self._calculate_sma(df['mid'], 20)
        elif indicator_name == 'sma_50':
            values = self._calculate_sma(df['mid'], 50)
        elif indicator_name == 'ema_12':
            values = self._calculate_ema(df['mid'], 12)
        elif indicator_name == 'ema_26':
            values = self._calculate_ema(df['mid'], 26)
        elif indicator_name == 'rsi_14':
            values = self._calculate_rsi(df['mid'], 14)
        elif indicator_name == 'macd':
            values = self._calculate_macd(df['mid'])
        elif indicator_name == 'bollinger_bands':
            values = self._calculate_bollinger_bands(df['mid'], 20, 2)
        elif indicator_name == 'atr_14':
            values = self._calculate_atr(df, 14)
        elif indicator_name == 'stochastic':
            values = self._calculate_stochastic(df, 14)
        elif indicator_name == 'williams_r':
            values = self._calculate_williams_r(df, 14)
        else:
            return indicators
        
        # Convert to TechnicalIndicator objects
        for i, (timestamp, value) in enumerate(zip(df.index, values)):
            if not np.isnan(value):
                indicator = TechnicalIndicator(
                    name=indicator_name,
                    value=float(value),
                    timestamp=timestamp,
                    parameters={}
                )
                indicators.append(indicator)
        
        return indicators
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, series: pd.Series) -> pd.Series:
        """Calculate MACD"""
        ema_12 = self._calculate_ema(series, 12)
        ema_26 = self._calculate_ema(series, 26)
        macd = ema_12 - ema_26
        return macd
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int, std_dev: float) -> pd.Series:
        """Calculate Bollinger Bands (return middle band)"""
        sma = self._calculate_sma(series, period)
        std = series.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return sma  # Return middle band
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['ask']  # Use ask as high
        low = df['bid']   # Use bid as low
        close = df['mid']  # Use mid as close
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic Oscillator"""
        high = df['ask']
        low = df['bid']
        close = df['mid']
        
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R"""
        high = df['ask']
        low = df['bid']
        close = df['mid']
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    async def _create_market_snapshot(self, 
                                    price_data: PriceData, 
                                    indicators: Dict[str, List[TechnicalIndicator]]) -> MarketSnapshot:
        """Create a complete market snapshot"""
        # Get current prices for all instruments
        current_prices = {price_data.instrument: price_data}
        
        # Calculate volatility (simplified)
        volatility = await self._calculate_volatility(price_data.instrument)
        
        # Calculate sentiment score (placeholder)
        sentiment_score = await self._calculate_sentiment_score(price_data)
        
        snapshot = MarketSnapshot(
            timestamp=price_data.time,
            prices=current_prices,
            indicators=indicators,
            sentiment_score=sentiment_score,
            volatility=volatility
        )
        
        return snapshot
    
    async def _calculate_volatility(self, instrument: str) -> Optional[float]:
        """Calculate volatility for an instrument"""
        if instrument not in self.price_history:
            return None
        
        price_data = self.price_history[instrument]
        if len(price_data) < 20:
            return None
        
        # Calculate price changes
        prices = [(p.bid + p.ask) / 2 for p in price_data[-20:]]
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        
        # Calculate standard deviation
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        return float(volatility)
    
    async def _calculate_sentiment_score(self, price_data: PriceData) -> Optional[float]:
        """Calculate sentiment score (placeholder)"""
        # This would typically integrate with news sentiment analysis
        # For now, return a simple score based on price movement
        if price_data.instrument not in self.price_history:
            return None
        
        price_data_list = self.price_history[price_data.instrument]
        if len(price_data_list) < 2:
            return None
        
        # Simple sentiment based on recent price movement
        current_price = (price_data.bid + price_data.ask) / 2
        previous_price = (price_data_list[-2].bid + price_data_list[-2].ask) / 2
        
        price_change = (current_price - previous_price) / previous_price
        sentiment = np.tanh(price_change * 100)  # Normalize to [-1, 1]
        
        return float(sentiment)
    
    def add_snapshot_callback(self, callback: Callable) -> None:
        """Add a callback for market snapshots"""
        self.snapshot_callbacks.append(callback)
    
    def add_indicator_callback(self, callback: Callable) -> None:
        """Add a callback for technical indicators"""
        self.indicator_callbacks.append(callback)
    
    def get_price_history(self, instrument: str) -> List[PriceData]:
        """Get price history for an instrument"""
        return self.price_history.get(instrument, [])
    
    def get_candle_history(self, time_key: str) -> List[CandleData]:
        """Get candle history for a time key"""
        return self.candle_history.get(time_key, [])
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            'status': self.status.value,
            'instruments_tracked': len(self.price_history),
            'total_price_points': sum(len(data) for data in self.price_history.values()),
            'total_candle_points': sum(len(data) for data in self.candle_history.values()),
            'enabled_indicators': self.enabled_indicators
        }

# Example usage and testing
async def test_data_processor():
    """Test the data processor"""
    from ..brokers.oanda_client import PriceData, CandleData
    
    processor = DataProcessor()
    
    # Add callbacks
    def snapshot_callback(snapshot):
        print(f"Snapshot: {snapshot.timestamp} - {len(snapshot.prices)} prices, {len(snapshot.indicators)} indicators")
    
    processor.add_snapshot_callback(snapshot_callback)
    
    # Simulate price data
    base_time = datetime.now()
    for i in range(50):
        price_data = PriceData(
            instrument="EUR_USD",
            time=base_time + timedelta(minutes=i),
            bid=1.1000 + i * 0.0001,
            ask=1.1002 + i * 0.0001,
            spread=0.0002
        )
        await processor.process_price_data(price_data)
    
    # Get status
    status = processor.get_status()
    print(f"Processor status: {status}")
    
    # Get price history
    history = processor.get_price_history("EUR_USD")
    print(f"Price history length: {len(history)}")

if __name__ == "__main__":
    asyncio.run(test_data_processor())
