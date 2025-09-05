"""
Technical indicators calculator using TA-Lib
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from ..models import TechnicalIndicator, Timeframe, OHLCVData

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """Calculator for technical indicators using TA-Lib"""
    
    def __init__(self):
        self.supported_indicators = {
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger_bands': self._calculate_bollinger_bands,
            'stochastic': self._calculate_stochastic,
            'atr': self._calculate_atr,
            'adx': self._calculate_adx,
            'williams_r': self._calculate_williams_r,
            'cci': self._calculate_cci,
            'mfi': self._calculate_mfi,
            'obv': self._calculate_obv,
            'vwap': self._calculate_vwap
        }
    
    def calculate_indicator(
        self,
        indicator_name: str,
        data: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        parameters: Dict[str, any]
    ) -> Optional[TechnicalIndicator]:
        """Calculate a single technical indicator"""
        try:
            if indicator_name not in self.supported_indicators:
                logger.error(f"Unsupported indicator: {indicator_name}")
                return None
            
            calculator_func = self.supported_indicators[indicator_name]
            values, timestamps = calculator_func(data, parameters)
            
            if values is None or len(values) == 0:
                logger.warning(f"No values calculated for {indicator_name}")
                return None
            
            return TechnicalIndicator(
                name=indicator_name,
                timeframe=timeframe,
                symbol=symbol,
                values=values.tolist() if isinstance(values, np.ndarray) else values,
                timestamps=timestamps,
                parameters=parameters,
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {str(e)}")
            return None
    
    def calculate_multiple_indicators(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        indicator_configs: List[Dict[str, any]]
    ) -> Dict[str, TechnicalIndicator]:
        """Calculate multiple indicators for the same dataset"""
        results = {}
        
        for config in indicator_configs:
            indicator_name = config['name']
            parameters = config.get('parameters', {})
            
            indicator = self.calculate_indicator(
                indicator_name, data, symbol, timeframe, parameters
            )
            
            if indicator:
                results[indicator_name] = indicator
        
        return results
    
    def _prepare_ohlcv_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare OHLCV data for TA-Lib"""
        try:
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns. Available: {data.columns.tolist()}")
                return None, None, None, None, None
            
            # Convert to numpy arrays
            open_prices = data['open'].values.astype(np.float64)
            high_prices = data['high'].values.astype(np.float64)
            low_prices = data['low'].values.astype(np.float64)
            close_prices = data['close'].values.astype(np.float64)
            volumes = data['volume'].values.astype(np.float64)
            
            return open_prices, high_prices, low_prices, close_prices, volumes
            
        except Exception as e:
            logger.error(f"Error preparing OHLCV data: {str(e)}")
            return None, None, None, None, None
    
    def _calculate_sma(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Simple Moving Average"""
        try:
            close_prices = data['close'].values.astype(np.float64)
            period = parameters.get('period', 20)
            
            sma_values = talib.SMA(close_prices, timeperiod=period)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return sma_values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            return None, []
    
    def _calculate_ema(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Exponential Moving Average"""
        try:
            close_prices = data['close'].values.astype(np.float64)
            period = parameters.get('period', 20)
            
            ema_values = talib.EMA(close_prices, timeperiod=period)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return ema_values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return None, []
    
    def _calculate_rsi(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Relative Strength Index"""
        try:
            close_prices = data['close'].values.astype(np.float64)
            period = parameters.get('period', 14)
            
            rsi_values = talib.RSI(close_prices, timeperiod=period)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return rsi_values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return None, []
    
    def _calculate_macd(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate MACD"""
        try:
            close_prices = data['close'].values.astype(np.float64)
            fast_period = parameters.get('fast_period', 12)
            slow_period = parameters.get('slow_period', 26)
            signal_period = parameters.get('signal_period', 9)
            
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices, 
                fastperiod=fast_period, 
                slowperiod=slow_period, 
                signalperiod=signal_period
            )
            
            # Return MACD line as the main indicator
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return macd, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return None, []
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Bollinger Bands (returns middle band)"""
        try:
            close_prices = data['close'].values.astype(np.float64)
            period = parameters.get('period', 20)
            std_dev = parameters.get('std_dev', 2)
            
            upper, middle, lower = talib.BBANDS(
                close_prices, 
                timeperiod=period, 
                nbdevup=std_dev, 
                nbdevdn=std_dev
            )
            
            # Return middle band as the main indicator
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return middle, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return None, []
    
    def _calculate_stochastic(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Stochastic Oscillator"""
        try:
            open_prices, high_prices, low_prices, close_prices, volumes = self._prepare_ohlcv_data(data)
            if open_prices is None:
                return None, []
            
            k_period = parameters.get('k_period', 14)
            d_period = parameters.get('d_period', 3)
            
            slowk, slowd = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=k_period,
                slowk_period=d_period,
                slowd_period=d_period
            )
            
            # Return %K as the main indicator
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return slowk, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            return None, []
    
    def _calculate_atr(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Average True Range"""
        try:
            open_prices, high_prices, low_prices, close_prices, volumes = self._prepare_ohlcv_data(data)
            if open_prices is None:
                return None, []
            
            period = parameters.get('period', 14)
            
            atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return atr_values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return None, []
    
    def _calculate_adx(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Average Directional Index"""
        try:
            open_prices, high_prices, low_prices, close_prices, volumes = self._prepare_ohlcv_data(data)
            if open_prices is None:
                return None, []
            
            period = parameters.get('period', 14)
            
            adx_values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return adx_values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return None, []
    
    def _calculate_williams_r(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Williams %R"""
        try:
            open_prices, high_prices, low_prices, close_prices, volumes = self._prepare_ohlcv_data(data)
            if open_prices is None:
                return None, []
            
            period = parameters.get('period', 14)
            
            williams_r = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=period)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return williams_r, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            return None, []
    
    def _calculate_cci(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Commodity Channel Index"""
        try:
            open_prices, high_prices, low_prices, close_prices, volumes = self._prepare_ohlcv_data(data)
            if open_prices is None:
                return None, []
            
            period = parameters.get('period', 14)
            
            cci_values = talib.CCI(high_prices, low_prices, close_prices, timeperiod=period)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return cci_values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating CCI: {str(e)}")
            return None, []
    
    def _calculate_mfi(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Money Flow Index"""
        try:
            open_prices, high_prices, low_prices, close_prices, volumes = self._prepare_ohlcv_data(data)
            if open_prices is None:
                return None, []
            
            period = parameters.get('period', 14)
            
            mfi_values = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=period)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return mfi_values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating MFI: {str(e)}")
            return None, []
    
    def _calculate_obv(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate On-Balance Volume"""
        try:
            close_prices = data['close'].values.astype(np.float64)
            volumes = data['volume'].values.astype(np.float64)
            
            obv_values = talib.OBV(close_prices, volumes)
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return obv_values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return None, []
    
    def _calculate_vwap(self, data: pd.DataFrame, parameters: Dict[str, any]) -> Tuple[Optional[np.ndarray], List[datetime]]:
        """Calculate Volume Weighted Average Price"""
        try:
            # VWAP = Σ(Price × Volume) / Σ(Volume)
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap_values = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
            
            timestamps = data.index.tolist() if hasattr(data.index, 'tolist') else data['timestamp'].tolist()
            
            return vwap_values.values, timestamps
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return None, []

