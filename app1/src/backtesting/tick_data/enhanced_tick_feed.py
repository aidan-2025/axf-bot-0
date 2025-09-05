#!/usr/bin/env python3
"""
Enhanced Tick Data Feed with Variable Spread Simulation

Integrates variable spread simulation with Backtrader tick data feeds
for more realistic backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

try:
    import backtrader as bt
    from backtrader.feed import DataBase
    from backtrader.utils import date2num
except ImportError:
    bt = None
    DataBase = None

from .variable_spread_simulator import VariableSpreadSimulator, SpreadConfig, SpreadModel
from .simple_tick_feed import SimpleTickDataFeed

logger = logging.getLogger(__name__)


class EnhancedTickDataFeed(DataBase):
    """
    Enhanced tick data feed with variable spread simulation.
    
    This feed provides realistic tick-level data with variable spreads
    that change based on market conditions, time of day, and volume.
    """
    
    def __init__(self, tick_data: pd.DataFrame, symbol: str = "EURUSD", 
                 spread_config: Optional[SpreadConfig] = None,
                 enable_spread_simulation: bool = True):
        """
        Initialize the enhanced tick data feed.
        
        Args:
            tick_data: DataFrame with columns ['timestamp', 'bid', 'ask', 'volume']
            symbol: Symbol name for the data
            spread_config: Configuration for spread simulation
            enable_spread_simulation: Whether to enable variable spread simulation
        """
        super().__init__()
        
        self.tick_data = tick_data.copy()
        self.symbol = symbol
        self.current_index = 0
        self.total_ticks = len(tick_data)
        self.enable_spread_simulation = enable_spread_simulation
        
        # Initialize spread simulator
        if enable_spread_simulation:
            if spread_config is None:
                spread_config = SpreadConfig(symbol=symbol)
            self.spread_simulator = VariableSpreadSimulator(spread_config)
        else:
            self.spread_simulator = None
        
        # Validate data
        self._validate_tick_data()
        
        # Prepare data for Backtrader
        self._prepare_data()
        
        # Simulate spreads if enabled
        if enable_spread_simulation:
            self._simulate_variable_spreads()
        
        logger.info(f"Initialized EnhancedTickDataFeed for {symbol} with {self.total_ticks} ticks")
        if enable_spread_simulation:
            logger.info("Variable spread simulation enabled")
    
    def _validate_tick_data(self):
        """Validate the tick data format"""
        required_columns = ['timestamp', 'bid', 'ask']
        missing_columns = [col for col in required_columns if col not in self.tick_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if self.tick_data.empty:
            raise ValueError("Tick data is empty")
        
        # Check for valid prices
        if (self.tick_data['bid'] <= 0).any() or (self.tick_data['ask'] <= 0).any():
            raise ValueError("Found non-positive bid or ask prices")
    
    def _prepare_data(self):
        """Prepare data for Backtrader consumption"""
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.tick_data['timestamp']):
            self.tick_data['timestamp'] = pd.to_datetime(self.tick_data['timestamp'])
        
        # Sort by timestamp
        self.tick_data = self.tick_data.sort_values('timestamp').reset_index(drop=True)
        
        # Add volume if not present
        if 'volume' not in self.tick_data.columns:
            self.tick_data['volume'] = 1
        
        # Calculate mid prices
        self.tick_data['mid'] = (self.tick_data['bid'] + self.tick_data['ask']) / 2
        
        # Convert timestamps to Backtrader format
        self.tick_data['datetime_num'] = self.tick_data['timestamp'].apply(
            lambda x: date2num(x.to_pydatetime()) if hasattr(x, 'to_pydatetime') else date2num(x)
        )
        
        logger.info(f"Prepared {len(self.tick_data)} ticks for Backtrader")
    
    def _simulate_variable_spreads(self):
        """Simulate variable spreads for the entire dataset"""
        
        if self.spread_simulator is None:
            return
        
        logger.info("Simulating variable spreads...")
        
        # Simulate spreads for the entire dataset
        self.tick_data = self.spread_simulator.simulate_spreads_for_dataframe(
            self.tick_data,
            mid_price_column='mid',
            volume_column='volume'
        )
        
        # Update bid and ask prices with simulated spreads
        self.tick_data['bid'] = self.tick_data['simulated_bid']
        self.tick_data['ask'] = self.tick_data['simulated_ask']
        
        # Calculate new mid prices
        self.tick_data['mid'] = (self.tick_data['bid'] + self.tick_data['ask']) / 2
        
        # Calculate actual spreads
        self.tick_data['spread'] = self.tick_data['ask'] - self.tick_data['bid']
        
        # Validate spread quality
        validation_results = self.spread_simulator.validate_spread_quality(self.tick_data)
        
        if validation_results.get('validation_passed', False):
            logger.info(f"Spread simulation completed successfully. Quality score: {validation_results['quality_score']:.1f}")
        else:
            logger.warning(f"Spread simulation quality issues detected. Quality score: {validation_results['quality_score']:.1f}")
            logger.warning(f"Unrealistic spreads: {validation_results['unrealistic_spreads']}")
            logger.warning(f"Extreme changes: {validation_results['extreme_changes']}")
    
    def start(self):
        """Start the data feed"""
        self.current_index = 0
        logger.debug(f"Started enhanced tick data feed for {self.symbol}")
    
    def stop(self):
        """Stop the data feed"""
        logger.debug(f"Stopped enhanced tick data feed for {self.symbol}")
    
    def _load(self):
        """Load the next tick"""
        
        if self.current_index >= self.total_ticks:
            return False
        
        # Get current tick data
        tick = self.tick_data.iloc[self.current_index]
        
        # Set the datetime
        if hasattr(self.lines.datetime, '__setitem__'):
            self.lines.datetime[0] = tick['datetime_num']
        else:
            self.lines.datetime[0] = tick['datetime_num']
        
        # Set the OHLCV data using bid/ask prices
        if hasattr(self.lines.open, '__setitem__'):
            self.lines.open[0] = tick['bid']    # Use bid as open
            self.lines.high[0] = tick['ask']    # Use ask as high
            self.lines.low[0] = tick['bid']     # Use bid as low
            self.lines.close[0] = tick['ask']   # Use ask as close
            self.lines.volume[0] = tick['volume']
        else:
            # Fallback for when lines are not properly initialized
            self.lines.open[0] = tick['bid']
            self.lines.high[0] = tick['ask']
            self.lines.low[0] = tick['bid']
            self.lines.close[0] = tick['ask']
            self.lines.volume[0] = tick['volume']
        
        # Move to next tick
        self.current_index += 1
        
        return True
    
    def get_current_tick(self) -> Optional[Dict[str, Any]]:
        """Get current tick data with spread information"""
        if self.current_index == 0 or self.current_index > self.total_ticks:
            return None
        
        current_tick = self.tick_data.iloc[self.current_index - 1]
        
        result = {
            'timestamp': current_tick['timestamp'],
            'bid': current_tick['bid'],
            'ask': current_tick['ask'],
            'spread': current_tick['spread'],
            'mid': current_tick['mid'],
            'volume': current_tick['volume']
        }
        
        # Add spread simulation information if available
        if self.enable_spread_simulation and 'simulated_spread' in current_tick:
            result.update({
                'simulated_spread': current_tick['simulated_spread'],
                'market_condition': current_tick.get('market_condition', 'normal'),
                'trading_session': current_tick.get('trading_session', 'unknown')
            })
        
        return result
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the data feed"""
        if self.tick_data.empty:
            return {}
        
        spreads = self.tick_data['spread']
        
        info = {
            'symbol': self.symbol,
            'total_ticks': self.total_ticks,
            'start_time': self.tick_data['timestamp'].min(),
            'end_time': self.tick_data['timestamp'].max(),
            'avg_spread': spreads.mean(),
            'min_spread': spreads.min(),
            'max_spread': spreads.max(),
            'current_index': self.current_index,
            'progress_percentage': (self.current_index / self.total_ticks) * 100,
            'spread_simulation_enabled': self.enable_spread_simulation
        }
        
        # Add spread simulation statistics if available
        if self.enable_spread_simulation and self.spread_simulator:
            spread_stats = self.spread_simulator.get_spread_statistics()
            info.update({
                'spread_simulation_stats': spread_stats,
                'market_conditions': self.tick_data.get('market_condition', {}).value_counts().to_dict(),
                'trading_sessions': self.tick_data.get('trading_session', {}).value_counts().to_dict()
            })
        
        return info
    
    def update_market_conditions(self, volatility: float = None, liquidity: float = None):
        """Update market conditions for spread simulation"""
        if self.spread_simulator:
            self.spread_simulator.update_market_conditions(volatility, liquidity)
            logger.info(f"Updated market conditions - Volatility: {volatility}, Liquidity: {liquidity}")
    
    def get_spread_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of spread patterns"""
        if not self.enable_spread_simulation or self.tick_data.empty:
            return {}
        
        spreads = self.tick_data['spread']
        
        # Time-based analysis
        self.tick_data['hour'] = self.tick_data['timestamp'].dt.hour
        hourly_spreads = self.tick_data.groupby('hour')['spread'].agg(['mean', 'std', 'min', 'max'])
        
        # Session-based analysis
        if 'trading_session' in self.tick_data.columns:
            session_spreads = self.tick_data.groupby('trading_session')['spread'].agg(['mean', 'std', 'min', 'max'])
        else:
            session_spreads = pd.DataFrame()
        
        # Market condition analysis
        if 'market_condition' in self.tick_data.columns:
            condition_spreads = self.tick_data.groupby('market_condition')['spread'].agg(['mean', 'std', 'min', 'max'])
        else:
            condition_spreads = pd.DataFrame()
        
        return {
            'overall_stats': {
                'mean': spreads.mean(),
                'median': spreads.median(),
                'std': spreads.std(),
                'min': spreads.min(),
                'max': spreads.max(),
                'q25': spreads.quantile(0.25),
                'q75': spreads.quantile(0.75)
            },
            'hourly_analysis': hourly_spreads.to_dict(),
            'session_analysis': session_spreads.to_dict() if not session_spreads.empty else {},
            'condition_analysis': condition_spreads.to_dict() if not condition_spreads.empty else {},
            'spread_distribution': {
                'histogram': np.histogram(spreads, bins=20)[0].tolist(),
                'bin_edges': np.histogram(spreads, bins=20)[1].tolist()
            }
        }


class EnhancedTickDataFeedFactory:
    """Factory for creating enhanced tick data feeds"""
    
    @staticmethod
    def from_dataframe(tick_data: pd.DataFrame, symbol: str = "EURUSD", 
                      spread_config: Optional[SpreadConfig] = None,
                      enable_spread_simulation: bool = True) -> EnhancedTickDataFeed:
        """Create enhanced feed from DataFrame"""
        return EnhancedTickDataFeed(tick_data, symbol, spread_config, enable_spread_simulation)
    
    @staticmethod
    def create_sample_feed(symbol: str = "EURUSD", days: int = 7,
                          spread_model: SpreadModel = SpreadModel.HYBRID,
                          enable_spread_simulation: bool = True) -> EnhancedTickDataFeed:
        """Create sample enhanced tick data feed for testing"""
        
        from .tick_data_loader import TickDataLoader, TickDataConfig, TickDataFormat
        
        # Create sample data
        config = TickDataConfig(data_format=TickDataFormat.INFLUXDB, symbol=symbol)
        loader = TickDataLoader(config)
        tick_data = loader.create_sample_data(symbol, days)
        
        # Configure spread simulation
        spread_config = SpreadConfig(
            symbol=symbol,
            model=spread_model,
            base_spread=0.0001,  # 1 pip
            volatility_multiplier=1.5,
            time_of_day_factor=0.3,
            volume_factor=0.2
        )
        
        return EnhancedTickDataFeed(tick_data, symbol, spread_config, enable_spread_simulation)
    
    @staticmethod
    def create_high_volatility_feed(symbol: str = "EURUSD", days: int = 7) -> EnhancedTickDataFeed:
        """Create feed with high volatility spread simulation"""
        
        spread_config = SpreadConfig(
            symbol=symbol,
            model=SpreadModel.MARKET_CONDITIONS,
            market_volatility=2.0,  # High volatility
            liquidity_level=0.5,    # Low liquidity
            base_spread=0.0002,     # 2 pip base spread
            volatility_multiplier=2.0
        )
        
        # Create sample data
        from .tick_data_loader import TickDataLoader, TickDataConfig, TickDataFormat
        
        config = TickDataConfig(data_format=TickDataFormat.INFLUXDB, symbol=symbol)
        loader = TickDataLoader(config)
        tick_data = loader.create_sample_data(symbol, days)
        
        return EnhancedTickDataFeed(tick_data, symbol, spread_config, True)
    
    @staticmethod
    def create_low_volatility_feed(symbol: str = "EURUSD", days: int = 7) -> EnhancedTickDataFeed:
        """Create feed with low volatility spread simulation"""
        
        spread_config = SpreadConfig(
            symbol=symbol,
            model=SpreadModel.MARKET_CONDITIONS,
            market_volatility=0.5,  # Low volatility
            liquidity_level=2.0,    # High liquidity
            base_spread=0.00005,    # 0.5 pip base spread
            volatility_multiplier=0.5
        )
        
        # Create sample data
        from .tick_data_loader import TickDataLoader, TickDataConfig, TickDataFormat
        
        config = TickDataConfig(data_format=TickDataFormat.INFLUXDB, symbol=symbol)
        loader = TickDataLoader(config)
        tick_data = loader.create_sample_data(symbol, days)
        
        return EnhancedTickDataFeed(tick_data, symbol, spread_config, True)
