#!/usr/bin/env python3
"""
Variable Spread Simulation Module

Simulates realistic variable spreads for backtesting using historical spread data
or statistical models. Integrates with Backtrader execution logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SpreadModel(Enum):
    """Available spread simulation models"""
    HISTORICAL = "historical"
    STATISTICAL = "statistical"
    HYBRID = "hybrid"
    MARKET_CONDITIONS = "market_conditions"


@dataclass
class SpreadConfig:
    """Configuration for spread simulation"""
    
    # Model configuration
    model: SpreadModel = SpreadModel.HYBRID
    symbol: str = "EURUSD"
    
    # Historical data configuration
    historical_data_path: Optional[str] = None
    lookback_days: int = 30
    
    # Statistical model parameters
    base_spread: float = 0.0001  # 1 pip base spread
    volatility_multiplier: float = 1.5
    time_of_day_factor: float = 0.3
    volume_factor: float = 0.2
    
    # Market conditions
    market_volatility: float = 1.0  # 1.0 = normal, >1.0 = high volatility
    liquidity_level: float = 1.0    # 1.0 = normal, <1.0 = low liquidity
    
    # Time-based adjustments
    trading_hours: Dict[str, Tuple[int, int]] = None  # Hour ranges for different sessions
    session_multipliers: Dict[str, float] = None     # Spread multipliers per session
    
    # Quality control
    min_spread: float = 0.00005  # 0.5 pips minimum
    max_spread: float = 0.001    # 10 pips maximum
    max_spread_change: float = 0.00005  # Max change per tick
    
    def __post_init__(self):
        if self.trading_hours is None:
            self.trading_hours = {
                'asian': (0, 8),
                'european': (8, 16),
                'american': (16, 24)
            }
        
        if self.session_multipliers is None:
            self.session_multipliers = {
                'asian': 1.2,
                'european': 1.0,
                'american': 1.1
            }


@dataclass
class SpreadData:
    """Spread data point"""
    timestamp: datetime
    spread: float
    bid: float
    ask: float
    mid: float
    volume: float
    market_condition: str
    session: str


class VariableSpreadSimulator:
    """
    Simulates realistic variable spreads for backtesting.
    
    This module provides multiple approaches to spread simulation:
    1. Historical-based: Uses actual historical spread data
    2. Statistical: Uses statistical models based on market conditions
    3. Hybrid: Combines historical patterns with statistical adjustments
    4. Market conditions: Adjusts spreads based on current market state
    """
    
    def __init__(self, config: SpreadConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.historical_spreads: Optional[pd.DataFrame] = None
        self.spread_model: Optional[Callable] = None
        self.scaler = StandardScaler()
        self.kmeans_model: Optional[KMeans] = None
        
        # Load historical data if available
        if config.historical_data_path:
            self._load_historical_data()
        
        # Initialize the spread model
        self._initialize_model()
        
        logger.info(f"Initialized VariableSpreadSimulator for {config.symbol} using {config.model.value} model")
    
    def _load_historical_data(self):
        """Load historical spread data"""
        try:
            if self.config.historical_data_path.endswith('.csv'):
                self.historical_spreads = pd.read_csv(self.config.historical_data_path)
            elif self.config.historical_data_path.endswith('.parquet'):
                self.historical_spreads = pd.read_parquet(self.config.historical_data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.config.historical_data_path}")
            
            # Preprocess historical data
            self._preprocess_historical_data()
            
            logger.info(f"Loaded {len(self.historical_spreads)} historical spread records")
            
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
            self.historical_spreads = None
    
    def _preprocess_historical_data(self):
        """Preprocess historical spread data"""
        if self.historical_spreads is None or self.historical_spreads.empty:
            return
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' in self.historical_spreads.columns:
            self.historical_spreads['timestamp'] = pd.to_datetime(self.historical_spreads['timestamp'])
        else:
            # Create timestamp if not present
            self.historical_spreads['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(days=self.config.lookback_days),
                periods=len(self.historical_spreads),
                freq='1S'
            )
        
        # Calculate spread if not present
        if 'spread' not in self.historical_spreads.columns:
            if 'bid' in self.historical_spreads.columns and 'ask' in self.historical_spreads.columns:
                self.historical_spreads['spread'] = (
                    self.historical_spreads['ask'] - self.historical_spreads['bid']
                )
            else:
                logger.warning("Cannot calculate spread from historical data")
                return
        
        # Add session information
        self.historical_spreads['session'] = self.historical_spreads['timestamp'].apply(
            self._get_trading_session
        )
        
        # Add market condition indicators
        self.historical_spreads['market_condition'] = self._classify_market_condition(
            self.historical_spreads
        )
        
        # Filter to recent data
        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_days)
        self.historical_spreads = self.historical_spreads[
            self.historical_spreads['timestamp'] >= cutoff_date
        ]
    
    def _get_trading_session(self, timestamp: datetime) -> str:
        """Determine trading session based on timestamp"""
        hour = timestamp.hour
        
        for session, (start_hour, end_hour) in self.config.trading_hours.items():
            if start_hour <= hour < end_hour:
                return session
        
        return 'asian'  # Default to asian session
    
    def _classify_market_condition(self, data: pd.DataFrame) -> pd.Series:
        """Classify market conditions based on spread patterns"""
        if 'spread' not in data.columns:
            return pd.Series(['normal'] * len(data), index=data.index)
        
        # Calculate rolling statistics
        window = min(100, len(data) // 10)  # Adaptive window size
        rolling_mean = data['spread'].rolling(window=window, min_periods=1).mean()
        rolling_std = data['spread'].rolling(window=window, min_periods=1).std()
        
        # Classify conditions
        conditions = []
        for i in range(len(data)):
            if i < window:
                conditions.append('normal')
            else:
                current_spread = data['spread'].iloc[i]
                mean_spread = rolling_mean.iloc[i]
                std_spread = rolling_std.iloc[i]
                
                if current_spread > mean_spread + 2 * std_spread:
                    conditions.append('high_volatility')
                elif current_spread < mean_spread - std_spread:
                    conditions.append('low_volatility')
                else:
                    conditions.append('normal')
        
        return pd.Series(conditions, index=data.index)
    
    def _initialize_model(self):
        """Initialize the spread simulation model"""
        if self.config.model == SpreadModel.HISTORICAL:
            self._initialize_historical_model()
        elif self.config.model == SpreadModel.STATISTICAL:
            self._initialize_statistical_model()
        elif self.config.model == SpreadModel.HYBRID:
            self._initialize_hybrid_model()
        elif self.config.model == SpreadModel.MARKET_CONDITIONS:
            self._initialize_market_conditions_model()
        else:
            raise ValueError(f"Unknown spread model: {self.config.model}")
    
    def _initialize_historical_model(self):
        """Initialize historical-based model"""
        if self.historical_spreads is None or self.historical_spreads.empty:
            logger.warning("No historical data available, falling back to statistical model")
            self._initialize_statistical_model()
            return
        
        # Create lookup tables for different conditions
        self.spread_lookup = {}
        
        for session in self.historical_spreads['session'].unique():
            for condition in self.historical_spreads['market_condition'].unique():
                key = (session, condition)
                session_data = self.historical_spreads[
                    (self.historical_spreads['session'] == session) &
                    (self.historical_spreads['market_condition'] == condition)
                ]
                
                if not session_data.empty:
                    self.spread_lookup[key] = {
                        'spreads': session_data['spread'].values,
                        'mean': session_data['spread'].mean(),
                        'std': session_data['spread'].std(),
                        'percentiles': session_data['spread'].quantile([0.1, 0.5, 0.9]).values
                    }
        
        logger.info(f"Initialized historical model with {len(self.spread_lookup)} condition combinations")
    
    def _initialize_statistical_model(self):
        """Initialize statistical model"""
        # This model uses statistical distributions based on market conditions
        self.spread_model = self._statistical_spread_function
        logger.info("Initialized statistical spread model")
    
    def _initialize_hybrid_model(self):
        """Initialize hybrid model combining historical and statistical approaches"""
        if self.historical_spreads is None or self.historical_spreads.empty:
            logger.warning("No historical data for hybrid model, using statistical model")
            self._initialize_statistical_model()
            return
        
        # Train a machine learning model on historical data
        self._train_ml_model()
        self.spread_model = self._hybrid_spread_function
        logger.info("Initialized hybrid spread model")
    
    def _initialize_market_conditions_model(self):
        """Initialize market conditions-based model"""
        self.spread_model = self._market_conditions_spread_function
        logger.info("Initialized market conditions spread model")
    
    def _train_ml_model(self):
        """Train machine learning model on historical data"""
        if self.historical_spreads is None or self.historical_spreads.empty:
            return
        
        # Prepare features
        features = []
        for _, row in self.historical_spreads.iterrows():
            feature_vector = [
                row['timestamp'].hour,  # Hour of day
                row['timestamp'].weekday(),  # Day of week
                row['volume'] if 'volume' in row else 1.0,  # Volume
                self.config.market_volatility,  # Market volatility
                self.config.liquidity_level,  # Liquidity level
            ]
            features.append(feature_vector)
        
        # Scale features
        features_array = np.array(features)
        self.scaler.fit(features_array)
        scaled_features = self.scaler.transform(features_array)
        
        # Train K-means clustering
        n_clusters = min(5, len(scaled_features) // 10)
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans_model.fit(scaled_features)
        
        # Calculate cluster statistics
        self.cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = self.kmeans_model.labels_ == i
            cluster_spreads = self.historical_spreads[cluster_mask]['spread']
            
            self.cluster_stats[i] = {
                'mean': cluster_spreads.mean(),
                'std': cluster_spreads.std(),
                'min': cluster_spreads.min(),
                'max': cluster_spreads.max()
            }
    
    def _statistical_spread_function(self, timestamp: datetime, mid_price: float, 
                                   volume: float = 1.0, **kwargs) -> float:
        """Statistical spread calculation"""
        
        # Base spread
        spread = self.config.base_spread
        
        # Time of day adjustment
        hour = timestamp.hour
        session = self._get_trading_session(timestamp)
        session_multiplier = self.config.session_multipliers.get(session, 1.0)
        spread *= session_multiplier
        
        # Volatility adjustment
        spread *= (1 + self.config.volatility_multiplier * self.config.market_volatility)
        
        # Volume adjustment (higher volume = tighter spreads)
        volume_factor = 1 - (self.config.volume_factor * min(volume / 100, 1.0))
        spread *= volume_factor
        
        # Add some randomness
        noise = np.random.normal(0, spread * 0.1)
        spread += noise
        
        # Apply constraints
        spread = max(self.config.min_spread, min(self.config.max_spread, spread))
        
        return spread
    
    def _hybrid_spread_function(self, timestamp: datetime, mid_price: float, 
                              volume: float = 1.0, **kwargs) -> float:
        """Hybrid spread calculation combining historical and statistical models"""
        
        # Get statistical base
        statistical_spread = self._statistical_spread_function(timestamp, mid_price, volume, **kwargs)
        
        # If we have historical data, blend with historical patterns
        if self.historical_spreads is not None and not self.historical_spreads.empty:
            session = self._get_trading_session(timestamp)
            condition = self._classify_market_condition_single(timestamp, volume)
            
            key = (session, condition)
            if key in self.spread_lookup:
                historical_data = self.spread_lookup[key]
                # Sample from historical distribution
                historical_spread = np.random.choice(historical_data['spreads'])
                
                # Blend statistical and historical (70% historical, 30% statistical)
                spread = 0.7 * historical_spread + 0.3 * statistical_spread
            else:
                spread = statistical_spread
        else:
            spread = statistical_spread
        
        # Apply constraints
        spread = max(self.config.min_spread, min(self.config.max_spread, spread))
        
        return spread
    
    def _market_conditions_spread_function(self, timestamp: datetime, mid_price: float, 
                                         volume: float = 1.0, **kwargs) -> float:
        """Market conditions-based spread calculation"""
        
        # Base spread adjusted for market conditions
        spread = self.config.base_spread
        
        # Market volatility impact
        volatility_impact = self.config.market_volatility ** 2
        spread *= (1 + volatility_impact)
        
        # Liquidity impact
        liquidity_impact = 1 / max(self.config.liquidity_level, 0.1)
        spread *= liquidity_impact
        
        # Time-based adjustments
        hour = timestamp.hour
        if 8 <= hour < 16:  # European session
            spread *= 0.9  # Tighter spreads during active hours
        elif 16 <= hour < 24:  # American session
            spread *= 1.0
        else:  # Asian session
            spread *= 1.2  # Wider spreads during less active hours
        
        # Volume impact
        if volume > 50:
            spread *= 0.8  # Tighter spreads with high volume
        elif volume < 10:
            spread *= 1.3  # Wider spreads with low volume
        
        # Add realistic noise
        noise_factor = np.random.normal(1.0, 0.1)
        spread *= noise_factor
        
        # Apply constraints
        spread = max(self.config.min_spread, min(self.config.max_spread, spread))
        
        return spread
    
    def _classify_market_condition_single(self, timestamp: datetime, volume: float) -> str:
        """Classify market condition for a single timestamp"""
        # Simple classification based on time and volume
        hour = timestamp.hour
        
        if hour in [8, 9, 14, 15]:  # Market open/close times
            return 'high_volatility'
        elif volume > 50:
            return 'high_volatility'
        elif volume < 10:
            return 'low_volatility'
        else:
            return 'normal'
    
    def simulate_spread(self, timestamp: datetime, mid_price: float, 
                       volume: float = 1.0, **kwargs) -> SpreadData:
        """
        Simulate spread for a given timestamp and market conditions.
        
        Args:
            timestamp: Current timestamp
            mid_price: Current mid price
            volume: Current volume
            **kwargs: Additional parameters
            
        Returns:
            SpreadData object with simulated spread information
        """
        
        # Calculate spread using the configured model
        spread = self.spread_model(timestamp, mid_price, volume, **kwargs)
        
        # Calculate bid and ask prices
        bid_price = mid_price - (spread / 2)
        ask_price = mid_price + (spread / 2)
        
        # Determine market condition and session
        market_condition = self._classify_market_condition_single(timestamp, volume)
        session = self._get_trading_session(timestamp)
        
        return SpreadData(
            timestamp=timestamp,
            spread=spread,
            bid=bid_price,
            ask=ask_price,
            mid=mid_price,
            volume=volume,
            market_condition=market_condition,
            session=session
        )
    
    def simulate_spreads_for_dataframe(self, data: pd.DataFrame, 
                                     mid_price_column: str = 'mid',
                                     volume_column: str = 'volume') -> pd.DataFrame:
        """
        Simulate spreads for an entire DataFrame.
        
        Args:
            data: DataFrame with timestamp and price data
            mid_price_column: Column name for mid prices
            volume_column: Column name for volumes
            
        Returns:
            DataFrame with added spread simulation columns
        """
        
        result_data = data.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in result_data.columns:
            if 'datetime' in result_data.columns:
                result_data['timestamp'] = pd.to_datetime(result_data['datetime'])
            else:
                raise ValueError("DataFrame must have 'timestamp' or 'datetime' column")
        
        # Simulate spreads for each row
        spreads = []
        bids = []
        asks = []
        market_conditions = []
        sessions = []
        
        for _, row in result_data.iterrows():
            mid_price = row[mid_price_column]
            volume = row.get(volume_column, 1.0)
            
            spread_data = self.simulate_spread(
                timestamp=row['timestamp'],
                mid_price=mid_price,
                volume=volume
            )
            
            spreads.append(spread_data.spread)
            bids.append(spread_data.bid)
            asks.append(spread_data.ask)
            market_conditions.append(spread_data.market_condition)
            sessions.append(spread_data.session)
        
        # Add spread simulation results
        result_data['simulated_spread'] = spreads
        result_data['simulated_bid'] = bids
        result_data['simulated_ask'] = asks
        result_data['market_condition'] = market_conditions
        result_data['trading_session'] = sessions
        
        return result_data
    
    def get_spread_statistics(self) -> Dict[str, float]:
        """Get statistics about the spread simulation"""
        
        if self.historical_spreads is not None and not self.historical_spreads.empty:
            spreads = self.historical_spreads['spread']
            return {
                'mean_spread': spreads.mean(),
                'median_spread': spreads.median(),
                'std_spread': spreads.std(),
                'min_spread': spreads.min(),
                'max_spread': spreads.max(),
                'q25_spread': spreads.quantile(0.25),
                'q75_spread': spreads.quantile(0.75)
            }
        else:
            return {
                'mean_spread': self.config.base_spread,
                'median_spread': self.config.base_spread,
                'std_spread': self.config.base_spread * 0.2,
                'min_spread': self.config.min_spread,
                'max_spread': self.config.max_spread
            }
    
    def update_market_conditions(self, volatility: float = None, liquidity: float = None):
        """Update market conditions for spread simulation"""
        
        if volatility is not None:
            self.config.market_volatility = volatility
            logger.info(f"Updated market volatility to {volatility}")
        
        if liquidity is not None:
            self.config.liquidity_level = liquidity
            logger.info(f"Updated liquidity level to {liquidity}")
    
    def validate_spread_quality(self, simulated_data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate the quality of simulated spreads.
        
        Args:
            simulated_data: DataFrame with simulated spread data
            
        Returns:
            Dictionary with validation results
        """
        
        if 'simulated_spread' not in simulated_data.columns:
            return {'error': 'No simulated spread data found'}
        
        spreads = simulated_data['simulated_spread']
        
        # Check for unrealistic spreads
        unrealistic_spreads = (spreads < self.config.min_spread) | (spreads > self.config.max_spread)
        unrealistic_count = unrealistic_spreads.sum()
        
        # Check for extreme spread changes
        spread_changes = spreads.diff().abs()
        extreme_changes = spread_changes > self.config.max_spread_change
        extreme_count = extreme_changes.sum()
        
        # Calculate quality metrics
        quality_score = 100.0
        quality_score -= (unrealistic_count / len(spreads)) * 50  # Penalty for unrealistic spreads
        quality_score -= (extreme_count / len(spreads)) * 30      # Penalty for extreme changes
        
        quality_score = max(0, quality_score)
        
        return {
            'quality_score': quality_score,
            'unrealistic_spreads': unrealistic_count,
            'extreme_changes': extreme_count,
            'mean_spread': spreads.mean(),
            'std_spread': spreads.std(),
            'min_spread': spreads.min(),
            'max_spread': spreads.max(),
            'validation_passed': quality_score >= 70
        }

