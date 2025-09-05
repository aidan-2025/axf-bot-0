"""
Test script for strategy generation engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategy_generation import (
    StrategyGenerationEngine, StrategyType, StrategyParameters,
    TrendStrategy, RangeStrategy, BreakoutStrategy, SentimentStrategy,
    NewsStrategy, MultiTimeframeStrategy, PairsStrategy
)
from src.strategy_generation.core.parameter_space import ParameterSpace, ParameterType, ParameterDefinition
from src.strategy_generation.validation.strategy_validator import StrategyValidator
from src.strategy_generation.modules.signal_processor import SignalProcessor
from src.strategy_generation.modules.feature_extractor import FeatureExtractor

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_strategy_templates():
    """Test all strategy templates"""
    logger.info("Testing strategy templates...")
    
    # Test parameters
    test_params = StrategyParameters(
        strategy_id="test_strategy",
        strategy_type=StrategyType.TREND,
        name="Test Strategy",
        description="Test strategy for validation",
        parameters={
            "fast_ma_period": 20,
            "slow_ma_period": 50,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "min_trend_strength": 0.6
        }
    )
    
    # Test Trend Strategy
    try:
        trend_strategy = TrendStrategy(test_params)
        assert trend_strategy.initialize(), "Trend strategy initialization failed"
        logger.info("✅ Trend Strategy: PASSED")
    except Exception as e:
        logger.error(f"❌ Trend Strategy: FAILED - {e}")
    
    # Test Range Strategy
    try:
        range_params = test_params
        range_params.strategy_type = StrategyType.RANGE
        range_params.parameters = {
            "lookback_period": 20,
            "support_threshold": 0.02,
            "resistance_threshold": 0.02,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "min_range_size": 0.005
        }
        range_strategy = RangeStrategy(range_params)
        assert range_strategy.initialize(), "Range strategy initialization failed"
        logger.info("✅ Range Strategy: PASSED")
    except Exception as e:
        logger.error(f"❌ Range Strategy: FAILED - {e}")
    
    # Test Breakout Strategy
    try:
        breakout_params = test_params
        breakout_params.strategy_type = StrategyType.BREAKOUT
        breakout_params.parameters = {
            "breakout_period": 20,
            "volume_threshold": 1.5,
            "price_threshold": 0.005,
            "confirmation_periods": 2,
            "atr_period": 14,
            "atr_multiplier": 2.0
        }
        breakout_strategy = BreakoutStrategy(breakout_params)
        assert breakout_strategy.initialize(), "Breakout strategy initialization failed"
        logger.info("✅ Breakout Strategy: PASSED")
    except Exception as e:
        logger.error(f"❌ Breakout Strategy: FAILED - {e}")

def test_parameter_space():
    """Test parameter space functionality"""
    logger.info("Testing parameter space...")
    
    try:
        param_space = ParameterSpace()
        
        # Add parameters
        param_space.add_parameter(ParameterDefinition(
            name="test_int",
            param_type=ParameterType.INTEGER,
            min_value=1,
            max_value=100,
            default_value=50,
            description="Test integer parameter"
        ))
        
        param_space.add_parameter(ParameterDefinition(
            name="test_float",
            param_type=ParameterType.FLOAT,
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
            description="Test float parameter"
        ))
        
        # Test validation
        test_values = {"test_int": 50, "test_float": 0.5}
        is_valid, errors = param_space.validate_parameters(test_values)
        assert is_valid, f"Parameter validation failed: {errors}"
        
        # Test encoding/decoding
        encoded = param_space.encode_parameters(test_values)
        decoded = param_space.decode_parameters(encoded)
        
        assert abs(decoded["test_int"] - test_values["test_int"]) <= 1, "Integer encoding/decoding failed"
        assert abs(decoded["test_float"] - test_values["test_float"]) < 0.1, "Float encoding/decoding failed"
        
        logger.info("✅ Parameter Space: PASSED")
    except Exception as e:
        logger.error(f"❌ Parameter Space: FAILED - {e}")

def test_strategy_engine():
    """Test strategy generation engine"""
    logger.info("Testing strategy generation engine...")
    
    try:
        # Create engine
        engine = StrategyGenerationEngine()
        
        # Test engine status
        status = engine.get_engine_status()
        assert status["engine_initialized"], "Engine not initialized"
        
        # Test strategy creation
        test_params = StrategyParameters(
            strategy_id="test_engine_strategy",
            strategy_type=StrategyType.TREND,
            name="Test Engine Strategy",
            description="Test strategy for engine validation",
            parameters={
                "fast_ma_period": 20,
                "slow_ma_period": 50,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "atr_period": 14,
                "atr_multiplier": 2.0,
                "min_trend_strength": 0.6
            }
        )
        
        strategy = engine.create_strategy(StrategyType.TREND, test_params)
        assert strategy is not None, "Strategy creation failed"
        assert strategy.is_initialized, "Strategy not initialized"
        
        logger.info("✅ Strategy Engine: PASSED")
    except Exception as e:
        logger.error(f"❌ Strategy Engine: FAILED - {e}")

def test_validation():
    """Test strategy validation"""
    logger.info("Testing strategy validation...")
    
    try:
        validator = StrategyValidator()
        
        # Create test strategy
        test_params = StrategyParameters(
            strategy_id="test_validation_strategy",
            strategy_type=StrategyType.TREND,
            name="Test Validation Strategy",
            description="Test strategy for validation testing",
            parameters={
                "fast_ma_period": 20,
                "slow_ma_period": 50,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "atr_period": 14,
                "atr_multiplier": 2.0,
                "min_trend_strength": 0.6
            }
        )
        
        strategy = TrendStrategy(test_params)
        strategy.initialize()
        
        # Test validation
        is_valid, errors = validator.validate(strategy)
        assert is_valid, f"Strategy validation failed: {errors}"
        
        # Test validation summary
        summary = validator.get_validation_summary(strategy)
        assert summary["is_valid"], "Validation summary shows invalid strategy"
        
        logger.info("✅ Strategy Validation: PASSED")
    except Exception as e:
        logger.error(f"❌ Strategy Validation: FAILED - {e}")

def test_signal_processing():
    """Test signal processing"""
    logger.info("Testing signal processing...")
    
    try:
        processor = SignalProcessor()
        
        # Test market data processing
        market_data = {
            "ohlcv": {
                "open": [1.0, 1.1, 1.2, 1.3, 1.4],
                "high": [1.05, 1.15, 1.25, 1.35, 1.45],
                "low": [0.95, 1.05, 1.15, 1.25, 1.35],
                "close": [1.1, 1.2, 1.3, 1.4, 1.5],
                "volume": [1000, 1100, 1200, 1300, 1400]
            }
        }
        
        processed_data = processor.process_market_data(market_data)
        assert "ohlcv" in processed_data, "OHLCV data not processed"
        
        logger.info("✅ Signal Processing: PASSED")
    except Exception as e:
        logger.error(f"❌ Signal Processing: FAILED - {e}")

def test_feature_extraction():
    """Test feature extraction"""
    logger.info("Testing feature extraction...")
    
    try:
        extractor = FeatureExtractor()
        
        # Test technical features
        ohlcv_data = {
            "open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "high": [1.05, 1.15, 1.25, 1.35, 1.45],
            "low": [0.95, 1.05, 1.15, 1.25, 1.35],
            "close": [1.1, 1.2, 1.3, 1.4, 1.5],
            "volume": [1000, 1100, 1200, 1300, 1400]
        }
        
        features = extractor.extract_technical_features(ohlcv_data)
        assert len(features) > 0, "No technical features extracted"
        
        # Test sentiment features
        sentiment_data = {
            "news": [
                {"sentiment_score": 0.5, "relevance": 0.8},
                {"sentiment_score": -0.3, "relevance": 0.6}
            ],
            "social": [
                {"sentiment_score": 0.2, "engagement": 0.7},
                {"sentiment_score": -0.1, "engagement": 0.5}
            ]
        }
        
        sentiment_features = extractor.extract_sentiment_features(sentiment_data)
        assert len(sentiment_features) > 0, "No sentiment features extracted"
        
        logger.info("✅ Feature Extraction: PASSED")
    except Exception as e:
        logger.error(f"❌ Feature Extraction: FAILED - {e}")

def main():
    """Run all tests"""
    logger.info("Starting strategy generation engine tests...")
    
    test_strategy_templates()
    test_parameter_space()
    test_strategy_engine()
    test_validation()
    test_signal_processing()
    test_feature_extraction()
    
    logger.info("Strategy generation engine tests completed!")

if __name__ == "__main__":
    main()

