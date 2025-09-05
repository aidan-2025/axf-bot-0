#!/usr/bin/env python3
"""
Test script for advanced signal processing and feature extraction engine
"""

import unittest
import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Suppress logging for cleaner test output
logging.getLogger('root').setLevel(logging.CRITICAL)

from src.strategy_generation.modules import (
    AdvancedSignalProcessor, ProcessedDataPoint, DataStreamConfig, DataQuality,
    AdvancedFeatureExtractor, FeatureDefinition, FeatureType,
    RealTimeIntegration, DataStream, DataSource
)

class TestAdvancedSignalProcessing(unittest.TestCase):
    """Test cases for advanced signal processing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.signal_processor = AdvancedSignalProcessor({
            'max_latency_ms': 1000,
            'quality_threshold': 0.7,
            'enable_interpolation': True,
            'enable_outlier_detection': True
        })
        
        # Create test market data
        self.test_market_data = {
            'ohlcv': {
                'open': [1.1000, 1.1010, 1.1020, 1.1015, 1.1025],
                'high': [1.1015, 1.1025, 1.1030, 1.1020, 1.1035],
                'low': [1.0995, 1.1005, 1.1015, 1.1010, 1.1020],
                'close': [1.1010, 1.1020, 1.1015, 1.1025, 1.1030],
                'volume': [1000, 1200, 1100, 1300, 1400],
                'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(5, 0, -1)]
            },
            'sentiment': {
                'news': [
                    {'sentiment_score': 0.6, 'relevance': 0.8, 'title': 'Positive news'},
                    {'sentiment_score': -0.3, 'relevance': 0.6, 'title': 'Negative news'}
                ],
                'social': [
                    {'sentiment_score': 0.4, 'engagement': 0.7, 'platform': 'twitter'},
                    {'sentiment_score': 0.2, 'engagement': 0.5, 'platform': 'reddit'}
                ]
            },
            'economic_events': [
                {
                    'title': 'NFP Release',
                    'event_time': datetime.now() + timedelta(hours=2),
                    'impact': 'high',
                    'market_impact_score': 0.9,
                    'volatility_expected': 0.8
                }
            ]
        }
    
    def test_signal_processor_initialization(self):
        """Test signal processor initialization"""
        self.assertIsNotNone(self.signal_processor)
        self.assertEqual(self.signal_processor.max_latency_ms, 1000)
        self.assertEqual(self.signal_processor.quality_threshold, 0.7)
        self.assertTrue(self.signal_processor.enable_interpolation)
        self.assertTrue(self.signal_processor.enable_outlier_detection)
    
    def test_data_stream_config_creation(self):
        """Test data stream configuration creation"""
        config = DataStreamConfig(
            symbol="EURUSD",
            data_types=['ohlcv', 'volume'],
            timeframes=['M1', 'M5'],
            buffer_size=1000,
            quality_threshold=0.8
        )
        
        self.assertEqual(config.symbol, "EURUSD")
        self.assertEqual(config.data_types, ['ohlcv', 'volume'])
        self.assertEqual(config.timeframes, ['M1', 'M5'])
        self.assertEqual(config.buffer_size, 1000)
        self.assertEqual(config.quality_threshold, 0.8)
    
    def test_processed_data_point_creation(self):
        """Test processed data point creation"""
        data_point = ProcessedDataPoint(
            timestamp=datetime.now(),
            symbol="EURUSD",
            data_type="price",
            value=1.1000,
            quality_score=0.9,
            confidence=0.8
        )
        
        self.assertEqual(data_point.symbol, "EURUSD")
        self.assertEqual(data_point.data_type, "price")
        self.assertEqual(data_point.value, 1.1000)
        self.assertEqual(data_point.quality_score, 0.9)
        self.assertEqual(data_point.confidence, 0.8)
        self.assertIsNotNone(data_point.metadata)
    
    def test_signal_processor_configure_stream(self):
        """Test stream configuration"""
        config = DataStreamConfig(
            symbol="EURUSD",
            data_types=['ohlcv'],
            timeframes=['M1'],
            buffer_size=500
        )
        
        self.signal_processor.configure_stream("EURUSD", config)
        
        self.assertIn("EURUSD", self.signal_processor.stream_configs)
        self.assertIn("EURUSD", self.signal_processor.data_buffers)
        self.assertIn("EURUSD", self.signal_processor.quality_metrics)
    
    async def test_async_market_data_processing(self):
        """Test asynchronous market data processing"""
        processed_data = await self.signal_processor.process_market_data_async(self.test_market_data)
        
        self.assertIsInstance(processed_data, dict)
        self.assertIn('ohlcv', processed_data)
        self.assertIn('sentiment', processed_data)
        self.assertIn('economic_events', processed_data)
        self.assertIn('processing_metadata', processed_data)
        
        # Check OHLCV processing
        ohlcv = processed_data['ohlcv']
        self.assertIn('quality_metrics', ohlcv)
        self.assertIn('quality_score', ohlcv['quality_metrics'])
        self.assertGreaterEqual(ohlcv['quality_metrics']['quality_score'], 0.0)
        self.assertLessEqual(ohlcv['quality_metrics']['quality_score'], 1.0)
    
    def test_sync_market_data_processing(self):
        """Test synchronous market data processing"""
        processed_data = self.signal_processor.process_market_data(self.test_market_data)
        
        self.assertIsInstance(processed_data, dict)
        self.assertIn('processing_metadata', processed_data)
    
    def test_ohlcv_quality_assessment(self):
        """Test OHLCV data quality assessment"""
        quality_score = self.signal_processor._assess_ohlcv_quality(self.test_market_data['ohlcv'])
        
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
    
    def test_missing_values_count(self):
        """Test missing values counting"""
        missing_count = self.signal_processor._count_missing_values(self.test_market_data['ohlcv'])
        
        self.assertIsInstance(missing_count, int)
        self.assertGreaterEqual(missing_count, 0)
    
    def test_processing_stats(self):
        """Test processing statistics"""
        stats = self.signal_processor.get_processing_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_processed', stats)
        self.assertIn('quality_failures', stats)
        self.assertIn('latency_failures', stats)
        self.assertIn('outliers_detected', stats)
        self.assertIn('interpolations_performed', stats)


class TestAdvancedFeatureExtraction(unittest.TestCase):
    """Test cases for advanced feature extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_extractor = AdvancedFeatureExtractor({
            'enable_real_time': True,
            'max_features': 1000,
            'feature_cache_size': 10000
        })
        
        # Create test processed market data
        self.test_processed_data = {
            'ohlcv': {
                'open': [1.1000, 1.1010, 1.1020, 1.1015, 1.1025, 1.1030, 1.1025, 1.1035, 1.1040, 1.1035],
                'high': [1.1015, 1.1025, 1.1030, 1.1020, 1.1035, 1.1040, 1.1035, 1.1045, 1.1050, 1.1045],
                'low': [1.0995, 1.1005, 1.1015, 1.1010, 1.1020, 1.1025, 1.1020, 1.1030, 1.1035, 1.1030],
                'close': [1.1010, 1.1020, 1.1015, 1.1025, 1.1030, 1.1035, 1.1030, 1.1040, 1.1045, 1.1040],
                'volume': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
                'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10, 0, -1)]
            },
            'sentiment': {
                'news': [
                    {'sentiment_score': 0.6, 'relevance': 0.8},
                    {'sentiment_score': -0.3, 'relevance': 0.6}
                ],
                'social': [
                    {'sentiment_score': 0.4, 'engagement': 0.7},
                    {'sentiment_score': 0.2, 'engagement': 0.5}
                ]
            },
            'economic_events': [
                {
                    'title': 'NFP Release',
                    'event_time': datetime.now() + timedelta(hours=2),
                    'market_impact_score': 0.9
                }
            ]
        }
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization"""
        self.assertIsNotNone(self.feature_extractor)
        self.assertTrue(self.feature_extractor.enable_real_time)
        self.assertEqual(self.feature_extractor.max_features, 1000)
        self.assertEqual(self.feature_extractor.feature_cache_size, 10000)
    
    def test_feature_definition_creation(self):
        """Test feature definition creation"""
        feature_def = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.TECHNICAL,
            description="Test feature",
            calculation_method="test_method",
            parameters={"param1": 10, "param2": 20}
        )
        
        self.assertEqual(feature_def.name, "test_feature")
        self.assertEqual(feature_def.feature_type, FeatureType.TECHNICAL)
        self.assertEqual(feature_def.description, "Test feature")
        self.assertEqual(feature_def.calculation_method, "test_method")
        self.assertEqual(feature_def.parameters, {"param1": 10, "param2": 20})
    
    def test_default_features_registration(self):
        """Test that default features are registered"""
        feature_definitions = self.feature_extractor.get_feature_definitions()
        
        # Check that some expected features are registered
        expected_features = ['sma_20', 'ema_12', 'rsi_14', 'macd', 'bollinger_bands']
        for feature_name in expected_features:
            self.assertIn(feature_name, feature_definitions)
    
    async def test_async_feature_extraction(self):
        """Test asynchronous feature extraction"""
        features = await self.feature_extractor.extract_features_async(self.test_processed_data)
        
        self.assertIsInstance(features, dict)
        self.assertIn('_metadata', features)
        
        # Check that some features were extracted
        feature_count = features['_metadata']['features_extracted']
        self.assertGreater(feature_count, 0)
    
    def test_sync_feature_extraction(self):
        """Test synchronous feature extraction"""
        features = self.feature_extractor.extract_features(self.test_processed_data)
        
        self.assertIsInstance(features, dict)
        self.assertIn('_metadata', features)
    
    async def test_specific_feature_extraction(self):
        """Test extraction of specific features"""
        specific_features = ['sma_20', 'rsi_14', 'news_sentiment']
        features = await self.feature_extractor.extract_features_async(
            self.test_processed_data, 
            specific_features
        )
        
        self.assertIsInstance(features, dict)
        # Check that requested features are present (if data is sufficient)
        for feature_name in specific_features:
            if feature_name in features:
                self.assertIsNotNone(features[feature_name])
    
    def test_feature_extraction_stats(self):
        """Test feature extraction statistics"""
        stats = self.feature_extractor.get_extraction_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('features_extracted', stats)
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('processing_time_ms', stats)
        self.assertIn('errors', stats)
    
    def test_feature_cache_management(self):
        """Test feature cache management"""
        # Extract some features to populate cache
        self.feature_extractor.extract_features(self.test_processed_data)
        
        initial_cache_size = len(self.feature_extractor.feature_cache)
        self.assertGreaterEqual(initial_cache_size, 0)
        
        # Clear cache
        self.feature_extractor.clear_cache()
        self.assertEqual(len(self.feature_extractor.feature_cache), 0)


class TestRealTimeIntegration(unittest.TestCase):
    """Test cases for real-time integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.integration = RealTimeIntegration({
            'signal_processing': {'quality_threshold': 0.7},
            'feature_extraction': {'enable_real_time': True}
        })
    
    def test_integration_initialization(self):
        """Test real-time integration initialization"""
        self.assertIsNotNone(self.integration)
        self.assertIsNotNone(self.integration.signal_processor)
        self.assertIsNotNone(self.integration.feature_extractor)
        self.assertFalse(self.integration.is_running)
    
    def test_data_stream_creation(self):
        """Test data stream creation"""
        stream = DataStream(
            source=DataSource.MARKET_DATA,
            symbol="EURUSD",
            endpoint="http://localhost:8000/api/v1/market-data",
            update_frequency=1000
        )
        
        self.assertEqual(stream.source, DataSource.MARKET_DATA)
        self.assertEqual(stream.symbol, "EURUSD")
        self.assertEqual(stream.endpoint, "http://localhost:8000/api/v1/market-data")
        self.assertEqual(stream.update_frequency, 1000)
        self.assertTrue(stream.is_active)
    
    def test_add_data_stream(self):
        """Test adding data stream to integration"""
        stream = DataStream(
            source=DataSource.MARKET_DATA,
            symbol="EURUSD",
            endpoint="http://localhost:8000/api/v1/market-data",
            update_frequency=1000
        )
        
        self.integration.add_data_stream("EURUSD_M1", stream)
        
        self.assertIn("EURUSD_M1", self.integration.data_streams)
        self.assertIn("EURUSD_M1", self.integration.data_buffers)
    
    def test_callback_registration(self):
        """Test callback registration"""
        def test_callback(data):
            pass
        
        self.integration.add_callback(test_callback)
        
        self.assertIn(test_callback, self.integration.callbacks)
    
    def test_integration_stats(self):
        """Test integration statistics"""
        stats = self.integration.get_integration_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_updates', stats)
        self.assertIn('successful_updates', stats)
        self.assertIn('failed_updates', stats)
        self.assertIn('average_latency_ms', stats)
        self.assertIn('active_streams', stats)
        self.assertIn('signal_processor_stats', stats)
        self.assertIn('feature_extractor_stats', stats)
    
    def test_data_buffer_sizes(self):
        """Test data buffer size reporting"""
        buffer_sizes = self.integration.get_data_buffer_sizes()
        
        self.assertIsInstance(buffer_sizes, dict)
        # Should be empty initially
        self.assertEqual(len(buffer_sizes), 0)


class TestIntegrationWorkflow(unittest.TestCase):
    """Test integrated workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.signal_processor = AdvancedSignalProcessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        
        self.test_data = {
            'ohlcv': {
                'open': [1.1000, 1.1010, 1.1020, 1.1015, 1.1025],
                'high': [1.1015, 1.1025, 1.1030, 1.1020, 1.1035],
                'low': [1.0995, 1.1005, 1.1015, 1.1010, 1.1020],
                'close': [1.1010, 1.1020, 1.1015, 1.1025, 1.1030],
                'volume': [1000, 1200, 1100, 1300, 1400],
                'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(5, 0, -1)]
            },
            'sentiment': {
                'news': [{'sentiment_score': 0.6, 'relevance': 0.8}]
            }
        }
    
    async def test_end_to_end_processing(self):
        """Test end-to-end processing workflow"""
        # Step 1: Process market data
        processed_data = await self.signal_processor.process_market_data_async(self.test_data)
        self.assertIsInstance(processed_data, dict)
        
        # Step 2: Extract features
        features = await self.feature_extractor.extract_features_async(processed_data)
        self.assertIsInstance(features, dict)
        
        # Step 3: Verify integration
        self.assertIn('processing_metadata', processed_data)
        self.assertIn('_metadata', features)
    
    def test_sync_end_to_end_processing(self):
        """Test synchronous end-to-end processing"""
        # Step 1: Process market data
        processed_data = self.signal_processor.process_market_data(self.test_data)
        self.assertIsInstance(processed_data, dict)
        
        # Step 2: Extract features
        features = self.feature_extractor.extract_features(processed_data)
        self.assertIsInstance(features, dict)
        
        # Step 3: Verify integration
        self.assertIn('processing_metadata', processed_data)
        self.assertIn('_metadata', features)


def run_async_tests():
    """Run asynchronous tests"""
    async def run_async_test_suite():
        # Create test instances
        signal_processor = TestAdvancedSignalProcessing()
        feature_extractor = TestAdvancedFeatureExtraction()
        integration_workflow = TestIntegrationWorkflow()
        
        # Set up test instances
        signal_processor.setUp()
        feature_extractor.setUp()
        integration_workflow.setUp()
        
        try:
            # Run async tests directly
            print("Running test_async_market_data_processing...")
            await signal_processor.test_async_market_data_processing()
            print("✓ test_async_market_data_processing passed")
            
            print("Running test_async_feature_extraction...")
            await feature_extractor.test_async_feature_extraction()
            print("✓ test_async_feature_extraction passed")
            
            print("Running test_specific_feature_extraction...")
            await feature_extractor.test_specific_feature_extraction()
            print("✓ test_specific_feature_extraction passed")
            
            print("Running test_end_to_end_processing...")
            await integration_workflow.test_end_to_end_processing()
            print("✓ test_end_to_end_processing passed")
            
            return True
        except Exception as e:
            print(f"❌ Async test failed: {e}")
            return False
        finally:
            # Clean up
            signal_processor.tearDown()
            feature_extractor.tearDown()
            integration_workflow.tearDown()
    
    return asyncio.run(run_async_test_suite())


if __name__ == '__main__':
    print("Testing Advanced Signal Processing and Feature Extraction Engine")
    print("=" * 60)
    
    # Run synchronous tests
    print("\nRunning synchronous tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run asynchronous tests
    print("\nRunning asynchronous tests...")
    success = run_async_tests()
    
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed!")
