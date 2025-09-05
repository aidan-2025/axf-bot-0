"""
Modules for signal processing and feature extraction
"""

from .signal_processor import SignalProcessor
from .feature_extractor import FeatureExtractor
from .advanced_signal_processor import AdvancedSignalProcessor, ProcessedDataPoint, DataStreamConfig, DataQuality
from .advanced_feature_extractor import AdvancedFeatureExtractor, FeatureDefinition, FeatureType
from .real_time_integration import RealTimeIntegration, DataStream, DataSource

__all__ = [
    'SignalProcessor',
    'FeatureExtractor',
    'AdvancedSignalProcessor',
    'ProcessedDataPoint',
    'DataStreamConfig',
    'DataQuality',
    'AdvancedFeatureExtractor',
    'FeatureDefinition',
    'FeatureType',
    'RealTimeIntegration',
    'DataStream',
    'DataSource'
]
