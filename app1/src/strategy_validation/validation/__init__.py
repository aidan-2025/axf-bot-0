"""
Validation Package

Comprehensive validation suite for assessing modeling quality, comparing backtest results
to MT4, and ensuring statistical accuracy and consistency.
"""

from .modeling_quality_validator import (
    ModelingQualityValidator,
    ModelingQualityConfig,
    QualityMetrics,
    QualityReport
)

from .cross_platform_comparator import (
    CrossPlatformComparator,
    ComparisonConfig,
    ComparisonResult,
    MetricComparison,
    TradeComparison
)

from .statistical_validator import (
    StatisticalValidator,
    StatisticalConfig,
    StatisticalTest,
    DistributionAnalysis,
    CorrelationAnalysis
)

from .regression_tester import (
    RegressionTester,
    RegressionConfig,
    TestSuite,
    TestResult,
    ToleranceThresholds
)

from .validation_reporter import (
    ValidationReporter,
    ReportConfig,
    ValidationReport,
    QualityDashboard
)

__all__ = [
    # Modeling Quality
    'ModelingQualityValidator',
    'ModelingQualityConfig', 
    'QualityMetrics',
    'QualityReport',
    
    # Cross-Platform Comparison
    'CrossPlatformComparator',
    'ComparisonConfig',
    'ComparisonResult',
    'MetricComparison',
    'TradeComparison',
    
    # Statistical Validation
    'StatisticalValidator',
    'StatisticalConfig',
    'StatisticalTest',
    'DistributionAnalysis',
    'CorrelationAnalysis',
    
    # Regression Testing
    'RegressionTester',
    'RegressionConfig',
    'TestSuite',
    'TestResult',
    'ToleranceThresholds',
    
    # Reporting
    'ValidationReporter',
    'ReportConfig',
    'ValidationReport',
    'QualityDashboard'
]

