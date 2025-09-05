"""
Statistical Validator

Performs statistical validation of backtesting results to ensure accuracy,
consistency, and reliability of the modeling framework.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json
from scipy import stats
from scipy.stats import (
    jarque_bera, shapiro, normaltest, anderson, 
    kstest, ks_2samp, pearsonr, spearmanr,
    chi2_contingency, fisher_exact
)

logger = logging.getLogger(__name__)


class StatisticalTestType(Enum):
    """Types of statistical tests"""
    NORMALITY = "normality"
    STATIONARITY = "stationarity"
    AUTOCORRELATION = "autocorrelation"
    HETEROSCEDASTICITY = "heteroscedasticity"
    COINTEGRATION = "cointegration"
    CORRELATION = "correlation"
    INDEPENDENCE = "independence"
    HOMOGENEITY = "homogeneity"


class TestResult(Enum):
    """Result of statistical test"""
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


@dataclass
class StatisticalTest:
    """Result of a statistical test"""
    
    test_name: str
    test_type: StatisticalTestType
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    result: TestResult = TestResult.INCONCLUSIVE
    significance_level: float = 0.05
    interpretation: str = ""
    assumptions_met: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Determine test result based on p-value"""
        if np.isnan(self.p_value) or np.isnan(self.statistic):
            self.result = TestResult.ERROR
            self.interpretation = "Test failed due to invalid data"
        elif self.p_value < self.significance_level:
            self.result = TestResult.FAIL
            self.interpretation = f"Reject null hypothesis at {self.significance_level:.3f} level"
        else:
            self.result = TestResult.PASS
            self.interpretation = f"Fail to reject null hypothesis at {self.significance_level:.3f} level"


@dataclass
class DistributionAnalysis:
    """Analysis of return distribution characteristics"""
    
    # Basic statistics
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Percentiles
    percentiles: Dict[str, float] = field(default_factory=dict)
    
    # Distribution tests
    normality_tests: List[StatisticalTest] = field(default_factory=list)
    
    # Distribution characteristics
    is_normal: bool = False
    is_stationary: bool = False
    has_fat_tails: bool = False
    is_symmetric: bool = False
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    def __post_init__(self):
        """Calculate derived characteristics"""
        # Check for fat tails (kurtosis > 3)
        self.has_fat_tails = self.kurtosis > 3.0
        
        # Check for symmetry (skewness close to 0)
        self.is_symmetric = abs(self.skewness) < 0.5
        
        # Determine if distribution is normal based on tests
        if self.normality_tests:
            normal_tests_passed = sum(1 for test in self.normality_tests if test.result == TestResult.PASS)
            self.is_normal = normal_tests_passed > len(self.normality_tests) / 2


@dataclass
class CorrelationAnalysis:
    """Analysis of correlations between different metrics"""
    
    # Correlation matrices
    price_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    return_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Correlation tests
    correlation_tests: List[StatisticalTest] = field(default_factory=list)
    
    # Autocorrelation analysis
    autocorrelation_lags: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    autocorrelation_values: Dict[int, float] = field(default_factory=dict)
    autocorrelation_tests: List[StatisticalTest] = field(default_factory=list)
    
    # Cross-correlation analysis
    cross_correlations: Dict[str, float] = field(default_factory=dict)
    
    # Summary statistics
    max_correlation: float = 0.0
    min_correlation: float = 0.0
    avg_correlation: float = 0.0
    significant_correlations: int = 0


@dataclass
class StatisticalConfig:
    """Configuration for statistical validation"""
    
    # Significance levels
    significance_level: float = 0.05
    multiple_comparison_correction: str = "bonferroni"  # bonferroni, fdr_bh, none
    
    # Normality test settings
    enable_normality_tests: bool = True
    normality_tests: List[str] = field(default_factory=lambda: ["jarque_bera", "shapiro", "normaltest"])
    
    # Stationarity test settings
    enable_stationarity_tests: bool = True
    stationarity_tests: List[str] = field(default_factory=lambda: ["adf", "kpss", "pp"])
    
    # Autocorrelation test settings
    enable_autocorrelation_tests: bool = True
    max_lags: int = 20
    autocorrelation_lags: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Correlation test settings
    enable_correlation_tests: bool = True
    correlation_methods: List[str] = field(default_factory=lambda: ["pearson", "spearman"])
    
    # Risk metric settings
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    enable_risk_metrics: bool = True
    
    # Data requirements
    min_observations: int = 30
    max_missing_ratio: float = 0.05  # 5%


class StatisticalValidator:
    """Performs statistical validation of backtesting results"""
    
    def __init__(self, config: StatisticalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_distribution(self, returns: List[float]) -> DistributionAnalysis:
        """Validate return distribution characteristics"""
        
        if not returns or len(returns) < self.config.min_observations:
            self.logger.warning("Insufficient data for distribution analysis")
            return DistributionAnalysis()
        
        returns_array = np.array(returns)
        
        # Calculate basic statistics
        mean = np.mean(returns_array)
        median = np.median(returns_array)
        std = np.std(returns_array)
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array)
        
        # Calculate percentiles
        percentiles = {
            '1%': np.percentile(returns_array, 1),
            '5%': np.percentile(returns_array, 5),
            '10%': np.percentile(returns_array, 10),
            '25%': np.percentile(returns_array, 25),
            '50%': np.percentile(returns_array, 50),
            '75%': np.percentile(returns_array, 75),
            '90%': np.percentile(returns_array, 90),
            '95%': np.percentile(returns_array, 95),
            '99%': np.percentile(returns_array, 99)
        }
        
        # Perform normality tests
        normality_tests = []
        if self.config.enable_normality_tests:
            normality_tests = self._perform_normality_tests(returns_array)
        
        # Calculate risk metrics
        var_95 = np.percentile(returns_array, 5)  # 5th percentile for 95% VaR
        var_99 = np.percentile(returns_array, 1)  # 1st percentile for 99% VaR
        cvar_95 = np.mean(returns_array[returns_array <= var_95])
        cvar_99 = np.mean(returns_array[returns_array <= var_99])
        
        return DistributionAnalysis(
            mean=mean,
            median=median,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            percentiles=percentiles,
            normality_tests=normality_tests,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99
        )
    
    def validate_correlations(self, data: Dict[str, List[float]]) -> CorrelationAnalysis:
        """Validate correlations between different metrics"""
        
        if not data or len(data) < 2:
            self.logger.warning("Insufficient data for correlation analysis")
            return CorrelationAnalysis()
        
        # Prepare data
        df = pd.DataFrame(data)
        
        # Calculate correlation matrices
        price_correlations = {}
        return_correlations = {}
        
        if 'returns' in df.columns:
            return_correlations = df.corr().to_dict()
        
        # Calculate autocorrelations
        autocorrelation_values = {}
        autocorrelation_tests = []
        
        if 'returns' in df.columns and self.config.enable_autocorrelation_tests:
            returns = df['returns'].dropna()
            for lag in self.config.autocorrelation_lags:
                if len(returns) > lag:
                    autocorr = returns.autocorr(lag=lag)
                    autocorrelation_values[lag] = autocorr
                    
                    # Test for significant autocorrelation
                    test = self._test_autocorrelation(returns, lag)
                    autocorrelation_tests.append(test)
        
        # Calculate cross-correlations
        cross_correlations = {}
        correlation_tests = []
        
        if self.config.enable_correlation_tests:
            for col1 in df.columns:
                for col2 in df.columns:
                    if col1 != col2:
                        try:
                            corr, p_value = pearsonr(df[col1].dropna(), df[col2].dropna())
                            cross_correlations[f"{col1}_{col2}"] = corr
                            
                            # Test for significant correlation
                            test = StatisticalTest(
                                test_name=f"correlation_{col1}_{col2}",
                                test_type=StatisticalTestType.CORRELATION,
                                statistic=corr,
                                p_value=p_value,
                                significance_level=self.config.significance_level
                            )
                            correlation_tests.append(test)
                        except:
                            continue
        
        # Calculate summary statistics
        all_correlations = list(cross_correlations.values())
        max_correlation = max(all_correlations) if all_correlations else 0.0
        min_correlation = min(all_correlations) if all_correlations else 0.0
        avg_correlation = np.mean(all_correlations) if all_correlations else 0.0
        significant_correlations = sum(1 for test in correlation_tests if test.result == TestResult.PASS)
        
        return CorrelationAnalysis(
            price_correlations=price_correlations,
            return_correlations=return_correlations,
            correlation_tests=correlation_tests,
            autocorrelation_lags=self.config.autocorrelation_lags,
            autocorrelation_values=autocorrelation_values,
            autocorrelation_tests=autocorrelation_tests,
            cross_correlations=cross_correlations,
            max_correlation=max_correlation,
            min_correlation=min_correlation,
            avg_correlation=avg_correlation,
            significant_correlations=significant_correlations
        )
    
    def validate_stationarity(self, time_series: List[float]) -> List[StatisticalTest]:
        """Validate stationarity of time series"""
        
        if not time_series or len(time_series) < self.config.min_observations:
            self.logger.warning("Insufficient data for stationarity analysis")
            return []
        
        tests = []
        ts = pd.Series(time_series).dropna()
        
        if self.config.enable_stationarity_tests:
            # Augmented Dickey-Fuller test
            if "adf" in self.config.stationarity_tests:
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_stat, adf_p_value, _, _, adf_critical, _ = adfuller(ts)
                    
                    test = StatisticalTest(
                        test_name="augmented_dickey_fuller",
                        test_type=StatisticalTestType.STATIONARITY,
                        statistic=adf_stat,
                        p_value=adf_p_value,
                        critical_value=adf_critical.get('5%', None),
                        significance_level=self.config.significance_level
                    )
                    tests.append(test)
                except Exception as e:
                    self.logger.warning(f"ADF test failed: {e}")
            
            # KPSS test
            if "kpss" in self.config.stationarity_tests:
                try:
                    from statsmodels.tsa.stattools import kpss
                    kpss_stat, kpss_p_value, kpss_lags, kpss_critical = kpss(ts)
                    
                    test = StatisticalTest(
                        test_name="kpss",
                        test_type=StatisticalTestType.STATIONARITY,
                        statistic=kpss_stat,
                        p_value=kpss_p_value,
                        critical_value=kpss_critical.get('5%', None),
                        significance_level=self.config.significance_level
                    )
                    tests.append(test)
                except Exception as e:
                    self.logger.warning(f"KPSS test failed: {e}")
        
        return tests
    
    def validate_independence(self, returns: List[float]) -> List[StatisticalTest]:
        """Validate independence of returns"""
        
        if not returns or len(returns) < self.config.min_observations:
            self.logger.warning("Insufficient data for independence analysis")
            return []
        
        tests = []
        returns_array = np.array(returns)
        
        # Ljung-Box test for serial correlation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_stat, lb_p_value = acorr_ljungbox(returns_array, lags=10, return_df=False)
            
            test = StatisticalTest(
                test_name="ljung_box",
                test_type=StatisticalTestType.INDEPENDENCE,
                statistic=lb_stat[0],
                p_value=lb_p_value[0],
                significance_level=self.config.significance_level
            )
            tests.append(test)
        except Exception as e:
            self.logger.warning(f"Ljung-Box test failed: {e}")
        
        # Runs test
        try:
            runs_stat, runs_p_value = self._runs_test(returns_array)
            
            test = StatisticalTest(
                test_name="runs_test",
                test_type=StatisticalTestType.INDEPENDENCE,
                statistic=runs_stat,
                p_value=runs_p_value,
                significance_level=self.config.significance_level
            )
            tests.append(test)
        except Exception as e:
            self.logger.warning(f"Runs test failed: {e}")
        
        return tests
    
    def validate_heteroscedasticity(self, returns: List[float]) -> List[StatisticalTest]:
        """Validate heteroscedasticity in returns"""
        
        if not returns or len(returns) < self.config.min_observations:
            self.logger.warning("Insufficient data for heteroscedasticity analysis")
            return []
        
        tests = []
        returns_array = np.array(returns)
        
        # ARCH test
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_stat, arch_p_value, _, _ = het_arch(returns_array)
            
            test = StatisticalTest(
                test_name="arch_test",
                test_type=StatisticalTestType.HETEROSCEDASTICITY,
                statistic=arch_stat,
                p_value=arch_p_value,
                significance_level=self.config.significance_level
            )
            tests.append(test)
        except Exception as e:
            self.logger.warning(f"ARCH test failed: {e}")
        
        # Breusch-Pagan test
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_p_value, _, _ = het_breuschpagan(returns_array, np.ones((len(returns_array), 1)))
            
            test = StatisticalTest(
                test_name="breusch_pagan",
                test_type=StatisticalTestType.HETEROSCEDASTICITY,
                statistic=bp_stat,
                p_value=bp_p_value,
                significance_level=self.config.significance_level
            )
            tests.append(test)
        except Exception as e:
            self.logger.warning(f"Breusch-Pagan test failed: {e}")
        
        return tests
    
    def _perform_normality_tests(self, data: np.ndarray) -> List[StatisticalTest]:
        """Perform normality tests on data"""
        
        tests = []
        
        # Jarque-Bera test
        if "jarque_bera" in self.config.normality_tests:
            try:
                jb_stat, jb_p_value = jarque_bera(data)
                test = StatisticalTest(
                    test_name="jarque_bera",
                    test_type=StatisticalTestType.NORMALITY,
                    statistic=jb_stat,
                    p_value=jb_p_value,
                    significance_level=self.config.significance_level
                )
                tests.append(test)
            except Exception as e:
                self.logger.warning(f"Jarque-Bera test failed: {e}")
        
        # Shapiro-Wilk test (for small samples)
        if "shapiro" in self.config.normality_tests and len(data) <= 5000:
            try:
                sw_stat, sw_p_value = shapiro(data)
                test = StatisticalTest(
                    test_name="shapiro_wilk",
                    test_type=StatisticalTestType.NORMALITY,
                    statistic=sw_stat,
                    p_value=sw_p_value,
                    significance_level=self.config.significance_level
                )
                tests.append(test)
            except Exception as e:
                self.logger.warning(f"Shapiro-Wilk test failed: {e}")
        
        # D'Agostino's normality test
        if "normaltest" in self.config.normality_tests:
            try:
                da_stat, da_p_value = normaltest(data)
                test = StatisticalTest(
                    test_name="dagostino",
                    test_type=StatisticalTestType.NORMALITY,
                    statistic=da_stat,
                    p_value=da_p_value,
                    significance_level=self.config.significance_level
                )
                tests.append(test)
            except Exception as e:
                self.logger.warning(f"D'Agostino test failed: {e}")
        
        return tests
    
    def _test_autocorrelation(self, data: pd.Series, lag: int) -> StatisticalTest:
        """Test for significant autocorrelation at given lag"""
        
        try:
            autocorr = data.autocorr(lag=lag)
            n = len(data)
            
            # Standard error for autocorrelation
            se = 1.0 / np.sqrt(n)
            
            # Test statistic
            test_stat = autocorr / se
            
            # P-value (approximate)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
            
            return StatisticalTest(
                test_name=f"autocorrelation_lag_{lag}",
                test_type=StatisticalTestType.AUTOCORRELATION,
                statistic=test_stat,
                p_value=p_value,
                significance_level=self.config.significance_level
            )
        except Exception as e:
            self.logger.warning(f"Autocorrelation test failed for lag {lag}: {e}")
            return StatisticalTest(
                test_name=f"autocorrelation_lag_{lag}",
                test_type=StatisticalTestType.AUTOCORRELATION,
                statistic=0.0,
                p_value=1.0,
                significance_level=self.config.significance_level
            )
    
    def _runs_test(self, data: np.ndarray) -> Tuple[float, float]:
        """Perform runs test for independence"""
        
        # Convert to binary sequence (positive/negative)
        binary = (data > 0).astype(int)
        
        # Count runs
        runs = 0
        n1 = np.sum(binary)  # Number of 1s
        n2 = len(binary) - n1  # Number of 0s
        
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        
        # Count actual runs
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        runs += 1  # Add 1 for the first run
        
        # Expected runs
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        
        # Variance of runs
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        if variance <= 0:
            return 0.0, 1.0
        
        # Test statistic
        z_stat = (runs - expected_runs) / np.sqrt(variance)
        
        # P-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return z_stat, p_value

