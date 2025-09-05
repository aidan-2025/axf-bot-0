"""
Validation Reporter

Generates comprehensive reports for validation results, including quality dashboards,
comparison reports, and statistical analysis summaries.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report format options"""
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    PDF = "pdf"
    EXCEL = "excel"


class ReportType(Enum):
    """Report type options"""
    QUALITY_REPORT = "quality_report"
    COMPARISON_REPORT = "comparison_report"
    STATISTICAL_REPORT = "statistical_report"
    REGRESSION_REPORT = "regression_report"
    DASHBOARD = "dashboard"
    SUMMARY = "summary"


@dataclass
class ReportConfig:
    """Configuration for validation reporting"""
    
    # Report settings
    report_format: ReportFormat = ReportFormat.HTML
    include_charts: bool = True
    include_statistics: bool = True
    include_details: bool = True
    
    # Chart settings
    chart_style: str = "seaborn-v0_8"
    chart_size: Tuple[int, int] = (12, 8)
    chart_dpi: int = 300
    chart_colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    
    # Output settings
    output_directory: str = "validation_reports"
    filename_template: str = "validation_report_{timestamp}_{report_type}"
    include_timestamp: bool = True
    
    # Content settings
    max_trades_displayed: int = 100
    include_trade_details: bool = True
    include_performance_metrics: bool = True
    include_risk_metrics: bool = True
    
    # Template settings
    custom_template_path: Optional[str] = None
    use_custom_styling: bool = True


@dataclass
class QualityDashboard:
    """Quality dashboard data structure"""
    
    # Overall metrics
    overall_quality_score: float = 0.0
    quality_level: str = "unknown"
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    
    # Quality breakdown
    data_quality_score: float = 0.0
    execution_quality_score: float = 0.0
    statistical_quality_score: float = 0.0
    
    # Trend data
    quality_trend: List[Dict[str, Any]] = field(default_factory=list)
    
    # Issues and warnings
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Performance metrics
    average_validation_time: float = 0.0
    validation_success_rate: float = 0.0
    
    # Charts data
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    metric_scores: Dict[str, float] = field(default_factory=dict)
    trend_data: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    
    # Report metadata
    report_id: str
    report_type: ReportType
    generated_at: datetime
    generated_by: str = "validation_system"
    
    # Validation data
    strategy_id: str = ""
    validation_period: Optional[Tuple[datetime, datetime]] = None
    
    # Quality metrics
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Comparison results
    comparison_results: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical analysis
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Regression test results
    regression_results: Dict[str, Any] = field(default_factory=dict)
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Charts and visualizations
    charts: Dict[str, str] = field(default_factory=dict)  # Base64 encoded images
    
    # Raw data
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'generated_at': self.generated_at.isoformat(),
            'generated_by': self.generated_by,
            'strategy_id': self.strategy_id,
            'validation_period': {
                'start': self.validation_period[0].isoformat() if self.validation_period else None,
                'end': self.validation_period[1].isoformat() if self.validation_period else None
            } if self.validation_period else None,
            'quality_metrics': self.quality_metrics,
            'comparison_results': self.comparison_results,
            'statistical_analysis': self.statistical_analysis,
            'regression_results': self.regression_results,
            'summary': self.summary,
            'charts': self.charts,
            'raw_data': self.raw_data
        }


class ValidationReporter:
    """Generates comprehensive validation reports"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        if config.chart_style:
            plt.style.use(config.chart_style)
    
    def generate_quality_report(
        self,
        quality_data: Dict[str, Any],
        strategy_id: str = "",
        validation_period: Optional[Tuple[datetime, datetime]] = None
    ) -> ValidationReport:
        """Generate quality validation report"""
        
        report_id = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process quality data
        quality_metrics = self._process_quality_metrics(quality_data)
        summary = self._generate_quality_summary(quality_metrics)
        
        # Generate charts
        charts = {}
        if self.config.include_charts:
            charts = self._generate_quality_charts(quality_data)
        
        return ValidationReport(
            report_id=report_id,
            report_type=ReportType.QUALITY_REPORT,
            generated_at=datetime.now(),
            strategy_id=strategy_id,
            validation_period=validation_period,
            quality_metrics=quality_metrics,
            summary=summary,
            charts=charts,
            raw_data=quality_data
        )
    
    def generate_comparison_report(
        self,
        comparison_data: Dict[str, Any],
        strategy_id: str = "",
        validation_period: Optional[Tuple[datetime, datetime]] = None
    ) -> ValidationReport:
        """Generate cross-platform comparison report"""
        
        report_id = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process comparison data
        comparison_results = self._process_comparison_data(comparison_data)
        summary = self._generate_comparison_summary(comparison_results)
        
        # Generate charts
        charts = {}
        if self.config.include_charts:
            charts = self._generate_comparison_charts(comparison_data)
        
        return ValidationReport(
            report_id=report_id,
            report_type=ReportType.COMPARISON_REPORT,
            generated_at=datetime.now(),
            strategy_id=strategy_id,
            validation_period=validation_period,
            comparison_results=comparison_results,
            summary=summary,
            charts=charts,
            raw_data=comparison_data
        )
    
    def generate_statistical_report(
        self,
        statistical_data: Dict[str, Any],
        strategy_id: str = "",
        validation_period: Optional[Tuple[datetime, datetime]] = None
    ) -> ValidationReport:
        """Generate statistical validation report"""
        
        report_id = f"statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process statistical data
        statistical_analysis = self._process_statistical_data(statistical_data)
        summary = self._generate_statistical_summary(statistical_analysis)
        
        # Generate charts
        charts = {}
        if self.config.include_charts:
            charts = self._generate_statistical_charts(statistical_data)
        
        return ValidationReport(
            report_id=report_id,
            report_type=ReportType.STATISTICAL_REPORT,
            generated_at=datetime.now(),
            strategy_id=strategy_id,
            validation_period=validation_period,
            statistical_analysis=statistical_analysis,
            summary=summary,
            charts=charts,
            raw_data=statistical_data
        )
    
    def generate_regression_report(
        self,
        regression_data: Dict[str, Any],
        strategy_id: str = "",
        validation_period: Optional[Tuple[datetime, datetime]] = None
    ) -> ValidationReport:
        """Generate regression test report"""
        
        report_id = f"regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process regression data
        regression_results = self._process_regression_data(regression_data)
        summary = self._generate_regression_summary(regression_results)
        
        # Generate charts
        charts = {}
        if self.config.include_charts:
            charts = self._generate_regression_charts(regression_data)
        
        return ValidationReport(
            report_id=report_id,
            report_type=ReportType.REGRESSION_REPORT,
            generated_at=datetime.now(),
            strategy_id=strategy_id,
            validation_period=validation_period,
            regression_results=regression_results,
            summary=summary,
            charts=charts,
            raw_data=regression_data
        )
    
    def generate_dashboard(
        self,
        dashboard_data: QualityDashboard,
        strategy_id: str = "",
        validation_period: Optional[Tuple[datetime, datetime]] = None
    ) -> ValidationReport:
        """Generate quality dashboard"""
        
        report_id = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process dashboard data
        dashboard_metrics = self._process_dashboard_data(dashboard_data)
        summary = self._generate_dashboard_summary(dashboard_data)
        
        # Generate charts
        charts = {}
        if self.config.include_charts:
            charts = self._generate_dashboard_charts(dashboard_data)
        
        return ValidationReport(
            report_id=report_id,
            report_type=ReportType.DASHBOARD,
            generated_at=datetime.now(),
            strategy_id=strategy_id,
            validation_period=validation_period,
            quality_metrics=dashboard_metrics,
            summary=summary,
            charts=charts,
            raw_data=dashboard_data.__dict__
        )
    
    def save_report(self, report: ValidationReport, filename: Optional[str] = None) -> str:
        """Save report to file"""
        
        if filename is None:
            timestamp = report.generated_at.strftime('%Y%m%d_%H%M%S')
            filename = f"validation_report_{timestamp}_{report.report_type.value}.{self.config.report_format.value}"
        
        filepath = self.output_dir / filename
        
        if self.config.report_format == ReportFormat.JSON:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        
        elif self.config.report_format == ReportFormat.HTML:
            html_content = self._generate_html_report(report)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        elif self.config.report_format == ReportFormat.CSV:
            self._generate_csv_report(report, filepath)
        
        self.logger.info(f"Saved validation report: {filepath}")
        return str(filepath)
    
    def _process_quality_metrics(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process quality metrics data"""
        
        metrics = {}
        
        # Extract quality scores
        if 'metrics' in quality_data:
            metrics['quality_scores'] = quality_data['metrics']
        
        # Extract issues and warnings
        if 'issues' in quality_data:
            metrics['issues'] = quality_data['issues']
        
        if 'warnings' in quality_data:
            metrics['warnings'] = quality_data['warnings']
        
        # Extract recommendations
        if 'recommendations' in quality_data:
            metrics['recommendations'] = quality_data['recommendations']
        
        # Calculate quality level
        if 'metrics' in quality_data and 'overall_quality_score' in quality_data['metrics']:
            score = quality_data['metrics']['overall_quality_score']
            if score >= 0.99:
                metrics['quality_level'] = 'excellent'
            elif score >= 0.95:
                metrics['quality_level'] = 'good'
            elif score >= 0.90:
                metrics['quality_level'] = 'acceptable'
            else:
                metrics['quality_level'] = 'poor'
        
        return metrics
    
    def _process_comparison_data(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process comparison data"""
        
        results = {}
        
        # Extract metric comparisons
        if 'metric_comparisons' in comparison_data:
            results['metric_comparisons'] = comparison_data['metric_comparisons']
        
        # Extract trade comparisons
        if 'trade_comparisons' in comparison_data:
            results['trade_comparisons'] = comparison_data['trade_comparisons']
        
        # Extract overall statistics
        if 'overall_status' in comparison_data:
            results['overall_status'] = comparison_data['overall_status']
        
        if 'overall_agreement' in comparison_data:
            results['overall_agreement'] = comparison_data['overall_agreement']
        
        # Extract critical differences
        if 'critical_differences' in comparison_data:
            results['critical_differences'] = comparison_data['critical_differences']
        
        return results
    
    def _process_statistical_data(self, statistical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process statistical data"""
        
        analysis = {}
        
        # Extract distribution analysis
        if 'distribution_analysis' in statistical_data:
            analysis['distribution'] = statistical_data['distribution_analysis']
        
        # Extract correlation analysis
        if 'correlation_analysis' in statistical_data:
            analysis['correlation'] = statistical_data['correlation_analysis']
        
        # Extract statistical tests
        if 'statistical_tests' in statistical_data:
            analysis['tests'] = statistical_data['statistical_tests']
        
        return analysis
    
    def _process_regression_data(self, regression_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process regression data"""
        
        results = {}
        
        # Extract test results
        if 'test_results' in regression_data:
            results['test_results'] = regression_data['test_results']
        
        # Extract suite statistics
        if 'suite_statistics' in regression_data:
            results['suite_statistics'] = regression_data['suite_statistics']
        
        # Extract overall status
        if 'overall_status' in regression_data:
            results['overall_status'] = regression_data['overall_status']
        
        return results
    
    def _process_dashboard_data(self, dashboard_data: QualityDashboard) -> Dict[str, Any]:
        """Process dashboard data"""
        
        return {
            'overall_quality_score': dashboard_data.overall_quality_score,
            'quality_level': dashboard_data.quality_level,
            'total_validations': dashboard_data.total_validations,
            'passed_validations': dashboard_data.passed_validations,
            'failed_validations': dashboard_data.failed_validations,
            'data_quality_score': dashboard_data.data_quality_score,
            'execution_quality_score': dashboard_data.execution_quality_score,
            'statistical_quality_score': dashboard_data.statistical_quality_score,
            'critical_issues': dashboard_data.critical_issues,
            'warnings': dashboard_data.warnings,
            'recommendations': dashboard_data.recommendations,
            'average_validation_time': dashboard_data.average_validation_time,
            'validation_success_rate': dashboard_data.validation_success_rate
        }
    
    def _generate_quality_summary(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality summary"""
        
        summary = {
            'overall_quality': quality_metrics.get('quality_level', 'unknown'),
            'quality_score': quality_metrics.get('quality_scores', {}).get('overall_quality_score', 0.0),
            'total_issues': len(quality_metrics.get('issues', [])),
            'total_warnings': len(quality_metrics.get('warnings', [])),
            'total_recommendations': len(quality_metrics.get('recommendations', []))
        }
        
        return summary
    
    def _generate_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary"""
        
        summary = {
            'overall_status': comparison_results.get('overall_status', 'unknown'),
            'overall_agreement': comparison_results.get('overall_agreement', 0.0),
            'total_metric_comparisons': len(comparison_results.get('metric_comparisons', [])),
            'total_trade_comparisons': len(comparison_results.get('trade_comparisons', [])),
            'critical_differences': len(comparison_results.get('critical_differences', []))
        }
        
        return summary
    
    def _generate_statistical_summary(self, statistical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical summary"""
        
        summary = {
            'distribution_analysis_completed': 'distribution' in statistical_analysis,
            'correlation_analysis_completed': 'correlation' in statistical_analysis,
            'statistical_tests_completed': 'tests' in statistical_analysis
        }
        
        return summary
    
    def _generate_regression_summary(self, regression_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate regression summary"""
        
        summary = {
            'overall_status': regression_results.get('overall_status', 'unknown'),
            'total_tests': len(regression_results.get('test_results', [])),
            'suite_statistics': regression_results.get('suite_statistics', {})
        }
        
        return summary
    
    def _generate_dashboard_summary(self, dashboard_data: QualityDashboard) -> Dict[str, Any]:
        """Generate dashboard summary"""
        
        summary = {
            'overall_quality_score': dashboard_data.overall_quality_score,
            'quality_level': dashboard_data.quality_level,
            'validation_success_rate': dashboard_data.validation_success_rate,
            'total_validations': dashboard_data.total_validations,
            'critical_issues_count': len(dashboard_data.critical_issues),
            'warnings_count': len(dashboard_data.warnings),
            'recommendations_count': len(dashboard_data.recommendations)
        }
        
        return summary
    
    def _generate_quality_charts(self, quality_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate quality charts"""
        
        charts = {}
        
        try:
            # Quality score distribution
            if 'metrics' in quality_data:
                fig, ax = plt.subplots(figsize=self.config.chart_size)
                
                metrics = quality_data['metrics']
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                
                # Filter out non-numeric values
                numeric_data = []
                numeric_names = []
                for name, value in zip(metric_names, metric_values):
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        numeric_data.append(value)
                        numeric_names.append(name)
                
                if numeric_data:
                    bars = ax.bar(numeric_names, numeric_data, color=self.config.chart_colors[:len(numeric_names)])
                    ax.set_title('Quality Metrics Distribution')
                    ax.set_ylabel('Score')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, numeric_data):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    charts['quality_metrics'] = self._chart_to_base64(fig)
                    plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Failed to generate quality charts: {e}")
        
        return charts
    
    def _generate_comparison_charts(self, comparison_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate comparison charts"""
        
        charts = {}
        
        # Metric comparison chart
        if 'metric_comparisons' in comparison_data:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            comparisons = comparison_data['metric_comparisons']
            metric_names = [comp['metric_name'] for comp in comparisons]
            platform_a_values = [comp['platform_a_value'] for comp in comparisons]
            platform_b_values = [comp['platform_b_value'] for comp in comparisons]
            
            x = np.arange(len(metric_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, platform_a_values, width, label='Platform A', 
                          color=self.config.chart_colors[0])
            bars2 = ax.bar(x + width/2, platform_b_values, width, label='Platform B', 
                          color=self.config.chart_colors[1])
            
            ax.set_title('Metric Comparison')
            ax.set_ylabel('Value')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names, rotation=45)
            ax.legend()
            
            plt.tight_layout()
            charts['metric_comparison'] = self._chart_to_base64(fig)
            plt.close(fig)
        
        return charts
    
    def _generate_statistical_charts(self, statistical_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate statistical charts"""
        
        charts = {}
        
        # Return distribution chart
        if 'returns' in statistical_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.chart_size)
            
            returns = statistical_data['returns']
            
            # Histogram
            ax1.hist(returns, bins=50, alpha=0.7, color=self.config.chart_colors[0])
            ax1.set_title('Return Distribution')
            ax1.set_xlabel('Returns')
            ax1.set_ylabel('Frequency')
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot')
            
            plt.tight_layout()
            charts['return_distribution'] = self._chart_to_base64(fig)
            plt.close(fig)
        
        return charts
    
    def _generate_regression_charts(self, regression_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate regression charts"""
        
        charts = {}
        
        # Test results chart
        if 'test_results' in regression_data:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            test_results = regression_data['test_results']
            status_counts = {}
            for result in test_results:
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            labels = list(status_counts.keys())
            sizes = list(status_counts.values())
            colors = self.config.chart_colors[:len(labels)]
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title('Test Results Distribution')
            
            charts['test_results'] = self._chart_to_base64(fig)
            plt.close(fig)
        
        return charts
    
    def _generate_dashboard_charts(self, dashboard_data: QualityDashboard) -> Dict[str, str]:
        """Generate dashboard charts"""
        
        charts = {}
        
        # Quality trend chart
        if dashboard_data.quality_trend:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            dates = [item['date'] for item in dashboard_data.quality_trend]
            scores = [item['score'] for item in dashboard_data.quality_trend]
            
            ax.plot(dates, scores, marker='o', color=self.config.chart_colors[0])
            ax.set_title('Quality Score Trend')
            ax.set_ylabel('Quality Score')
            ax.set_ylim(0, 1)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            charts['quality_trend'] = self._chart_to_base64(fig)
            plt.close(fig)
        
        # Quality distribution chart
        if dashboard_data.quality_distribution:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            labels = list(dashboard_data.quality_distribution.keys())
            sizes = list(dashboard_data.quality_distribution.values())
            colors = self.config.chart_colors[:len(labels)]
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title('Quality Level Distribution')
            
            charts['quality_distribution'] = self._chart_to_base64(fig)
            plt.close(fig)
        
        return charts
    
    def _chart_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=self.config.chart_dpi, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML report"""
        
        try:
            # Load template
            template_path = self.config.custom_template_path or "templates/validation_report.html"
            
            try:
                with open(template_path, 'r') as f:
                    template_content = f.read()
            except FileNotFoundError:
                # Use default template
                template_content = self._get_default_html_template()
            
            template = Template(template_content)
            
            # Prepare template data
            template_data = {
                'report': report.to_dict(),
                'config': self.config.__dict__,
                'generated_at': report.generated_at.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return template.render(**template_data)
        except Exception as e:
            self.logger.warning(f"Failed to generate HTML report: {e}")
            # Return a simple HTML report
            return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report</title>
</head>
<body>
    <h1>Validation Report</h1>
    <p>Report ID: {report.report_id}</p>
    <p>Generated At: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Strategy ID: {report.strategy_id}</p>
    <p>Report Type: {report.report_type.value}</p>
    <p>Error generating detailed report: {str(e)}</p>
</body>
</html>
            """
    
    def _generate_csv_report(self, report: ValidationReport, filepath: Path):
        """Generate CSV report"""
        
        # Create CSV data
        csv_data = []
        
        # Add summary data
        if report.summary:
            for key, value in report.summary.items():
                csv_data.append({'metric': key, 'value': value, 'category': 'summary'})
        
        # Add quality metrics
        if report.quality_metrics:
            for key, value in report.quality_metrics.items():
                csv_data.append({'metric': key, 'value': value, 'category': 'quality'})
        
        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
    
    def _get_default_html_template(self) -> str:
        """Get default HTML template"""
        
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; }
        .summary { background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007acc; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Validation Report</h1>
        <p><strong>Report ID:</strong> {{ report.report_id }}</p>
        <p><strong>Report Type:</strong> {{ report.report_type }}</p>
        <p><strong>Generated At:</strong> {{ generated_at }}</p>
        <p><strong>Strategy ID:</strong> {{ report.strategy_id }}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="summary">
            {% for key, value in report.summary.items() %}
            <p><strong>{{ key }}:</strong> {{ value }}</p>
            {% endfor %}
        </div>
    </div>
    
    {% if report.charts %}
    <div class="section">
        <h2>Charts</h2>
        {% for chart_name, chart_data in report.charts.items() %}
        <div class="chart">
            <h3>{{ chart_name.replace('_', ' ').title() }}</h3>
            <img src="{{ chart_data }}" alt="{{ chart_name }}">
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="section">
        <h2>Quality Metrics</h2>
        {% for key, value in report.quality_metrics.items() %}
        <div class="metric">
            <strong>{{ key }}:</strong> {{ value }}
        </div>
        {% endfor %}
    </div>
</body>
</html>
        """
