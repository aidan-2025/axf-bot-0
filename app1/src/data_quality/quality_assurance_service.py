#!/usr/bin/env python3
"""
Data Quality Assurance Service
Main orchestrator for data quality monitoring, validation, and failover
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from enum import Enum

from ..data_ingestion.storage.influxdb_writer import InfluxDBWriter
from ..data_ingestion.cache.redis_cache import RedisCacheManager
from .real_time_validator import RealTimeValidator
from .data_corrector import DataCorrector
from .failover_manager import FailoverManager
from .quality_metrics import QualityMetrics
from .audit_scheduler import AuditScheduler
from .timezone_synchronizer import TimezoneSynchronizer
from .outlier_detector import OutlierDetector
from .missing_data_handler import MissingDataHandler

logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class DataSourceStatus(Enum):
    """Data source status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class QualityThresholds:
    """Quality thresholds for different metrics"""
    # Data completeness thresholds
    min_completeness: float = 0.95
    max_missing_data: float = 0.05
    
    # Data accuracy thresholds
    max_outlier_ratio: float = 0.02
    max_price_deviation: float = 0.05  # 5% deviation from expected range
    
    # Timeliness thresholds
    max_delay_seconds: int = 30
    max_staleness_minutes: int = 5
    
    # Consistency thresholds
    min_cross_source_agreement: float = 0.90
    max_timezone_offset_hours: int = 1

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    timestamp: datetime
    overall_quality: QualityLevel
    data_sources: Dict[str, DataSourceStatus]
    metrics: Dict[str, float]
    issues: List[str]
    corrections_applied: List[str]
    recommendations: List[str]
    next_audit: datetime

class DataQualityAssuranceService:
    """Main data quality assurance service"""
    
    def __init__(self, 
                 influxdb_writer: InfluxDBWriter,
                 redis_cache: RedisCacheManager,
                 thresholds: Optional[QualityThresholds] = None):
        self.influxdb_writer = influxdb_writer
        self.redis_cache = redis_cache
        self.thresholds = thresholds or QualityThresholds()
        
        # Initialize components
        self.validator = RealTimeValidator(thresholds=self.thresholds)
        self.corrector = DataCorrector()
        self.failover_manager = FailoverManager()
        self.metrics = QualityMetrics()
        self.audit_scheduler = AuditScheduler()
        self.timezone_sync = TimezoneSynchronizer()
        self.outlier_detector = OutlierDetector()
        self.missing_data_handler = MissingDataHandler()
        
        # Service state
        self._running = False
        self._last_audit = None
        self._quality_history = []
        
    async def initialize(self):
        """Initialize the quality assurance service"""
        try:
            logger.info("Initializing data quality assurance service")
            
            # Initialize all components
            await self.validator.initialize()
            await self.corrector.initialize()
            await self.failover_manager.initialize()
            await self.metrics.initialize()
            await self.audit_scheduler.initialize()
            await self.timezone_sync.initialize()
            await self.outlier_detector.initialize()
            await self.missing_data_handler.initialize()
            
            # Start background tasks
            self._running = True
            asyncio.create_task(self._monitor_quality())
            asyncio.create_task(self._process_audits())
            
            logger.info("Data quality assurance service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize quality assurance service: {e}")
            raise
    
    async def validate_data(self, data: Dict[str, Any], source: str) -> Tuple[bool, List[str]]:
        """Validate incoming data in real-time"""
        try:
            is_valid, issues = await self.validator.validate(data, source)
            
            if not is_valid:
                logger.warning(f"Data validation failed for {source}: {issues}")
                # Attempt automatic correction
                corrected_data = await self.corrector.correct(data, issues)
                if corrected_data:
                    logger.info(f"Data corrected for {source}")
                    return True, []
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating data from {source}: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    async def monitor_data_source(self, source: str) -> DataSourceStatus:
        """Monitor a specific data source health"""
        try:
            # Check recent data quality
            quality_score = await self.metrics.get_source_quality(source)
            
            # Check for recent failures
            recent_failures = await self.failover_manager.get_recent_failures(source)
            
            # Check data freshness
            last_update = await self.metrics.get_last_update(source)
            staleness = datetime.now() - last_update if last_update else timedelta(hours=1)
            
            if quality_score < 0.5 or recent_failures > 3:
                status = DataSourceStatus.FAILED
            elif quality_score < 0.8 or staleness > timedelta(minutes=self.thresholds.max_staleness_minutes):
                status = DataSourceStatus.DEGRADED
            elif recent_failures > 0:
                status = DataSourceStatus.RECOVERING
            else:
                status = DataSourceStatus.HEALTHY
            
            # Update failover manager
            await self.failover_manager.update_source_status(source, status)
            
            return status
            
        except Exception as e:
            logger.error(f"Error monitoring data source {source}: {e}")
            return DataSourceStatus.FAILED
    
    async def run_quality_audit(self) -> QualityReport:
        """Run comprehensive quality audit"""
        try:
            logger.info("Starting data quality audit")
            audit_start = datetime.now()
            
            # Get all data sources
            sources = await self._get_active_sources()
            
            # Monitor each source
            source_statuses = {}
            all_issues = []
            all_corrections = []
            
            for source in sources:
                status = await self.monitor_data_source(source)
                source_statuses[source] = status
                
                # Check for specific issues
                issues = await self._check_source_issues(source)
                all_issues.extend(issues)
                
                # Apply corrections if needed
                if status in [DataSourceStatus.DEGRADED, DataSourceStatus.FAILED]:
                    corrections = await self._apply_corrections(source)
                    all_corrections.extend(corrections)
            
            # Calculate overall quality
            overall_quality = await self._calculate_overall_quality(source_statuses)
            
            # Generate metrics
            metrics = await self.metrics.get_comprehensive_metrics()
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(overall_quality, all_issues)
            
            # Create report
            report = QualityReport(
                timestamp=audit_start,
                overall_quality=overall_quality,
                data_sources=source_statuses,
                metrics=metrics,
                issues=all_issues,
                corrections_applied=all_corrections,
                recommendations=recommendations,
                next_audit=audit_start + timedelta(hours=1)
            )
            
            # Store report
            await self._store_quality_report(report)
            self._last_audit = audit_start
            
            logger.info(f"Quality audit completed. Overall quality: {overall_quality.value}")
            return report
            
        except Exception as e:
            logger.error(f"Error running quality audit: {e}")
            raise
    
    async def handle_data_failure(self, source: str, error: str):
        """Handle data source failure"""
        try:
            logger.warning(f"Handling data failure for {source}: {error}")
            
            # Update failover manager
            await self.failover_manager.record_failure(source, error)
            
            # Check if failover is needed
            if await self.failover_manager.should_failover(source):
                backup_source = await self.failover_manager.get_backup_source(source)
                if backup_source:
                    logger.info(f"Failing over from {source} to {backup_source}")
                    await self.failover_manager.activate_backup(source, backup_source)
                else:
                    logger.error(f"No backup available for {source}")
            
            # Update quality metrics
            await self.metrics.record_failure(source, error)
            
        except Exception as e:
            logger.error(f"Error handling data failure for {source}: {e}")
    
    async def _monitor_quality(self):
        """Background quality monitoring"""
        while self._running:
            try:
                # Check for critical issues
                critical_issues = await self._check_critical_issues()
                if critical_issues:
                    logger.warning(f"Critical quality issues detected: {critical_issues}")
                
                # Update quality metrics
                await self.metrics.update_realtime_metrics()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in quality monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _process_audits(self):
        """Process scheduled audits"""
        while self._running:
            try:
                if await self.audit_scheduler.should_run_audit():
                    await self.run_quality_audit()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error processing audits: {e}")
                await asyncio.sleep(600)
    
    async def _get_active_sources(self) -> List[str]:
        """Get list of active data sources"""
        # This would typically query your data sources
        return ["oanda", "alpha_vantage", "forex_factory", "central_bank", "twitter"]
    
    async def _check_source_issues(self, source: str) -> List[str]:
        """Check for specific issues with a data source"""
        issues = []
        
        try:
            # Check data completeness
            completeness = await self.metrics.get_completeness(source)
            if completeness < self.thresholds.min_completeness:
                issues.append(f"Low data completeness: {completeness:.2%}")
            
            # Check for outliers
            outlier_ratio = await self.outlier_detector.get_outlier_ratio(source)
            if outlier_ratio > self.thresholds.max_outlier_ratio:
                issues.append(f"High outlier ratio: {outlier_ratio:.2%}")
            
            # Check timezone consistency
            timezone_issues = await self.timezone_sync.check_timezone_consistency(source)
            if timezone_issues:
                issues.extend(timezone_issues)
            
        except Exception as e:
            issues.append(f"Error checking source issues: {str(e)}")
        
        return issues
    
    async def _apply_corrections(self, source: str) -> List[str]:
        """Apply corrections to a data source"""
        corrections = []
        
        try:
            # Handle missing data
            missing_corrections = await self.missing_data_handler.handle_missing_data(source)
            corrections.extend(missing_corrections)
            
            # Correct outliers
            outlier_corrections = await self.outlier_detector.correct_outliers(source)
            corrections.extend(outlier_corrections)
            
            # Synchronize timezones
            timezone_corrections = await self.timezone_sync.synchronize_timezones(source)
            corrections.extend(timezone_corrections)
            
        except Exception as e:
            corrections.append(f"Error applying corrections: {str(e)}")
        
        return corrections
    
    async def _calculate_overall_quality(self, source_statuses: Dict[str, DataSourceStatus]) -> QualityLevel:
        """Calculate overall data quality level"""
        if not source_statuses:
            return QualityLevel.CRITICAL
        
        # Count statuses
        status_counts = {}
        for status in source_statuses.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_sources = len(source_statuses)
        
        # Calculate quality based on status distribution
        if status_counts.get(DataSourceStatus.FAILED, 0) > total_sources * 0.5:
            return QualityLevel.CRITICAL
        elif status_counts.get(DataSourceStatus.FAILED, 0) > 0 or status_counts.get(DataSourceStatus.DEGRADED, 0) > total_sources * 0.3:
            return QualityLevel.POOR
        elif status_counts.get(DataSourceStatus.DEGRADED, 0) > 0:
            return QualityLevel.FAIR
        elif status_counts.get(DataSourceStatus.RECOVERING, 0) > 0:
            return QualityLevel.GOOD
        else:
            return QualityLevel.EXCELLENT
    
    async def _generate_recommendations(self, quality: QualityLevel, issues: List[str]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if quality == QualityLevel.CRITICAL:
            recommendations.append("Immediate attention required - multiple data sources failing")
            recommendations.append("Consider activating all backup sources")
        
        if quality in [QualityLevel.POOR, QualityLevel.FAIR]:
            recommendations.append("Review data source configurations")
            recommendations.append("Consider increasing monitoring frequency")
        
        if "outlier" in " ".join(issues).lower():
            recommendations.append("Review outlier detection thresholds")
        
        if "timezone" in " ".join(issues).lower():
            recommendations.append("Standardize timezone handling across sources")
        
        if "completeness" in " ".join(issues).lower():
            recommendations.append("Investigate data source reliability")
        
        return recommendations
    
    async def _check_critical_issues(self) -> List[str]:
        """Check for critical quality issues that need immediate attention"""
        critical_issues = []
        
        try:
            # Check if any critical sources are down
            sources = await self._get_active_sources()
            for source in sources:
                status = await self.monitor_data_source(source)
                if status == DataSourceStatus.FAILED:
                    critical_issues.append(f"Critical source {source} is down")
            
            # Check overall data quality
            if self._last_audit:
                time_since_audit = datetime.now() - self._last_audit
                if time_since_audit > timedelta(hours=2):
                    critical_issues.append("No recent quality audit completed")
        
        except Exception as e:
            critical_issues.append(f"Error checking critical issues: {str(e)}")
        
        return critical_issues
    
    async def _store_quality_report(self, report: QualityReport):
        """Store quality report for historical analysis"""
        try:
            # Store in Redis for quick access
            await self.redis_cache.cache_quality_report(report)
            
            # Store in InfluxDB for historical analysis
            await self.influxdb_writer.write_quality_metrics({
                "timestamp": report.timestamp,
                "overall_quality": report.overall_quality.value,
                "total_sources": len(report.data_sources),
                "healthy_sources": sum(1 for s in report.data_sources.values() if s == DataSourceStatus.HEALTHY),
                "issues_count": len(report.issues),
                "corrections_count": len(report.corrections_applied)
            })
            
        except Exception as e:
            logger.error(f"Error storing quality report: {e}")
    
    async def shutdown(self):
        """Shutdown the quality assurance service"""
        logger.info("Shutting down data quality assurance service")
        self._running = False
        
        # Shutdown all components
        await self.validator.shutdown()
        await self.corrector.shutdown()
        await self.failover_manager.shutdown()
        await self.metrics.shutdown()
        await self.audit_scheduler.shutdown()
        await self.timezone_sync.shutdown()
        await self.outlier_detector.shutdown()
        await self.missing_data_handler.shutdown()
        
        logger.info("Data quality assurance service shutdown complete")

