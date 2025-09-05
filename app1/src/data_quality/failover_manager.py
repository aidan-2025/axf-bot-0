#!/usr/bin/env python3
"""
Failover Manager
Manages automatic failover to backup data sources
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SourceStatus(Enum):
    """Source status for failover management"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class FailoverConfig:
    """Failover configuration"""
    max_consecutive_failures: int = 3
    recovery_timeout_minutes: int = 30
    health_check_interval_seconds: int = 60
    backup_sources: Dict[str, List[str]] = None

class FailoverManager:
    """Manages automatic failover to backup sources"""
    
    def __init__(self, config: Optional[FailoverConfig] = None):
        self.config = config or FailoverConfig()
        self.source_statuses = {}
        self.failure_counts = {}
        self.last_failure_times = {}
        self.active_backups = {}
        
    async def initialize(self):
        """Initialize failover manager"""
        logger.info("Initializing failover manager")
        # Initialize backup source mappings
        self.config.backup_sources = {
            'oanda': ['alpha_vantage', 'forex_factory'],
            'alpha_vantage': ['oanda', 'forex_factory'],
            'forex_factory': ['central_bank'],
            'central_bank': ['twitter'],
            'twitter': ['forex_factory']
        }
        
    async def record_failure(self, source: str, error: str):
        """Record a failure for a source"""
        try:
            current_time = datetime.now()
            
            # Update failure count
            self.failure_counts[source] = self.failure_counts.get(source, 0) + 1
            self.last_failure_times[source] = current_time
            
            # Update source status
            if self.failure_counts[source] >= self.config.max_consecutive_failures:
                self.source_statuses[source] = SourceStatus.FAILED
                logger.warning(f"Source {source} marked as FAILED after {self.failure_counts[source]} consecutive failures")
            else:
                self.source_statuses[source] = SourceStatus.DEGRADED
                logger.warning(f"Source {source} marked as DEGRADED after {self.failure_counts[source]} failures")
            
        except Exception as e:
            logger.error(f"Error recording failure for {source}: {e}")
    
    async def should_failover(self, source: str) -> bool:
        """Check if failover should be triggered for a source"""
        try:
            status = self.source_statuses.get(source, SourceStatus.HEALTHY)
            return status == SourceStatus.FAILED
            
        except Exception as e:
            logger.error(f"Error checking failover condition for {source}: {e}")
            return False
    
    async def get_backup_source(self, primary_source: str) -> Optional[str]:
        """Get the best backup source for a primary source"""
        try:
            backup_sources = self.config.backup_sources.get(primary_source, [])
            
            for backup in backup_sources:
                backup_status = self.source_statuses.get(backup, SourceStatus.HEALTHY)
                if backup_status == SourceStatus.HEALTHY:
                    return backup
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting backup source for {primary_source}: {e}")
            return None
    
    async def activate_backup(self, primary_source: str, backup_source: str):
        """Activate a backup source"""
        try:
            self.active_backups[primary_source] = backup_source
            logger.info(f"Activated backup {backup_source} for primary {primary_source}")
            
        except Exception as e:
            logger.error(f"Error activating backup {backup_source} for {primary_source}: {e}")
    
    async def update_source_status(self, source: str, status: SourceStatus):
        """Update source status"""
        try:
            self.source_statuses[source] = status
            
            # Reset failure count if source is healthy
            if status == SourceStatus.HEALTHY:
                self.failure_counts[source] = 0
                # Remove from active backups if it was a backup
                self.active_backups = {k: v for k, v in self.active_backups.items() if v != source}
            
        except Exception as e:
            logger.error(f"Error updating status for {source}: {e}")
    
    async def get_recent_failures(self, source: str) -> int:
        """Get recent failure count for a source"""
        return self.failure_counts.get(source, 0)
    
    async def shutdown(self):
        """Shutdown failover manager"""
        logger.info("Shutting down failover manager")

