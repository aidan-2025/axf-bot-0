"""
Health Check Routes for App1 (Strategy Generator)
Provides comprehensive health monitoring and system status
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from src.database.connection import get_db, get_database_status, get_influx_client
import psutil
import time
import os
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import text

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    Returns 200 if service is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "app1-strategy-generator",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with system metrics
    Returns comprehensive system status
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Service-specific checks
        checks = {
            "memory_usage": f"{memory.percent}%",
            "cpu_usage": f"{cpu_percent}%",
            "disk_usage": f"{disk.percent}%",
            "available_memory": f"{memory.available / (1024**3):.2f} GB",
            "free_disk_space": f"{disk.free / (1024**3):.2f} GB"
        }
        
        # Overall health status
        overall_status = "healthy"
        if memory.percent > 90:
            overall_status = "warning"
        if memory.percent > 95 or disk.percent > 95:
            overall_status = "critical"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "app1-strategy-generator",
            "version": "1.0.0",
            "checks": checks,
            "uptime": time.time() - psutil.boot_time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes/Docker
    Returns 200 if service is ready to accept traffic
    """
    try:
        # Check if required directories exist
        required_dirs = ["data", "models", "logs"]
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "app1-strategy-generator"
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@router.get("/health/live")
async def liveness_check():
    """
    Liveness check for Kubernetes/Docker
    Returns 200 if service is alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "app1-strategy-generator"
    }

@router.get("/health/database")
async def database_health_check(db: Session = Depends(get_db)):
    """
    Database health check endpoint
    Returns status of PostgreSQL and InfluxDB connections
    """
    try:
        # Get database status
        db_status = get_database_status()
        
        # Test PostgreSQL with a simple query
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
            postgres_status = "connected"
        except Exception as e:
            postgres_status = f"error: {str(e)}"
            db_status['postgresql']['connected'] = False
        
        # Test InfluxDB
        influx_client = get_influx_client()
        if influx_client:
            influx_status = "connected"
        else:
            influx_status = "not_available"
            db_status['influxdb']['connected'] = False
        
        return {
            "status": "healthy" if db_status['postgresql']['connected'] else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "app1-strategy-generator",
            "databases": {
                "postgresql": {
                    "status": postgres_status,
                    "connected": db_status['postgresql']['connected'],
                    "url": db_status['postgresql']['url']
                },
                "influxdb": {
                    "status": influx_status,
                    "connected": db_status['influxdb']['connected'],
                    "url": db_status['influxdb']['url'],
                    "org": db_status['influxdb']['org'],
                    "bucket": db_status['influxdb']['bucket']
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database health check failed: {str(e)}")

@router.get("/metrics")
async def get_metrics():
    """
    Prometheus-style metrics endpoint
    Returns system and application metrics
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        metrics = {
            "system_cpu_percent": cpu_percent,
            "system_memory_percent": memory.percent,
            "system_memory_available_bytes": memory.available,
            "system_disk_percent": disk.percent,
            "system_disk_free_bytes": disk.free,
            "process_memory_rss_bytes": process_memory.rss,
            "process_memory_vms_bytes": process_memory.vms,
            "process_cpu_percent": process.cpu_percent(),
            "process_num_threads": process.num_threads(),
            "process_create_time": process.create_time()
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "app1-strategy-generator",
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

@router.get("/health/validation")
async def validation_stats() -> Dict[str, Any]:
    """Return data validation anomaly counters if available."""
    try:
        # Lazy import to avoid circular deps
        from src.data_ingestion.engines.ingestion_service import DataIngestionService  # type: ignore
    except Exception:
        DataIngestionService = None  # type: ignore

    stats: Dict[str, Any] = {
        "status": "ok",
        "anomalies": {}
    }

    try:
        # We may not have a running singleton; attempt to introspect via module state
        if DataIngestionService is not None:
            # Try to read a global/reference if one exists
            service_ref = None
            try:
                # Some apps maintain a module-level service
                from src.api.state import ingestion_service  # type: ignore
                service_ref = ingestion_service
            except Exception:
                service_ref = None

            if service_ref and getattr(service_ref, "data_validator", None):
                stats["anomalies"] = getattr(service_ref.data_validator, "anomaly_counts", {})
            else:
                stats["anomalies"] = {}
        else:
            stats["anomalies"] = {}
    except Exception as e:
        # Do not fail health just because stats are unavailable
        stats["error"] = str(e)

    return stats