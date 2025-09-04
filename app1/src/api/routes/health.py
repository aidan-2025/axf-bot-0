"""
Health Check Routes for App1 (Strategy Generator)
Provides comprehensive health monitoring and system status
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app1.src.database.connection import get_db
import psutil
import time
import os
from datetime import datetime
from typing import Dict, Any

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
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Detailed health check with system metrics
    Returns comprehensive system status
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Database connectivity check
        db_status = "healthy"
        try:
            db.execute("SELECT 1")
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Service-specific checks
        checks = {
            "database": db_status,
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
        if db_status != "healthy":
            overall_status = "unhealthy"
        
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
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness check for Kubernetes/Docker
    Returns 200 if service is ready to accept traffic
    """
    try:
        # Check database connectivity
        db.execute("SELECT 1")
        
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
