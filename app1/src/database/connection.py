"""
Database connection and session management for App1
Supports both PostgreSQL and InfluxDB
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# PostgreSQL configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/axf_bot_db"
)

# InfluxDB configuration
INFLUX_CONFIG = {
    'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
    'token': os.getenv('INFLUXDB_TOKEN', ''),
    'org': os.getenv('INFLUXDB_ORG', 'axf-bot'),
    'bucket': os.getenv('INFLUXDB_BUCKET', 'forex_data')
}

# Create SQLAlchemy engine for PostgreSQL
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# InfluxDB client (lazy initialization)
_influx_client = None

def get_db() -> Session:
    """
    Dependency to get PostgreSQL database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """
    Get a PostgreSQL database session (for use outside of FastAPI dependency injection)
    """
    return SessionLocal()

def get_influx_client():
    """
    Get InfluxDB client (lazy initialization)
    """
    global _influx_client
    
    if _influx_client is None:
        try:
            from .influx_client import InfluxDBClientWrapper
            _influx_client = InfluxDBClientWrapper()
            
            # Try to connect
            if not _influx_client.connect():
                logger.warning("InfluxDB connection failed - time series features may not be available")
                _influx_client = None
        except Exception as e:
            logger.warning(f"Could not initialize InfluxDB client: {e}")
            _influx_client = None
    
    return _influx_client

def init_db():
    """
    Initialize database tables
    """
    try:
        # Import all models to ensure they are registered
        from . import models
        
        # Create all PostgreSQL tables
        Base.metadata.create_all(bind=engine)
        logger.info("PostgreSQL database tables created successfully")
        
        # Initialize InfluxDB if available
        influx_client = get_influx_client()
        if influx_client:
            logger.info("InfluxDB client initialized successfully")
        else:
            logger.warning("InfluxDB client not available - time series features disabled")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def check_db_connection() -> bool:
    """
    Check if PostgreSQL database connection is working
    """
    try:
        with engine.connect() as connection:
            from sqlalchemy import text
            connection.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False

def check_influx_connection() -> bool:
    """
    Check if InfluxDB connection is working
    """
    try:
        influx_client = get_influx_client()
        if influx_client:
            return True
        return False
    except Exception as e:
        logger.error(f"InfluxDB connection failed: {e}")
        return False

def get_database_status() -> dict:
    """
    Get status of all database connections
    """
    return {
        'postgresql': {
            'connected': check_db_connection(),
            'url': DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'unknown'
        },
        'influxdb': {
            'connected': check_influx_connection(),
            'url': INFLUX_CONFIG['url'],
            'org': INFLUX_CONFIG['org'],
            'bucket': INFLUX_CONFIG['bucket']
        }
    }
