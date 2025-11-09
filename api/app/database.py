"""
Database Connection Pool Module

Provides SQLAlchemy connection pooling for PostgreSQL database.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging
from typing import Generator

from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create database engine with connection pooling
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=20,  # Number of connections to maintain in the pool
    max_overflow=10,  # Additional connections allowed above pool_size
    pool_pre_ping=True,  # Test connections before using them
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False,  # Set to True for SQL query logging (development only)
)


@event.listens_for(Engine, "connect")
def set_search_path(dbapi_conn, connection_record):
    """Set PostgreSQL search_path on connection."""
    existing_autocommit = dbapi_conn.autocommit
    dbapi_conn.autocommit = True
    cursor = dbapi_conn.cursor()
    cursor.execute("SET search_path TO public")
    cursor.close()
    dbapi_conn.autocommit = existing_autocommit


@contextmanager
def get_db_connection() -> Generator:
    """
    Get database connection from pool.

    Usage:
        with get_db_connection() as conn:
            result = conn.execute("SELECT * FROM events LIMIT 10")

    Yields:
        SQLAlchemy connection object
    """
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()


def test_database_connection() -> bool:
    """
    Test database connectivity.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            result = conn.execute("SELECT 1")
            result.fetchone()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_database_info() -> dict:
    """
    Get database information for health checks.

    Returns:
        Dictionary with database version, size, and connection stats
    """
    try:
        with get_db_connection() as conn:
            # PostgreSQL version
            version_result = conn.execute("SELECT version()")
            version = version_result.fetchone()[0]

            # Database size
            size_result = conn.execute(
                "SELECT pg_size_pretty(pg_database_size('ntsb_aviation'))"
            )
            db_size = size_result.fetchone()[0]

            # Connection pool stats
            pool_stats = {
                "size": engine.pool.size(),
                "checked_out": engine.pool.checkedout(),
                "overflow": engine.pool.overflow(),
                "checkedin": engine.pool.checkedin(),
            }

            # Event count
            count_result = conn.execute("SELECT COUNT(*) FROM events")
            event_count = count_result.fetchone()[0]

            return {
                "version": version.split(" ")[1],  # Extract version number
                "size": db_size,
                "event_count": event_count,
                "pool_stats": pool_stats,
            }
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {"error": str(e)}
