"""Database connection utilities for Streamlit dashboard.

This module provides connection pooling and database utilities for the
NTSB Aviation Accident Database dashboard.

Migrated to SQLAlchemy (2025-11-09) to eliminate pandas UserWarning
about DBAPI2 connections.
"""

import os
from typing import Any
import streamlit as st
from sqlalchemy import create_engine, Engine
from sqlalchemy.pool import QueuePool


@st.cache_resource
def get_engine() -> Engine:
    """Get or create SQLAlchemy engine with connection pooling.

    Uses Streamlit's @cache_resource to create engine once and reuse across sessions.
    Connection pooling is handled by SQLAlchemy's QueuePool.

    Returns:
        Engine: SQLAlchemy engine instance with connection pooling

    Configuration:
        - Pool size: 10 connections
        - Max overflow: 5 additional connections
        - Pool pre-ping: Enabled (auto-reconnect on stale connections)
        - Echo: Disabled (set to True for SQL logging)
    """
    # Get database parameters from environment or use defaults
    db_user = os.getenv("DB_USER", os.getenv("USER", "parobek"))
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "ntsb_aviation")
    db_password = os.getenv("DB_PASSWORD", "")

    # Build connection string
    if db_password:
        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
    else:
        connection_string = f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"

    # Create engine with connection pooling
    engine = create_engine(
        connection_string,
        poolclass=QueuePool,
        pool_size=10,  # Number of connections to keep in pool
        max_overflow=5,  # Additional connections beyond pool_size
        pool_pre_ping=True,  # Verify connections before using
        echo=False,  # Set to True for SQL query logging
    )

    return engine


def get_connection() -> Engine:
    """Get database connection for use with pandas.

    Returns SQLAlchemy engine which is compatible with pd.read_sql().
    This eliminates the UserWarning about DBAPI2 connections.

    Returns:
        Engine: SQLAlchemy engine (compatible with pandas)

    Example:
        >>> conn = get_connection()
        >>> df = pd.read_sql("SELECT * FROM events LIMIT 10", conn)
        >>> release_connection(conn)  # No-op for SQLAlchemy
    """
    return get_engine()


def release_connection(conn: Any) -> None:
    """Release connection back to pool.

    This is a no-op for SQLAlchemy engines as connection pooling
    is handled automatically. Kept for API compatibility with
    previous psycopg2 implementation.

    Args:
        conn: Connection or engine to release (ignored)
    """
    # No-op: SQLAlchemy handles connection pooling automatically
    pass


def close_all_connections() -> None:
    """Close all connections in the pool.

    Call this when shutting down the application.
    Disposes of the SQLAlchemy engine and all pooled connections.
    """
    engine = get_engine()
    engine.dispose()
