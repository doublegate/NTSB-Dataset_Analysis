"""Database connection utilities for Streamlit dashboard.

This module provides connection pooling and database utilities for the
NTSB Aviation Accident Database dashboard.
"""

import os
from typing import Optional
from psycopg2 import pool
from psycopg2.extensions import connection


# Connection pool (singleton pattern)
_pool: Optional[pool.SimpleConnectionPool] = None


def get_connection_pool() -> pool.SimpleConnectionPool:
    """Get or create database connection pool.

    Returns:
        SimpleConnectionPool: Connection pool instance
    """
    global _pool

    if _pool is None:
        # Get database parameters from environment or use defaults
        db_user = os.getenv("DB_USER", os.getenv("USER", "parobek"))
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "ntsb_aviation")
        db_password = os.getenv("DB_PASSWORD", "")

        # Create connection pool
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password if db_password else None,
        )

    return _pool


def get_connection() -> connection:
    """Get a connection from the pool.

    Returns:
        connection: PostgreSQL connection object
    """
    pool_instance = get_connection_pool()
    return pool_instance.getconn()


def release_connection(conn: connection) -> None:
    """Return a connection to the pool.

    Args:
        conn: Connection to return to pool
    """
    pool_instance = get_connection_pool()
    pool_instance.putconn(conn)


def close_all_connections() -> None:
    """Close all connections in the pool.

    Call this when shutting down the application.
    """
    global _pool

    if _pool is not None:
        _pool.closeall()
        _pool = None
