"""
Health Check Router

Endpoints for API and database health monitoring.
"""

from fastapi import APIRouter, HTTPException
from app.schemas.common import HealthResponse, DatabaseHealthResponse
from app.database import test_database_connection, get_database_info
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthResponse, summary="API Health Check")
async def health_check():
    """
    Simple health check endpoint.

    Returns API status and database connectivity.
    """
    db_connected = test_database_connection()

    return HealthResponse(
        status="healthy" if db_connected else "unhealthy",
        database="connected" if db_connected else "disconnected",
        version="1.0.0",
    )


@router.get(
    "/database",
    response_model=DatabaseHealthResponse,
    summary="Database Health Check",
)
async def database_health():
    """
    Detailed database health check.

    Returns PostgreSQL version, database size, event count, and connection pool stats.
    """
    db_info = get_database_info()

    if "error" in db_info:
        raise HTTPException(
            status_code=503, detail=f"Database error: {db_info['error']}"
        )

    return DatabaseHealthResponse(
        status="healthy",
        version=db_info["version"],
        size=db_info["size"],
        event_count=db_info["event_count"],
        pool_stats=db_info["pool_stats"],
    )
