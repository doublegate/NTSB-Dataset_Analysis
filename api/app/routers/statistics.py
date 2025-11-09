"""
Statistics Router

Endpoints for statistical summaries and trends.
"""

from fastapi import APIRouter, HTTPException, Query
from app.schemas.statistics import YearlyStats, StateStats, AircraftStats, SummaryStats
from app.database import get_db_connection
from app.crud.statistics import (
    get_summary_stats,
    get_yearly_stats,
    get_state_stats,
    get_aircraft_stats,
    get_decade_stats,
    get_seasonal_patterns,
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/summary",
    response_model=SummaryStats,
    summary="Overall Statistics",
    description="Get overall database summary statistics",
)
async def get_summary():
    """
    Get overall database summary.

    Returns:
    - Total events
    - Date range coverage
    - Total fatalities
    - Total aircraft
    - States covered
    - Most dangerous/safest years
    """
    try:
        with get_db_connection() as conn:
            stats = get_summary_stats(conn)

        if not stats:
            raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

        # Convert dates to strings
        stats["date_range_start"] = str(stats["date_range_start"])
        stats["date_range_end"] = str(stats["date_range_end"])

        return SummaryStats(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summary stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/yearly",
    response_model=list[YearlyStats],
    summary="Yearly Statistics",
    description="Get accident statistics by year",
)
async def get_yearly():
    """
    Get yearly statistics from materialized view.

    Returns accident counts, fatalities, and trends for each year.
    """
    try:
        with get_db_connection() as conn:
            stats = get_yearly_stats(conn)

        return [YearlyStats(**s) for s in stats]

    except Exception as e:
        logger.error(f"Error getting yearly stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/states",
    response_model=list[StateStats],
    summary="State Statistics",
    description="Get accident statistics by state",
)
async def get_states():
    """
    Get state-level statistics from materialized view.

    Returns accident counts and geographic averages for each state.
    """
    try:
        with get_db_connection() as conn:
            stats = get_state_stats(conn)

        return [StateStats(**s) for s in stats]

    except Exception as e:
        logger.error(f"Error getting state stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/aircraft",
    response_model=list[AircraftStats],
    summary="Aircraft Statistics",
    description="Get accident statistics by aircraft make/model",
)
async def get_aircraft(
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
):
    """
    Get aircraft make/model statistics from materialized view.

    Returns top aircraft types by accident count with fatal rates.
    """
    try:
        with get_db_connection() as conn:
            stats = get_aircraft_stats(conn, limit=limit)

        return [AircraftStats(**s) for s in stats]

    except Exception as e:
        logger.error(f"Error getting aircraft stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/decades",
    summary="Decade Statistics",
    description="Get accident statistics by decade",
)
async def get_decades():
    """
    Get decade-level statistics from materialized view.

    Returns aggregated statistics for each 10-year period.
    """
    try:
        with get_db_connection() as conn:
            stats = get_decade_stats(conn)

        return stats

    except Exception as e:
        logger.error(f"Error getting decade stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/seasonal",
    summary="Seasonal Patterns",
    description="Get seasonal accident patterns (aggregated by month)",
)
async def get_seasonal():
    """
    Get seasonal patterns aggregated across all years.

    Returns monthly averages and fatal rates.
    """
    try:
        with get_db_connection() as conn:
            stats = get_seasonal_patterns(conn)

        return stats

    except Exception as e:
        logger.error(f"Error getting seasonal patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
