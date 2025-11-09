"""
Statistics CRUD Operations

Database query functions for statistics endpoints.
"""

from typing import List, Dict, Any
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


def get_summary_stats(conn) -> Dict[str, Any]:
    """
    Get overall database summary statistics.

    Args:
        conn: Database connection

    Returns:
        Dictionary with summary statistics
    """
    query = text(
        """
        WITH stats AS (
            SELECT
                COUNT(*) as total_events,
                MIN(ev_date) as date_range_start,
                MAX(ev_date) as date_range_end,
                SUM(COALESCE(inj_tot_f, 0)) as total_fatalities,
                COUNT(DISTINCT ev_state) as states_covered
            FROM events
        ),
        aircraft_count AS (
            SELECT COUNT(*) as total_aircraft FROM aircraft
        ),
        worst_year AS (
            SELECT ev_year, SUM(COALESCE(inj_tot_f, 0)) as fatalities
            FROM events
            GROUP BY ev_year
            ORDER BY fatalities DESC
            LIMIT 1
        ),
        safest_year AS (
            SELECT ev_year, COUNT(*) as accidents
            FROM events
            GROUP BY ev_year
            ORDER BY accidents ASC
            LIMIT 1
        )
        SELECT
            s.total_events,
            s.date_range_start,
            s.date_range_end,
            EXTRACT(YEAR FROM s.date_range_end) - EXTRACT(YEAR FROM s.date_range_start) + 1 as years_coverage,
            s.total_fatalities,
            ac.total_aircraft,
            s.states_covered,
            wy.ev_year as most_dangerous_year,
            sy.ev_year as safest_year
        FROM stats s
        CROSS JOIN aircraft_count ac
        CROSS JOIN worst_year wy
        CROSS JOIN safest_year sy
        """
    )

    result = conn.execute(query)
    row = result.first()

    if row:
        return dict(row._mapping)
    return {}


def get_yearly_stats(conn) -> List[Dict[str, Any]]:
    """
    Get statistics by year from materialized view.

    Args:
        conn: Database connection

    Returns:
        List of yearly statistics
    """
    query = text(
        """
        SELECT
            ev_year,
            total_accidents,
            fatal_accidents,
            total_fatalities,
            avg_fatalities_per_accident,
            serious_injury_accidents,
            destroyed_aircraft
        FROM mv_yearly_stats
        ORDER BY ev_year
        """
    )

    result = conn.execute(query)
    return [dict(row._mapping) for row in result]


def get_state_stats(conn) -> List[Dict[str, Any]]:
    """
    Get statistics by state from materialized view.

    Args:
        conn: Database connection

    Returns:
        List of state statistics
    """
    query = text(
        """
        SELECT
            ev_state,
            accident_count,
            fatal_count,
            avg_latitude,
            avg_longitude
        FROM mv_state_stats
        ORDER BY accident_count DESC
        """
    )

    result = conn.execute(query)
    return [dict(row._mapping) for row in result]


def get_aircraft_stats(conn, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get aircraft make/model statistics from materialized view.

    Args:
        conn: Database connection
        limit: Maximum results

    Returns:
        List of aircraft statistics
    """
    query = text(
        """
        SELECT
            acft_make as make,
            acft_model as model,
            event_count as accident_count,
            fatal_rate
        FROM mv_aircraft_stats
        ORDER BY event_count DESC
        LIMIT :limit
        """
    )

    result = conn.execute(query, {"limit": limit})
    return [dict(row._mapping) for row in result]


def get_decade_stats(conn) -> List[Dict[str, Any]]:
    """
    Get statistics by decade from materialized view.

    Args:
        conn: Database connection

    Returns:
        List of decade statistics
    """
    query = text(
        """
        SELECT
            decade,
            event_count,
            total_fatalities,
            avg_fatalities,
            fatal_event_count
        FROM mv_decade_stats
        ORDER BY decade
        """
    )

    result = conn.execute(query)
    return [dict(row._mapping) for row in result]


def get_monthly_trends(conn, year: int) -> List[Dict[str, Any]]:
    """
    Get monthly event trends for a specific year.

    Args:
        conn: Database connection
        year: Year to analyze

    Returns:
        List of monthly statistics
    """
    query = text(
        """
        SELECT
            ev_month as month,
            COUNT(*) as accident_count,
            SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
            SUM(COALESCE(inj_tot_f, 0)) as total_fatalities
        FROM events
        WHERE ev_year = :year
        GROUP BY ev_month
        ORDER BY ev_month
        """
    )

    result = conn.execute(query, {"year": year})
    return [dict(row._mapping) for row in result]


def get_seasonal_patterns(conn) -> List[Dict[str, Any]]:
    """
    Get seasonal patterns (aggregated across all years).

    Args:
        conn: Database connection

    Returns:
        List of monthly averages
    """
    query = text(
        """
        SELECT
            ev_month as month,
            COUNT(*) as total_accidents,
            AVG(CASE WHEN ev_highest_injury = 'FATL' THEN 1.0 ELSE 0.0 END) as fatal_rate,
            COUNT(*) * 1.0 / COUNT(DISTINCT ev_year) as avg_per_year
        FROM events
        GROUP BY ev_month
        ORDER BY ev_month
        """
    )

    result = conn.execute(query)
    return [dict(row._mapping) for row in result]
