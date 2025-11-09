"""Database query utilities for Streamlit dashboard.

This module provides cached query functions for retrieving data
from the NTSB Aviation Accident Database.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import date
from .database import get_connection, release_connection


@st.cache_data(ttl=3600)
def get_yearly_stats() -> pd.DataFrame:
    """Get yearly statistics from materialized view (cached 1 hour).

    Returns:
        DataFrame with columns: ev_year, total_accidents, fatal_accidents,
                               total_fatalities, avg_fatalities_per_accident, etc.
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            ev_year,
            total_accidents,
            fatal_accidents,
            fatal_accident_rate,
            total_fatalities,
            avg_fatalities_per_accident,
            serious_injury_accidents,
            total_serious_injuries,
            total_minor_injuries
        FROM mv_yearly_stats
        ORDER BY ev_year
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_state_stats() -> pd.DataFrame:
    """Get state statistics from materialized view (cached 1 hour).

    Returns:
        DataFrame with columns: ev_state, accident_count, fatal_count,
                               avg_latitude, avg_longitude
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            ev_state,
            accident_count,
            fatal_count,
            avg_latitude,
            avg_longitude
        FROM mv_state_stats
        ORDER BY accident_count DESC
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_aircraft_stats(min_accidents: int = 5) -> pd.DataFrame:
    """Get aircraft statistics from materialized view.

    Args:
        min_accidents: Minimum accident count filter

    Returns:
        DataFrame with aircraft statistics
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            acft_make,
            acft_model,
            accident_count,
            fatal_accident_count,
            total_fatalities,
            avg_fatalities_per_accident
        FROM mv_aircraft_stats
        WHERE accident_count >= %s
        ORDER BY accident_count DESC
        """
        df = pd.read_sql(query, conn, params=(min_accidents,))
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_decade_stats() -> pd.DataFrame:
    """Get decade statistics from materialized view.

    Returns:
        DataFrame with decade-level statistics
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            decade,
            total_accidents,
            fatal_accidents,
            total_fatalities,
            avg_fatalities_per_accident
        FROM mv_decade_stats
        ORDER BY decade
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_finding_stats(min_occurrences: int = 10) -> pd.DataFrame:
    """Get finding statistics from materialized view.

    Args:
        min_occurrences: Minimum occurrence count

    Returns:
        DataFrame with finding code statistics
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            finding_code,
            occurrence_count,
            in_probable_cause_count,
            in_pc_percentage
        FROM mv_finding_stats
        WHERE occurrence_count >= %s
        ORDER BY occurrence_count DESC
        """
        df = pd.read_sql(query, conn, params=(min_occurrences,))
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_monthly_stats() -> pd.DataFrame:
    """Get monthly event statistics.

    Returns:
        DataFrame with monthly event counts
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            ev_month,
            COUNT(*) as event_count,
            SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
            SUM(COALESCE(inj_tot_f, 0)) as total_fatalities
        FROM events
        GROUP BY ev_month
        ORDER BY ev_month
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_dow_stats() -> pd.DataFrame:
    """Get day of week statistics.

    Returns:
        DataFrame with day-of-week event counts
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            EXTRACT(DOW FROM ev_date) as dow,
            COUNT(*) as event_count,
            SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count
        FROM events
        WHERE ev_date IS NOT NULL
        GROUP BY EXTRACT(DOW FROM ev_date)
        ORDER BY dow
        """
        df = pd.read_sql(query, conn)

        # Map day numbers to names
        day_map = {
            0: "Sunday",
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday",
        }
        df["day_name"] = df["dow"].map(day_map)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_summary_stats() -> Dict[str, Any]:
    """Get overall summary statistics.

    Returns:
        Dictionary with summary metrics
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            COUNT(*) as total_events,
            MIN(ev_year) as min_year,
            MAX(ev_year) as max_year,
            SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_events,
            SUM(COALESCE(inj_tot_f, 0)) as total_fatalities,
            SUM(COALESCE(inj_tot_s, 0)) as total_serious_injuries,
            COUNT(DISTINCT ev_state) as states_count
        FROM events
        """
        df = pd.read_sql(query, conn)
        return df.iloc[0].to_dict()
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_events(
    limit: int = 10000,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    states: Optional[List[str]] = None,
    severity: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Get events with optional filters.

    Args:
        limit: Maximum number of events to return
        start_date: Start date filter
        end_date: End date filter
        states: List of state codes to filter
        severity: List of severity levels

    Returns:
        DataFrame with event data
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            ev_id,
            ev_date,
            ev_state,
            ev_city,
            dec_latitude,
            dec_longitude,
            inj_tot_f,
            inj_tot_s,
            ev_highest_injury,
            ev_year
        FROM events
        WHERE 1=1
        """
        params = []

        if start_date is not None and end_date is not None:
            query += " AND ev_date BETWEEN %s AND %s"
            params.extend([start_date, end_date])

        if states and len(states) > 0:
            placeholders = ",".join(["%s"] * len(states))
            query += f" AND ev_state IN ({placeholders})"
            params.extend(states)

        if severity and len(severity) > 0:
            placeholders = ",".join(["%s"] * len(severity))
            query += f" AND ev_highest_injury IN ({placeholders})"
            params.extend(severity)

        query += f" LIMIT {limit}"

        df = pd.read_sql(query, conn, params=params if params else None)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_states_list() -> List[str]:
    """Get list of all states with events.

    Returns:
        List of state codes
    """
    conn = get_connection()
    try:
        query = """
        SELECT DISTINCT ev_state
        FROM events
        WHERE ev_state IS NOT NULL
        ORDER BY ev_state
        """
        df = pd.read_sql(query, conn)
        return df["ev_state"].tolist()
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_weather_stats() -> pd.DataFrame:
    """Get weather condition statistics.

    Returns:
        DataFrame with weather statistics
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            wx_cond_basic,
            COUNT(*) as event_count,
            SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
            SUM(COALESCE(inj_tot_f, 0)) as total_fatalities,
            ROUND(AVG(COALESCE(inj_tot_f, 0))::numeric, 2) as avg_fatalities
        FROM events
        WHERE wx_cond_basic IS NOT NULL
        GROUP BY wx_cond_basic
        ORDER BY event_count DESC
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_phase_stats() -> pd.DataFrame:
    """Get flight phase statistics.

    Returns:
        DataFrame with flight phase statistics
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            flight_phase,
            COUNT(*) as event_count,
            SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
            ROUND(100.0 * SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) / COUNT(*), 2) as fatality_rate
        FROM events
        WHERE flight_phase IS NOT NULL AND flight_phase != ''
        GROUP BY flight_phase
        HAVING COUNT(*) >= 10
        ORDER BY event_count DESC
        LIMIT 15
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_aircraft_category_stats() -> pd.DataFrame:
    """Get aircraft category statistics.

    Returns:
        DataFrame with aircraft category statistics
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            a.acft_category,
            COUNT(*) as event_count,
            SUM(CASE WHEN e.ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count,
            SUM(COALESCE(e.inj_tot_f, 0)) as total_fatalities
        FROM events e
        JOIN aircraft a ON e.ev_id = a.ev_id
        WHERE a.acft_category IS NOT NULL AND a.acft_category != ''
        GROUP BY a.acft_category
        ORDER BY event_count DESC
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        release_connection(conn)


@st.cache_data(ttl=3600)
def get_top_finding_codes(limit: int = 30) -> pd.DataFrame:
    """Get top finding codes with descriptions.

    Args:
        limit: Number of top codes to return

    Returns:
        DataFrame with finding code statistics
    """
    conn = get_connection()
    try:
        query = """
        SELECT
            finding_code,
            finding_description,
            COUNT(*) as occurrence_count,
            SUM(CASE WHEN cm_inPC = TRUE THEN 1 ELSE 0 END) as in_probable_cause_count
        FROM Findings
        WHERE finding_code IS NOT NULL
        GROUP BY finding_code, finding_description
        ORDER BY occurrence_count DESC
        LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    finally:
        release_connection(conn)
