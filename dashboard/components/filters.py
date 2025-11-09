"""Reusable filter components for Streamlit dashboard.

This module provides common filter widgets used across multiple pages.
"""

import streamlit as st
from datetime import date
from typing import List, Optional, Tuple
from dashboard.utils.queries import get_states_list


def date_range_filter(
    key: str = "date_range",
    default_start: date = date(1962, 1, 1),
    default_end: date = date(2025, 12, 31),
) -> Tuple[date, date]:
    """Date range filter widget.

    Args:
        key: Unique key for widget
        default_start: Default start date
        default_end: Default end date

    Returns:
        Tuple of (start_date, end_date)
    """
    dates = st.date_input(
        "Date Range",
        value=(default_start, default_end),
        min_value=date(1962, 1, 1),
        max_value=date(2025, 12, 31),
        key=key,
        help="Filter events by date range",
    )

    # Handle single date selection
    if isinstance(dates, tuple) and len(dates) == 2:
        return dates[0], dates[1]
    elif isinstance(dates, date):
        return dates, dates
    else:
        return default_start, default_end


def severity_filter(
    key: str = "severity", default: Optional[List[str]] = None
) -> List[str]:
    """Severity filter widget.

    Args:
        key: Unique key for widget
        default: Default selected values

    Returns:
        List of selected severity levels
    """
    if default is None:
        default = ["FATL", "SERS", "MINR", "NONE"]

    severity_options = ["FATL", "SERS", "MINR", "NONE"]
    severity_labels = {
        "FATL": "Fatal",
        "SERS": "Serious",
        "MINR": "Minor",
        "NONE": "None",
    }

    selected = st.multiselect(
        "Injury Severity",
        options=severity_options,
        default=default,
        format_func=lambda x: severity_labels.get(x, x),
        key=key,
        help="Filter by highest injury level",
    )

    return selected


def state_filter(key: str = "state", default: Optional[List[str]] = None) -> List[str]:
    """State filter widget.

    Args:
        key: Unique key for widget
        default: Default selected states

    Returns:
        List of selected state codes
    """
    states = get_states_list()

    selected = st.multiselect(
        "States",
        options=states,
        default=default,
        key=key,
        help="Filter by state (leave empty for all states)",
    )

    return selected


def event_type_filter(
    key: str = "event_type", default: Optional[List[str]] = None
) -> List[str]:
    """Event type filter widget.

    Args:
        key: Unique key for widget
        default: Default selected types

    Returns:
        List of selected event types
    """
    event_types = ["ACC", "INC"]  # Accident, Incident
    type_labels = {"ACC": "Accident", "INC": "Incident"}

    if default is None:
        default = event_types

    selected = st.multiselect(
        "Event Type",
        options=event_types,
        default=default,
        format_func=lambda x: type_labels.get(x, x),
        key=key,
        help="Filter by event type",
    )

    return selected


def year_range_slider(
    key: str = "year_range",
    min_year: int = 1962,
    max_year: int = 2025,
    default_min: Optional[int] = None,
    default_max: Optional[int] = None,
) -> Tuple[int, int]:
    """Year range slider widget.

    Args:
        key: Unique key for widget
        min_year: Minimum year
        max_year: Maximum year
        default_min: Default minimum year
        default_max: Default maximum year

    Returns:
        Tuple of (start_year, end_year)
    """
    if default_min is None:
        default_min = min_year
    if default_max is None:
        default_max = max_year

    years = st.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(default_min, default_max),
        key=key,
        help="Filter by year range",
    )

    return years


def limit_selector(
    key: str = "limit", default: int = 10000, options: Optional[List[int]] = None
) -> int:
    """Data limit selector widget.

    Args:
        key: Unique key for widget
        default: Default limit
        options: List of limit options

    Returns:
        Selected limit
    """
    if options is None:
        options = [1000, 5000, 10000, 25000, 50000]

    limit = st.selectbox(
        "Data Limit",
        options=options,
        index=options.index(default) if default in options else 0,
        key=key,
        help="Maximum number of events to load (for performance)",
    )

    return limit
