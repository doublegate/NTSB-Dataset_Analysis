"""NTSB Aviation Accident Database - Interactive Dashboard

Main entry point for the Streamlit multi-page dashboard.

Usage:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.utils.queries import get_summary_stats

# Page configuration
st.set_page_config(
    page_title="NTSB Aviation Accident Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.title("🛩️ NTSB Aviation Database")
    st.markdown("**64 Years of Aviation Safety Data**")
    st.markdown("1962-2025 • 179,809 Events")

    st.markdown("---")

    # Navigation
    st.markdown("### Navigation")
    st.markdown("Use the sidebar to explore:")
    st.markdown("- 📊 **Overview**: Summary statistics")
    st.markdown("- 📈 **Temporal Trends**: Time series analysis")
    st.markdown("- 🗺️ **Geographic**: Interactive maps")
    st.markdown("- ✈️ **Aircraft Safety**: Type analysis")
    st.markdown("- 🔍 **Cause Factors**: Root causes")

    st.markdown("---")

    # About
    st.markdown("### About")
    st.markdown("Interactive dashboard for NTSB aviation accident data")
    st.markdown("**Database**: PostgreSQL 18.0 + PostGIS")
    st.markdown("**Coverage**: 179,809 events (1962-2025)")
    st.markdown("**Size**: 801 MB")
    st.markdown("**Health Score**: 98/100")

# Main page
st.title("NTSB Aviation Accident Dashboard")
st.markdown("### Interactive Analytics Platform")

# Introduction
st.markdown(
    """
Welcome to the NTSB Aviation Accident Dashboard. This interactive platform provides
comprehensive analytics for 64 years of aviation safety data from the National
Transportation Safety Board.

**Features**:
- 📊 **Overview**: High-level statistics and key findings
- 📈 **Temporal Analysis**: Trends, seasonality, and forecasting
- 🗺️ **Geographic Insights**: Interactive maps and regional analysis
- ✈️ **Aircraft Safety**: Type-specific risk assessment
- 🔍 **Cause Analysis**: Root cause identification and patterns

Select a page from the sidebar to begin exploring.
"""
)

# Quick metrics
try:
    stats = get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Events", f"{stats['total_events']:,}")

    with col2:
        years_coverage = stats["max_year"] - stats["min_year"] + 1
        st.metric("Years Coverage", f"{years_coverage}")

    with col3:
        st.metric("Total Fatalities", f"{stats['total_fatalities']:,}")

    with col4:
        st.metric("States", f"{stats['states_count']}")

except Exception as e:
    st.error(f"Error loading summary statistics: {e}")
    st.info("Please check database connection settings.")

# Getting Started
st.markdown("---")
st.markdown("### Getting Started")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    **Explore the Data**:

    1. Navigate to the **Overview** page for high-level statistics
    2. Use the **Temporal Trends** page to analyze accident patterns over time
    3. Explore geographic patterns in the **Geographic Analysis** page
    4. Investigate aircraft-specific risks in **Aircraft Safety**
    5. Examine root causes in the **Cause Factors** page
    """
    )

with col2:
    st.markdown(
        """
    **Interactive Features**:

    - Filter data by date range, state, and severity
    - Hover over charts for detailed information
    - Click on map markers to see event details
    - Download filtered data as CSV
    - Zoom and pan on all interactive visualizations
    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <p>Data Source: National Transportation Safety Board (NTSB)</p>
    <p>Dashboard Version: 1.0.0 | Database: PostgreSQL 18.0 + PostGIS</p>
</div>
""",
    unsafe_allow_html=True,
)
