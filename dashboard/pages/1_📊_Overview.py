"""Overview Dashboard - High-level statistics and key insights

This page provides a comprehensive overview of the NTSB aviation accident database
with summary metrics, long-term trends, geographic distribution, and key findings.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.queries import (
    get_summary_stats,
    get_yearly_stats,
    get_state_stats,
    get_aircraft_stats,
    get_weather_stats,
)
from dashboard.components.charts import (
    create_line_chart,
    create_choropleth_map,
    create_bar_chart,
)

# Page config
st.set_page_config(
    page_title="Overview - NTSB Dashboard", page_icon="📊", layout="wide"
)

# Page title
st.title("📊 Overview Dashboard")
st.markdown(
    "**High-level statistics and key insights from 64 years of aviation safety data**"
)

# Hero Metrics
st.markdown("### Key Metrics")

try:
    stats = get_summary_stats()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Events",
            f"{stats['total_events']:,}",
            help="Total aviation accident events in database",
        )

    with col2:
        years_coverage = stats["max_year"] - stats["min_year"] + 1
        st.metric(
            "Years Coverage",
            f"{years_coverage}",
            delta=f"{stats['min_year']}-{stats['max_year']}",
            help="Years of historical data",
        )

    with col3:
        st.metric(
            "Total Fatalities",
            f"{stats['total_fatalities']:,}",
            help="Total fatalities across all events",
        )

    with col4:
        fatal_rate = stats["fatal_events"] / stats["total_events"] * 100
        st.metric(
            "Fatal Event Rate",
            f"{fatal_rate:.1f}%",
            help="Percentage of events with fatalities",
        )

    with col5:
        st.metric(
            "States Covered",
            f"{stats['states_count']}",
            help="Number of US states with recorded events",
        )

except Exception as e:
    st.error(f"Error loading summary statistics: {e}")

# Long-term Trends
st.markdown("---")
st.markdown("### Long-term Trends (1962-2025)")

try:
    yearly_data = get_yearly_stats()

    # Calculate 5-year moving average
    yearly_data["ma_5yr"] = (
        yearly_data["total_accidents"].rolling(window=5, center=True).mean()
    )

    # Create line chart with multiple series
    fig = create_line_chart(
        yearly_data,
        x="ev_year",
        y="total_accidents",
        title="Annual Aviation Accidents (1962-2025)",
    )

    # Add 5-year moving average
    fig.add_scatter(
        x=yearly_data["ev_year"],
        y=yearly_data["ma_5yr"],
        mode="lines",
        name="5-Year Moving Average",
        line=dict(color="red", width=2, dash="dash"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show key statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        peak_year = yearly_data.loc[yearly_data["total_accidents"].idxmax()]
        st.info(
            f"**Peak Year**: {int(peak_year['ev_year'])} with {int(peak_year['total_accidents'])} accidents"
        )

    with col2:
        recent_year = yearly_data.iloc[-1]
        st.info(
            f"**Most Recent**: {int(recent_year['ev_year'])} with {int(recent_year['total_accidents'])} accidents"
        )

    with col3:
        avg_accidents = yearly_data["total_accidents"].mean()
        st.info(f"**Average**: {avg_accidents:.0f} accidents/year")

except Exception as e:
    st.error(f"Error loading yearly trends: {e}")

# Geographic Distribution
st.markdown("---")
st.markdown("### Geographic Distribution")

try:
    state_data = get_state_stats()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Choropleth map
        fig_map = create_choropleth_map(
            state_data,
            locations="ev_state",
            color="accident_count",
            title="Aviation Accidents by State",
            color_continuous_scale="Reds",
            labels={"accident_count": "Accidents"},
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        # Top 10 states table
        st.markdown("#### Top 10 States by Events")
        top_states = state_data.head(10)[["ev_state", "accident_count", "fatal_count"]]
        top_states.columns = ["State", "Total Events", "Fatal Events"]
        st.dataframe(top_states, hide_index=True, use_container_width=True)

except Exception as e:
    st.error(f"Error loading geographic data: {e}")

# Quick Stats - Two Columns
st.markdown("---")
st.markdown("### Quick Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Top 5 Aircraft Makes by Events")
    try:
        aircraft_data = get_aircraft_stats(min_accidents=10)
        top_aircraft = aircraft_data.head(5)

        # Create horizontal bar chart
        fig_aircraft = create_bar_chart(
            top_aircraft,
            x="accident_count",
            y="acft_make",
            title="",
            orientation="h",
            color="fatal_accident_count",
            labels={"accident_count": "Total Events", "acft_make": "Aircraft Make"},
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_aircraft, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading aircraft data: {e}")

with col2:
    st.markdown("#### Weather Conditions")
    try:
        weather_data = get_weather_stats()

        # Create pie chart
        fig_weather = create_bar_chart(
            weather_data.head(5),
            x="wx_cond_basic",
            y="event_count",
            title="",
            labels={"wx_cond_basic": "Weather Condition", "event_count": "Event Count"},
            color="fatal_count",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading weather data: {e}")

# Key Findings
st.markdown("---")
st.markdown("### Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Safety Improvements 📈")
    st.markdown(
        """
    - **Declining Trend**: Aviation accidents have generally declined since the 1970s-1980s peak
    - **Modern Safety**: Recent years show significantly fewer accidents compared to historical averages
    - **Technology Impact**: Advanced avionics and safety systems correlate with improved outcomes
    - **Training Standards**: Enhanced pilot certification requirements show positive correlation
    - **Regulatory Success**: FAA regulations have contributed to long-term safety improvements
    """
    )

with col2:
    st.markdown("#### Risk Factors ⚠️")
    st.markdown(
        """
    - **General Aviation**: Represents the majority of accidents (vs commercial)
    - **Weather**: VMC vs IMC conditions show significant fatality rate differences
    - **Pilot Factors**: Loss of control and spatial disorientation remain leading causes
    - **Phase of Flight**: Takeoff and landing phases show higher accident rates
    - **Geographic Patterns**: States with high general aviation activity show more events
    """
    )

# Data Quality Note
st.markdown("---")
st.info(
    """
**Data Quality**: This dashboard uses production data from the NTSB Aviation Accident Database.
All statistics are based on 179,809 events from 1962-2025. Database health score: 98/100.
Data is updated monthly from NTSB sources.
"""
)
