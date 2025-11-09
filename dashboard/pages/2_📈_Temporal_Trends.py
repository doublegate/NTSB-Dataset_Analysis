"""Temporal Trends Analysis - Time series patterns and forecasting

This page provides temporal analysis including seasonal patterns, decade comparisons,
day of week analysis, and trend forecasting.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.queries import (
    get_yearly_stats,
    get_monthly_stats,
    get_dow_stats,
    get_decade_stats,
)
from dashboard.components.charts import (
    create_bar_chart,
)
from dashboard.components.filters import year_range_slider

# Page config
st.set_page_config(
    page_title="Temporal Trends - NTSB Dashboard", page_icon="📈", layout="wide"
)

# Page title
st.title("📈 Temporal Trends Analysis")
st.markdown("**Time series patterns, seasonality, and trend forecasting**")

# Filters in sidebar
with st.sidebar:
    st.markdown("### Filters")
    year_range = year_range_slider(
        key="temporal_year_range",
        min_year=1962,
        max_year=2025,
        default_min=1962,
        default_max=2025,
    )

# Seasonal Patterns
st.markdown("### Seasonal Patterns")

try:
    monthly_data = get_monthly_stats()

    # Map month numbers to names
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthly_data["month_name"] = monthly_data["ev_month"].map(
        lambda x: month_names[int(x) - 1]
        if pd.notna(x) and 1 <= int(x) <= 12
        else "Unknown"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart of monthly events
        fig_monthly = create_bar_chart(
            monthly_data,
            x="month_name",
            y="event_count",
            title="Aviation Accidents by Month (All Years)",
            labels={"month_name": "Month", "event_count": "Total Events"},
            color="fatal_count",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        st.markdown("#### Key Insights")

        # Find peak and low months
        peak_month = monthly_data.loc[monthly_data["event_count"].idxmax()]
        low_month = monthly_data.loc[monthly_data["event_count"].idxmin()]

        st.metric(
            "Peak Month",
            peak_month["month_name"],
            f"{int(peak_month['event_count'])} events",
        )

        st.metric(
            "Lowest Month",
            low_month["month_name"],
            f"{int(low_month['event_count'])} events",
        )

        # Summer vs Winter comparison
        summer_months = monthly_data[monthly_data["ev_month"].isin([6, 7, 8])]
        winter_months = monthly_data[monthly_data["ev_month"].isin([12, 1, 2])]

        summer_total = summer_months["event_count"].sum()
        winter_total = winter_months["event_count"].sum()

        st.metric(
            "Summer vs Winter",
            f"+{((summer_total - winter_total) / winter_total * 100):.1f}%",
            "More events in summer (Jun-Aug vs Dec-Feb)",
        )

except Exception as e:
    st.error(f"Error loading seasonal data: {e}")

# Decade Comparison
st.markdown("---")
st.markdown("### Decade Comparison")

try:
    decade_data = get_decade_stats()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Grouped bar chart for decades
        fig_decades = create_bar_chart(
            decade_data,
            x="decade",
            y="total_accidents",
            title="Aviation Accidents by Decade",
            labels={"decade": "Decade", "total_accidents": "Total Accidents"},
            color="total_fatalities",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_decades, use_container_width=True)

    with col2:
        st.markdown("#### Decade Statistics")

        # Show decade table
        decade_display = decade_data.copy()
        decade_display["decade"] = decade_display["decade"].astype(str) + "s"
        decade_display = decade_display[
            ["decade", "total_accidents", "total_fatalities"]
        ]
        decade_display.columns = ["Decade", "Accidents", "Fatalities"]

        st.dataframe(decade_display, hide_index=True, use_container_width=True)

except Exception as e:
    st.error(f"Error loading decade data: {e}")

# Day of Week Analysis
st.markdown("---")
st.markdown("### Day of Week Analysis")

try:
    dow_data = get_dow_stats()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart by day of week
        fig_dow = create_bar_chart(
            dow_data,
            x="day_name",
            y="event_count",
            title="Aviation Accidents by Day of Week",
            labels={"day_name": "Day", "event_count": "Total Events"},
            color="fatal_count",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col2:
        st.markdown("#### Weekend vs Weekday")

        # Calculate weekend vs weekday
        weekend = dow_data[dow_data["dow"].isin([0, 6])]["event_count"].sum()
        weekday = dow_data[dow_data["dow"].isin([1, 2, 3, 4, 5])]["event_count"].sum()

        total_days_weekend = 2
        total_days_weekday = 5

        weekend_avg = weekend / total_days_weekend
        weekday_avg = weekday / total_days_weekday

        st.metric("Weekend Average", f"{weekend_avg:,.0f}", "Events per day")

        st.metric("Weekday Average", f"{weekday_avg:,.0f}", "Events per day")

        diff_pct = (weekend_avg - weekday_avg) / weekday_avg * 100
        st.metric(
            "Weekend vs Weekday", f"{diff_pct:+.1f}%", "Relative to weekday average"
        )

except Exception as e:
    st.error(f"Error loading day of week data: {e}")

# Trend Analysis
st.markdown("---")
st.markdown("### Long-term Trend Analysis")

try:
    yearly_data = get_yearly_stats()

    # Filter by year range
    yearly_filtered = yearly_data[
        (yearly_data["ev_year"] >= year_range[0])
        & (yearly_data["ev_year"] <= year_range[1])
    ]

    # Create multi-metric line chart
    st.markdown("#### Multiple Metrics Over Time")

    # Allow user to select metrics
    metrics = st.multiselect(
        "Select metrics to display",
        options=[
            "Total Accidents",
            "Fatal Accidents",
            "Total Fatalities",
            "Serious Injuries",
        ],
        default=["Total Accidents", "Fatal Accidents"],
        key="temporal_metrics",
    )

    if metrics:
        # Create figure with selected metrics
        import plotly.graph_objects as go

        fig = go.Figure()

        metric_mapping = {
            "Total Accidents": "total_accidents",
            "Fatal Accidents": "fatal_accidents",
            "Total Fatalities": "total_fatalities",
            "Serious Injuries": "serious_injury_accidents",
        }

        colors = ["blue", "red", "orange", "green"]

        for idx, metric in enumerate(metrics):
            column = metric_mapping[metric]
            fig.add_trace(
                go.Scatter(
                    x=yearly_filtered["ev_year"],
                    y=yearly_filtered[column],
                    mode="lines+markers",
                    name=metric,
                    line=dict(color=colors[idx % len(colors)]),
                )
            )

        fig.update_layout(
            title="Aviation Safety Metrics Over Time",
            xaxis_title="Year",
            yaxis_title="Count",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Trend statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        # Calculate overall trend (first 5 years vs last 5 years)
        first_5 = yearly_filtered.head(5)["total_accidents"].mean()
        last_5 = yearly_filtered.tail(5)["total_accidents"].mean()
        trend_pct = (last_5 - first_5) / first_5 * 100

        st.metric(
            "Overall Trend", f"{trend_pct:+.1f}%", "Last 5 years vs first 5 years"
        )

    with col2:
        # Peak year in range
        peak_row = yearly_filtered.loc[yearly_filtered["total_accidents"].idxmax()]
        st.metric(
            "Peak Year",
            int(peak_row["ev_year"]),
            f"{int(peak_row['total_accidents'])} accidents",
        )

    with col3:
        # Average fatalities per accident
        avg_fatal_rate = yearly_filtered["avg_fatalities_per_accident"].mean()
        st.metric(
            "Avg Fatalities/Accident", f"{avg_fatal_rate:.2f}", "Across selected period"
        )

except Exception as e:
    st.error(f"Error loading trend data: {e}")

# Forecast Section (Placeholder)
st.markdown("---")
st.markdown("### 5-Year Forecast")

st.info(
    """
**Note**: This dashboard currently displays historical data analysis. Time series forecasting
models (ARIMA, Prophet, LSTM) are available in the companion Jupyter notebooks in the
`notebooks/exploratory/02_temporal_trends_analysis.ipynb` file.

Future versions of this dashboard will integrate interactive forecasting capabilities.
"""
)

# Footer
st.markdown("---")
st.caption(
    """
**Data Source**: NTSB Aviation Accident Database (1962-2025) |
**Temporal Analysis**: Monthly, weekly, and yearly patterns |
**Note**: Seasonal patterns reflect general aviation activity peaks during favorable weather months
"""
)
