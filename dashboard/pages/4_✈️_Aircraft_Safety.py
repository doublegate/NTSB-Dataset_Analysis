"""Aircraft Safety Analysis - Aircraft type-specific risk assessment

This page provides aircraft safety analysis including top makes/models,
aircraft categories, and safety statistics.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.queries import get_aircraft_stats, get_aircraft_category_stats
from dashboard.components.charts import (
    create_bar_chart,
    create_pie_chart,
    create_scatter_plot,
)

# Page config
st.set_page_config(
    page_title="Aircraft Safety - NTSB Dashboard", page_icon="✈️", layout="wide"
)

# Page title
st.title("✈️ Aircraft Safety Analysis")
st.markdown("**Aircraft type-specific risk assessment and safety patterns**")

# Filters in sidebar
with st.sidebar:
    st.markdown("### Filters")

    min_accidents = st.slider(
        "Minimum Accident Count",
        min_value=1,
        max_value=50,
        value=5,
        help="Filter aircraft with at least this many accidents",
    )

# Top Aircraft Makes
st.markdown("### Top Aircraft Makes by Events")

try:
    aircraft_data = get_aircraft_stats(min_accidents=min_accidents)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Horizontal bar chart of top 20 makes
        top_makes = aircraft_data.head(20)

        fig_makes = create_bar_chart(
            top_makes,
            x="accident_count",
            y="acft_make",
            title=f"Top 20 Aircraft Makes (≥{min_accidents} accidents)",
            orientation="h",
            labels={"accident_count": "Total Accidents", "acft_make": "Aircraft Make"},
            color="fatal_accident_count",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_makes, use_container_width=True)

    with col2:
        st.markdown("#### Top 5 Makes")

        top_5 = aircraft_data.head(5)
        display_top5 = top_5[["acft_make", "accident_count", "fatal_accident_count"]]
        display_top5.columns = ["Make", "Accidents", "Fatal"]

        st.dataframe(display_top5, hide_index=True, use_container_width=True)

        # Calculate total
        total_in_top20 = top_makes["accident_count"].sum()
        total_all = aircraft_data["accident_count"].sum()
        pct_top20 = (total_in_top20 / total_all * 100) if total_all > 0 else 0

        st.metric("Top 20 Share", f"{pct_top20:.1f}%", "of all aircraft accidents")

except Exception as e:
    st.error(f"Error loading aircraft data: {e}")

# Aircraft Categories
st.markdown("---")
st.markdown("### Aircraft Categories")

try:
    category_data = get_aircraft_category_stats()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart of categories
        fig_categories = create_pie_chart(
            category_data,
            values="event_count",
            names="acft_category",
            title="Event Distribution by Aircraft Category",
        )
        st.plotly_chart(fig_categories, use_container_width=True)

    with col2:
        # Bar chart with fatalities
        fig_cat_fatal = create_bar_chart(
            category_data,
            x="acft_category",
            y="total_fatalities",
            title="Total Fatalities by Aircraft Category",
            labels={"acft_category": "Category", "total_fatalities": "Fatalities"},
            color="total_fatalities",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_cat_fatal, use_container_width=True)

except Exception as e:
    st.error(f"Error loading category data: {e}")

# Aircraft Make/Model Analysis
st.markdown("---")
st.markdown("### Detailed Aircraft Analysis")

try:
    aircraft_data = get_aircraft_stats(min_accidents=min_accidents)

    # Calculate severity score (fatalities per accident)
    aircraft_data["severity_score"] = (
        aircraft_data["total_fatalities"] / aircraft_data["accident_count"]
    ).round(2)

    # Scatter plot: accidents vs fatalities
    col1, col2 = st.columns([2, 1])

    with col1:
        # Scatter plot
        fig_scatter = create_scatter_plot(
            aircraft_data.head(50),
            x="accident_count",
            y="total_fatalities",
            title="Accidents vs Fatalities (Top 50 Aircraft)",
            labels={
                "accident_count": "Total Accidents",
                "total_fatalities": "Total Fatalities",
            },
            hover_data=["acft_make", "acft_model"],
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.markdown("#### Severity Analysis")

        # Highest severity aircraft
        high_severity = aircraft_data.nlargest(10, "severity_score")

        st.markdown("**Top 10 by Severity Score**")
        st.caption("(Avg fatalities per accident)")

        display_severity = high_severity[
            ["acft_make", "acft_model", "severity_score", "accident_count"]
        ]
        display_severity.columns = ["Make", "Model", "Severity", "Accidents"]

        st.dataframe(
            display_severity, hide_index=True, use_container_width=True, height=300
        )

except Exception as e:
    st.error(f"Error in detailed analysis: {e}")

# Aircraft Statistics Table
st.markdown("---")
st.markdown("### Complete Aircraft Statistics")

try:
    aircraft_data = get_aircraft_stats(min_accidents=1)  # Get all aircraft

    # Prepare display table
    display_table = aircraft_data.copy()
    display_table["fatal_rate"] = (
        display_table["fatal_accident_count"] / display_table["accident_count"] * 100
    ).round(2)

    display_table = display_table[
        [
            "acft_make",
            "acft_model",
            "accident_count",
            "fatal_accident_count",
            "total_fatalities",
            "fatal_rate",
        ]
    ]
    display_table.columns = [
        "Make",
        "Model",
        "Total Accidents",
        "Fatal Accidents",
        "Total Fatalities",
        "Fatal Rate (%)",
    ]

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        search_make = st.text_input("Search by aircraft make", key="search_make")

    with col2:
        search_model = st.text_input("Search by aircraft model", key="search_model")

    # Apply filters
    if search_make:
        display_table = display_table[
            display_table["Make"].str.contains(search_make, case=False, na=False)
        ]

    if search_model:
        display_table = display_table[
            display_table["Model"].str.contains(search_model, case=False, na=False)
        ]

    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        options=[
            "Total Accidents",
            "Fatal Accidents",
            "Total Fatalities",
            "Fatal Rate (%)",
        ],
        key="aircraft_sort",
    )

    display_table = display_table.sort_values(by=sort_by, ascending=False)

    # Display table
    st.dataframe(display_table, hide_index=True, use_container_width=True, height=400)

    # Download button
    csv = display_table.to_csv(index=False)
    st.download_button(
        label="Download Aircraft Data as CSV",
        data=csv,
        file_name="ntsb_aircraft_statistics.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Error creating aircraft table: {e}")

# Key Findings
st.markdown("---")
st.markdown("### Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### General Aviation Dominance")
    st.markdown(
        """
    - **Cessna**: Consistently highest accident count (largest fleet size)
    - **Piper**: Second largest, similar safety profile to Cessna
    - **Beechcraft**: Strong presence in general aviation accidents
    - **Fleet Size Effect**: Higher accident counts often reflect larger active fleets
    - **Single-Engine**: Represents majority of general aviation accidents
    """
    )

with col2:
    st.markdown("#### Safety Insights")
    st.markdown(
        """
    - **Fatal Rate Variance**: Different aircraft types show varying fatal accident rates
    - **Category Differences**: Commercial vs general aviation safety profiles differ
    - **Training Requirements**: Higher certification aircraft show different patterns
    - **Modern Safety**: Newer aircraft with advanced avionics show improved safety
    - **Operational Environment**: Mission profile affects accident characteristics
    """
    )

# Disclaimers
st.markdown("---")
st.info(
    """
**Important Notes**:
- Aircraft accident counts are influenced by fleet size and operational exposure
- Higher accident counts do not necessarily indicate lower safety (may reflect larger fleets)
- Severity scores must be interpreted with operational context (mission profile, environment)
- Data includes all event types: accidents and incidents
- Comparison should account for total flight hours, which are not available in this dataset
"""
)

# Footer
st.markdown("---")
st.caption(
    """
**Aircraft Safety Analysis**: Based on NTSB aircraft data |
**Note**: Statistics reflect raw counts and may not account for fleet size or flight hours |
**Filter**: Adjust minimum accident count to focus on specific aircraft types
"""
)
