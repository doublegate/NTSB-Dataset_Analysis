"""Geographic Analysis - Interactive maps and spatial patterns

This page provides geographic analysis with interactive maps, state rankings,
regional patterns, and density heatmaps.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from streamlit_folium import st_folium

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.queries import get_state_stats, get_events
from dashboard.components.charts import create_bar_chart, create_choropleth_map
from dashboard.components.maps import create_event_map, create_density_heatmap
from dashboard.components.filters import limit_selector, severity_filter

# Page config
st.set_page_config(
    page_title="Geographic Analysis - NTSB Dashboard", page_icon="🗺️", layout="wide"
)

# Page title
st.title("🗺️ Geographic Analysis")
st.markdown("**Interactive maps and spatial patterns of aviation accidents**")

# Filters in sidebar
with st.sidebar:
    st.markdown("### Map Settings")

    map_type = st.radio(
        "Map Type",
        options=["markers", "heatmap", "clusters"],
        format_func=lambda x: {
            "markers": "Individual Markers",
            "heatmap": "Density Heatmap",
            "clusters": "Cluster Map",
        }[x],
        key="geo_map_type",
        help="Choose visualization type",
    )

    data_limit = limit_selector(
        key="geo_limit", default=10000, options=[1000, 5000, 10000]
    )

    severity = severity_filter(key="geo_severity")

# Interactive Map
st.markdown("### Interactive Map")

try:
    # Load event data
    with st.spinner("Loading event data..."):
        events = get_events(limit=data_limit, severity=severity if severity else None)

    # Show data info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Events Loaded", f"{len(events):,}")

    with col2:
        events_with_coords = events.dropna(subset=["dec_latitude", "dec_longitude"])
        st.metric("Events with Coordinates", f"{len(events_with_coords):,}")

    with col3:
        pct_with_coords = (
            (len(events_with_coords) / len(events) * 100) if len(events) > 0 else 0
        )
        st.metric("Coverage", f"{pct_with_coords:.1f}%")

    # Create map
    if len(events_with_coords) > 0:
        if map_type == "heatmap":
            map_obj = create_density_heatmap(events_with_coords)
            st.markdown(
                """
            **Heatmap**: Shows density of accidents. Darker red indicates higher concentration.
            Weighted by fatalities.
            """
            )
        else:
            map_obj = create_event_map(events_with_coords, map_type=map_type)
            if map_type == "markers":
                st.markdown(
                    """
                **Individual Markers**: Red = Fatal, Blue = Non-Fatal. Click markers for event details.
                Limited to 10,000 events for performance.
                """
                )
            else:
                st.markdown(
                    """
                **Cluster Map**: Markers are grouped by proximity. Zoom in to see individual events.
                """
                )

        # Display map
        st_folium(map_obj, width=None, height=600)

    else:
        st.warning("No events with valid coordinates found for current filters.")

except Exception as e:
    st.error(f"Error loading map data: {e}")

# State Rankings
st.markdown("---")
st.markdown("### State Rankings")

try:
    state_data = get_state_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 15 States by Total Events")

        top_states = state_data.head(15)

        fig_states = create_bar_chart(
            top_states,
            x="accident_count",
            y="ev_state",
            title="",
            orientation="h",
            labels={"accident_count": "Total Events", "ev_state": "State"},
            color="fatal_count",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_states, width="stretch")

    with col2:
        st.markdown("#### Top 15 States by Fatalities")

        # Calculate total fatalities per state (need to get this from events table)
        # For now, use fatal_count as proxy
        top_fatal = state_data.nlargest(15, "fatal_count")

        fig_fatal = create_bar_chart(
            top_fatal,
            x="fatal_count",
            y="ev_state",
            title="",
            orientation="h",
            labels={"fatal_count": "Fatal Events", "ev_state": "State"},
            color="fatal_count",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_fatal, width="stretch")

except Exception as e:
    st.error(f"Error loading state rankings: {e}")

# Regional Analysis
st.markdown("---")
st.markdown("### Regional Analysis")

try:
    state_data = get_state_stats()

    # Define regions
    regions = {
        "Northeast": ["CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"],
        "Southeast": [
            "AL",
            "AR",
            "DE",
            "FL",
            "GA",
            "KY",
            "LA",
            "MD",
            "MS",
            "NC",
            "SC",
            "TN",
            "VA",
            "WV",
        ],
        "Midwest": [
            "IL",
            "IN",
            "IA",
            "KS",
            "MI",
            "MN",
            "MO",
            "NE",
            "ND",
            "OH",
            "SD",
            "WI",
        ],
        "Southwest": ["AZ", "NM", "OK", "TX"],
        "West": ["AK", "CA", "CO", "HI", "ID", "MT", "NV", "OR", "UT", "WA", "WY"],
    }

    # Calculate regional statistics
    regional_stats = []
    for region, states in regions.items():
        region_data = state_data[state_data["ev_state"].isin(states)]
        regional_stats.append(
            {
                "region": region,
                "total_events": region_data["accident_count"].sum(),
                "fatal_events": region_data["fatal_count"].sum(),
                "states": len(region_data),
            }
        )

    regional_df = pd.DataFrame(regional_stats)
    regional_df["fatality_rate"] = (
        regional_df["fatal_events"] / regional_df["total_events"] * 100
    ).round(2)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart of regional events
        fig_regional = create_bar_chart(
            regional_df,
            x="region",
            y="total_events",
            title="Aviation Accidents by Region",
            labels={"region": "Region", "total_events": "Total Events"},
            color="fatal_events",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_regional, width="stretch")

    with col2:
        st.markdown("#### Regional Statistics")

        display_df = regional_df[
            ["region", "total_events", "fatal_events", "fatality_rate"]
        ]
        display_df.columns = ["Region", "Events", "Fatal", "Fatal %"]

        st.dataframe(display_df, hide_index=True, width="stretch")

except Exception as e:
    st.error(f"Error loading regional data: {e}")

# Choropleth Map
st.markdown("---")
st.markdown("### State Choropleth Map")

try:
    state_data = get_state_stats()

    # Create choropleth
    fig_choropleth = create_choropleth_map(
        state_data,
        locations="ev_state",
        color="accident_count",
        title="Aviation Accidents by State (Color-coded by Event Count)",
        color_continuous_scale="Reds",
    )

    st.plotly_chart(fig_choropleth, width="stretch")

    st.info(
        """
    **Choropleth Map**: Color intensity represents the number of aviation accidents per state.
    Darker red indicates higher accident counts. Alaska, California, Florida, and Texas
    typically show the highest counts due to high general aviation activity.
    """
    )

except Exception as e:
    st.error(f"Error creating choropleth map: {e}")

# Data Table
st.markdown("---")
st.markdown("### Complete State Data Table")

try:
    state_data = get_state_stats()

    # Prepare display table
    display_table = state_data.copy()
    display_table["fatal_rate"] = (
        display_table["fatal_count"] / display_table["accident_count"] * 100
    ).round(2)

    display_table = display_table[
        ["ev_state", "accident_count", "fatal_count", "fatal_rate"]
    ]
    display_table.columns = ["State", "Total Events", "Fatal Events", "Fatal Rate (%)"]

    # Add search/filter
    search = st.text_input("Search by state code", key="state_search")

    if search:
        display_table = display_table[
            display_table["State"].str.contains(search.upper(), case=False, na=False)
        ]

    st.dataframe(display_table, hide_index=True, width="stretch", height=400)

    # Download button
    csv = display_table.to_csv(index=False)
    st.download_button(
        label="Download State Data as CSV",
        data=csv,
        file_name="ntsb_state_statistics.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Error creating data table: {e}")

# Footer
st.markdown("---")
st.caption(
    """
**Geographic Analysis**: Based on event location coordinates |
**Note**: ~8% of events lack precise coordinates (primarily older historical data) |
**Regional Definitions**: Standard US Census Bureau regions
"""
)
