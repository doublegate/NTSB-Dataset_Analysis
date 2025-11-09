"""Cause Factor Analysis - Root cause identification and patterns

This page provides analysis of accident causes including finding codes,
weather impact, pilot factors, and flight phase analysis.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.queries import (
    get_top_finding_codes,
    get_weather_stats,
    get_phase_stats,
    get_finding_stats,
)
from dashboard.components.charts import (
    create_bar_chart,
    create_pie_chart,
    create_treemap,
)

# Page config
st.set_page_config(
    page_title="Cause Factors - NTSB Dashboard", page_icon="🔍", layout="wide"
)

# Page title
st.title("🔍 Cause Factor Analysis")
st.markdown("**Root cause identification and contributing factor patterns**")

# Top Finding Codes
st.markdown("### Top Finding Codes")
st.caption("NTSB investigation findings and probable causes")

try:
    finding_data = get_top_finding_codes(limit=30)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Horizontal bar chart of top findings
        fig_findings = create_bar_chart(
            finding_data,
            x="occurrence_count",
            y="finding_code",
            title="Top 30 Finding Codes by Frequency",
            orientation="h",
            labels={"occurrence_count": "Occurrences", "finding_code": "Finding Code"},
            color="in_probable_cause_count",
            color_continuous_scale="Reds",
            hover_data=["finding_description"],
        )
        st.plotly_chart(fig_findings, use_container_width=True)

    with col2:
        st.markdown("#### Top 5 Findings")

        top_5_findings = finding_data.head(5)
        display_findings = top_5_findings[
            ["finding_code", "occurrence_count", "finding_description"]
        ]
        display_findings.columns = ["Code", "Count", "Description"]

        # Truncate descriptions for display
        display_findings["Description"] = (
            display_findings["Description"].str[:50] + "..."
        )

        st.dataframe(display_findings, hide_index=True, use_container_width=True)

        st.info(
            """
        **Finding Codes**: NTSB uses hierarchical codes to categorize causes.
        Codes starting with numbers indicate specific aircraft systems or factors.
        """
        )

except Exception as e:
    st.error(f"Error loading finding codes: {e}")

# Weather Impact
st.markdown("---")
st.markdown("### Weather Impact Analysis")

try:
    weather_data = get_weather_stats()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart of weather conditions
        fig_weather_dist = create_pie_chart(
            weather_data,
            values="event_count",
            names="wx_cond_basic",
            title="Event Distribution by Weather Conditions",
        )
        st.plotly_chart(fig_weather_dist, use_container_width=True)

    with col2:
        # Bar chart of fatality rates
        weather_data["fatal_rate"] = (
            weather_data["fatal_count"] / weather_data["event_count"] * 100
        ).round(2)

        fig_weather_fatal = create_bar_chart(
            weather_data,
            x="wx_cond_basic",
            y="fatal_rate",
            title="Fatality Rate by Weather Condition",
            labels={"wx_cond_basic": "Weather", "fatal_rate": "Fatal Rate (%)"},
            color="fatal_rate",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_weather_fatal, use_container_width=True)

    # Weather statistics table
    st.markdown("#### Weather Condition Statistics")

    display_weather = weather_data[
        [
            "wx_cond_basic",
            "event_count",
            "fatal_count",
            "total_fatalities",
            "avg_fatalities",
        ]
    ]
    display_weather.columns = [
        "Condition",
        "Events",
        "Fatal Events",
        "Total Fatalities",
        "Avg Fatal/Event",
    ]

    st.dataframe(display_weather, hide_index=True, use_container_width=True)

    st.caption(
        """
    **VMC**: Visual Meteorological Conditions (good weather) |
    **IMC**: Instrument Meteorological Conditions (poor visibility/weather) |
    **UNK**: Unknown weather conditions
    """
    )

except Exception as e:
    st.error(f"Error loading weather data: {e}")

# Phase of Flight Analysis
st.markdown("---")
st.markdown("### Phase of Flight Analysis")

try:
    phase_data = get_phase_stats()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Treemap of flight phases
        if len(phase_data) > 0:
            fig_phase_tree = create_treemap(
                phase_data,
                path=["flight_phase"],
                values="event_count",
                title="Events by Flight Phase",
                color="fatality_rate",
                color_continuous_scale="Reds",
                labels={"fatality_rate": "Fatal Rate (%)"},
            )
            st.plotly_chart(fig_phase_tree, use_container_width=True)
        else:
            st.warning("No flight phase data available")

    with col2:
        st.markdown("#### Top 10 Phases")

        if len(phase_data) > 0:
            top_phases = phase_data.head(10)
            display_phases = top_phases[
                ["flight_phase", "event_count", "fatal_count", "fatality_rate"]
            ]
            display_phases.columns = ["Phase", "Events", "Fatal", "Fatal Rate (%)"]

            st.dataframe(display_phases, hide_index=True, use_container_width=True)

        st.info(
            """
        **Critical Phases**: Takeoff and landing are typically high-risk phases.
        **Color Coding**: Darker red indicates higher fatality rates.
        """
        )

except Exception as e:
    st.error(f"Error loading phase data: {e}")

# Finding Statistics from Materialized View
st.markdown("---")
st.markdown("### Finding Code Statistics")

try:
    finding_stats = get_finding_stats(min_occurrences=10)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart of findings
        top_finding_stats = finding_stats.head(20)

        fig_finding_stats = create_bar_chart(
            top_finding_stats,
            x="occurrence_count",
            y="finding_code",
            title="Top 20 Finding Codes (≥10 occurrences)",
            orientation="h",
            labels={"occurrence_count": "Count", "finding_code": "Finding Code"},
            color="in_pc_percentage",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_finding_stats, use_container_width=True)

    with col2:
        st.markdown("#### Statistics")

        total_findings = len(finding_stats)
        total_occurrences = finding_stats["occurrence_count"].sum()

        st.metric("Unique Finding Codes", f"{total_findings:,}")
        st.metric("Total Occurrences", f"{total_occurrences:,}")

        avg_pc_pct = finding_stats["in_pc_percentage"].mean()
        st.metric(
            "Avg Probable Cause %",
            f"{avg_pc_pct:.1f}%",
            help="Average % of findings cited in probable cause",
        )

except Exception as e:
    st.error(f"Error loading finding statistics: {e}")

# Complete Finding Codes Table
st.markdown("---")
st.markdown("### Complete Finding Codes Reference")

try:
    all_findings = get_top_finding_codes(limit=100)

    # Search functionality
    search_code = st.text_input("Search by finding code", key="search_finding_code")
    search_desc = st.text_input("Search by description", key="search_finding_desc")

    # Apply filters
    filtered_findings = all_findings.copy()

    if search_code:
        filtered_findings = filtered_findings[
            filtered_findings["finding_code"]
            .astype(str)
            .str.contains(search_code, case=False, na=False)
        ]

    if search_desc:
        filtered_findings = filtered_findings[
            filtered_findings["finding_description"].str.contains(
                search_desc, case=False, na=False
            )
        ]

    # Prepare display
    display_findings_table = filtered_findings[
        [
            "finding_code",
            "finding_description",
            "occurrence_count",
            "in_probable_cause_count",
        ]
    ]
    display_findings_table.columns = [
        "Code",
        "Description",
        "Occurrences",
        "In Probable Cause",
    ]

    # Calculate percentage
    display_findings_table["In PC %"] = (
        display_findings_table["In Probable Cause"]
        / display_findings_table["Occurrences"]
        * 100
    ).round(1)

    st.dataframe(
        display_findings_table, hide_index=True, use_container_width=True, height=400
    )

    # Download button
    csv = display_findings_table.to_csv(index=False)
    st.download_button(
        label="Download Finding Codes as CSV",
        data=csv,
        file_name="ntsb_finding_codes.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Error creating findings table: {e}")

# Key Insights
st.markdown("---")
st.markdown("### Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Common Causal Factors")
    st.markdown(
        """
    - **Pilot Error**: Loss of control and spatial disorientation are leading causes
    - **Weather**: IMC conditions associated with higher fatality rates
    - **Mechanical**: Engine failures and system malfunctions are significant factors
    - **Flight Phase**: Takeoff and landing show elevated risk
    - **Maintenance**: Inadequate maintenance appears in many findings
    """
    )

with col2:
    st.markdown("#### Safety Recommendations")
    st.markdown(
        """
    - **Training**: Enhanced pilot training for IMC and emergency procedures
    - **Maintenance**: Rigorous maintenance schedules and inspections
    - **Technology**: Advanced avionics and safety systems show benefits
    - **Regulation**: FAA regulations address many identified risk factors
    - **Weather Awareness**: Better weather briefing and decision-making tools
    """
    )

# Reference Information
st.markdown("---")
st.info(
    """
**Finding Codes**: NTSB uses a hierarchical coding system for investigation findings:
- **10000-21000**: Aircraft/Equipment subjects (airframe, systems, powerplant)
- **22000-25000**: Performance/Operations (pilot technique, procedures)
- **30000-84000**: Direct underlying causes (failure modes, pilot errors)
- **90000-93000**: Indirect underlying causes (design, maintenance, organizational)

Refer to the NTSB Aviation Coding Manual (`ref_docs/codman.pdf`) for complete code definitions.
"""
)

# Footer
st.markdown("---")
st.caption(
    """
**Cause Factor Analysis**: Based on NTSB investigation findings and probable causes |
**Note**: Finding codes may appear multiple times per event (multiple contributing factors) |
**Reference**: See Aviation Coding Manual in ref_docs/ for complete code descriptions
"""
)
