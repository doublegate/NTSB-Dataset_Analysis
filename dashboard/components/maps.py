"""Map visualization components for Streamlit dashboard.

This module provides Folium map components for geographic analysis.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd


def create_event_map(
    events: pd.DataFrame,
    map_type: str = "markers",
    center_lat: float = 39.8,
    center_lon: float = -98.5,
    zoom_start: int = 4,
) -> folium.Map:
    """Create Folium map with aviation accident events.

    Args:
        events: DataFrame with event data (must have dec_latitude, dec_longitude)
        map_type: Type of map ('markers', 'heatmap', 'clusters')
        center_lat: Center latitude
        center_lon: Center longitude
        zoom_start: Initial zoom level

    Returns:
        Folium Map object
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap"
    )

    # Filter out events with no coordinates
    events_with_coords = events.dropna(subset=["dec_latitude", "dec_longitude"])

    if map_type == "markers":
        # Add individual markers (limit for performance)
        for idx, event in events_with_coords.head(10000).iterrows():
            # Color based on fatalities
            if event.get("inj_tot_f", 0) > 0:
                color = "red"
                icon = "exclamation-triangle"
            else:
                color = "blue"
                icon = "info-sign"

            # Create popup text
            popup_text = f"""
            <b>Event ID:</b> {event['ev_id']}<br>
            <b>Date:</b> {event.get('ev_date', 'N/A')}<br>
            <b>Location:</b> {event.get('ev_city', 'N/A')}, {event.get('ev_state', 'N/A')}<br>
            <b>Fatalities:</b> {event.get('inj_tot_f', 0)}<br>
            <b>Serious:</b> {event.get('inj_tot_s', 0)}
            """

            folium.CircleMarker(
                location=[event["dec_latitude"], event["dec_longitude"]],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                icon=icon,
            ).add_to(m)

    elif map_type == "heatmap":
        # Create heatmap layer
        heat_data = [
            [row["dec_latitude"], row["dec_longitude"]]
            for _, row in events_with_coords.iterrows()
        ]

        HeatMap(
            heat_data,
            radius=15,
            blur=20,
            max_zoom=13,
            gradient={0.4: "blue", 0.6: "yellow", 0.8: "orange", 1: "red"},
        ).add_to(m)

    elif map_type == "clusters":
        # Create marker clusters
        marker_cluster = MarkerCluster(
            name="Event Clusters", overlay=True, control=True
        ).add_to(m)

        for idx, event in events_with_coords.iterrows():
            popup_text = f"""
            <b>Event ID:</b> {event['ev_id']}<br>
            <b>Date:</b> {event.get('ev_date', 'N/A')}<br>
            <b>Location:</b> {event.get('ev_city', 'N/A')}, {event.get('ev_state', 'N/A')}<br>
            <b>Fatalities:</b> {event.get('inj_tot_f', 0)}
            """

            # Color based on severity
            if event.get("inj_tot_f", 0) > 0:
                icon_color = "red"
            else:
                icon_color = "blue"

            folium.Marker(
                location=[event["dec_latitude"], event["dec_longitude"]],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=icon_color, icon="plane", prefix="fa"),
            ).add_to(marker_cluster)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


def create_state_choropleth(
    state_data: pd.DataFrame, value_column: str, title: str = "Events by State"
) -> folium.Map:
    """Create US state choropleth map.

    Args:
        state_data: DataFrame with state statistics (must have ev_state column)
        value_column: Column name for color scale
        title: Map title

    Returns:
        Folium Map object
    """
    # Create base map centered on US
    m = folium.Map(location=[37.8, -96], zoom_start=4, tiles="OpenStreetMap")

    # Load US states GeoJSON (built-in Folium dataset)
    us_states_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"

    folium.Choropleth(
        geo_data=us_states_url,
        name="choropleth",
        data=state_data,
        columns=["ev_state", value_column],
        key_on="feature.id",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    return m


def create_density_heatmap(events: pd.DataFrame, grid_size: int = 50) -> folium.Map:
    """Create grid-based density heatmap.

    Args:
        events: DataFrame with event coordinates
        grid_size: Number of grid cells (size x size)

    Returns:
        Folium Map object
    """
    # Filter events with coordinates
    events_with_coords = events.dropna(subset=["dec_latitude", "dec_longitude"])

    # Create base map
    m = folium.Map(location=[39.8, -98.5], zoom_start=4, tiles="OpenStreetMap")

    # Prepare heatmap data with weights (fatalities)
    heat_data = []
    for _, row in events_with_coords.iterrows():
        weight = max(1, row.get("inj_tot_f", 1))  # Weight by fatalities, minimum 1
        heat_data.append([row["dec_latitude"], row["dec_longitude"], weight])

    # Add weighted heatmap
    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        max_zoom=13,
        gradient={0.4: "blue", 0.6: "yellow", 0.8: "orange", 1: "red"},
    ).add_to(m)

    return m
