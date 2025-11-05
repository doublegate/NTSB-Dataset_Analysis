#!/usr/bin/env python3
"""Geospatial analysis of aviation accidents"""

import pandas as pd
import duckdb
from pathlib import Path

try:
    import geopandas as gpd
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("⚠️  Geospatial libraries not installed")
    print("   Install with: pip install geopandas folium")


def load_events_with_coordinates(year_filter: int = 2020):
    """Load events that have valid coordinates"""
    query = f"""
    SELECT
        ev_id,
        ev_date,
        ev_year,
        ev_state,
        ev_city,
        latitude,
        longitude,
        inj_tot_f as fatalities,
        inj_tot_s as serious_injuries,
        ev_type
    FROM 'data/avall-events.csv'
    WHERE latitude IS NOT NULL
      AND longitude IS NOT NULL
      AND latitude != 0
      AND longitude != 0
      AND ev_year >= {year_filter}
    """

    df = duckdb.query(query).to_df()
    print(f"📍 Loaded {len(df)} events with coordinates (>= {year_filter})")
    return df


def create_accident_map(df: pd.DataFrame, output_file: str = 'outputs/accident_map.html'):
    """Create interactive map of accident locations"""
    if not GEOSPATIAL_AVAILABLE:
        print("❌ Cannot create map: geopandas/folium not installed")
        return

    # Create base map centered on US
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles='OpenStreetMap'
    )

    # Create marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers for each accident
    for idx, row in df.iterrows():
        # Create popup text
        popup_text = f"""
        <b>Event ID:</b> {row['ev_id']}<br>
        <b>Date:</b> {row['ev_date']}<br>
        <b>Location:</b> {row['ev_city']}, {row['ev_state']}<br>
        <b>Type:</b> {row['ev_type']}<br>
        <b>Fatalities:</b> {row['fatalities']}<br>
        <b>Serious Injuries:</b> {row['serious_injuries']}
        """

        # Color code by severity
        if row['fatalities'] > 0:
            color = 'red'
            icon = 'exclamation-triangle'
        elif row['serious_injuries'] > 0:
            color = 'orange'
            icon = 'warning'
        else:
            color = 'blue'
            icon = 'info-sign'

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['ev_city']}, {row['ev_state']}",
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(marker_cluster)

    # Create output directory
    Path('outputs').mkdir(exist_ok=True)

    # Save map
    m.save(output_file)
    print(f"✅ Map saved to: {output_file}")
    print(f"   Open in browser to view {len(df)} accident locations")


def create_heatmap(df: pd.DataFrame, output_file: str = 'outputs/accident_heatmap.html'):
    """Create heatmap of accident density"""
    if not GEOSPATIAL_AVAILABLE:
        print("❌ Cannot create heatmap: geopandas/folium not installed")
        return

    # Create base map
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles='OpenStreetMap'
    )

    # Prepare data for heatmap
    heat_data = [[row['latitude'], row['longitude']] for idx, row in df.iterrows()]

    # Add heatmap layer
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

    # Create output directory
    Path('outputs').mkdir(exist_ok=True)

    # Save map
    m.save(output_file)
    print(f"✅ Heatmap saved to: {output_file}")


def create_fatal_accidents_map(df: pd.DataFrame, output_file: str = 'outputs/fatal_accidents_map.html'):
    """Create map showing only fatal accidents"""
    if not GEOSPATIAL_AVAILABLE:
        print("❌ Cannot create map: geopandas/folium not installed")
        return

    # Filter for fatal accidents
    fatal_df = df[df['fatalities'] > 0].copy()
    print(f"🔴 Mapping {len(fatal_df)} fatal accidents")

    # Create base map
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles='OpenStreetMap'
    )

    # Add markers for fatal accidents
    for idx, row in fatal_df.iterrows():
        popup_text = f"""
        <b>Event ID:</b> {row['ev_id']}<br>
        <b>Date:</b> {row['ev_date']}<br>
        <b>Location:</b> {row['ev_city']}, {row['ev_state']}<br>
        <b>Fatalities:</b> {row['fatalities']}<br>
        """

        # Size based on number of fatalities
        radius = min(row['fatalities'] * 2 + 5, 30)

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['fatalities']} fatalities",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6
        ).add_to(m)

    # Create output directory
    Path('outputs').mkdir(exist_ok=True)

    # Save map
    m.save(output_file)
    print(f"✅ Fatal accidents map saved to: {output_file}")


def analyze_by_region():
    """Analyze accidents by geographic region"""
    query = """
    SELECT
        CASE
            WHEN ev_state IN ('WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM', 'AK', 'HI') THEN 'West'
            WHEN ev_state IN ('ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH') THEN 'Midwest'
            WHEN ev_state IN ('TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'TN', 'KY', 'WV', 'VA', 'NC', 'SC', 'GA', 'FL') THEN 'South'
            WHEN ev_state IN ('PA', 'NY', 'VT', 'NH', 'ME', 'MA', 'RI', 'CT', 'NJ', 'DE', 'MD', 'DC') THEN 'Northeast'
            ELSE 'Other'
        END as region,
        COUNT(*) as accident_count,
        SUM(inj_tot_f) as total_fatalities
    FROM 'data/avall-events.csv'
    WHERE ev_state IS NOT NULL
    GROUP BY region
    ORDER BY accident_count DESC
    """

    df = duckdb.query(query).to_df()
    print("\n📊 Accidents by Region:")
    print(df.to_string(index=False))
    return df


if __name__ == '__main__':
    print("🗺️  NTSB Aviation Accident Database - Geospatial Analysis")
    print("=" * 60)

    # Check if data exists
    if not Path('data/avall-events.csv').exists():
        print("\n❌ Error: data/avall-events.csv not found")
        print("   Extract data first with: ./scripts/extract_all_tables.fish datasets/avall.mdb")
        exit(1)

    try:
        # Load data
        df = load_events_with_coordinates(year_filter=2020)

        if len(df) == 0:
            print("⚠️  No events with coordinates found")
            exit(1)

        # Regional analysis
        analyze_by_region()

        # Create maps
        if GEOSPATIAL_AVAILABLE:
            print("\n🗺️  Creating interactive maps...")
            create_accident_map(df)
            create_heatmap(df)
            create_fatal_accidents_map(df)

            print("\n✅ Geospatial analysis complete!")
            print("\n📂 Output files:")
            print("   - outputs/accident_map.html")
            print("   - outputs/accident_heatmap.html")
            print("   - outputs/fatal_accidents_map.html")
        else:
            print("\n⚠️  Install geospatial libraries to create maps:")
            print("   pip install geopandas folium")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
