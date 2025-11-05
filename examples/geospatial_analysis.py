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
    # Validate year parameter
    if not isinstance(year_filter, int) or year_filter < 1962 or year_filter > 2100:
        raise ValueError(f"Invalid year filter: {year_filter}. Must be between 1962 and 2100")

    query = f"""
    SELECT
        ev_id,
        ev_date,
        ev_year,
        TRIM(ev_state) as ev_state,
        TRIM(ev_city) as ev_city,
        TRY_CAST(dec_latitude AS DOUBLE) as latitude,
        TRY_CAST(dec_longitude AS DOUBLE) as longitude,
        COALESCE(inj_tot_f, 0) as fatalities,
        COALESCE(inj_tot_s, 0) as serious_injuries,
        ev_type
    FROM 'data/avall-events.csv'
    WHERE dec_latitude IS NOT NULL
      AND dec_longitude IS NOT NULL
      AND TRY_CAST(dec_latitude AS DOUBLE) IS NOT NULL
      AND TRY_CAST(dec_longitude AS DOUBLE) IS NOT NULL
      AND TRY_CAST(dec_latitude AS DOUBLE) BETWEEN -90 AND 90
      AND TRY_CAST(dec_longitude AS DOUBLE) BETWEEN -180 AND 180
      AND TRY_CAST(dec_latitude AS DOUBLE) != 0
      AND TRY_CAST(dec_longitude AS DOUBLE) != 0
      AND ev_year >= {year_filter}
      AND ev_year IS NOT NULL
      AND ev_id IS NOT NULL
    """

    try:
        df = duckdb.query(query).to_df()
        print(f"📍 Loaded {len(df):,} events with valid coordinates (>= {year_filter})")

        if len(df) == 0:
            print(f"⚠️  No events with coordinates found for year >= {year_filter}")

        return df

    except Exception as e:
        print(f"❌ Error loading coordinate data: {e}")
        raise


def create_accident_map(df: pd.DataFrame, output_file: str = 'outputs/accident_map.html'):
    """Create interactive map of accident locations"""
    if not GEOSPATIAL_AVAILABLE:
        print("❌ Cannot create map: geopandas/folium not installed")
        return

    if df is None or len(df) == 0:
        print("⚠️  No data to map")
        return

    try:
        # Create base map centered on US
        m = folium.Map(
            location=[39.8283, -98.5795],
            zoom_start=4,
            tiles='OpenStreetMap'
        )

        # Create marker cluster
        marker_cluster = MarkerCluster().add_to(m)

        # Add markers for each accident
        markers_added = 0
        for idx, row in df.iterrows():
            try:
                # Validate coordinates
                lat = float(row['latitude'])
                lon = float(row['longitude'])

                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue

                # Create popup text with safe string conversion
                popup_text = f"""
                <b>Event ID:</b> {row['ev_id']}<br>
                <b>Date:</b> {row.get('ev_date', 'N/A')}<br>
                <b>Location:</b> {row.get('ev_city', 'N/A')}, {row.get('ev_state', 'N/A')}<br>
                <b>Type:</b> {row.get('ev_type', 'N/A')}<br>
                <b>Fatalities:</b> {row.get('fatalities', 0)}<br>
                <b>Serious Injuries:</b> {row.get('serious_injuries', 0)}
                """

                # Color code by severity
                if row.get('fatalities', 0) > 0:
                    color = 'red'
                    icon = 'exclamation-triangle'
                elif row.get('serious_injuries', 0) > 0:
                    color = 'orange'
                    icon = 'warning'
                else:
                    color = 'blue'
                    icon = 'info-sign'

                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{row.get('ev_city', 'Unknown')}, {row.get('ev_state', 'N/A')}",
                    icon=folium.Icon(color=color, icon=icon)
                ).add_to(marker_cluster)

                markers_added += 1

            except (ValueError, TypeError, KeyError) as e:
                # Skip rows with invalid data
                continue

        # Create output directory
        Path('outputs').mkdir(exist_ok=True)

        # Save map
        m.save(output_file)
        print(f"✅ Map saved to: {output_file}")
        print(f"   {markers_added:,} accident locations plotted (from {len(df):,} total events)")

    except Exception as e:
        print(f"❌ Error creating map: {e}")
        raise


def create_heatmap(df: pd.DataFrame, output_file: str = 'outputs/accident_heatmap.html'):
    """Create heatmap of accident density"""
    if not GEOSPATIAL_AVAILABLE:
        print("❌ Cannot create heatmap: geopandas/folium not installed")
        return

    if df is None or len(df) == 0:
        print("⚠️  No data for heatmap")
        return

    try:
        # Create base map
        m = folium.Map(
            location=[39.8283, -98.5795],
            zoom_start=4,
            tiles='OpenStreetMap'
        )

        # Prepare data for heatmap with validation
        heat_data = []
        for idx, row in df.iterrows():
            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])

                # Validate coordinates
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    heat_data.append([lat, lon])
            except (ValueError, TypeError, KeyError):
                continue

        if len(heat_data) == 0:
            print("⚠️  No valid coordinates for heatmap")
            return

        # Add heatmap layer
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

        # Create output directory
        Path('outputs').mkdir(exist_ok=True)

        # Save map
        m.save(output_file)
        print(f"✅ Heatmap saved to: {output_file}")
        print(f"   {len(heat_data):,} valid coordinates plotted")

    except Exception as e:
        print(f"❌ Error creating heatmap: {e}")
        raise


def create_fatal_accidents_map(df: pd.DataFrame, output_file: str = 'outputs/fatal_accidents_map.html'):
    """Create map showing only fatal accidents"""
    if not GEOSPATIAL_AVAILABLE:
        print("❌ Cannot create map: geopandas/folium not installed")
        return

    if df is None or len(df) == 0:
        print("⚠️  No data for fatal accidents map")
        return

    try:
        # Filter for fatal accidents
        fatal_df = df[df['fatalities'] > 0].copy()

        if len(fatal_df) == 0:
            print("⚠️  No fatal accidents found in dataset")
            return

        print(f"🔴 Mapping {len(fatal_df):,} fatal accidents")

        # Create base map
        m = folium.Map(
            location=[39.8283, -98.5795],
            zoom_start=4,
            tiles='OpenStreetMap'
        )

        # Add markers for fatal accidents
        markers_added = 0
        for idx, row in fatal_df.iterrows():
            try:
                # Validate coordinates
                lat = float(row['latitude'])
                lon = float(row['longitude'])

                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue

                fatalities = int(row.get('fatalities', 0))
                if fatalities <= 0:
                    continue

                popup_text = f"""
                <b>Event ID:</b> {row['ev_id']}<br>
                <b>Date:</b> {row.get('ev_date', 'N/A')}<br>
                <b>Location:</b> {row.get('ev_city', 'N/A')}, {row.get('ev_state', 'N/A')}<br>
                <b>Fatalities:</b> {fatalities}<br>
                """

                # Size based on number of fatalities
                radius = min(fatalities * 2 + 5, 30)

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{fatalities} fatalities",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.6
                ).add_to(m)

                markers_added += 1

            except (ValueError, TypeError, KeyError):
                continue

        # Create output directory
        Path('outputs').mkdir(exist_ok=True)

        # Save map
        m.save(output_file)
        print(f"✅ Fatal accidents map saved to: {output_file}")
        print(f"   {markers_added:,} fatal accidents plotted")

    except Exception as e:
        print(f"❌ Error creating fatal accidents map: {e}")
        raise


def analyze_by_region():
    """Analyze accidents by geographic region"""
    query = """
    SELECT
        CASE
            WHEN TRIM(ev_state) IN ('WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM', 'AK', 'HI') THEN 'West'
            WHEN TRIM(ev_state) IN ('ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH') THEN 'Midwest'
            WHEN TRIM(ev_state) IN ('TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'TN', 'KY', 'WV', 'VA', 'NC', 'SC', 'GA', 'FL') THEN 'South'
            WHEN TRIM(ev_state) IN ('PA', 'NY', 'VT', 'NH', 'ME', 'MA', 'RI', 'CT', 'NJ', 'DE', 'MD', 'DC') THEN 'Northeast'
            ELSE 'Other'
        END as region,
        COUNT(*) as accident_count,
        COALESCE(SUM(inj_tot_f), 0) as total_fatalities
    FROM 'data/avall-events.csv'
    WHERE ev_state IS NOT NULL
      AND TRIM(ev_state) != ''
      AND LENGTH(TRIM(ev_state)) > 0
    GROUP BY region
    ORDER BY accident_count DESC
    """

    try:
        df = duckdb.query(query).to_df()

        if len(df) == 0:
            print("⚠️  No regional data found")
            return None

        print("\n📊 Accidents by Region:")
        print(df.to_string(index=False))
        return df

    except Exception as e:
        print(f"❌ Error in regional analysis: {e}")
        raise


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
