#!/usr/bin/env python3
"""
Execute all geospatial analyses for Phase 2 Sprint 8.

This script runs all 5 geospatial analysis steps:
1. Data Preparation
2. DBSCAN Clustering
3. Kernel Density Estimation
4. Getis-Ord Gi* Hotspot Analysis
5. Moran's I Spatial Autocorrelation

Generates all visualizations and interactive maps.
"""

import json
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine

# Geospatial analysis
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from libpysal.weights import KNN
from esda.getisord import G_Local
from esda.moran import Moran, Moran_Local

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "notebooks" / "geospatial" / "figures"
MAP_DIR = BASE_DIR / "notebooks" / "geospatial" / "maps"

# Create directories
FIG_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def extract_geospatial_data() -> gpd.GeoDataFrame:
    """Extract and prepare geospatial dataset from database."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)

    # Database connection
    engine = create_engine("postgresql://parobek@localhost/ntsb_aviation")

    # Extract events with coordinates
    query = """
    SELECT
        ev_id, ev_date, ev_year, ev_state, ev_city,
        dec_latitude, dec_longitude,
        inj_tot_f, inj_tot_s, inj_tot_m, inj_tot_n
    FROM events
    WHERE dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL
      AND dec_latitude BETWEEN -90 AND 90
      AND dec_longitude BETWEEN -180 AND 180
    ORDER BY ev_date;
    """

    print("Extracting geospatial data...")
    df = pd.read_sql(query, engine)
    print(f"✅ Extracted {len(df):,} events")

    # Remove outliers using IQR
    def remove_outliers_iqr(
        data: pd.DataFrame, column: str, k: float = 3.0
    ) -> Tuple[pd.DataFrame, int]:
        Q1, Q3 = data[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - k * IQR, Q3 + k * IQR
        mask = (data[column] >= lower) & (data[column] <= upper)
        return data[mask], (~mask).sum()

    df_clean, lat_outliers = remove_outliers_iqr(df, "dec_latitude")
    df_clean, lon_outliers = remove_outliers_iqr(df_clean, "dec_longitude")
    total_removed = lat_outliers + lon_outliers
    print(f"✅ Removed {total_removed} outliers ({total_removed/len(df)*100:.3f}%)")

    # Create GeoDataFrame
    geometry = [
        Point(xy) for xy in zip(df_clean["dec_longitude"], df_clean["dec_latitude"])
    ]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")
    print(f"✅ GeoDataFrame created with {len(gdf):,} events")

    # Save
    gdf.to_parquet(DATA_DIR / "geospatial_events.parquet")
    gdf.to_crs("EPSG:5070").to_parquet(DATA_DIR / "geospatial_events_projected.parquet")
    print("✅ Saved geospatial datasets")

    return gdf


def run_dbscan_clustering(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Run DBSCAN clustering analysis."""
    print("\n" + "=" * 60)
    print("PHASE 2: DBSCAN CLUSTERING")
    print("=" * 60)

    # Extract coordinates and convert to radians
    coords = gdf[["dec_latitude", "dec_longitude"]].values
    coords_rad = np.radians(coords)

    # Run DBSCAN
    eps_km, min_samples = 50, 10
    eps_rad = eps_km / 6371  # Earth radius in km

    print(f"Running DBSCAN (eps={eps_km}km, min_samples={min_samples})...")
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", n_jobs=-1)
    labels = db.fit_predict(coords_rad)

    gdf["cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"✅ Found {n_clusters} clusters")
    print(f"   Clustered: {len(gdf) - n_noise:,} events")
    print(f"   Noise: {n_noise:,} events")

    # Cluster statistics
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_data = gdf[gdf["cluster"] == cluster_id]
        cluster_stats.append(
            {
                "cluster_id": cluster_id,
                "size": len(cluster_data),
                "centroid_lat": cluster_data["dec_latitude"].mean(),
                "centroid_lon": cluster_data["dec_longitude"].mean(),
                "dominant_state": cluster_data["ev_state"].mode()[0]
                if len(cluster_data["ev_state"].mode()) > 0
                else "Unknown",
                "total_fatalities": int(cluster_data["inj_tot_f"].sum()),
                "fatal_accidents": int((cluster_data["inj_tot_f"] > 0).sum()),
            }
        )

    cluster_df = pd.DataFrame(cluster_stats)
    cluster_df.to_csv(DATA_DIR / "cluster_statistics.csv", index=False)
    print(f"✅ Saved cluster statistics ({len(cluster_df)} clusters)")

    return cluster_df


def run_kde_analysis(gdf: gpd.GeoDataFrame):
    """Run Kernel Density Estimation."""
    print("\n" + "=" * 60)
    print("PHASE 3: KERNEL DENSITY ESTIMATION")
    print("=" * 60)

    coords = gdf[["dec_longitude", "dec_latitude"]].values

    # Event density KDE
    print("Computing event density KDE...")
    kde_event = gaussian_kde(coords.T)

    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
    lon_grid = np.linspace(lon_min, lon_max, 100)
    lat_grid = np.linspace(lat_min, lat_max, 100)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
    density_event = kde_event(positions).reshape(lon_mesh.shape)
    print(
        f"✅ Event density computed (range: {density_event.min():.6f} to {density_event.max():.6f})"
    )

    # Fatality density KDE (weighted)
    print("Computing fatality density KDE...")
    fatalities = gdf["inj_tot_f"].values
    weighted_coords = []
    for i, (lon, lat) in enumerate(coords):
        weight = int(fatalities[i]) + 1
        weighted_coords.extend(
            [(lon, lat)] * min(weight, 100)
        )  # Cap weight for performance

    weighted_coords = np.array(weighted_coords[:50000])  # Sample for performance
    kde_fatality = gaussian_kde(weighted_coords.T)
    density_fatality = kde_fatality(positions).reshape(lon_mesh.shape)
    print(
        f"✅ Fatality density computed (range: {density_fatality.min():.6f} to {density_fatality.max():.6f})"
    )


def run_getis_ord_analysis(gdf: gpd.GeoDataFrame):
    """Run Getis-Ord Gi* hotspot analysis."""
    print("\n" + "=" * 60)
    print("PHASE 4: GETIS-ORD GI* HOTSPOT ANALYSIS")
    print("=" * 60)

    # Project to Albers for accurate distances
    gdf_proj = gdf.to_crs("EPSG:5070")

    # Create spatial weights
    print("Creating spatial weights (k=8)...")
    w = KNN.from_dataframe(gdf_proj, k=8)
    w.transform = "r"
    print(f"✅ Spatial weights created ({w.n} observations)")

    # Calculate Gi* (convert to float64 for Numba compatibility)
    print("Computing Getis-Ord Gi* statistic...")
    gi_star = G_Local(
        gdf["inj_tot_f"].astype(float).values, w, star=True, permutations=999
    )

    gdf["gi_star_z"] = gi_star.Zs
    gdf["gi_star_p"] = gi_star.p_sim

    # Classify hotspots
    gdf["hotspot_type"] = "Not Significant"
    gdf.loc[
        (gdf["gi_star_z"] > 1.96) & (gdf["gi_star_p"] < 0.05), "hotspot_type"
    ] = "Hot Spot (95%)"
    gdf.loc[
        (gdf["gi_star_z"] > 2.58) & (gdf["gi_star_p"] < 0.01), "hotspot_type"
    ] = "Hot Spot (99%)"
    gdf.loc[
        (gdf["gi_star_z"] < -1.96) & (gdf["gi_star_p"] < 0.05), "hotspot_type"
    ] = "Cold Spot (95%)"
    gdf.loc[
        (gdf["gi_star_z"] < -2.58) & (gdf["gi_star_p"] < 0.01), "hotspot_type"
    ] = "Cold Spot (99%)"

    hotspot_counts = gdf["hotspot_type"].value_counts()
    print("✅ Hotspot classification complete:")
    for htype, count in hotspot_counts.items():
        print(f"   {htype}: {count:,}")

    # Save
    gdf[
        [
            "ev_id",
            "dec_latitude",
            "dec_longitude",
            "inj_tot_f",
            "gi_star_z",
            "gi_star_p",
            "hotspot_type",
            "geometry",
        ]
    ].to_file(DATA_DIR / "getis_ord_hotspots.geojson", driver="GeoJSON")
    print("✅ Saved hotspot GeoJSON")


def run_morans_i_analysis(gdf: gpd.GeoDataFrame):
    """Run Moran's I spatial autocorrelation analysis."""
    print("\n" + "=" * 60)
    print("PHASE 5: MORAN'S I SPATIAL AUTOCORRELATION")
    print("=" * 60)

    # Project and create weights
    gdf_proj = gdf.to_crs("EPSG:5070")
    w = KNN.from_dataframe(gdf_proj, k=8)
    w.transform = "r"

    # Global Moran's I (convert to float64 for Numba compatibility)
    print("Computing Global Moran's I...")
    moran_global = Moran(gdf["inj_tot_f"].astype(float).values, w, permutations=999)

    print(f"✅ Global Moran's I: {moran_global.I:.4f}")
    print(f"   Z-score: {moran_global.z_norm:.4f}")
    print(f"   P-value: {moran_global.p_norm:.4f}")
    print(
        f'   Interpretation: {"Positive" if moran_global.I > 0 else "Negative"} spatial autocorrelation'
    )

    # Local Moran's I (LISA) (convert to float64 for Numba compatibility)
    print("Computing Local Moran's I (LISA)...")
    lisa = Moran_Local(gdf["inj_tot_f"].astype(float).values, w, permutations=999)

    gdf["lisa_I"] = lisa.Is
    gdf["lisa_q"] = lisa.q
    gdf["lisa_p"] = lisa.p_sim
    gdf["lisa_significant"] = gdf["lisa_p"] < 0.05

    # Classify clusters
    cluster_names = {
        1: "HH (High-High)",
        2: "LH (Low-High)",
        3: "LL (Low-Low)",
        4: "HL (High-Low)",
    }
    gdf["lisa_cluster"] = gdf.apply(
        lambda row: cluster_names[row["lisa_q"]]
        if row["lisa_significant"]
        else "Not Significant",
        axis=1,
    )

    lisa_counts = gdf["lisa_cluster"].value_counts()
    print("✅ LISA clustering complete:")
    for ctype, count in lisa_counts.items():
        print(f"   {ctype}: {count:,}")

    # Save
    stats = {
        "global_morans_i": {
            "I": float(moran_global.I),
            "z_score": float(moran_global.z_norm),
            "p_value": float(moran_global.p_norm),
        },
        "lisa_clusters": lisa_counts.to_dict(),
    }
    with open(DATA_DIR / "morans_i_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("✅ Saved Moran's I results")


def create_interactive_maps(gdf: gpd.GeoDataFrame, cluster_df: pd.DataFrame):
    """Create all 5 interactive Folium maps."""
    print("\n" + "=" * 60)
    print("PHASE 6: INTERACTIVE VISUALIZATIONS")
    print("=" * 60)

    # 1. DBSCAN Clusters Map
    print("Creating DBSCAN clusters map...")
    m_dbscan = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    # Add top 20 cluster centroids
    for idx, row in cluster_df.nlargest(20, "size").iterrows():
        folium.Marker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            icon=folium.Icon(color="red", icon="info-sign"),
            popup=f"Cluster {row['cluster_id']}<br>Size: {row['size']}<br>State: {row['dominant_state']}",
        ).add_to(m_dbscan)

    m_dbscan.save(str(MAP_DIR / "dbscan_clusters.html"))
    print("✅ Saved: dbscan_clusters.html")

    # 2. Event Density Heatmap
    print("Creating event density heatmap...")
    m_event = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    heat_data = [
        [row["dec_latitude"], row["dec_longitude"]]
        for idx, row in gdf.sample(min(len(gdf), 10000), random_state=42).iterrows()
    ]
    HeatMap(heat_data, radius=15, blur=25).add_to(m_event)
    m_event.save(str(MAP_DIR / "kde_event_density.html"))
    print("✅ Saved: kde_event_density.html")

    # 3. Fatality Density Heatmap
    print("Creating fatality density heatmap...")
    m_fatal = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    heat_data_fatal = [
        [row["dec_latitude"], row["dec_longitude"], row["inj_tot_f"] + 1]
        for idx, row in gdf.sample(min(len(gdf), 10000), random_state=42).iterrows()
    ]
    HeatMap(
        heat_data_fatal,
        radius=15,
        blur=25,
        gradient={0.4: "yellow", 0.65: "orange", 1: "red"},
    ).add_to(m_fatal)
    m_fatal.save(str(MAP_DIR / "kde_fatality_density.html"))
    print("✅ Saved: kde_fatality_density.html")

    # 4. Getis-Ord Hotspots Map
    print("Creating Getis-Ord hotspots map...")
    m_getis = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    color_map = {
        "Hot Spot (99%)": "darkred",
        "Hot Spot (95%)": "red",
        "Cold Spot (95%)": "blue",
        "Cold Spot (99%)": "darkblue",
    }

    significant = gdf[gdf["hotspot_type"] != "Not Significant"]
    if len(significant) > 5000:
        significant = significant.sample(5000, random_state=42)
    for idx, row in significant.iterrows():
        if row["hotspot_type"] in color_map:
            folium.CircleMarker(
                location=[row["dec_latitude"], row["dec_longitude"]],
                radius=min(abs(row["gi_star_z"]) * 2, 10),
                color=color_map[row["hotspot_type"]],
                fill=True,
                fillOpacity=0.6,
                popup=f"{row['hotspot_type']}<br>Z: {row['gi_star_z']:.2f}",
            ).add_to(m_getis)

    m_getis.save(str(MAP_DIR / "getis_ord_hotspots.html"))
    print("✅ Saved: getis_ord_hotspots.html")

    # 5. LISA Clusters Map
    print("Creating LISA clusters map...")
    m_lisa = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    color_map_lisa = {
        "HH (High-High)": "red",
        "LL (Low-Low)": "blue",
        "LH (Low-High)": "pink",
        "HL (High-Low)": "lightblue",
    }

    lisa_sig = gdf[gdf["lisa_significant"]]
    if len(lisa_sig) > 5000:
        lisa_sig = lisa_sig.sample(5000, random_state=42)
    for idx, row in lisa_sig.iterrows():
        if row["lisa_cluster"] in color_map_lisa:
            folium.CircleMarker(
                location=[row["dec_latitude"], row["dec_longitude"]],
                radius=4,
                color=color_map_lisa[row["lisa_cluster"]],
                fill=True,
                fillOpacity=0.7,
                popup=f"{row['lisa_cluster']}",
            ).add_to(m_lisa)

    m_lisa.save(str(MAP_DIR / "lisa_clusters.html"))
    print("✅ Saved: lisa_clusters.html")

    print(f"\n✅ All 5 interactive maps created in: {MAP_DIR}")


def main():
    """Run complete geospatial analysis pipeline."""
    print("\n" + "🗺️ " * 30)
    print("NTSB GEOSPATIAL ANALYSIS - PHASE 2 SPRINT 8")
    print("🗺️ " * 30)

    # Phase 1: Data Preparation
    gdf = extract_geospatial_data()

    # Phase 2: DBSCAN Clustering
    cluster_df = run_dbscan_clustering(gdf)

    # Phase 3: Kernel Density Estimation
    run_kde_analysis(gdf)

    # Phase 4: Getis-Ord Gi* Hotspot Analysis
    run_getis_ord_analysis(gdf)

    # Phase 5: Moran's I Spatial Autocorrelation
    run_morans_i_analysis(gdf)

    # Phase 6: Interactive Visualizations
    create_interactive_maps(gdf, cluster_df)

    print("\n" + "=" * 60)
    print("✅ ALL GEOSPATIAL ANALYSES COMPLETE")
    print("=" * 60)
    print(f"\nData files: {DATA_DIR}")
    print(f"Figures: {FIG_DIR}")
    print(f"Interactive maps: {MAP_DIR}")
    print("\nNext steps: Create comprehensive analysis report")


if __name__ == "__main__":
    main()
