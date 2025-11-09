"""
Geospatial Router

PostGIS-powered spatial queries and GeoJSON export.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.database import get_db_connection
from app.crud.geospatial import (
    get_events_within_radius,
    get_events_in_bbox,
    get_event_density_grid,
    get_event_clusters,
    get_events_for_geojson,
    get_state_boundaries,
)
from app.schemas.geojson import events_to_geojson, GeoJSONFeatureCollection
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/radius",
    summary="Events Within Radius",
    description="Get events within a specified radius of a point (km)",
)
async def events_within_radius(
    lat: float = Query(..., description="Center point latitude", ge=-90, le=90),
    lon: float = Query(..., description="Center point longitude", ge=-180, le=180),
    radius_km: float = Query(..., description="Radius in kilometers", gt=0, le=1000),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
):
    """
    Get events within radius of a point.

    Uses PostGIS ST_DWithin for efficient spatial queries.

    Example:
    - Los Angeles (50km): lat=34.0522, lon=-118.2437, radius_km=50
    - New York (100km): lat=40.7128, lon=-74.0060, radius_km=100
    """
    try:
        with get_db_connection() as conn:
            results = get_events_within_radius(
                conn, latitude=lat, longitude=lon, radius_km=radius_km, limit=limit
            )

        return {
            "center": {"latitude": lat, "longitude": lon},
            "radius_km": radius_km,
            "total": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error getting events within radius: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/bbox",
    summary="Events in Bounding Box",
    description="Get events within a rectangular bounding box",
)
async def events_in_bbox(
    min_lat: float = Query(..., description="Southwest corner latitude", ge=-90, le=90),
    min_lon: float = Query(
        ..., description="Southwest corner longitude", ge=-180, le=180
    ),
    max_lat: float = Query(..., description="Northeast corner latitude", ge=-90, le=90),
    max_lon: float = Query(
        ..., description="Northeast corner longitude", ge=-180, le=180
    ),
    limit: int = Query(1000, ge=1, le=5000, description="Maximum results"),
):
    """
    Get events within bounding box.

    Coordinates define a rectangular area (min/max lat/lon).

    Example:
    - California: min_lat=32.5, min_lon=-124.5, max_lat=42.0, max_lon=-114.0
    - Texas: min_lat=25.8, min_lon=-106.6, max_lat=36.5, max_lon=-93.5
    """
    try:
        # Validate bounding box
        if min_lat >= max_lat:
            raise HTTPException(
                status_code=400, detail="min_lat must be less than max_lat"
            )
        if min_lon >= max_lon:
            raise HTTPException(
                status_code=400, detail="min_lon must be less than max_lon"
            )

        with get_db_connection() as conn:
            results = get_events_in_bbox(
                conn,
                min_lat=min_lat,
                min_lon=min_lon,
                max_lat=max_lat,
                max_lon=max_lon,
                limit=limit,
            )

        return {
            "bbox": {
                "min_lat": min_lat,
                "min_lon": min_lon,
                "max_lat": max_lat,
                "max_lon": max_lon,
            },
            "total": len(results),
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting events in bounding box: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/density",
    summary="Event Density Heatmap",
    description="Get event density data for heatmap visualization",
)
async def event_density(
    grid_size_degrees: float = Query(
        0.5, ge=0.1, le=5.0, description="Grid cell size in degrees (~0.5 = ~50km)"
    ),
):
    """
    Get event density grid for heatmap visualization.

    Returns grid cells with event counts and density levels.

    Grid sizes:
    - 0.5 degrees ≈ 50km (recommended for national view)
    - 0.1 degrees ≈ 10km (recommended for state view)
    - 1.0 degrees ≈ 100km (recommended for continental view)
    """
    try:
        with get_db_connection() as conn:
            results = get_event_density_grid(conn, grid_size_degrees=grid_size_degrees)

        return {
            "grid_size_degrees": grid_size_degrees,
            "total_cells": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error calculating event density: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/clusters",
    summary="Spatial Event Clusters",
    description="Find spatial clusters of events using DBSCAN algorithm",
)
async def event_clusters(
    cluster_distance_km: float = Query(
        10.0,
        ge=1.0,
        le=100.0,
        description="Maximum distance between cluster members (km)",
    ),
    min_cluster_size: int = Query(
        5, ge=2, le=50, description="Minimum events to form a cluster"
    ),
):
    """
    Find spatial clusters using DBSCAN (Density-Based Spatial Clustering).

    Returns clusters with:
    - Center coordinates (average of cluster members)
    - Event count
    - Date range (earliest to latest event)
    - Fatal event count
    - List of event IDs

    Useful for identifying:
    - Accident-prone airports or regions
    - Geographic patterns in aviation safety
    - Hotspots for targeted interventions
    """
    try:
        with get_db_connection() as conn:
            results = get_event_clusters(
                conn,
                cluster_distance_km=cluster_distance_km,
                min_cluster_size=min_cluster_size,
            )

        return {
            "cluster_distance_km": cluster_distance_km,
            "min_cluster_size": min_cluster_size,
            "total_clusters": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error finding event clusters: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/geojson",
    response_model=GeoJSONFeatureCollection,
    summary="Export as GeoJSON",
    description="Export events as GeoJSON FeatureCollection for mapping",
)
async def export_geojson(
    state: Optional[str] = Query(None, description="Filter by state code"),
    severity: Optional[str] = Query(
        None, description="Filter by severity (FATL/SERS/MINR/NONE)"
    ),
    limit: int = Query(10000, ge=1, le=50000, description="Maximum features"),
):
    """
    Export events as GeoJSON FeatureCollection.

    GeoJSON format is compatible with:
    - Leaflet.js
    - Mapbox GL JS
    - OpenLayers
    - QGIS
    - ArcGIS

    Each feature includes:
    - Point geometry (longitude, latitude)
    - Event properties (ID, date, state, severity, casualties, NTSB number)
    """
    try:
        with get_db_connection() as conn:
            events = get_events_for_geojson(
                conn, state=state, severity=severity, limit=limit
            )

        geojson = events_to_geojson(events)

        return geojson

    except Exception as e:
        logger.error(f"Error exporting GeoJSON: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/state/{state_code}",
    summary="Events by State",
    description="Get events within a US state's bounding box",
)
async def events_by_state(state_code: str):
    """
    Get events within a US state.

    Returns state bounding box and all events within it.

    Note: Uses approximate bounding box, not exact state polygons.
    """
    try:
        # Validate state code (2 letters)
        if len(state_code) != 2:
            raise HTTPException(
                status_code=400, detail="State code must be 2 letters (e.g., CA, TX)"
            )

        state_code = state_code.upper()

        with get_db_connection() as conn:
            state_info = get_state_boundaries(conn, state_code)

            if not state_info:
                raise HTTPException(
                    status_code=404, detail=f"No events found for state {state_code}"
                )

            # Get events within state bounding box
            events = get_events_in_bbox(
                conn,
                min_lat=state_info["min_lat"],
                min_lon=state_info["min_lon"],
                max_lat=state_info["max_lat"],
                max_lon=state_info["max_lon"],
                limit=10000,
            )

        return {
            "state": state_code,
            "bbox": {
                "min_lat": state_info["min_lat"],
                "min_lon": state_info["min_lon"],
                "max_lat": state_info["max_lat"],
                "max_lon": state_info["max_lon"],
            },
            "total_events": state_info["event_count"],
            "results": events,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting events for state {state_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
