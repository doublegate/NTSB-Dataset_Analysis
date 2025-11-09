"""
Geospatial CRUD Operations

PostGIS spatial query functions for geospatial endpoints.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


def get_events_within_radius(
    conn, latitude: float, longitude: float, radius_km: float, limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get events within radius of a point (in kilometers).

    Uses PostGIS ST_DWithin for efficient spatial queries.

    Args:
        conn: Database connection
        latitude: Center point latitude
        longitude: Center point longitude
        radius_km: Radius in kilometers
        limit: Maximum results

    Returns:
        List of events with distance
    """
    query = text(
        """
        SELECT
            ev_id, ev_date, ev_state, ev_city,
            dec_latitude, dec_longitude,
            ev_highest_injury, inj_tot_f, inj_tot_s,
            ntsb_no,
            ST_Distance(
                location_geom::geography,
                ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326)::geography
            ) / 1000.0 as distance_km
        FROM events
        WHERE location_geom IS NOT NULL
          AND ST_DWithin(
              location_geom::geography,
              ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326)::geography,
              :radius_meters
          )
        ORDER BY distance_km
        LIMIT :limit
        """
    )

    result = conn.execute(
        query,
        {
            "latitude": latitude,
            "longitude": longitude,
            "radius_meters": radius_km * 1000,  # Convert km to meters
            "limit": limit,
        },
    )

    return [dict(row._mapping) for row in result]


def get_events_in_bbox(
    conn,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Get events within bounding box.

    Args:
        conn: Database connection
        min_lat: Minimum latitude (southwest corner)
        min_lon: Minimum longitude (southwest corner)
        max_lat: Maximum latitude (northeast corner)
        max_lon: Maximum longitude (northeast corner)
        limit: Maximum results

    Returns:
        List of events within bounding box
    """
    query = text(
        """
        SELECT
            ev_id, ev_date, ev_state, ev_city,
            dec_latitude, dec_longitude,
            ev_highest_injury, inj_tot_f, inj_tot_s,
            ntsb_no
        FROM events
        WHERE location_geom IS NOT NULL
          AND location_geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        ORDER BY ev_date DESC
        LIMIT :limit
        """
    )

    result = conn.execute(
        query,
        {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon,
            "limit": limit,
        },
    )

    return [dict(row._mapping) for row in result]


def get_event_density_grid(
    conn, grid_size_degrees: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Calculate event density on a grid (for heatmaps).

    Args:
        conn: Database connection
        grid_size_degrees: Grid cell size in degrees (~0.5 = ~50km at equator)

    Returns:
        List of grid cells with event counts
    """
    query = text(
        """
        WITH grid AS (
            SELECT
                FLOOR(dec_latitude / :grid_size) * :grid_size as lat_bin,
                FLOOR(dec_longitude / :grid_size) * :grid_size as lon_bin,
                COUNT(*) as event_count,
                SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_count
            FROM events
            WHERE dec_latitude IS NOT NULL AND dec_longitude IS NOT NULL
            GROUP BY lat_bin, lon_bin
        )
        SELECT
            lat_bin,
            lon_bin,
            event_count,
            fatal_count,
            CASE
                WHEN event_count > 500 THEN 'very_high'
                WHEN event_count > 200 THEN 'high'
                WHEN event_count > 50 THEN 'medium'
                ELSE 'low'
            END as density_level
        FROM grid
        ORDER BY event_count DESC
        """
    )

    result = conn.execute(query, {"grid_size": grid_size_degrees})
    return [dict(row._mapping) for row in result]


def get_event_clusters(
    conn, cluster_distance_km: float = 10.0, min_cluster_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Find spatial clusters of events using ST_ClusterDBSCAN.

    DBSCAN (Density-Based Spatial Clustering) groups nearby events.

    Args:
        conn: Database connection
        cluster_distance_km: Maximum distance between cluster members (km)
        min_cluster_size: Minimum events to form a cluster

    Returns:
        List of clusters with center coordinates and event lists
    """
    # Convert km to degrees (approximate: 1 degree ≈ 111 km)
    eps_degrees = cluster_distance_km / 111.0

    query = text(
        """
        WITH clustered AS (
            SELECT
                ev_id,
                dec_latitude,
                dec_longitude,
                ev_date,
                ev_highest_injury,
                ST_ClusterDBSCAN(
                    location_geom::geometry,
                    eps := :eps_degrees,
                    minpoints := :min_cluster_size
                ) OVER () as cluster_id
            FROM events
            WHERE location_geom IS NOT NULL
        )
        SELECT
            cluster_id,
            COUNT(*) as event_count,
            AVG(dec_latitude) as center_lat,
            AVG(dec_longitude) as center_lon,
            MIN(ev_date) as earliest_event,
            MAX(ev_date) as latest_event,
            SUM(CASE WHEN ev_highest_injury = 'FATL' THEN 1 ELSE 0 END) as fatal_events,
            array_agg(ev_id ORDER BY ev_date DESC) as event_ids
        FROM clustered
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
        HAVING COUNT(*) >= :min_cluster_size
        ORDER BY event_count DESC
        """
    )

    result = conn.execute(
        query, {"eps_degrees": eps_degrees, "min_cluster_size": min_cluster_size}
    )

    return [dict(row._mapping) for row in result]


def get_events_for_geojson(
    conn,
    state: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 10000,
) -> List[Dict[str, Any]]:
    """
    Get events formatted for GeoJSON export.

    Args:
        conn: Database connection
        state: Optional state filter
        severity: Optional severity filter
        limit: Maximum results

    Returns:
        List of events with coordinates and properties
    """
    where_clauses = ["dec_latitude IS NOT NULL", "dec_longitude IS NOT NULL"]
    params = {"limit": limit}

    if state:
        where_clauses.append("ev_state = :state")
        params["state"] = state

    if severity:
        where_clauses.append("ev_highest_injury = :severity")
        params["severity"] = severity

    where_sql = " AND ".join(where_clauses)

    query = text(
        f"""
        SELECT
            ev_id,
            ev_date,
            ev_state,
            ev_city,
            dec_latitude,
            dec_longitude,
            ev_highest_injury,
            inj_tot_f,
            inj_tot_s,
            ntsb_no,
            probable_cause
        FROM events
        WHERE {where_sql}
        ORDER BY ev_date DESC
        LIMIT :limit
        """
    )

    result = conn.execute(query, params)
    return [dict(row._mapping) for row in result]


def get_state_boundaries(conn, state_code: str) -> Optional[Dict[str, Any]]:
    """
    Get approximate bounding box for a US state.

    Note: This returns a simple bounding box, not actual state polygons.
    For production, consider using a dedicated geodata table.

    Args:
        conn: Database connection
        state_code: 2-letter state code

    Returns:
        Bounding box coordinates
    """
    query = text(
        """
        SELECT
            ev_state,
            MIN(dec_latitude) as min_lat,
            MAX(dec_latitude) as max_lat,
            MIN(dec_longitude) as min_lon,
            MAX(dec_longitude) as max_lon,
            COUNT(*) as event_count
        FROM events
        WHERE ev_state = :state_code
          AND dec_latitude IS NOT NULL
          AND dec_longitude IS NOT NULL
        GROUP BY ev_state
        """
    )

    result = conn.execute(query, {"state_code": state_code})
    row = result.first()

    if row:
        return dict(row._mapping)
    return None
