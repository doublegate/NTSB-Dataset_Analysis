"""
Event CRUD Operations

Database query functions for event-related endpoints.
"""

from datetime import date
from typing import Optional, List, Dict, Any
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


def get_events(
    conn,
    page: int = 1,
    page_size: int = 100,
    state: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    severity: Optional[str] = None,
    ev_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get paginated events with optional filters.

    Args:
        conn: Database connection
        page: Page number (1-indexed)
        page_size: Items per page
        state: Filter by state code (e.g., 'CA')
        start_date: Filter by minimum date
        end_date: Filter by maximum date
        severity: Filter by injury severity (FATL/SERS/MINR/NONE)
        ev_type: Filter by event type (ACC/INC)

    Returns:
        Dictionary with 'total', 'results' keys
    """
    # Build WHERE clauses
    where_clauses = []
    params = {}

    if state:
        where_clauses.append("ev_state = :state")
        params["state"] = state

    if start_date:
        where_clauses.append("ev_date >= :start_date")
        params["start_date"] = start_date

    if end_date:
        where_clauses.append("ev_date <= :end_date")
        params["end_date"] = end_date

    if severity:
        where_clauses.append("ev_highest_injury = :severity")
        params["severity"] = severity

    if ev_type:
        where_clauses.append("ev_type = :ev_type")
        params["ev_type"] = ev_type

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Count total matching events
    count_query = text(f"SELECT COUNT(*) FROM events {where_sql}")
    count_result = conn.execute(count_query, params)
    total = count_result.scalar()

    # Get paginated results
    offset = (page - 1) * page_size
    params["limit"] = page_size
    params["offset"] = offset

    query = text(
        f"""
        SELECT
            ev_id, ev_date, ev_time, ev_year, ev_month,
            ev_city, ev_state, ev_country, ev_site_zipcode,
            dec_latitude, dec_longitude,
            ev_type, ev_highest_injury,
            inj_tot_f, inj_tot_s, inj_tot_m, inj_tot_n,
            wx_cond_basic, ntsb_no, report_status
        FROM events
        {where_sql}
        ORDER BY ev_date DESC, ev_id
        LIMIT :limit OFFSET :offset
        """
    )

    result = conn.execute(query, params)
    events = [dict(row._mapping) for row in result]

    return {"total": total, "results": events}


def get_event_by_id(conn, ev_id: str) -> Optional[Dict[str, Any]]:
    """
    Get single event by ID with full details.

    Args:
        conn: Database connection
        ev_id: Event ID

    Returns:
        Event dictionary or None if not found
    """
    query = text(
        """
        SELECT
            ev_id, ev_date, ev_time, ev_year, ev_month, ev_dow,
            ev_city, ev_state, ev_country, ev_site_zipcode,
            dec_latitude, dec_longitude,
            ev_type, ev_highest_injury, ev_nr_apt_id, ev_nr_apt_loc, ev_nr_apt_dist,
            inj_tot_f, inj_tot_s, inj_tot_m, inj_tot_n,
            wx_cond_basic, wx_temp, wx_wind_dir, wx_wind_speed, wx_vis,
            flight_plan_filed, flight_activity, flight_phase,
            ntsb_no, report_status, probable_cause
        FROM events
        WHERE ev_id = :ev_id
        """
    )

    result = conn.execute(query, {"ev_id": ev_id})
    row = result.first()

    if row:
        return dict(row._mapping)
    return None


def get_event_aircraft(conn, ev_id: str) -> List[Dict[str, Any]]:
    """
    Get aircraft involved in event.

    Args:
        conn: Database connection
        ev_id: Event ID

    Returns:
        List of aircraft dictionaries
    """
    query = text(
        """
        SELECT
            Aircraft_Key, acft_serial_number, regis_no,
            acft_make, acft_model, acft_series, acft_category, acft_type_code,
            far_part, oper_country, owner_city, owner_state,
            damage, cert_max_gr_wt, num_eng, fixed_retractable
        FROM aircraft
        WHERE ev_id = :ev_id
        ORDER BY Aircraft_Key
        """
    )

    result = conn.execute(query, {"ev_id": ev_id})
    return [dict(row._mapping) for row in result]


def get_event_findings(conn, ev_id: str) -> List[Dict[str, Any]]:
    """
    Get findings for event.

    Args:
        conn: Database connection
        ev_id: Event ID

    Returns:
        List of finding dictionaries
    """
    query = text(
        """
        SELECT
            finding_code, finding_description, cm_inPC, modifier_code
        FROM Findings
        WHERE ev_id = :ev_id
        ORDER BY cm_inPC DESC, finding_code
        """
    )

    result = conn.execute(query, {"ev_id": ev_id})
    return [dict(row._mapping) for row in result]


def get_event_narratives(conn, ev_id: str) -> List[Dict[str, Any]]:
    """
    Get narratives for event.

    Args:
        conn: Database connection
        ev_id: Event ID

    Returns:
        List of narrative dictionaries
    """
    query = text(
        """
        SELECT
            narr_accp, narr_cause, narr_rectification
        FROM narratives
        WHERE ev_id = :ev_id
        """
    )

    result = conn.execute(query, {"ev_id": ev_id})
    return [dict(row._mapping) for row in result]


def search_events(conn, search_term: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Full-text search across event narratives.

    Args:
        conn: Database connection
        search_term: Search query
        limit: Maximum results

    Returns:
        List of matching events with relevance ranking
    """
    query = text(
        """
        SELECT
            e.ev_id, e.ev_date, e.ev_state, e.ev_city,
            e.ev_highest_injury, e.inj_tot_f,
            n.narr_accp,
            ts_rank(n.search_vector, to_tsquery('english', :search_term)) as relevance
        FROM events e
        JOIN narratives n ON e.ev_id = n.ev_id
        WHERE n.search_vector @@ to_tsquery('english', :search_term)
        ORDER BY relevance DESC
        LIMIT :limit
        """
    )

    # Convert search term to tsquery format (replace spaces with &)
    formatted_term = " & ".join(search_term.split())

    result = conn.execute(query, {"search_term": formatted_term, "limit": limit})
    return [dict(row._mapping) for row in result]
