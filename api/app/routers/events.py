"""
Events Router

Endpoints for querying aviation accident events.
"""

from typing import Optional
from datetime import date
from fastapi import APIRouter, HTTPException, Query
from app.schemas.common import PaginatedResponse
from app.schemas.event import (
    EventDetail,
    EventSummary,
    AircraftBase,
    FindingBase,
    NarrativeBase,
)
from app.database import get_db_connection
from app.crud.events import (
    get_events,
    get_event_by_id,
    get_event_aircraft,
    get_event_findings,
    get_event_narratives,
)
import logging
import math

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    response_model=PaginatedResponse[EventSummary],
    summary="List Events",
    description="Get paginated list of events with optional filters",
)
async def list_events(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
    state: Optional[str] = Query(None, description="Filter by state code (e.g., CA)"),
    start_date: Optional[date] = Query(None, description="Filter by minimum date"),
    end_date: Optional[date] = Query(None, description="Filter by maximum date"),
    severity: Optional[str] = Query(
        None, description="Filter by injury severity (FATL/SERS/MINR/NONE)"
    ),
    ev_type: Optional[str] = Query(None, description="Filter by event type (ACC/INC)"),
):
    """
    Get paginated list of events.

    Supports filtering by:
    - State (2-letter code)
    - Date range (start_date, end_date)
    - Injury severity (FATL/SERS/MINR/NONE)
    - Event type (ACC=Accident, INC=Incident)

    Results are ordered by date (descending).
    """
    try:
        with get_db_connection() as conn:
            result = get_events(
                conn,
                page=page,
                page_size=page_size,
                state=state,
                start_date=start_date,
                end_date=end_date,
                severity=severity,
                ev_type=ev_type,
            )

        total_pages = math.ceil(result["total"] / page_size)

        # Convert results to EventSummary objects
        events = [EventSummary(**event) for event in result["results"]]

        return PaginatedResponse(
            total=result["total"],
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            results=events,
        )
    except Exception as e:
        logger.error(f"Error listing events: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/{ev_id}",
    response_model=EventDetail,
    summary="Get Event Details",
    description="Get full details for a specific event including aircraft, findings, and narratives",
)
async def get_event(ev_id: str):
    """
    Get detailed event information by event ID.

    Includes:
    - Event details (date, location, injuries, weather, etc.)
    - Aircraft involved
    - Investigation findings
    - Accident narratives
    """
    try:
        with get_db_connection() as conn:
            # Get event base data
            event = get_event_by_id(conn, ev_id)
            if not event:
                raise HTTPException(status_code=404, detail=f"Event {ev_id} not found")

            # Get related data
            aircraft = get_event_aircraft(conn, ev_id)
            findings = get_event_findings(conn, ev_id)
            narratives = get_event_narratives(conn, ev_id)

        # Build EventDetail response
        event_detail = EventDetail(
            **event,
            aircraft=[AircraftBase(**a) for a in aircraft],
            findings=[FindingBase(**f) for f in findings],
            narratives=[NarrativeBase(**n) for n in narratives],
        )

        return event_detail

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting event {ev_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/{ev_id}/aircraft",
    response_model=list[AircraftBase],
    summary="Get Event Aircraft",
    description="Get aircraft involved in a specific event",
)
async def get_event_aircraft_list(ev_id: str):
    """Get list of aircraft involved in event."""
    try:
        with get_db_connection() as conn:
            # Verify event exists
            event = get_event_by_id(conn, ev_id)
            if not event:
                raise HTTPException(status_code=404, detail=f"Event {ev_id} not found")

            aircraft = get_event_aircraft(conn, ev_id)

        return [AircraftBase(**a) for a in aircraft]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting aircraft for event {ev_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/{ev_id}/findings",
    response_model=list[FindingBase],
    summary="Get Event Findings",
    description="Get investigation findings for a specific event",
)
async def get_event_findings_list(ev_id: str):
    """Get list of investigation findings for event."""
    try:
        with get_db_connection() as conn:
            # Verify event exists
            event = get_event_by_id(conn, ev_id)
            if not event:
                raise HTTPException(status_code=404, detail=f"Event {ev_id} not found")

            findings = get_event_findings(conn, ev_id)

        return [FindingBase(**f) for f in findings]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting findings for event {ev_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get(
    "/{ev_id}/narratives",
    response_model=list[NarrativeBase],
    summary="Get Event Narratives",
    description="Get accident narratives for a specific event",
)
async def get_event_narratives_list(ev_id: str):
    """Get accident narratives for event."""
    try:
        with get_db_connection() as conn:
            # Verify event exists
            event = get_event_by_id(conn, ev_id)
            if not event:
                raise HTTPException(status_code=404, detail=f"Event {ev_id} not found")

            narratives = get_event_narratives(conn, ev_id)

        return [NarrativeBase(**n) for n in narratives]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting narratives for event {ev_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
