"""
Search Router

Full-text search across accident narratives.
"""

from fastapi import APIRouter, HTTPException, Query
from app.database import get_db_connection
from app.crud.events import search_events
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    summary="Search Events",
    description="Full-text search across accident narratives",
)
async def search(
    q: str = Query(
        ..., min_length=3, description="Search query (minimum 3 characters)"
    ),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
):
    """
    Full-text search across accident narratives.

    Uses PostgreSQL's full-text search capabilities with relevance ranking.

    Example queries:
    - "engine failure"
    - "weather IMC"
    - "pilot error stall"

    Multiple words are combined with AND logic.
    """
    try:
        with get_db_connection() as conn:
            results = search_events(conn, search_term=q, limit=limit)

        return {
            "query": q,
            "total": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error searching events: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
