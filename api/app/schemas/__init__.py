"""
Pydantic Schemas Package

Data models for API request/response validation.
"""

from app.schemas.common import (
    PaginationParams,
    PaginatedResponse,
    ErrorResponse,
    HealthResponse,
    DatabaseHealthResponse,
)
from app.schemas.event import (
    EventBase,
    EventDetail,
    EventSummary,
    AircraftBase,
    FindingBase,
    NarrativeBase,
)
from app.schemas.statistics import (
    YearlyStats,
    StateStats,
    AircraftStats,
    SummaryStats,
)

__all__ = [
    # Common
    "PaginationParams",
    "PaginatedResponse",
    "ErrorResponse",
    "HealthResponse",
    "DatabaseHealthResponse",
    # Events
    "EventBase",
    "EventDetail",
    "EventSummary",
    "AircraftBase",
    "FindingBase",
    "NarrativeBase",
    # Statistics
    "YearlyStats",
    "StateStats",
    "AircraftStats",
    "SummaryStats",
]
