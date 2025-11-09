"""
Common Pydantic Schemas

Shared schemas used across multiple routers (pagination, responses, etc.).
"""

from typing import Generic, TypeVar, List, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(
        default=100, ge=1, le=1000, description="Number of items per page"
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    total: int = Field(..., description="Total number of items matching query")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    results: List[T] = Field(..., description="List of result items")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status (healthy/unhealthy)")
    database: str = Field(..., description="Database connection status")
    version: Optional[str] = Field(None, description="API version")


class DatabaseHealthResponse(BaseModel):
    """Detailed database health response."""

    status: str = Field(..., description="Database status")
    version: str = Field(..., description="PostgreSQL version")
    size: str = Field(..., description="Database size (human-readable)")
    event_count: int = Field(..., description="Total number of events")
    pool_stats: dict = Field(..., description="Connection pool statistics")
