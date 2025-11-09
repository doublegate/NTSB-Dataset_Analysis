"""
Statistics Pydantic Schemas

Data models for statistics API responses.
"""

from typing import Optional
from pydantic import BaseModel, Field


class YearlyStats(BaseModel):
    """Yearly statistics schema."""

    ev_year: int = Field(..., description="Year")
    total_accidents: int = Field(..., description="Total accidents")
    fatal_accidents: int = Field(..., description="Fatal accidents")
    total_fatalities: int = Field(..., description="Total fatalities")
    avg_fatalities_per_accident: float = Field(
        ..., description="Average fatalities per accident"
    )
    serious_injury_accidents: int = Field(..., description="Serious injury accidents")
    destroyed_aircraft: int = Field(..., description="Destroyed aircraft")

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class StateStats(BaseModel):
    """State-level statistics schema."""

    ev_state: str = Field(..., description="State code (2 letters)")
    accident_count: int = Field(..., description="Total accidents")
    fatal_count: int = Field(..., description="Fatal accidents")
    avg_latitude: Optional[float] = Field(None, description="Average latitude")
    avg_longitude: Optional[float] = Field(None, description="Average longitude")

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class AircraftStats(BaseModel):
    """Aircraft make/model statistics schema."""

    make: str = Field(..., description="Aircraft make")
    model: str = Field(..., description="Aircraft model")
    accident_count: int = Field(..., description="Number of accidents")
    fatal_rate: float = Field(..., description="Fatal accident rate (0.0-1.0)")

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class SummaryStats(BaseModel):
    """Overall database summary statistics."""

    total_events: int = Field(..., description="Total events in database")
    date_range_start: str = Field(..., description="Earliest event date")
    date_range_end: str = Field(..., description="Latest event date")
    years_coverage: int = Field(..., description="Years of data coverage")
    total_fatalities: int = Field(..., description="Total fatalities (all time)")
    total_aircraft: int = Field(..., description="Total aircraft involved")
    states_covered: int = Field(..., description="Number of states with events")
    most_dangerous_year: int = Field(..., description="Year with most fatalities")
    safest_year: int = Field(..., description="Year with fewest accidents")
