"""
Event Pydantic Schemas

Data models for event-related API responses.
"""

from datetime import date, time
from typing import Optional, List
from pydantic import BaseModel, Field


class EventBase(BaseModel):
    """Base event schema with core fields."""

    ev_id: str = Field(..., description="Event ID")
    ev_date: date = Field(..., description="Event date")
    ev_time: Optional[time] = Field(None, description="Event time (HH:MM:SS)")
    ev_year: int = Field(..., description="Event year")
    ev_month: int = Field(..., description="Event month (1-12)")

    # Location
    ev_city: Optional[str] = Field(None, description="Event city")
    ev_state: Optional[str] = Field(None, description="Event state (2-letter code)")
    ev_country: Optional[str] = Field(None, description="Event country (3-letter code)")
    ev_site_zipcode: Optional[str] = Field(None, description="ZIP code")
    dec_latitude: Optional[float] = Field(None, description="Latitude (decimal)")
    dec_longitude: Optional[float] = Field(None, description="Longitude (decimal)")

    # Classification
    ev_type: Optional[str] = Field(None, description="Event type (ACC/INC)")
    ev_highest_injury: Optional[str] = Field(
        None, description="Highest injury level (FATL/SERS/MINR/NONE)"
    )

    # Injury totals
    inj_tot_f: Optional[int] = Field(None, description="Total fatalities")
    inj_tot_s: Optional[int] = Field(None, description="Total serious injuries")
    inj_tot_m: Optional[int] = Field(None, description="Total minor injuries")
    inj_tot_n: Optional[int] = Field(None, description="Total uninjured")

    # Weather
    wx_cond_basic: Optional[str] = Field(
        None, description="Basic weather condition (VMC/IMC)"
    )

    # Investigation
    ntsb_no: Optional[str] = Field(None, description="NTSB number")
    report_status: Optional[str] = Field(None, description="Report status")

    class Config:
        """Pydantic configuration."""

        from_attributes = True  # Allow ORM mode


class AircraftBase(BaseModel):
    """Aircraft schema."""

    Aircraft_Key: str = Field(..., description="Aircraft key")
    acft_make: Optional[str] = Field(None, description="Aircraft make")
    acft_model: Optional[str] = Field(None, description="Aircraft model")
    acft_series: Optional[str] = Field(None, description="Aircraft series")
    acft_category: Optional[str] = Field(None, description="Aircraft category")
    regis_no: Optional[str] = Field(None, description="Registration number")
    damage: Optional[str] = Field(
        None, description="Damage level (DEST/SUBS/MINR/NONE)"
    )
    num_eng: Optional[int] = Field(None, description="Number of engines")

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class FindingBase(BaseModel):
    """Finding schema."""

    finding_code: Optional[str] = Field(None, description="Finding code")
    finding_description: Optional[str] = Field(None, description="Finding description")
    cm_inPC: Optional[bool] = Field(
        None, description="Cited in probable cause (TRUE/FALSE)"
    )

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class NarrativeBase(BaseModel):
    """Narrative schema."""

    narr_accp: Optional[str] = Field(None, description="Accident description")
    narr_cause: Optional[str] = Field(None, description="Cause/contributing factors")

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class EventDetail(EventBase):
    """Detailed event with nested related data."""

    aircraft: List[AircraftBase] = Field(
        default_factory=list, description="Aircraft involved"
    )
    findings: List[FindingBase] = Field(
        default_factory=list, description="Investigation findings"
    )
    narratives: List[NarrativeBase] = Field(
        default_factory=list, description="Accident narratives"
    )

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class EventSummary(BaseModel):
    """Lightweight event summary for lists."""

    ev_id: str = Field(..., description="Event ID")
    ev_date: date = Field(..., description="Event date")
    ev_state: Optional[str] = Field(None, description="Event state")
    ev_city: Optional[str] = Field(None, description="Event city")
    ev_highest_injury: Optional[str] = Field(None, description="Highest injury level")
    inj_tot_f: Optional[int] = Field(None, description="Total fatalities")
    ntsb_no: Optional[str] = Field(None, description="NTSB number")

    class Config:
        """Pydantic configuration."""

        from_attributes = True
