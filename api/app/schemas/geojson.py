"""
GeoJSON Pydantic Schemas

Data models for GeoJSON responses.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


class GeoJSONGeometry(BaseModel):
    """GeoJSON geometry object."""

    type: Literal["Point"] = Field(default="Point", description="Geometry type")
    coordinates: List[float] = Field(..., description="[longitude, latitude]")


class GeoJSONFeature(BaseModel):
    """GeoJSON feature object."""

    type: Literal["Feature"] = Field(default="Feature", description="Feature type")
    geometry: GeoJSONGeometry = Field(..., description="Geometry")
    properties: Dict[str, Any] = Field(..., description="Feature properties")


class GeoJSONFeatureCollection(BaseModel):
    """GeoJSON feature collection."""

    type: Literal["FeatureCollection"] = Field(
        default="FeatureCollection", description="Collection type"
    )
    features: List[GeoJSONFeature] = Field(..., description="List of features")


def events_to_geojson(events: List[Dict[str, Any]]) -> GeoJSONFeatureCollection:
    """
    Convert list of events to GeoJSON FeatureCollection.

    Args:
        events: List of event dictionaries with coordinates

    Returns:
        GeoJSON FeatureCollection
    """
    features = []

    for event in events:
        if event.get("dec_latitude") and event.get("dec_longitude"):
            feature = GeoJSONFeature(
                geometry=GeoJSONGeometry(
                    coordinates=[
                        float(event["dec_longitude"]),
                        float(event["dec_latitude"]),
                    ]
                ),
                properties={
                    "ev_id": event["ev_id"],
                    "ev_date": str(event["ev_date"]),
                    "ev_state": event.get("ev_state"),
                    "ev_city": event.get("ev_city"),
                    "severity": event.get("ev_highest_injury"),
                    "fatalities": event.get("inj_tot_f", 0),
                    "injuries": event.get("inj_tot_s", 0),
                    "ntsb_no": event.get("ntsb_no"),
                },
            )
            features.append(feature)

    return GeoJSONFeatureCollection(features=features)
