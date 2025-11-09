"""
Events Endpoint Tests

Tests for /api/v1/events endpoints.
"""

import pytest
from datetime import date


def test_list_events_default(client):
    """Test listing events with default pagination."""
    response = client.get("/api/v1/events/")
    assert response.status_code == 200

    data = response.json()
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert "results" in data

    # Verify pagination
    assert data["page"] == 1
    assert data["page_size"] == 100
    assert len(data["results"]) <= 100

    # Verify event structure
    if data["results"]:
        event = data["results"][0]
        assert "ev_id" in event
        assert "ev_date" in event
        assert "ev_state" in event


def test_list_events_pagination(client):
    """Test pagination parameters."""
    response = client.get("/api/v1/events/?page=2&page_size=50")
    assert response.status_code == 200

    data = response.json()
    assert data["page"] == 2
    assert data["page_size"] == 50
    assert len(data["results"]) <= 50


def test_list_events_filter_by_state(client):
    """Test filtering events by state."""
    response = client.get("/api/v1/events/?state=CA")
    assert response.status_code == 200

    data = response.json()
    # Verify all results are from California
    for event in data["results"]:
        if event["ev_state"]:
            assert event["ev_state"] == "CA"


def test_list_events_filter_by_date(client):
    """Test filtering events by date range."""
    response = client.get("/api/v1/events/?start_date=2020-01-01&end_date=2020-12-31")
    assert response.status_code == 200

    data = response.json()
    # Verify all results are within date range
    for event in data["results"]:
        event_date = date.fromisoformat(event["ev_date"])
        assert date(2020, 1, 1) <= event_date <= date(2020, 12, 31)


def test_list_events_filter_by_severity(client):
    """Test filtering events by injury severity."""
    response = client.get("/api/v1/events/?severity=FATL")
    assert response.status_code == 200

    data = response.json()
    # Verify all results have fatal injuries
    for event in data["results"]:
        if event["ev_highest_injury"]:
            assert event["ev_highest_injury"] == "FATL"


def test_get_event_details(client, sample_ev_id):
    """Test getting single event details."""
    if not sample_ev_id:
        pytest.skip("No events in database")

    response = client.get(f"/api/v1/events/{sample_ev_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["ev_id"] == sample_ev_id
    assert "aircraft" in data
    assert "findings" in data
    assert "narratives" in data

    # Verify nested data are lists
    assert isinstance(data["aircraft"], list)
    assert isinstance(data["findings"], list)
    assert isinstance(data["narratives"], list)


def test_get_event_not_found(client):
    """Test getting non-existent event."""
    response = client.get("/api/v1/events/INVALID_ID")
    assert response.status_code == 404


def test_get_event_aircraft(client, sample_ev_id):
    """Test getting aircraft for event."""
    if not sample_ev_id:
        pytest.skip("No events in database")

    response = client.get(f"/api/v1/events/{sample_ev_id}/aircraft")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


def test_get_event_findings(client, sample_ev_id):
    """Test getting findings for event."""
    if not sample_ev_id:
        pytest.skip("No events in database")

    response = client.get(f"/api/v1/events/{sample_ev_id}/findings")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


def test_get_event_narratives(client, sample_ev_id):
    """Test getting narratives for event."""
    if not sample_ev_id:
        pytest.skip("No events in database")

    response = client.get(f"/api/v1/events/{sample_ev_id}/narratives")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
