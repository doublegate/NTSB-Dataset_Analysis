"""
Statistics Endpoint Tests

Tests for /api/v1/statistics endpoints.
"""



def test_get_summary(client):
    """Test overall summary statistics."""
    response = client.get("/api/v1/statistics/summary")
    assert response.status_code == 200

    data = response.json()
    assert "total_events" in data
    assert "date_range_start" in data
    assert "date_range_end" in data
    assert "years_coverage" in data
    assert "total_fatalities" in data
    assert "states_covered" in data

    # Verify reasonable values
    assert data["total_events"] > 100000
    assert data["years_coverage"] >= 60


def test_get_yearly_stats(client):
    """Test yearly statistics."""
    response = client.get("/api/v1/statistics/yearly")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    # Verify yearly stat structure
    stat = data[0]
    assert "ev_year" in stat
    assert "total_accidents" in stat
    assert "fatal_accidents" in stat
    assert "total_fatalities" in stat


def test_get_state_stats(client):
    """Test state statistics."""
    response = client.get("/api/v1/statistics/states")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    # Verify state stat structure
    stat = data[0]
    assert "ev_state" in stat
    assert "accident_count" in stat
    assert "fatal_count" in stat


def test_get_aircraft_stats(client):
    """Test aircraft statistics."""
    response = client.get("/api/v1/statistics/aircraft?limit=10")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10

    # Verify aircraft stat structure
    if data:
        stat = data[0]
        assert "make" in stat
        assert "model" in stat
        assert "accident_count" in stat
        assert "fatal_rate" in stat


def test_get_seasonal_patterns(client):
    """Test seasonal patterns."""
    response = client.get("/api/v1/statistics/seasonal")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 12  # Should have 12 months
