"""
Health Endpoint Tests

Tests for /api/v1/health endpoints.
"""


def test_health_check(client):
    """Test basic health check endpoint."""
    response = client.get("/api/v1/health/")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "version" in data
    assert data["status"] in ["healthy", "unhealthy"]


def test_database_health(client):
    """Test detailed database health check."""
    response = client.get("/api/v1/health/database")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "size" in data
    assert "event_count" in data
    assert "pool_stats" in data

    # Verify event count is reasonable
    assert data["event_count"] > 100000  # Should have ~179K events
