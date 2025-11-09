"""
Pytest Configuration and Fixtures

Shared test fixtures for API testing.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db_connection


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def db_connection():
    """Database connection for direct queries."""
    with get_db_connection() as conn:
        yield conn


@pytest.fixture
def sample_ev_id(db_connection):
    """Get a sample event ID from database."""
    result = db_connection.execute(
        "SELECT ev_id FROM events ORDER BY ev_date DESC LIMIT 1"
    )
    row = result.first()
    return row[0] if row else None
