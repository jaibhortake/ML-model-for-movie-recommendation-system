"""Integration tests for the Movie Recommendation System FastAPI REST API.

Tests synchronous endpoints using FastAPI TestClient and asynchronous endpoints
using httpx.AsyncClient.
"""

from typing import Any
import httpx
import pytest
from fastapi.testclient import TestClient

# Attempt import of FastAPI app
try:
    from src.api import app
except ImportError:
    try:
        from api import app  # type: ignore
    except ImportError:
        try:
            from src.main import app  # type: ignore
        except ImportError:
            app = None


# ============================================================================
# Synchronous Endpoint Tests (TestClient)
# ============================================================================

def test_health_check(api_client: TestClient):
    """Test GET /health endpoint returns 200 OK and healthy status."""
    response = api_client.get("/health")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    data = response.json()
    assert "status" in data, "Health response must contain 'status' field"
    assert data["status"] in ["healthy", "ok", "UP", "success"]


def test_get_movies_pagination(api_client: TestClient):
    """Test GET /movies with pagination parameters page and page_size."""
    response = api_client.get("/movies?page=1&page_size=5")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    data = response.json()

    # Accommodate list response or paginated dictionary response
    if isinstance(data, dict):
        assert "items" in data or "movies" in data or "data" in data, "Paginated response should contain items list"
        items = data.get("items") or data.get("movies") or data.get("data")
    else:
        items = data

    assert isinstance(items, list), "Movie items must be returned in a list"
    assert len(items) <= 5, f"Expected at most 5 movies per page, got {len(items)}"
    assert len(items) > 0, "Expected at least 1 movie in page 1"


def test_get_movie_by_id(api_client: TestClient):
    """Test GET /movies/{movie_id} returns movie details for an existing movie."""
    movie_id = 1
    response = api_client.get(f"/movies/{movie_id}")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    data = response.json()

    retrieved_id = data.get("id") or data.get("movie_id") or data.get("item_id")
    assert retrieved_id == movie_id, f"Expected movie ID {movie_id}, got {retrieved_id}"
    assert "title" in data, "Movie detail must contain 'title' field"


def test_get_movie_not_found(api_client: TestClient):
    """Test GET /movies/{movie_id} returns 404 for a non-existent movie ID."""
    non_existent_id = 999999
    response = api_client.get(f"/movies/{non_existent_id}")
    assert response.status_code == 404, f"Expected 404 Not Found, got {response.status_code}"


def test_get_recommendations(api_client: TestClient):
    """Test GET /recommendations/{user_id} returns recommendations list."""
    user_id = 1
    # Test both standard paths
    response = api_client.get(f"/recommendations/{user_id}?n=5")
    if response.status_code == 404:
        response = api_client.get(f"/recommend/{user_id}?n=5")

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    data = response.json()

    recs = data.get("recommendations") or data.get("items") or data
    assert isinstance(recs, list), "Recommendations must be a list"
    assert len(recs) > 0, "Expected non-empty recommendations"


def test_get_similar_movies(api_client: TestClient):
    """Test GET /movies/{movie_id}/similar returns similar movies."""
    movie_id = 1
    response = api_client.get(f"/movies/{movie_id}/similar?n=4")
    if response.status_code == 404:
        response = api_client.get(f"/similar/{movie_id}?n=4")

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    data = response.json()

    similar = data.get("similar_movies") or data.get("items") or data
    assert isinstance(similar, list), "Similar movies must be a list"
    assert len(similar) > 0, "Expected non-empty similar movies list"


def test_post_rating_valid(api_client: TestClient):
    """Test POST /ratings with valid rating data returns success."""
    payload = {
        "user_id": 1,
        "movie_id": 2,
        "rating": 4.5,
    }
    response = api_client.post("/ratings", json=payload)
    assert response.status_code in [200, 201], f"Expected 200 or 201, got {response.status_code}"
    data = response.json()
    assert data is not None


def test_post_rating_invalid_range(api_client: TestClient):
    """Test POST /ratings with rating outside 1.0 - 5.0 returns validation error (422/400)."""
    # Test rating > 5.0
    payload_high = {"user_id": 1, "movie_id": 2, "rating": 6.5}
    resp_high = api_client.post("/ratings", json=payload_high)
    assert resp_high.status_code in [400, 422], (
        f"Rating > 5.0 must return 400 or 422, got {resp_high.status_code}"
    )

    # Test rating < 1.0
    payload_low = {"user_id": 1, "movie_id": 2, "rating": 0.0}
    resp_low = api_client.post("/ratings", json=payload_low)
    assert resp_low.status_code in [400, 422], (
        f"Rating < 1.0 must return 400 or 422, got {resp_low.status_code}"
    )


# ============================================================================
# Asynchronous Endpoint Tests (httpx.AsyncClient)
# ============================================================================

@pytest.mark.asyncio
async def test_async_health_and_movies(api_client: TestClient):
    """Test async interaction with FastAPI application using httpx.AsyncClient."""
    target_app = api_client.app

    # Support httpx ASGITransport or direct app parameter across httpx versions
    try:
        transport = httpx.ASGITransport(app=target_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp_health = await client.get("/health")
            assert resp_health.status_code == 200

            resp_movies = await client.get("/movies")
            assert resp_movies.status_code == 200
    except (AttributeError, TypeError):
        async with httpx.AsyncClient(app=target_app, base_url="http://testserver") as client:
            resp_health = await client.get("/health")
            assert resp_health.status_code == 200

            resp_movies = await client.get("/movies")
            assert resp_movies.status_code == 200
