"""Tests for Cache-Control headers added by CacheControlMiddleware."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from any_llm.gateway.server import CacheControlMiddleware


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with CacheControlMiddleware for testing."""
    app = FastAPI()
    app.add_middleware(CacheControlMiddleware)

    @app.get("/v1/users")
    def list_users() -> list[str]:
        return ["alice", "bob"]

    @app.get("/v1/keys")
    def list_keys() -> list[str]:
        return ["key-1"]

    @app.get("/v1/budgets")
    def list_budgets() -> list[str]:
        return ["budget-1"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/health/liveness")
    def liveness() -> str:
        return "I'm alive!"

    @app.get("/health/readiness")
    def readiness() -> dict[str, str]:
        return {"status": "healthy", "database": "connected"}

    return app


def test_authenticated_endpoint_has_cache_control() -> None:
    """Authenticated endpoints must include Cache-Control: private, no-store, no-cache."""
    with TestClient(_make_app()) as client:
        response = client.get("/v1/users")
        assert response.headers["Cache-Control"] == "private, no-store, no-cache"


def test_authenticated_endpoint_has_vary_authorization() -> None:
    """Authenticated endpoints must include Vary: Authorization."""
    with TestClient(_make_app()) as client:
        response = client.get("/v1/users")
        assert "Authorization" in response.headers.get("Vary", "")


def test_health_endpoint_no_cache_headers() -> None:
    """Health endpoints should not have restrictive cache headers."""
    with TestClient(_make_app()) as client:
        response = client.get("/health")
        assert "Cache-Control" not in response.headers


def test_health_liveness_no_cache_headers() -> None:
    """Liveness probe should not have restrictive cache headers."""
    with TestClient(_make_app()) as client:
        response = client.get("/health/liveness")
        assert "Cache-Control" not in response.headers


def test_health_readiness_no_cache_headers() -> None:
    """Readiness probe should not have restrictive cache headers."""
    with TestClient(_make_app()) as client:
        response = client.get("/health/readiness")
        assert "Cache-Control" not in response.headers


def test_cache_headers_on_keys_endpoint() -> None:
    """Keys endpoint returns sensitive data and must have cache headers."""
    with TestClient(_make_app()) as client:
        response = client.get("/v1/keys")
        assert response.headers["Cache-Control"] == "private, no-store, no-cache"
        assert "Authorization" in response.headers.get("Vary", "")


def test_cache_headers_on_budgets_endpoint() -> None:
    """Budgets endpoint returns sensitive data and must have cache headers."""
    with TestClient(_make_app()) as client:
        response = client.get("/v1/budgets")
        assert response.headers["Cache-Control"] == "private, no-store, no-cache"
        assert "Authorization" in response.headers.get("Vary", "")
