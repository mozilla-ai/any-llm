"""Tests for CORS configuration."""

from collections.abc import AsyncGenerator
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.db import get_db
from any_llm.gateway.main import create_app


def _make_override(session: AsyncSession) -> Any:
    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        yield session

    return override_get_db


def test_cors_disabled_by_default(postgres_url: str, test_db: AsyncSession) -> None:
    """Test that CORS middleware is not added when cors_allow_origins is empty."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
    )

    app = create_app(config)
    app.dependency_overrides[get_db] = _make_override(test_db)

    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "https://evil.com"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


def test_cors_with_specific_origins(postgres_url: str, test_db: AsyncSession) -> None:
    """Test that CORS allows only configured origins."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        cors_allow_origins=["https://trusted.com"],
    )

    app = create_app(config)
    app.dependency_overrides[get_db] = _make_override(test_db)

    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "https://trusted.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://trusted.com"
        assert response.headers.get("access-control-allow-credentials") == "true"

        response = client.get("/health", headers={"Origin": "https://evil.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") != "https://evil.com"


def test_cors_wildcard_disables_credentials(postgres_url: str, test_db: AsyncSession) -> None:
    """Test that wildcard origin disables allow_credentials per CORS spec."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        cors_allow_origins=["*"],
    )

    app = create_app(config)
    app.dependency_overrides[get_db] = _make_override(test_db)

    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "https://any-site.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        assert response.headers.get("access-control-allow-credentials") != "true"
