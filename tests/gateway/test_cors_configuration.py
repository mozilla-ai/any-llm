"""Tests for CORS configuration."""

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import get_db
from any_llm.gateway.server import create_app


def test_cors_disabled_by_default(postgres_url: str, test_db: Session) -> None:
    """Test that CORS middleware is not added when cors_allow_origins is empty."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
    )

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "https://evil.com"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


def test_cors_with_specific_origins(postgres_url: str, test_db: Session) -> None:
    """Test that CORS allows only configured origins."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        cors_allow_origins=["https://trusted.com"],
    )

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        # Trusted origin should get CORS headers
        response = client.get("/health", headers={"Origin": "https://trusted.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://trusted.com"
        assert response.headers.get("access-control-allow-credentials") == "true"

        # Untrusted origin should not get CORS headers
        response = client.get("/health", headers={"Origin": "https://evil.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") != "https://evil.com"


def test_cors_wildcard_disables_credentials(postgres_url: str, test_db: Session) -> None:
    """Test that wildcard origin disables allow_credentials per CORS spec."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        cors_allow_origins=["*"],
    )

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "https://any-site.com"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        # Credentials should NOT be allowed with wildcard
        assert response.headers.get("access-control-allow-credentials") != "true"
