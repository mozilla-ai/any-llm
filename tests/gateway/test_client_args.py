from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import get_db
from any_llm.gateway.server import create_app
from tests.gateway.conftest import _run_alembic_migrations


class MockCompletionError(Exception):
    """Exception raised to short-circuit mock acompletion calls."""


@pytest.fixture
def config_with_client_args(postgres_url: str) -> GatewayConfig:
    """Create a test configuration with client_args for openai provider."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        providers={
            "openai": {
                "api_key": "test-openai-key",
                "api_base": "https://api.openai.com/v1",
                "client_args": {
                    "custom_headers": {"X-Custom-Header": "custom-value"},
                    "timeout": 60,
                },
            }
        },
    )


@pytest.fixture
def client_with_client_args(config_with_client_args: GatewayConfig) -> Generator[TestClient]:
    """Create a test client with client_args configured."""
    from sqlalchemy import text

    from any_llm.gateway.db import Base

    _run_alembic_migrations(config_with_client_args.database_url)
    engine = create_engine(config_with_client_args.database_url, pool_pre_ping=True)
    app = create_app(config_with_client_args)

    def override_get_db() -> Generator[Session]:
        testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()


@pytest.mark.asyncio
async def test_client_args_passed_to_acompletion(
    client_with_client_args: TestClient,
) -> None:
    """Verify client_args from config flows through to acompletion call."""
    captured_kwargs: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise MockCompletionError

    master_key_header = {"X-AnyLLM-Key": "Bearer test-master-key"}

    response = client_with_client_args.post(
        "/v1/users",
        json={"user_id": "test-user", "alias": "Test User"},
        headers=master_key_header,
    )
    assert response.status_code == 200

    with patch("any_llm.gateway.routes.chat.acompletion", new=mock_acompletion):
        client_with_client_args.post(
            "/v1/chat/completions",
            json={
                "model": "openai:gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test-user",
            },
            headers=master_key_header,
        )

    assert "client_args" in captured_kwargs, "client_args not passed to acompletion"
    assert captured_kwargs["client_args"]["custom_headers"]["X-Custom-Header"] == "custom-value"
    assert captured_kwargs["client_args"]["timeout"] == 60
    assert captured_kwargs["api_key"] == "test-openai-key"
    assert captured_kwargs["api_base"] == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_provider_config_without_client_args(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Verify completion works when client_args is not configured."""
    captured_kwargs: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise MockCompletionError

    with patch("any_llm.gateway.routes.chat.acompletion", new=mock_acompletion):
        client.post(
            "/v1/chat/completions",
            json={
                "model": "openai:gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": test_user["user_id"],
            },
            headers=master_key_header,
        )

    assert "client_args" not in captured_kwargs or captured_kwargs.get("client_args") is None
