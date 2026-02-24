"""Tests for provider kwargs not overriding user request fields."""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from any_llm import LLMProvider
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import Base, get_db
from any_llm.gateway.routes.chat import _get_provider_kwargs
from any_llm.gateway.server import create_app
from tests.gateway.conftest import _run_alembic_migrations


class _MockCompletionError(Exception):
    """Exception raised to short-circuit mock acompletion calls."""


@pytest.fixture
def config_with_model_in_provider(postgres_url: str) -> GatewayConfig:
    """Create a test configuration where the provider config contains a 'model' key."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        providers={
            "openai": {
                "api_key": "test-openai-key",
                "model": "provider-default-model",
            }
        },
    )


@pytest.fixture
def client_with_model_in_provider(config_with_model_in_provider: GatewayConfig) -> Generator[TestClient]:
    """Create a test client whose provider config contains a conflicting 'model' key."""
    _run_alembic_migrations(config_with_model_in_provider.database_url)
    engine = create_engine(config_with_model_in_provider.database_url, pool_pre_ping=True)
    app = create_app(config_with_model_in_provider)

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


def test_provider_kwargs_do_not_contain_model() -> None:
    """Test that provider kwargs from a typical config don't include request-level fields."""
    config = GatewayConfig(
        database_url="postgresql://localhost/test",
        master_key="test",
        providers={
            "openai": {
                "api_key": "sk-test-key",
            },
        },
    )

    kwargs = _get_provider_kwargs(config, LLMProvider.OPENAI)
    assert "api_key" in kwargs
    assert "model" not in kwargs
    assert "messages" not in kwargs
    assert "stream" not in kwargs


@pytest.mark.asyncio
async def test_user_model_not_overridden_by_provider_config(
    client_with_model_in_provider: TestClient,
) -> None:
    """Verify the user's model value is not overwritten when provider config also contains 'model'."""
    captured_kwargs: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise _MockCompletionError

    master_key_header = {"X-AnyLLM-Key": "Bearer test-master-key"}

    response = client_with_model_in_provider.post(
        "/v1/users",
        json={"user_id": "test-user", "alias": "Test User"},
        headers=master_key_header,
    )
    assert response.status_code == 200

    with patch("any_llm.gateway.routes.chat.acompletion", new=mock_acompletion):
        client_with_model_in_provider.post(
            "/v1/chat/completions",
            json={
                "model": "openai:gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test-user",
            },
            headers=master_key_header,
        )

    assert captured_kwargs["model"] == "openai:gpt-4"
    assert captured_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert captured_kwargs["api_key"] == "test-openai-key"


@pytest.mark.asyncio
async def test_unset_optional_fields_do_not_override_provider_defaults(
    client_with_model_in_provider: TestClient,
) -> None:
    """Verify that optional request fields the user didn't set don't appear in kwargs."""
    captured_kwargs: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise _MockCompletionError

    master_key_header = {"X-AnyLLM-Key": "Bearer test-master-key"}

    with patch("any_llm.gateway.routes.chat.acompletion", new=mock_acompletion):
        client_with_model_in_provider.post(
            "/v1/chat/completions",
            json={
                "model": "openai:gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test-user",
            },
            headers=master_key_header,
        )

    # The user didn't set temperature, so it should not be in the kwargs
    # (exclude_unset=True prevents None from overwriting provider defaults)
    assert "temperature" not in captured_kwargs
    assert "max_tokens" not in captured_kwargs
    assert "tools" not in captured_kwargs
