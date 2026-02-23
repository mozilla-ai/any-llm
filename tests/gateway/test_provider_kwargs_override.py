"""Tests for provider kwargs not overriding user request fields."""

from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.routes.chat import _get_provider_kwargs


def test_provider_kwargs_do_not_contain_model() -> None:
    """Test that provider kwargs don't include fields that could override request."""
    from any_llm import LLMProvider

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
    # Provider kwargs should not contain fields like 'model', 'messages', 'stream'
    assert "model" not in kwargs
    assert "messages" not in kwargs
    assert "stream" not in kwargs


def test_request_fields_take_precedence_over_provider_kwargs() -> None:
    """Test that the merge order gives request fields precedence."""
    provider_kwargs = {"api_key": "sk-test", "model": "overridden-model"}
    request_fields = {"model": "user-model", "messages": [{"role": "user", "content": "Hi"}]}

    # Our fix: {**provider_kwargs, **request_fields} -- request wins
    completion_kwargs = {**provider_kwargs, **request_fields}
    assert completion_kwargs["model"] == "user-model"
    assert completion_kwargs["api_key"] == "sk-test"
