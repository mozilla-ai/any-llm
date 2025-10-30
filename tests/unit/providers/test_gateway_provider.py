import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.gateway.gateway import GATEWAY_HEADER_NAME, GatewayProvider


def test_gateway_init_requires_api_base() -> None:
    with pytest.raises(ValueError, match="api_base is required"):
        GatewayProvider(api_key="test-key")


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_init_with_api_key(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(api_key="test-key", api_base="https://gateway.example.com")

    mock_openai_class.assert_called_once()
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["base_url"] == "https://gateway.example.com"
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer test-key"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_init_without_api_key(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(api_base="https://gateway.example.com")

    mock_openai_class.assert_called_once()
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["base_url"] == "https://gateway.example.com"
    assert call_kwargs["api_key"] == ""


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch("any_llm.providers.gateway.gateway.logger")
def test_gateway_init_header_override_warning(mock_logger: MagicMock, mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(
        api_key="new-key",
        api_base="https://gateway.example.com",
        default_headers={GATEWAY_HEADER_NAME: "Bearer old-key"},
    )

    mock_logger.info.assert_called_once()
    assert "already set" in mock_logger.info.call_args[0][0]
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer new-key"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch.dict(os.environ, {"GATEWAY_API_KEY": "env-key"}, clear=False)
def test_gateway_init_with_env_api_key(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(api_base="https://gateway.example.com")

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["api_key"] == "env-key"
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer env-key"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch.dict(os.environ, {}, clear=True)
def test_gateway_init_without_any_api_key(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(api_base="https://gateway.example.com")

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["api_key"] == ""
    assert "default_headers" not in call_kwargs or GATEWAY_HEADER_NAME not in call_kwargs.get("default_headers", {})


def test_verify_api_key_with_provided_key() -> None:
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(api_key="test-key", api_base="https://gateway.example.com")
        result = provider._verify_and_set_api_key("provided-key")
        assert result == "provided-key"


@patch.dict(os.environ, {"GATEWAY_API_KEY": "env-key"}, clear=False)
def test_verify_api_key_with_env_variable() -> None:
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(api_key="test-key", api_base="https://gateway.example.com")
        result = provider._verify_and_set_api_key(None)
        assert result == "env-key"


@patch.dict(os.environ, {}, clear=True)
def test_verify_api_key_none_returns_empty() -> None:
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(api_base="https://gateway.example.com")
        result = provider._verify_and_set_api_key(None)
        assert result == ""


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_client_initialization_with_custom_headers(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    custom_headers = {"X-Custom-Header": "custom-value"}
    GatewayProvider(api_key="test-key", api_base="https://gateway.example.com", default_headers=custom_headers)

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer test-key"
    assert call_kwargs["default_headers"]["X-Custom-Header"] == "custom-value"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_passes_kwargs_to_parent(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(
        api_key="test-key",
        api_base="https://gateway.example.com",
        timeout=30,
        max_retries=5,
        default_headers={},
    )

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["timeout"] == 30
    assert call_kwargs["max_retries"] == 5
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer test-key"
