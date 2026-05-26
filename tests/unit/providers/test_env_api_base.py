"""Tests for ENV_API_BASE_NAME environment variable support."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from any_llm.providers.llamafile.llamafile import LlamafileProvider
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.otari.otari import OtariProvider

pytest.importorskip("otari")


def test_resolve_api_base_returns_parameter_when_provided() -> None:
    """When api_base parameter is provided, it should be used."""

    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "test"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        ENV_API_BASE_NAME = "TEST_API_BASE"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://default.example.com/v1"

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = AsyncMock()
        provider = TestProvider(api_key="test-key", api_base="https://custom.example.com/v1")
        result = provider._resolve_api_base("https://custom.example.com/v1")
        assert result == "https://custom.example.com/v1"


@patch.dict(os.environ, {"TEST_API_BASE": "https://env.example.com/v1"}, clear=False)
def test_resolve_api_base_returns_env_var_when_no_parameter() -> None:
    """When api_base is not provided, env var should be used."""

    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "test"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        ENV_API_BASE_NAME = "TEST_API_BASE"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://default.example.com/v1"

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = AsyncMock()
        provider = TestProvider(api_key="test-key")
        result = provider._resolve_api_base(None)
        assert result == "https://env.example.com/v1"


@patch.dict(os.environ, {}, clear=True)
def test_resolve_api_base_returns_none_when_no_env_var() -> None:
    """When neither api_base nor env var is set, return None for class default fallback."""

    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "test"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        ENV_API_BASE_NAME = "TEST_API_BASE"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://default.example.com/v1"

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = AsyncMock()
        provider = TestProvider(api_key="test-key")
        result = provider._resolve_api_base(None)
        assert result is None


def test_resolve_api_base_returns_none_when_no_env_api_base_name() -> None:
    """When ENV_API_BASE_NAME is not defined, return None."""

    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "test"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://default.example.com/v1"

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = AsyncMock()
        provider = TestProvider(api_key="test-key")
        result = provider._resolve_api_base(None)
        assert result is None


@patch.dict(os.environ, {"LLAMAFILE_API_BASE": "http://custom-llamafile:9000/v1"}, clear=False)
def test_llamafile_uses_env_var_for_api_base() -> None:
    """LlamafileProvider should use LLAMAFILE_API_BASE env var."""
    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = AsyncMock()
        LlamafileProvider()
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["base_url"] == "http://custom-llamafile:9000/v1"


def test_llamafile_uses_default_when_no_env_var() -> None:
    """LlamafileProvider should use default API_BASE when env var is not set."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = AsyncMock()
            LlamafileProvider()
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://127.0.0.1:8080/v1"


def test_llamafile_parameter_overrides_env_var() -> None:
    """Direct api_base parameter should override env var."""
    with patch.dict(os.environ, {"LLAMAFILE_API_BASE": "http://env-llamafile:9000/v1"}, clear=False):
        with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = AsyncMock()
            LlamafileProvider(api_base="http://param-llamafile:8000/v1")
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://param-llamafile:8000/v1"


@patch.dict(os.environ, {"OTARI_API_BASE": "https://env-otari.example.com"}, clear=False)
def test_otari_uses_env_var_for_api_base() -> None:
    """OtariProvider should use OTARI_API_BASE env var."""
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
        mock_otari_client.return_value = type("Client", (), {"openai": AsyncMock(), "platform_mode": False})()
        OtariProvider()
        call_kwargs = mock_otari_client.call_args.kwargs
        assert call_kwargs["api_base"] == "https://env-otari.example.com"


@patch.dict(os.environ, {}, clear=True)
def test_otari_requires_api_base_without_env_var() -> None:
    """OtariProvider should raise error when neither api_base nor env vars are set."""
    with pytest.raises(ValueError, match="api_base is required"):
        OtariProvider()


def test_otari_parameter_overrides_env_var() -> None:
    """Direct api_base parameter should override OTARI_API_BASE env var."""
    with patch.dict(os.environ, {"OTARI_API_BASE": "https://env-otari.example.com"}, clear=False):
        with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
            mock_otari_client.return_value = type("Client", (), {"openai": AsyncMock(), "platform_mode": False})()
            OtariProvider(api_base="https://param-otari.example.com")
            call_kwargs = mock_otari_client.call_args.kwargs
            assert call_kwargs["api_base"] == "https://param-otari.example.com"


@patch.dict(os.environ, {"GATEWAY_API_BASE": "https://legacy-gateway.example.com"}, clear=False)
def test_otari_uses_legacy_gateway_api_base_as_fallback() -> None:
    """OtariProvider should fallback to GATEWAY_API_BASE when OTARI_API_BASE is unset."""
    with patch.dict(os.environ, {"OTARI_API_BASE": ""}, clear=False):
        with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
            mock_otari_client.return_value = type("Client", (), {"openai": AsyncMock(), "platform_mode": False})()
            OtariProvider()
            call_kwargs = mock_otari_client.call_args.kwargs
            assert call_kwargs["api_base"] == "https://legacy-gateway.example.com"


def test_provider_metadata_includes_env_api_base() -> None:
    """Provider metadata should include env_api_base field."""

    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "test"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        ENV_API_BASE_NAME = "TEST_API_BASE"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    metadata = TestProvider.get_provider_metadata()
    assert metadata.env_api_base == "TEST_API_BASE"


def test_provider_metadata_env_api_base_none_when_not_defined() -> None:
    """Provider metadata env_api_base should be None when not defined."""

    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "test"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    metadata = TestProvider.get_provider_metadata()
    assert metadata.env_api_base is None
