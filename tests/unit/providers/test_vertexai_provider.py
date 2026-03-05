from unittest.mock import patch

from google.genai import types

from any_llm.providers.vertexai import VertexaiProvider


def test_vertexai_initialization_without_api_key() -> None:
    """Test that the VertexaiProvider initializes correctly without API Key."""
    with patch("any_llm.providers.vertexai.vertexai.genai.Client"):
        provider = VertexaiProvider()
        assert provider.client is not None


def test_vertexai_timeout_in_client_args_routed_to_http_options() -> None:
    """Test that timeout in client_args is converted to HttpOptions at client init."""
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as mock_client:
        VertexaiProvider(timeout=30.0)
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "http_options" in call_kwargs
        assert call_kwargs["http_options"].timeout == 30_000


def test_vertexai_timeout_does_not_override_explicit_http_options() -> None:
    """Test that explicit http_options timeout takes precedence over client_args timeout."""
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as mock_client:
        VertexaiProvider(timeout=30.0, http_options=types.HttpOptions(timeout=10_000))
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs["http_options"].timeout == 10_000
