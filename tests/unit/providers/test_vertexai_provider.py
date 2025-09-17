from unittest.mock import patch

from any_llm.config import ClientConfig
from any_llm.providers.vertexai import VertexaiProvider


def test_vertexai_initialization_without_api_key() -> None:
    """Test that the VertexaiProvider initializes correctly without API Key."""
    with patch("any_llm.providers.vertexai.vertexai.genai.Client"):
        provider = VertexaiProvider(ClientConfig())
        assert provider.config.api_key is None
