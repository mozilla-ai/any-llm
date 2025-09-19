import pytest

from any_llm import AnyLLM
from any_llm.providers.perplexity import PerplexityProvider


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "sk-perplexity-test-123")


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    p = PerplexityProvider(api_key="sk-test")
    assert p.PROVIDER_NAME == "perplexity"
    assert p.API_BASE == "https://api.perplexity.ai"  # No override, just default
    assert p.SUPPORTS_COMPLETION is True
    assert p.SUPPORTS_COMPLETION_STREAMING is True
    assert p.SUPPORTS_RESPONSES is False
    assert p.SUPPORTS_EMBEDDING is False


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    # Test provider creation
    p = AnyLLM.create("perplexity", api_key="sk-1")
    assert isinstance(p, PerplexityProvider)
    assert p.PROVIDER_NAME == "perplexity"

    # Test that perplexity is in supported providers
    supported = AnyLLM.get_supported_providers()
    assert "perplexity" in supported


def test_model_provider_split() -> None:
    """Test that model string parsing works correctly."""
    provider_enum, model_name = AnyLLM.split_model_provider("perplexity/llama-3.1-sonar-small-128k-chat")
    assert provider_enum.value == "perplexity"
    assert model_name == "llama-3.1-sonar-small-128k-chat"


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = PerplexityProvider.get_provider_metadata()
    assert metadata.name == "perplexity"
    assert metadata.env_key == "PERPLEXITY_API_KEY"
    assert metadata.doc_url == "https://docs.perplexity.ai/"
    assert metadata.completion is True
    assert metadata.streaming is True
    assert metadata.embedding is False
    assert metadata.responses is False
