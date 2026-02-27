import pytest

from any_llm import AnyLLM
from any_llm.providers.moonshot.moonshot import MoonshotProvider


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moonshot-test-123")


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    p = MoonshotProvider(api_key="sk-test")
    assert p.PROVIDER_NAME == "moonshot"
    assert p.API_BASE == "https://api.moonshot.ai/v1"
    assert p.SUPPORTS_COMPLETION is True
    assert p.SUPPORTS_COMPLETION_STREAMING is True
    assert p.SUPPORTS_EMBEDDING is False
    assert p.SUPPORTS_COMPLETION_IMAGE is False
    assert p.SUPPORTS_COMPLETION_PDF is False
    assert p.SUPPORTS_COMPLETION_REASONING is True


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    p = AnyLLM.create("moonshot", api_key="sk-1")
    assert isinstance(p, MoonshotProvider)
    assert p.PROVIDER_NAME == "moonshot"

    supported = AnyLLM.get_supported_providers()
    assert "moonshot" in supported


def test_model_provider_split() -> None:
    """Test that model string parsing works correctly."""
    provider_enum, model_name = AnyLLM.split_model_provider("moonshot:moonshot-v1-8k")
    assert provider_enum.value == "moonshot"
    assert model_name == "moonshot-v1-8k"


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = MoonshotProvider.get_provider_metadata()
    assert metadata.name == "moonshot"
    assert metadata.env_key == "MOONSHOT_API_KEY"
    assert metadata.doc_url == "https://platform.moonshot.ai/"
    assert metadata.completion is True
    assert metadata.streaming is True
    assert metadata.embedding is False
    assert metadata.image is False
