from any_llm import AnyLLM
from any_llm.providers.lmstudio.lmstudio import LmstudioProvider


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    p = LmstudioProvider(api_key="sk-test")
    assert p.PROVIDER_NAME == "lmstudio"
    assert p.API_BASE == "http://localhost:1234/v1"
    assert p.SUPPORTS_COMPLETION is True
    assert p.SUPPORTS_COMPLETION_STREAMING is True
    assert p.SUPPORTS_COMPLETION_REASONING is True
    assert p.SUPPORTS_RESPONSES is True


def test_api_key_not_required() -> None:
    """Test that LM Studio does not require an API key."""
    p = LmstudioProvider()
    assert p.PROVIDER_NAME == "lmstudio"


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    p = AnyLLM.create("lmstudio")
    assert isinstance(p, LmstudioProvider)
    assert p.PROVIDER_NAME == "lmstudio"

    supported = AnyLLM.get_supported_providers()
    assert "lmstudio" in supported


def test_model_provider_split() -> None:
    """Test that model string parsing works correctly."""
    provider_enum, model_name = AnyLLM.split_model_provider("lmstudio:google/gemma-3-4b")
    assert provider_enum.value == "lmstudio"
    assert model_name == "google/gemma-3-4b"


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = LmstudioProvider.get_provider_metadata()
    assert metadata.name == "lmstudio"
    assert metadata.env_key == "LM_STUDIO_API_KEY"
    assert metadata.doc_url == "https://lmstudio.ai/"
    assert metadata.completion is True
    assert metadata.streaming is True
    assert metadata.responses is True
