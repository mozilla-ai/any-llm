import pytest

from any_llm import AnyLLM
from any_llm.providers.qiniu.qiniu import QiniuProvider


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QINIU_API_KEY", "sk-qiniu-test-123")


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    p = QiniuProvider(api_key="sk-test")
    assert p.PROVIDER_NAME == "qiniu"
    assert p.API_BASE == "https://api.qnaigc.com/v1"
    assert p.ENV_API_KEY_NAME == "QINIU_API_KEY"
    assert p.ENV_API_BASE_NAME == "QINIU_API_BASE"
    assert p.SUPPORTS_COMPLETION is True
    assert p.SUPPORTS_COMPLETION_STREAMING is True


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    p = AnyLLM.create("qiniu", api_key="sk-1")
    assert isinstance(p, QiniuProvider)
    assert p.PROVIDER_NAME == "qiniu"

    supported = AnyLLM.get_supported_providers()
    assert "qiniu" in supported


def test_model_provider_split() -> None:
    """Test that model string parsing works correctly."""
    provider_enum, model_name = AnyLLM.split_model_provider("qiniu:deepseek-v3")
    assert provider_enum.value == "qiniu"
    assert model_name == "deepseek-v3"


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = QiniuProvider.get_provider_metadata()
    assert metadata.name == "qiniu"
    assert metadata.env_key == "QINIU_API_KEY"
    assert metadata.env_api_base == "QINIU_API_BASE"
    assert metadata.doc_url == "https://www.qiniu.com/"
    assert metadata.completion is True
    assert metadata.streaming is True
