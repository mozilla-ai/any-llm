import pytest

from any_llm import AnyLLM
from any_llm.providers.qiniu.qiniu import QiniuProvider


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QINIU_API_KEY", "sk-qiniu-test-123")


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    p = AnyLLM.create("qiniu", api_key="sk-1")
    assert isinstance(p, QiniuProvider)
    assert p.PROVIDER_NAME == "qiniu"

    supported = AnyLLM.get_supported_providers()
    assert "qiniu" in supported
