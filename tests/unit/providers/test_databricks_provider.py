import pytest

from any_llm import AnyLLM
from any_llm.providers.databricks.databricks import DatabricksProvider


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test-123")
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.com")


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    p = DatabricksProvider(api_key="dapi-test", api_base="https://test.databricks.com")
    assert p.PROVIDER_NAME == "databricks"
    assert p.ENV_API_KEY_NAME == "DATABRICKS_TOKEN"
    assert p.ENV_API_BASE_NAME == "DATABRICKS_HOST"
    assert p.SUPPORTS_COMPLETION is True
    assert p.SUPPORTS_COMPLETION_STREAMING is True
    assert p.SUPPORTS_COMPLETION_REASONING is True
    assert p.SUPPORTS_COMPLETION_IMAGE is False
    assert p.SUPPORTS_COMPLETION_PDF is False
    assert p.SUPPORTS_LIST_MODELS is False


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    p = AnyLLM.create("databricks", api_key="dapi-1", api_base="https://test.databricks.com")
    assert isinstance(p, DatabricksProvider)
    assert p.PROVIDER_NAME == "databricks"

    supported = AnyLLM.get_supported_providers()
    assert "databricks" in supported


def test_model_provider_split() -> None:
    """Test that model string parsing works correctly."""
    provider_enum, model_name = AnyLLM.split_model_provider("databricks:databricks-meta-llama-3-70b-instruct")
    assert provider_enum.value == "databricks"
    assert model_name == "databricks-meta-llama-3-70b-instruct"


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = DatabricksProvider.get_provider_metadata()
    assert metadata.name == "databricks"
    assert metadata.env_key == "DATABRICKS_TOKEN"
    assert metadata.doc_url == "https://docs.databricks.com/"
    assert metadata.completion is True
    assert metadata.streaming is True
    assert metadata.list_models is False
    assert metadata.image is False
