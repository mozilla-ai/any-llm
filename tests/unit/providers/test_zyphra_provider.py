from any_llm import AnyLLM
from any_llm.providers.zyphra.zyphra import ZyphraProvider
from any_llm.types.completion import CompletionParams


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    provider = ZyphraProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "zyphra"
    assert provider.ENV_API_KEY_NAME == "ZYPHRA_API_KEY"
    assert provider.ENV_API_BASE_NAME == "ZYPHRA_API_BASE"
    assert provider.API_BASE == "https://api.zyphracloud.com/api/v1"


def test_zyphra_supports_flags() -> None:
    """Test that ZyphraProvider has correct feature support flags."""
    provider = ZyphraProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_COMPLETION_REASONING is True
    assert provider.SUPPORTS_COMPLETION_IMAGE is False
    assert provider.SUPPORTS_COMPLETION_PDF is False
    assert provider.SUPPORTS_LIST_MODELS is True


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    provider = AnyLLM.create("zyphra", api_key="sk-1")
    assert isinstance(provider, ZyphraProvider)
    assert provider.PROVIDER_NAME == "zyphra"
    assert "zyphra" in AnyLLM.get_supported_providers()


def test_zyphra_convert_completion_params_renames_max_tokens() -> None:
    """max_tokens should be remapped to max_completion_tokens by the base provider."""
    params = CompletionParams(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    result = ZyphraProvider._convert_completion_params(params)
    assert result["max_completion_tokens"] == 100
    assert "max_tokens" not in result


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = ZyphraProvider.get_provider_metadata()
    assert metadata.name == "zyphra"
    assert metadata.env_key == "ZYPHRA_API_KEY"
    assert metadata.env_api_base == "ZYPHRA_API_BASE"
    assert metadata.completion is True


def test_zyphra_item_to_model_maps_all_fields() -> None:
    """Verify the Zyphra /models item shape is mapped to the OpenAI Model contract."""
    item = {
        "modelId": "deepseek-ai/DeepSeek-V3.2",
        "organization": "DeepSeek",
        "releaseDate": "2025-12-01",
        "name": "DeepSeek-V3.2",
    }
    model = ZyphraProvider._zyphra_item_to_model(item)
    assert model.id == "deepseek-ai/DeepSeek-V3.2"
    assert model.object == "model"
    assert model.owned_by == "DeepSeek"
    assert model.created > 0


def test_zyphra_item_to_model_handles_missing_release_date() -> None:
    """Items missing or with malformed releaseDate should still convert with created=0."""
    item = {"modelId": "zyphra/ZAYA1-8B", "organization": "Zyphra"}
    model = ZyphraProvider._zyphra_item_to_model(item)
    assert model.id == "zyphra/ZAYA1-8B"
    assert model.created == 0
    assert model.owned_by == "Zyphra"

    bad_item = {"modelId": "x/y", "organization": "x", "releaseDate": "not-a-date"}
    bad_model = ZyphraProvider._zyphra_item_to_model(bad_item)
    assert bad_model.created == 0


def test_zyphra_item_to_model_defaults_owned_by_when_missing() -> None:
    """Items without an organization key fall back to 'zyphra'."""
    item = {"modelId": "x/y", "releaseDate": "2026-01-01"}
    model = ZyphraProvider._zyphra_item_to_model(item)
    assert model.owned_by == "zyphra"


def test_zyphra_convert_completion_params_drops_reasoning_effort_none() -> None:
    """reasoning_effort='none' should be stripped before sending to Zyphra."""
    params = CompletionParams(
        model_id="deepseek-ai/DeepSeek-V3.2",
        messages=[{"role": "user", "content": "Hi"}],
        reasoning_effort="none",
    )
    result = ZyphraProvider._convert_completion_params(params)
    assert "reasoning_effort" not in result


def test_zyphra_convert_completion_params_preserves_non_none_reasoning_effort() -> None:
    """Other reasoning_effort values should pass through unchanged."""
    params = CompletionParams(
        model_id="moonshotai/Kimi-K2.6",
        messages=[{"role": "user", "content": "Hi"}],
        reasoning_effort="medium",
    )
    result = ZyphraProvider._convert_completion_params(params)
    assert result["reasoning_effort"] == "medium"
