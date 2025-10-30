import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError, UnsupportedProviderError


def test_all_providers_in_enum() -> None:
    """Test that all provider directories are accounted for in the ProviderName enum."""
    providers_dir = Path(__file__).parent.parent.parent / "src" / "any_llm" / "providers"

    provider_dirs = []
    for item in providers_dir.iterdir():
        if item.is_dir() and item.name not in ("__pycache__", "google"):
            provider_dirs.append(item.name)

    enum_values = [provider.value for provider in LLMProvider]

    provider_dirs.sort()
    enum_values.sort()

    missing_from_enum = set(provider_dirs) - set(enum_values)
    missing_from_dirs = set(enum_values) - set(provider_dirs)

    assert not missing_from_enum, f"Provider directories missing from ProviderName enum: {missing_from_enum}"
    assert not missing_from_dirs, f"ProviderName enum values missing provider directories: {missing_from_dirs}"

    assert provider_dirs == enum_values, f"Provider directories {provider_dirs} don't match enum values {enum_values}"


def test_provider_enum_values_match_directory_names() -> None:
    """Test that enum values exactly match the provider directory names."""
    providers_dir = Path(__file__).parent.parent.parent / "src" / "any_llm" / "providers"

    actual_providers = set()
    for item in providers_dir.iterdir():
        if item.is_dir() and item.name not in ("__pycache__", "google"):
            actual_providers.add(item.name)

    enum_providers = {provider.value for provider in LLMProvider}

    assert actual_providers == enum_providers, (
        f"Provider directories and enum values don't match!\n"
        f"In directories but not enum: {actual_providers - enum_providers}\n"
        f"In enum but not directories: {enum_providers - actual_providers}"
    )


def test_provider_model_split() -> None:
    """Test that model strings are split correctly into provider and model name."""
    model_str = "ollama:model:tag"
    provider, model_name = AnyLLM.split_model_provider(model_str)
    assert provider == LLMProvider.OLLAMA
    assert model_name == "model:tag"

    model_str = "ollama/model:tag"
    provider, model_name = AnyLLM.split_model_provider(model_str)
    assert provider == LLMProvider.OLLAMA
    assert model_name == "model:tag"

    model_str = "ollama:models/model-tag"
    provider, model_name = AnyLLM.split_model_provider(model_str)
    assert provider == LLMProvider.OLLAMA
    assert model_name == "models/model-tag"

    model_str = "ollama/models/model-tag"
    provider, model_name = AnyLLM.split_model_provider(model_str)
    assert provider == LLMProvider.OLLAMA
    assert model_name == "models/model-tag"  # legacy format


def test_get_provider_enum_valid_provider() -> None:
    """Test get_provider_enum returns correct enum for valid provider."""
    provider_enum = AnyLLM.get_provider_enum("openai")
    assert provider_enum == LLMProvider.OPENAI


def test_get_provider_enum_invalid_provider() -> None:
    """Test get_provider_enum raises UnsupportedProviderError for invalid provider."""
    with pytest.raises(UnsupportedProviderError) as exc_info:
        AnyLLM.get_provider_enum("invalid_provider")

    exception = exc_info.value
    assert exception.provider_key == "invalid_provider"
    assert isinstance(exception.supported_providers, list)
    assert len(exception.supported_providers) > 0
    assert "openai" in exception.supported_providers


def test_unsupported_provider_error_message() -> None:
    """Test UnsupportedProviderError has correct message format."""
    with pytest.raises(UnsupportedProviderError, match="'invalid_provider' is not a supported provider"):
        AnyLLM.get_provider_enum("invalid_provider")


def test_unsupported_provider_error_attributes() -> None:
    """Test UnsupportedProviderError has correct attributes."""
    with pytest.raises(UnsupportedProviderError) as exc_info:
        AnyLLM.get_provider_enum("nonexistent")

    e = exc_info.value
    assert e.provider_key == "nonexistent"
    assert e.supported_providers == AnyLLM.get_supported_providers()
    assert "Supported providers:" in str(e)


@pytest.mark.asyncio
async def test_all_providers_have_required_attributes(provider: LLMProvider) -> None:
    """Test that all supported providers can be loaded with sample config parameters.

    This test verifies that providers can handle common configuration parameters
    like api_key and api_base without throwing errors during instantiation.
    """
    kwargs: dict[str, Any] = {"api_key": "test_key", "api_base": "https://test.example.com"}
    if provider == "bedrock":
        kwargs["region_name"] = "us-east-1"
    if provider == "vertexai":
        kwargs["project"] = "test-project"
        kwargs["location"] = "test-location"
    if provider == "sagemaker":
        pytest.skip("sagemaker requires AWS credentials on instantiation")

    provider_instance = AnyLLM.create(provider.value, **kwargs)

    assert provider_instance.PROVIDER_NAME is not None
    assert provider_instance.PROVIDER_DOCUMENTATION_URL is not None
    assert provider_instance.MISSING_PACKAGES_ERROR is None
    assert provider_instance.SUPPORTS_COMPLETION is not None
    assert provider_instance.SUPPORTS_COMPLETION_STREAMING is not None
    assert provider_instance.SUPPORTS_COMPLETION_REASONING is not None
    assert provider_instance.SUPPORTS_EMBEDDING is not None
    assert provider_instance.SUPPORTS_RESPONSES is not None


def test_providers_raise_MissingApiKeyError(provider: LLMProvider) -> None:
    if provider in (
        LLMProvider.BEDROCK,
        LLMProvider.LLAMACPP,
        LLMProvider.LLAMAFILE,
        LLMProvider.LMSTUDIO,
        LLMProvider.OLLAMA,
        LLMProvider.SAGEMAKER,
        LLMProvider.VERTEXAI,
        LLMProvider.GATEWAY,
    ):
        pytest.skip("This provider handles `api_key` differently.")
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(MissingApiKeyError):
            AnyLLM.create(provider.value)


@pytest.mark.parametrize(
    ("provider_name", "module_name"),
    [
        ("anthropic", "anthropic"),
        ("bedrock", "boto3"),
        ("azure", "azure"),
        ("cerebras", "cerebras"),
        ("cohere", "cohere"),
        ("gemini", "google"),
        ("groq", "groq"),
        ("huggingface", "huggingface_hub"),
        ("mistral", "mistralai"),
        ("ollama", "ollama"),
        ("together", "together"),
        ("voyage", "voyageai"),
        ("watsonx", "ibm_watsonx_ai"),
        ("xai", "xai_sdk"),
    ],
)
def test_providers_raise_ImportError_from_original(provider_name: str, module_name: str) -> None:
    with patch.dict(sys.modules, {module_name: None}):
        for mod in list(sys.modules):
            if mod.startswith((f"any_llm.providers.{provider_name}", f"{module_name}.")):
                sys.modules.pop(mod)
        with pytest.raises(ImportError) as e:
            AnyLLM.create(provider_name, api_key="test_key")
        original_error = e.value.__cause__
        assert any(
            msg in str(original_error)
            for msg in [f"import of {module_name} halted", f"'{module_name}' is not a package"]
        )
