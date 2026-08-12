import sys
from collections.abc import AsyncIterator
from inspect import signature
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.api import acompletion, aresponses, completion, responses
from any_llm.constants import LLMProvider
from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.completion import ChatCompletion, CompletionParams

_PYTHON_314_INCOMPATIBLE_PROVIDERS = {"voyage", "watsonx"}


def test_completion_params_exposes_prompt_cache_key() -> None:
    params = CompletionParams(
        model_id="gpt-5.6",
        messages=[{"role": "user", "content": "Hello"}],
        prompt_cache_key="tenant-1",
    )

    assert params.prompt_cache_key == "tenant-1"
    assert "prompt_cache_key" in CompletionParams.model_json_schema()["properties"]
    assert "prompt_cache_key" in signature(completion).parameters
    assert "prompt_cache_key" in signature(acompletion).parameters
    assert "prompt_cache_key" in signature(AnyLLM.completion).parameters
    assert "prompt_cache_key" in signature(AnyLLM.acompletion).parameters

    mock_provider = Mock()
    with patch("any_llm.any_llm.AnyLLM.create", return_value=mock_provider):
        completion(
            model="openai:gpt-5.6",
            messages=[{"role": "user", "content": "Hello"}],
            prompt_cache_key="tenant-1",
        )

    assert mock_provider.completion.call_args.kwargs["prompt_cache_key"] == "tenant-1"


def test_completion_exposes_timeout_parameter() -> None:
    assert "timeout" in signature(completion).parameters
    assert "timeout" in signature(acompletion).parameters
    assert "timeout" in signature(AnyLLM.completion).parameters
    assert "timeout" in signature(AnyLLM.acompletion).parameters


@pytest.mark.asyncio
async def test_acompletion_forwards_timeout_to_provider() -> None:
    # Through the public acompletion() helper: a set timeout reaches the provider, None is dropped.
    provider = AnyLLM.create("openai", api_key="sk-test")
    with (
        patch("any_llm.any_llm.AnyLLM.create", return_value=provider),
        patch.object(provider, "_acompletion", new=AsyncMock(return_value=Mock(spec=ChatCompletion))) as mock_ac,
    ):
        await acompletion(model="openai:gpt-4", messages=[{"role": "user", "content": "Hi"}], timeout=600)
        assert mock_ac.call_args.kwargs.get("timeout") == 600

        mock_ac.reset_mock()
        await acompletion(model="openai:gpt-4", messages=[{"role": "user", "content": "Hi"}], timeout=None)
        assert "timeout" not in mock_ac.call_args.kwargs


def test_completion_forwards_timeout_to_provider_sync() -> None:
    # Through the public completion() helper on the synchronous non-streaming path.
    provider = AnyLLM.create("openai", api_key="sk-test")
    with (
        patch("any_llm.any_llm.AnyLLM.create", return_value=provider),
        patch.object(provider, "_acompletion", new=AsyncMock(return_value=Mock(spec=ChatCompletion))) as mock_ac,
    ):
        completion(model="openai:gpt-4", messages=[{"role": "user", "content": "Hi"}], timeout=600, stream=False)

    assert mock_ac.call_args.kwargs.get("timeout") == 600


def test_completion_forwards_timeout_on_streaming_path() -> None:
    # The synchronous streaming path forwards timeout too; consume the iterator to drive the call.
    provider = AnyLLM.create("openai", api_key="sk-test")

    async def _empty_stream() -> AsyncIterator[Any]:
        if False:
            yield None

    with (
        patch("any_llm.any_llm.AnyLLM.create", return_value=provider),
        patch.object(provider, "_acompletion", new=AsyncMock(return_value=_empty_stream())) as mock_ac,
    ):
        stream = completion(
            model="openai:gpt-4", messages=[{"role": "user", "content": "Hi"}], timeout=600, stream=True
        )
        list(stream)

    assert mock_ac.call_args.kwargs.get("timeout") == 600


@pytest.mark.asyncio
async def test_acompletion_rejects_prompt_cache_key_for_unsupported_provider() -> None:
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    client = Mock()
    provider = BedrockProvider(client=client)

    with pytest.raises(UnsupportedParameterError, match="prompt_cache_key"):
        await provider.acompletion(
            model="anthropic.claude-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            prompt_cache_key="tenant-1",
        )

    client.converse.assert_not_called()


@pytest.mark.asyncio
async def test_acompletion_parameter_capture() -> None:
    """Test that acompletion correctly captures and passes all parameters."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock(return_value=Mock())

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await acompletion(
            model="openai:gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
            reasoning_effort="high",
            prompt_cache_key="tenant-1",
            api_key="sk-test-key-123",
            api_base="https://custom-openai.example.com/v1",
            custom_param="custom_value",
        )

        mock_create.assert_called_once_with(
            LLMProvider.OPENAI,
            api_key="sk-test-key-123",
            api_base="https://custom-openai.example.com/v1",
        )

        mock_provider.acompletion.assert_called_once()
        call_args = mock_provider.acompletion.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args.kwargs["tools"] == [{"type": "function", "function": {"name": "test"}}]
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_tokens"] == 100
        assert call_args.kwargs["stream"] is False
        assert call_args.kwargs["reasoning_effort"] == "high"
        assert call_args.kwargs["prompt_cache_key"] == "tenant-1"
        assert call_args.kwargs["custom_param"] == "custom_value"


@pytest.mark.asyncio
async def test_aresponses_parameter_capture() -> None:
    """Test that aresponses correctly captures and passes all parameters."""
    mock_provider = Mock()
    mock_provider.aresponses = AsyncMock(return_value=Mock())

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await aresponses(
            model="mistral:mistral-large",
            input_data="test input",
            tools=[{"type": "function", "function": {"name": "search"}}],
            temperature=0.5,
            stream=True,
            instructions="Be helpful",
            parallel_tool_calls=True,
            context_management=[{"type": "compaction", "compact_threshold": 200_000}],
            api_key="mistral-key-456",
            api_base="https://custom-mistral.example.com/v1",
            another_custom_param="another_value",
        )

        mock_create.assert_called_once_with(
            LLMProvider.MISTRAL,
            api_key="mistral-key-456",
            api_base="https://custom-mistral.example.com/v1",
        )

        mock_provider.aresponses.assert_called_once()
        call_args = mock_provider.aresponses.call_args
        assert call_args.kwargs["model"] == "mistral-large"
        assert call_args.kwargs["input_data"] == "test input"
        assert call_args.kwargs["tools"] == [{"type": "function", "function": {"name": "search"}}]
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["stream"] is True
        assert call_args.kwargs["instructions"] == "Be helpful"
        assert call_args.kwargs["parallel_tool_calls"] is True
        assert call_args.kwargs["context_management"] == [{"type": "compaction", "compact_threshold": 200_000}]
        assert call_args.kwargs["another_custom_param"] == "another_value"


def test_responses_context_management_parameter_capture() -> None:
    """Test that the synchronous responses wrapper forwards context_management."""
    mock_provider = Mock()
    mock_provider.responses = Mock(return_value=Mock())

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        context_management = [{"type": "compaction", "compact_threshold": 200_000}]

        responses(
            model="openai:gpt-5.3-codex",
            input_data="Continue the coding task.",
            context_management=context_management,
            api_key="openai-key",
        )

        mock_create.assert_called_once_with(
            LLMProvider.OPENAI,
            api_key="openai-key",
            api_base=None,
        )
        assert mock_provider.responses.call_args.kwargs["context_management"] == context_management


@pytest.mark.asyncio
async def test_completion_invalid_model_format_no_slash() -> None:
    """Test completion raises ValueError for model without separator."""
    with pytest.raises(
        ValueError, match=r"Invalid model format. Expected 'provider:model' or 'provider/model', got 'gpt-4'"
    ):
        await acompletion("gpt-4", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_empty_provider() -> None:
    """Test completion raises ValueError for model with empty provider."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await acompletion("/model", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_empty_model() -> None:
    """Test completion raises ValueError for model with empty model name."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await acompletion("provider/", messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_completion_invalid_model_format_multiple_slashes() -> None:
    """Test completion handles multiple slashes correctly (should work - takes first split)."""
    mock_provider = Mock()
    mock_provider.acompletion = AsyncMock()

    with (
        patch("any_llm.any_llm.AnyLLM.split_model_provider") as mock_split,
        patch("any_llm.any_llm.AnyLLM.create") as mock_create,
    ):
        mock_split.return_value = (LLMProvider.OPENAI, "model/extra")
        mock_create.return_value = mock_provider

        await acompletion("openai/model/extra", messages=[{"role": "user", "content": "Hello"}])

        mock_provider.acompletion.assert_called_once()
        _, kwargs = mock_provider.acompletion.call_args
        assert kwargs["model"] == "model/extra"


@pytest.mark.asyncio
async def test_all_providers_can_be_loaded(provider: str) -> None:
    """Test that all supported providers can be loaded successfully.

    This test uses the provider fixture which iterates over all providers
    returned by AnyLLM.get_supported_providers(). It verifies that:
    1. Each provider can be imported and instantiated
    2. The created instance is actually a AnyLLM
    3. No ImportError or other exceptions are raised during loading

    This test will automatically include new providers when they're added
    without requiring any code changes.
    """
    kwargs: dict[str, Any] = {"api_key": "test_key"}
    if provider in ("azure", "azureopenai"):
        kwargs["api_base"] = "test_api_base"
    if provider == "bedrock":
        kwargs["region_name"] = "us-east-1"
    if provider == "sagemaker":
        pytest.skip("sagemaker requires AWS credentials on instantiation")
    if sys.version_info >= (3, 14) and provider in _PYTHON_314_INCOMPATIBLE_PROVIDERS:
        pytest.skip(f"{provider} is not compatible with Python 3.14+")
    if provider == "vertexai":
        kwargs["project"] = "test-project"
        kwargs["location"] = "test-location"
    if provider == "otari":
        kwargs["api_base"] = "http://127.0.0.1:8080/v1"

    provider_instance = AnyLLM.create(provider, **kwargs)

    assert isinstance(provider_instance, AnyLLM), f"Provider {provider} did not create a valid AnyLLM instance"

    assert hasattr(provider_instance, "acompletion"), f"Provider {provider} does not have an acompletion method"
    assert callable(provider_instance._acompletion), f"Provider {provider} acompletion is not callable"


@pytest.mark.asyncio
async def test_all_providers_can_be_loaded_with_config(provider: str) -> None:
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
    if sys.version_info >= (3, 14) and provider in _PYTHON_314_INCOMPATIBLE_PROVIDERS:
        pytest.skip(f"{provider} is not compatible with Python 3.14+")

    provider_instance = AnyLLM.create(provider, **kwargs)

    assert isinstance(provider_instance, AnyLLM), (
        f"Provider {provider} did not create a valid AnyLLM instance with config"
    )


@pytest.mark.asyncio
async def test_provider_factory_can_create_all_supported_providers() -> None:
    """Test that AnyLLM can create instances of all providers it claims to support."""
    supported_providers = AnyLLM.get_supported_providers()

    for provider_name in supported_providers:
        kwargs: dict[str, Any] = {"api_key": "test_key"}
        if provider_name in ("azure", "azureopenai"):
            kwargs["api_base"] = "test_api_base"
        if provider_name == "azureanthropic":
            kwargs["resource"] = "test-resource"
        if provider_name == "bedrock":
            kwargs["region_name"] = "us-east-1"
        if provider_name == "vertexai":
            kwargs["project"] = "test-project"
            kwargs["location"] = "test-location"
        if provider_name == "vertexaianthropic":
            kwargs["project_id"] = "test-project"
        if provider_name == "sagemaker":
            continue
        if sys.version_info >= (3, 14) and provider_name in _PYTHON_314_INCOMPATIBLE_PROVIDERS:
            continue
        if provider_name == "otari":
            kwargs["api_base"] = "http://127.0.0.1:8080/v1"
        provider_instance = AnyLLM.create(provider_name, **kwargs)

        assert isinstance(provider_instance, AnyLLM), f"Failed to create valid AnyLLM instance for {provider_name}"
