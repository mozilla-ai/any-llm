from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from any_llm import AnyLLM
from any_llm.exceptions import AuthenticationError
from any_llm.providers.openai import OpenaiProvider
from any_llm.providers.openai.custom import OpenAICompatibleProvider
from any_llm.types.completion import ChatCompletionChunk


@pytest.mark.parametrize("enabled", [True, False, None])
@pytest.mark.parametrize("factory", ["create", "constructor", "compatible"])
def test_instance_option_is_not_forwarded_to_sdk(enabled: bool | None, factory: str) -> None:
    provider_class = OpenAICompatibleProvider if factory == "compatible" else OpenaiProvider
    with patch.object(provider_class, "_init_client") as init_client:
        if factory == "create":
            provider = AnyLLM.create("openai", api_key="test", unified_exceptions=enabled, timeout=3)
        elif factory == "constructor":
            provider = OpenaiProvider(api_key="test", unified_exceptions=enabled, timeout=3)
        else:
            provider = AnyLLM.create_openai_compatible(
                "gateway", "https://example.com/v1", api_key="test", unified_exceptions=enabled, timeout=3
            )

    assert provider._unified_exceptions is enabled
    assert "unified_exceptions" not in init_client.call_args.kwargs
    assert init_client.call_args.kwargs["timeout"] == 3


@pytest.mark.parametrize("env_value", ["", "1"])
@pytest.mark.parametrize("enabled", [True, False, None])
@pytest.mark.parametrize("operation", ["completion", "embedding", "stream"])
@pytest.mark.asyncio
async def test_instance_exception_precedence(
    monkeypatch: pytest.MonkeyPatch, env_value: str, enabled: bool | None, operation: str
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", env_value)
    with patch.object(OpenaiProvider, "_init_client"):
        provider = AnyLLM.create("openai", api_key="test", unified_exceptions=enabled)
    original = RuntimeError("Invalid API key")
    chunk = ChatCompletionChunk(id="test", choices=[], created=0, model="test", object="chat.completion.chunk")

    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield chunk
        raise original

    async def invoke() -> None:
        if operation == "embedding":
            monkeypatch.setattr(provider, "_aembedding", AsyncMock(side_effect=original))
            await provider.aembedding("test", "hello")
        elif operation == "completion":
            monkeypatch.setattr(provider, "_acompletion", AsyncMock(side_effect=original))
            await provider.acompletion(model="test", messages=[{"role": "user", "content": "Hello"}])
        else:
            monkeypatch.setattr(provider, "_acompletion", AsyncMock(return_value=stream()))
            result = await provider.acompletion(
                model="test", messages=[{"role": "user", "content": "Hello"}], stream=True
            )
            assert await anext(result) is chunk
            await anext(result)

    expected_enabled = enabled if enabled is not None else env_value == "1"
    if expected_enabled:
        with pytest.raises(AuthenticationError) as error:
            await invoke()
        assert error.value.original_exception is original
        assert error.value.__cause__ is original
        assert error.value.provider_name == "openai"
    else:
        with (
            pytest.warns(DeprecationWarning, match="Provider-specific exceptions"),
            pytest.raises(RuntimeError) as raw_error,
        ):
            await invoke()
        assert raw_error.value is original


def test_clients_keep_independent_settings_for_sync_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANY_LLM_UNIFIED_EXCEPTIONS", raising=False)
    with patch.object(OpenaiProvider, "_init_client"):
        unified = AnyLLM.create("openai", api_key="test", unified_exceptions=True)
        original_errors = AnyLLM.create("openai", api_key="test", unified_exceptions=False)
    original = RuntimeError("Invalid API key")
    monkeypatch.setattr(unified, "_acompletion", AsyncMock(side_effect=original))
    monkeypatch.setattr(original_errors, "_acompletion", AsyncMock(side_effect=original))

    for provider in [unified, original_errors, unified]:
        if provider is unified:
            with pytest.raises(AuthenticationError):
                provider.completion(model="test", messages=[{"role": "user", "content": "Hello"}])
        else:
            with (
                pytest.warns(DeprecationWarning, match="Provider-specific exceptions"),
                pytest.raises(RuntimeError) as error,
            ):
                provider.completion(model="test", messages=[{"role": "user", "content": "Hello"}])
            assert error.value is original


@pytest.mark.asyncio
async def test_omitted_option_keeps_dynamic_environment_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANY_LLM_UNIFIED_EXCEPTIONS", raising=False)
    with patch.object(OpenaiProvider, "_init_client"):
        provider = AnyLLM.create("openai", api_key="test")
    monkeypatch.setattr(provider, "_aembedding", AsyncMock(side_effect=RuntimeError("Invalid API key")))
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    with pytest.raises(AuthenticationError):
        await provider.aembedding("test", "hello")


@pytest.mark.asyncio
async def test_successful_response_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    with patch.object(OpenaiProvider, "_init_client"):
        provider = AnyLLM.create("openai", api_key="test", unified_exceptions=True)
    response: Any = object()
    monkeypatch.setattr(provider, "_acompletion", AsyncMock(return_value=response))

    assert await provider.acompletion(model="test", messages=[{"role": "user", "content": "Hello"}]) is response
