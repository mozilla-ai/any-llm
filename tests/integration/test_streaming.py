import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.completion import ChatCompletionChunk
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_streaming_completion_async(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that streaming completion works for supported providers."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming completion")
        model_id = provider_model_map[provider]
        output = ""
        reasoning = ""
        num_chunks = 0
        stream = await llm.acompletion(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that exactly follows the user request."},
                {"role": "user", "content": "Say the exact phrase:'Hello World' with no fancy formatting"},
            ],
            stream=True,
        )

        if isinstance(stream, AsyncIterator):
            async for result in stream:
                num_chunks += 1
                assert isinstance(result, ChatCompletionChunk)
                if len(result.choices) > 0:
                    output += result.choices[0].delta.content or ""
                    if result.choices[0].delta.reasoning:
                        reasoning += result.choices[0].delta.reasoning.content or ""
            assert num_chunks >= 1, f"Expected at least 1 chunk, got {num_chunks}"
            assert "hello world" in output.lower()
        else:
            msg = f"Expected AsyncIterator[ChatCompletionChunk], not {type(stream)}"
            raise TypeError(msg)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"Streaming is not supported for {provider.value}")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise


def test_streaming_completion_sync(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that the sync streaming completion path works for supported providers.

    Mirrors test_streaming_completion_async, but drives the sync `completion()`
    wrapper instead of `acompletion()`. The sync streaming bridge (opening the
    response and consuming it on the same worker event loop) has no dedicated
    integration coverage otherwise; see #1253/#1260.
    """
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming completion")
        model_id = provider_model_map[provider]
        output = ""
        reasoning = ""
        num_chunks = 0
        stream = llm.completion(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that exactly follows the user request."},
                {"role": "user", "content": "Say the exact phrase:'Hello World' with no fancy formatting"},
            ],
            stream=True,
        )

        for result in stream:
            num_chunks += 1
            assert isinstance(result, ChatCompletionChunk)
            if len(result.choices) > 0:
                output += result.choices[0].delta.content or ""
                if result.choices[0].delta.reasoning:
                    reasoning += result.choices[0].delta.reasoning.content or ""
        assert num_chunks >= 1, f"Expected at least 1 chunk, got {num_chunks}"
        assert "hello world" in output.lower()
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"Streaming is not supported for {provider.value}")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise


@pytest.mark.asyncio
async def test_streaming_tool_call_keeps_one_id(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    """A streamed tool call's deltas share one id and name its function once.

    Clients join a call's deltas by id, so a delta that carries a different id
    or an empty name splits one call into several broken ones.
    """
    if provider in (*LOCAL_PROVIDERS, LLMProvider.PERPLEXITY):
        pytest.skip(f"{provider} does not support tools, skipping")

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming completion")
        stream = await llm.acompletion(
            model=provider_model_map[provider],
            messages=[{"role": "user", "content": "What is the weather in Paris, France?"}],
            tools=tools,
            stream=True,
        )
        assert isinstance(stream, AsyncIterator)

        ids: dict[int, set[str]] = {}
        names: dict[int, list[str]] = {}
        arguments: dict[int, str] = {}
        async for chunk in stream:
            for choice in chunk.choices:
                for call in choice.delta.tool_calls or []:
                    if call.id is not None:
                        ids.setdefault(call.index, set()).add(call.id)
                    if call.function is not None:
                        if call.function.name is not None:
                            names.setdefault(call.index, []).append(call.function.name)
                        arguments[call.index] = arguments.get(call.index, "") + (call.function.arguments or "")

        assert arguments, "Expected the model to call get_weather"
        for index, args in arguments.items():
            assert len(ids.get(index, set())) == 1, f"Tool call {index} carried ids {ids.get(index)}"
            assert names.get(index) == ["get_weather"], f"Tool call {index} carried names {names.get(index)}"
            assert "location" in json.loads(args)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"Streaming tool calls are not supported for {provider.value}")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
