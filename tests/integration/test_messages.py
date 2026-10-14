import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, NoReturn

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    MessageResponse,
    MessageStreamEvent,
    ParsedMessage,
    ThinkingBlock,
    ToolUseBlock,
)
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_messages_non_streaming(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all providers support the Messages API (non-streaming)."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        model_id = provider_model_map[provider]
        result = await llm.amessages(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            max_tokens=64,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, MessageResponse)
    assert result.role == "assistant"
    assert len(result.content) >= 1


@pytest.mark.asyncio
async def test_messages_streaming(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all providers support the Messages API (streaming)."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming")
        model_id = provider_model_map[provider]
        result = await llm.amessages(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            max_tokens=64,
            stream=True,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise

    assert isinstance(result, AsyncIterator)

    event_types: list[str] = []
    async for event in result:
        assert isinstance(event, MessageStreamEvent)
        event_types.append(event.type)

    assert "message_start" in event_types
    assert "message_stop" in event_types


@pytest.mark.asyncio
async def test_anthropic_messages_skills_container_with_typed_output_format() -> None:
    class City(BaseModel):
        city_name: str

    try:
        provider = AnthropicProvider(max_retries=0, timeout=90)
    except MissingApiKeyError:
        if LLMProvider.ANTHROPIC in EXPECTED_PROVIDERS:
            raise
        pytest.skip("ANTHROPIC_API_KEY is not configured")

    try:
        result = await provider.amessages(
            model=os.environ.get("ANTHROPIC_SKILLS_TEST_MODEL", "claude-sonnet-4-6"),
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=256,
            # Anthropic requires the code execution tool whenever a container loads Skills.
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            container={"skills": [{"type": "anthropic", "skill_id": "xlsx", "version": "latest"}]},
            output_format=City,
        )
    finally:
        # The container itself has no delete endpoint; it expires on its own at container.expires_at.
        await provider.client.close()

    assert isinstance(result, ParsedMessage)
    parsed = result.parsed_output
    assert isinstance(parsed, City)
    assert "paris" in parsed.city_name.lower()


def test_messages_streaming_sync(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that the sync Messages streaming path works for supported providers.

    Mirrors test_messages_streaming, but drives the sync `messages()` wrapper
    instead of `amessages()`. This is the path that regressed in #1253 (the
    response was opened and consumed on two different event loops); see #1260.
    """
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        if not llm.SUPPORTS_COMPLETION_STREAMING:
            pytest.skip(f"{provider.value} does not support streaming")
        model_id = provider_model_map[provider]
        result = llm.messages(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            max_tokens=64,
            stream=True,
        )
        assert isinstance(result, Iterator)

        # The sync bridge opens the connection lazily: request/auth/connection
        # errors surface from the first next() here, not from the messages() call
        # above, so the skip guards must cover this loop too.
        event_types: list[str] = []
        for event in result:
            assert isinstance(event, MessageStreamEvent)
            event_types.append(event.type)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise

    assert "message_start" in event_types
    assert "message_stop" in event_types


@pytest.mark.asyncio
async def test_messages_with_system_prompt(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test Messages API with a system prompt."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        model_id = provider_model_map[provider]
        result = await llm.amessages(
            model=model_id,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=64,
            system="You are a math tutor. Always answer with just the number.",
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, MessageResponse)
    assert len(result.content) >= 1
    assert result.usage.input_tokens >= 0
    assert result.usage.output_tokens >= 0


_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "The city name."}},
        "required": ["location"],
    },
}


@pytest.mark.asyncio
async def test_messages_tool_result_is_error(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """A failed tool result replays through the Messages API without the provider rejecting the request."""
    if provider in (*LOCAL_PROVIDERS, LLMProvider.PERPLEXITY):
        pytest.skip(f"{provider} does not support tools, skipping")

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        model_id = provider_model_map[provider]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."},
        ]
        first = await llm.amessages(model=model_id, messages=messages, max_tokens=1024, tools=[_WEATHER_TOOL])
        assert isinstance(first, MessageResponse)
        tool_use = next((block for block in first.content if isinstance(block, ToolUseBlock)), None)
        assert tool_use is not None, f"Expected a get_weather tool call, got: {first.content}"

        messages.append(
            {"role": "assistant", "content": [block.model_dump(exclude_none=True) for block in first.content]}
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": "weather service unavailable",
                        "is_error": True,
                    }
                ],
            }
        )
        result = await llm.amessages(model=model_id, messages=messages, max_tokens=1024, tools=[_WEATHER_TOOL])
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, MessageResponse)
    assert len(result.content) >= 1


# Long enough to clear the 4096-token minimum Claude Haiku 4.5 needs before it caches a prefix.
_CACHEABLE_SYSTEM_PROMPT = "You are a weather assistant. Answer in one short sentence. " * 500


def _skip_without_bedrock_credentials(exc: Exception) -> NoReturn:
    # boto3 raises these itself (not MissingApiKeyError) when ambient AWS credentials are missing.
    if type(exc).__name__ in {"NoCredentialsError", "PartialCredentialsError", "NoRegionError"}:
        pytest.skip(f"bedrock AWS credentials not available, skipping: {exc}")
    raise exc


@pytest.mark.asyncio
async def test_messages_bedrock_claude_keeps_cache_control_thinking_and_betas(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """A signed thinking block must come back with its tool call, which the Converse bridge could not carry."""
    if provider != LLMProvider.BEDROCK:
        pytest.skip("Targets the native Messages path for Claude models on Bedrock")

    request: dict[str, Any] = {
        "model": provider_reasoning_model_map[provider],
        "max_tokens": 2048,
        "system": [{"type": "text", "text": _CACHEABLE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tools": [_WEATHER_TOOL],
        "betas": ["interleaved-thinking-2025-05-14"],
        "metadata": {"user_id": "any-llm-integration"},
    }
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."},
    ]
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        first = await llm.amessages(messages=messages, **request)
        assert isinstance(first, MessageResponse)
        thinking = next((block for block in first.content if isinstance(block, ThinkingBlock)), None)
        tool_use = next((block for block in first.content if isinstance(block, ToolUseBlock)), None)
        assert thinking is not None, f"Expected a thinking block, got: {first.content}"
        assert thinking.signature
        assert tool_use is not None, f"Expected a get_weather tool call, got: {first.content}"
        assert (first.usage.cache_creation_input_tokens or 0) + (first.usage.cache_read_input_tokens or 0) > 0

        messages.append(
            {"role": "assistant", "content": [block.model_dump(exclude_none=True) for block in first.content]}
        )
        messages.append(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "Sunny, 21C"}]}
        )
        second = await llm.amessages(messages=messages, **request)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except Exception as exc:
        _skip_without_bedrock_credentials(exc)

    assert isinstance(second, MessageResponse)
    assert (second.usage.cache_read_input_tokens or 0) > 0


@pytest.mark.asyncio
async def test_messages_bedrock_claude_streams_signed_thinking(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    if provider != LLMProvider.BEDROCK:
        pytest.skip("Targets the native Messages path for Claude models on Bedrock")

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        stream = await llm.amessages(
            model=provider_reasoning_model_map[provider],
            messages=[{"role": "user", "content": "What is 17 * 23?"}],
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            stream=True,
        )
        assert isinstance(stream, AsyncIterator)
        events = [event async for event in stream]
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except Exception as exc:
        _skip_without_bedrock_credentials(exc)

    assert events[0].type == "message_start"
    assert events[-1].type == "message_stop"
    assert any(
        isinstance(event, ContentBlockDeltaEvent) and event.delta.type == "signature_delta" and event.delta.signature
        for event in events
    )


@pytest.mark.asyncio
async def test_messages_bedrock_claude_takes_automatic_caching_and_drops_unsupported_betas(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """InvokeModel answers a top-level cache_control (Opus 4.6 and earlier) and an unknown beta with a 400."""
    if provider != LLMProvider.BEDROCK:
        pytest.skip("Targets the native Messages path for Claude models on Bedrock")

    request: dict[str, Any] = {
        "model": provider_reasoning_model_map[provider],
        "max_tokens": 64,
        "system": _CACHEABLE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": "What is the weather like in Lisbon in spring?"}],
        "cache_control": {"type": "ephemeral"},
        "betas": ["interleaved-thinking-2025-05-14", "code-execution-2025-08-25"],
        "metadata": {"user_id": "any-llm-integration"},
    }
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        first = await llm.amessages(**request)
        second = await llm.amessages(**request)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except Exception as exc:
        _skip_without_bedrock_credentials(exc)

    assert isinstance(first, MessageResponse)
    assert (first.usage.cache_creation_input_tokens or 0) + (first.usage.cache_read_input_tokens or 0) > 0
    assert isinstance(second, MessageResponse)
    assert (second.usage.cache_read_input_tokens or 0) > 0
