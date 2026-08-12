"""Integration tests for issue #1258: provider-reported timing must survive normalization.

Groq reports timing in seconds on its usage object, Ollama in nanoseconds on its chat
response. Both land in ``usage.model_extra``. The unit tests hand-build the provider SDK
objects, so only these tests prove the fields are actually on the wire, which matters most
for Groq streaming: groq 1.6.0 takes no ``stream_options``, so streaming usage arrives under
``x_groq.usage`` rather than top-level ``chunk.usage``.

Requires GROQ_API_KEY for the Groq tests and a reachable Ollama host for the Ollama ones.
"""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

_GROQ_TIMING_FIELDS = ("queue_time", "prompt_time", "completion_time", "total_time")
_OLLAMA_TIMING_FIELDS = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")

_PROMPT: list[dict[str, Any] | ChatCompletionMessage] = [{"role": "user", "content": "Reply with the single word OK."}]


@pytest.mark.asyncio
async def test_groq_timing_non_streaming(provider_model_map: dict[LLMProvider, str]) -> None:
    """Groq's per-request timing survives on a non-streaming completion."""
    try:
        llm = AnyLLM.create(LLMProvider.GROQ)
    except MissingApiKeyError:
        pytest.skip("Groq API key not provided, skipping")

    result = await llm.acompletion(model=provider_model_map[LLMProvider.GROQ], messages=_PROMPT)

    assert isinstance(result, ChatCompletion)
    assert result.usage is not None
    extras = result.usage.model_extra or {}
    assert set(_GROQ_TIMING_FIELDS) <= extras.keys(), f"missing Groq timing fields in {extras}"
    assert extras["total_time"] > 0


@pytest.mark.asyncio
async def test_groq_timing_streaming(provider_model_map: dict[LLMProvider, str]) -> None:
    """Groq's streaming usage and timing arrive on the final chunk via x_groq."""
    try:
        llm = AnyLLM.create(LLMProvider.GROQ)
    except MissingApiKeyError:
        pytest.skip("Groq API key not provided, skipping")

    stream = await llm.acompletion(model=provider_model_map[LLMProvider.GROQ], messages=_PROMPT, stream=True)
    assert isinstance(stream, AsyncIterator)

    usages = []
    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionChunk)
        if chunk.usage is not None:
            usages.append(chunk.usage)

    assert usages, "no chunk reported usage: streaming usage is not reaching the converter"
    extras = usages[-1].model_extra or {}
    assert set(_GROQ_TIMING_FIELDS) <= extras.keys(), f"missing Groq timing fields in {extras}"
    assert usages[-1].completion_tokens > 0


@pytest.mark.asyncio
async def test_ollama_timing_non_streaming(provider_model_map: dict[LLMProvider, str]) -> None:
    """Ollama's duration fields survive on a non-streaming completion."""
    llm = AnyLLM.create(LLMProvider.OLLAMA)

    try:
        result = await llm.acompletion(model=provider_model_map[LLMProvider.OLLAMA], messages=_PROMPT)
    # An unreachable host surfaces as a builtin ConnectionError from the ollama SDK on the
    # non-streaming call and as a raw httpx.ConnectError on the streaming one.
    except (ConnectionError, httpx.ConnectError, httpx.HTTPStatusError):
        pytest.skip("Local Ollama host is not set up, skipping")

    assert isinstance(result, ChatCompletion)
    assert result.usage is not None
    extras = result.usage.model_extra or {}
    assert set(_OLLAMA_TIMING_FIELDS) <= extras.keys(), f"missing Ollama timing fields in {extras}"
    assert extras["total_duration"] > 0


@pytest.mark.asyncio
async def test_ollama_timing_streaming(provider_model_map: dict[LLMProvider, str]) -> None:
    """Ollama reports its duration fields on the final streaming chunk."""
    llm = AnyLLM.create(LLMProvider.OLLAMA)

    try:
        stream = await llm.acompletion(model=provider_model_map[LLMProvider.OLLAMA], messages=_PROMPT, stream=True)
        assert isinstance(stream, AsyncIterator)

        usages = []
        async for chunk in stream:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.usage is not None:
                usages.append(chunk.usage)
    except (ConnectionError, httpx.ConnectError, httpx.HTTPStatusError):
        pytest.skip("Local Ollama host is not set up, skipping")

    assert usages, "no chunk reported usage"
    extras = usages[-1].model_extra or {}
    assert set(_OLLAMA_TIMING_FIELDS) <= extras.keys(), f"missing Ollama timing fields in {extras}"
