"""Integration test for cached-token reporting through the Messages-to-Completions bridge.

Providers reached via the bridge report caching the OpenAI way: ``prompt_tokens`` is the
whole prompt and ``prompt_tokens_details.cached_tokens`` is a subset of it. Anthropic's
Messages usage instead keeps ``input_tokens`` and ``cache_read_input_tokens`` disjoint, so
the bridge has to subtract. These tests exercise that against real automatic prefix caching.

Requires OPENAI_API_KEY to be set.
"""

import os
from collections.abc import AsyncIterator

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.types.messages import MessageDeltaEvent, MessageResponse

# OpenAI only caches prompt prefixes above 1,024 tokens, so pad well past that.
_LONG_PREFIX = (
    "You are a helpful assistant that answers questions concisely. "
    "The following is reference material that you should use to answer questions.\n\n"
) + (
    "Automatic prefix caching lets a provider reuse an already-processed prompt prefix "
    "across requests, so the cached portion is billed at a reduced rate rather than being "
    "re-processed from scratch. It engages without any explicit cache-control marking.\n"
) * 200

_PROMPT = _LONG_PREFIX + "\n\nReply with the single word OK."


@pytest.fixture
def openai_model(provider_model_map: dict[LLMProvider, str]) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    return provider_model_map[LLMProvider.OPENAI]


@pytest.mark.asyncio
async def test_messages_bridge_cached_tokens_non_streaming(openai_model: str) -> None:
    """A cache hit is reported as cache_read_input_tokens, subtracted out of input_tokens."""
    llm = AnyLLM.create(LLMProvider.OPENAI)
    messages = [{"role": "user", "content": _PROMPT}]

    # The first call populates the cache; the prefix is only eligible for a hit afterwards.
    first = await llm.amessages(model=openai_model, messages=messages, max_tokens=2000)
    assert isinstance(first, MessageResponse)
    prompt_total = first.usage.input_tokens + (first.usage.cache_read_input_tokens or 0)

    result = await llm.amessages(model=openai_model, messages=messages, max_tokens=2000)
    assert isinstance(result, MessageResponse)

    if not result.usage.cache_read_input_tokens:
        pytest.skip("provider served no cached tokens: automatic prefix caching did not engage for this prompt")

    # The two fields are disjoint, so they must still sum to the same prompt total the
    # uncached call reported. Copying cached_tokens across without subtracting inflates this sum.
    assert result.usage.input_tokens + result.usage.cache_read_input_tokens == prompt_total
    # Most of this prompt is the cached prefix, so the fresh remainder must be the smaller of
    # the two. Reporting the whole prompt as input_tokens alongside the cache count fails here.
    assert result.usage.input_tokens < result.usage.cache_read_input_tokens
    # Automatic caching has no write step to report, so this stays unset.
    assert result.usage.cache_creation_input_tokens is None


@pytest.mark.asyncio
async def test_messages_bridge_cached_tokens_streaming(openai_model: str) -> None:
    """A streamed call reports the same disjoint totals as a non-streamed one."""
    llm = AnyLLM.create(LLMProvider.OPENAI)
    messages = [{"role": "user", "content": _PROMPT}]

    non_streamed = await llm.amessages(model=openai_model, messages=messages, max_tokens=2000)
    assert isinstance(non_streamed, MessageResponse)
    prompt_total = non_streamed.usage.input_tokens + (non_streamed.usage.cache_read_input_tokens or 0)

    stream = await llm.amessages(model=openai_model, messages=messages, max_tokens=2000, stream=True)
    assert isinstance(stream, AsyncIterator)

    delta: MessageDeltaEvent | None = None
    async for event in stream:
        if isinstance(event, MessageDeltaEvent):
            delta = event

    assert delta is not None, "stream should end with a message_delta carrying usage"
    if not delta.usage.cache_read_input_tokens:
        pytest.skip("provider served no cached tokens: automatic prefix caching did not engage for this prompt")

    assert delta.usage.input_tokens is not None
    assert delta.usage.input_tokens + delta.usage.cache_read_input_tokens == prompt_total
    # Most of this prompt is the cached prefix, so the fresh remainder must be the smaller of
    # the two. Reporting the whole prompt as input_tokens alongside the cache count fails here.
    assert delta.usage.input_tokens < delta.usage.cache_read_input_tokens
    assert delta.usage.cache_creation_input_tokens is None
