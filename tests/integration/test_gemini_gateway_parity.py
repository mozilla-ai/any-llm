import uuid
from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage
from any_llm.types.messages import MessageResponse
from tests.constants import EXPECTED_PROVIDERS

MODEL = "gemini-3-flash-preview"
CODE_PROMPT = "Use code execution to compute the sum of the first 50 prime numbers. Reply with the number."
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def _provider() -> AnyLLM:
    try:
        return AnyLLM.create("gemini")
    except MissingApiKeyError:
        if "gemini" in EXPECTED_PROVIDERS:
            raise
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")


@pytest.mark.asyncio
async def test_gemini_code_execution_maps_into_completion_extra_content() -> None:
    provider = _provider()

    result = await provider.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": CODE_PROMPT}],
        tools=[{"code_execution": {}}],
    )

    assert isinstance(result, ChatCompletion)
    extra_content = result.choices[0].message.extra_content or {}
    items = extra_content["google"]["code_execution"]
    assert {item["type"] for item in items} >= {"executable_code", "code_execution_result"}


@pytest.mark.asyncio
async def test_gemini_code_execution_maps_into_messages_server_tool_blocks() -> None:
    provider = _provider()

    result = await provider.amessages(
        model=MODEL,
        messages=[{"role": "user", "content": CODE_PROMPT}],
        tools=[{"code_execution": {}}],
        max_tokens=2048,
    )

    assert isinstance(result, MessageResponse)
    types = [block.type for block in result.content]
    assert "server_tool_use" in types
    assert "code_execution_tool_result" in types


@pytest.mark.asyncio
async def test_gemini_messages_thought_signature_survives_a_tool_round_trip() -> None:
    provider = _provider()
    messages: list[dict[str, Any]] = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]

    first = await provider.amessages(model=MODEL, messages=messages, tools=[WEATHER_TOOL], max_tokens=1024)

    assert isinstance(first, MessageResponse)
    tool_uses = [block for block in first.content if block.type == "tool_use"]
    assert tool_uses
    assert (tool_uses[0].model_extra or {}).get("extra_content", {}).get("google", {}).get("thought_signature")
    messages.append({"role": "assistant", "content": [block.model_dump(exclude_none=True) for block in first.content]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": block.id, "content": "Sunny, 21C"} for block in tool_uses
            ],
        }
    )

    second = await provider.amessages(model=MODEL, messages=messages, tools=[WEATHER_TOOL], max_tokens=1024)

    assert isinstance(second, MessageResponse)
    assert any(block.type == "text" for block in second.content)


def _manual() -> str:
    # Gemini refuses caches under 1,024 tokens on this model, so the manual is padded well past that. The
    # run id keeps a cache left by an earlier run from answering for this one.
    rules = " ".join(f"Rule {i}: the answer to question {i} is {i * 7}." for i in range(1500))
    return f"Run {uuid.uuid4().hex}. Answer from this manual in one short sentence.\n{rules}"


@pytest.mark.asyncio
async def test_gemini_cache_control_on_completion_writes_then_reads_a_context_cache() -> None:
    provider = _provider()
    messages: list[dict[str, Any] | ChatCompletionMessage] = [
        {"role": "system", "content": [{"type": "text", "text": _manual(), "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": "What is the answer to question 12?"},
    ]

    first = await provider.acompletion(model=MODEL, messages=messages)
    second = await provider.acompletion(model=MODEL, messages=messages)

    assert isinstance(first, ChatCompletion)
    assert first.usage is not None
    assert first.usage.prompt_tokens_details is not None
    assert first.usage.prompt_tokens_details.cache_write_tokens
    assert first.usage.prompt_tokens_details.cached_tokens == 0
    assert isinstance(second, ChatCompletion)
    assert second.usage is not None
    assert second.usage.prompt_tokens_details is not None
    assert second.usage.prompt_tokens_details.cached_tokens
    assert second.usage.prompt_tokens_details.cache_write_tokens is None


@pytest.mark.asyncio
async def test_gemini_cache_control_on_streamed_completion_reports_the_cache_write() -> None:
    provider = _provider()
    messages: list[dict[str, Any] | ChatCompletionMessage] = [
        {"role": "system", "content": _manual(), "cache_control": {"type": "ephemeral"}},
        {"role": "user", "content": "What is the answer to question 3?"},
    ]

    stream = await provider.acompletion(model=MODEL, messages=messages, stream=True)

    assert not isinstance(stream, ChatCompletion)
    usages = [chunk.usage async for chunk in stream if chunk.usage is not None]
    assert usages
    details = usages[-1].prompt_tokens_details
    assert details is not None
    assert details.cache_write_tokens


@pytest.mark.asyncio
async def test_gemini_cache_control_through_messages_reports_creation_then_read() -> None:
    provider = _provider()
    system = [{"type": "text", "text": _manual(), "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": "What is the answer to question 40?"}]

    first = await provider.amessages(model=MODEL, system=system, messages=messages, max_tokens=512)
    second = await provider.amessages(model=MODEL, system=system, messages=messages, max_tokens=512)

    assert isinstance(first, MessageResponse)
    assert first.usage.cache_creation_input_tokens
    assert not first.usage.cache_read_input_tokens
    assert isinstance(second, MessageResponse)
    assert second.usage.cache_read_input_tokens
    assert not second.usage.cache_creation_input_tokens
