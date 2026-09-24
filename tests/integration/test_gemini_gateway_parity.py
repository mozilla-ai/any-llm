from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion
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

