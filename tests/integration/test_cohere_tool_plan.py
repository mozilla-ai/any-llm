from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage
from any_llm.utils.streaming import aaccumulate_chunks
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.COHERE])]

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


async def test_cohere_streamed_tool_call_turn_keeps_and_replays_tool_plan(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """A streamed tool-call turn carries Cohere's tool plan as content, and the turn replays as its tool_plan.

    https://docs.cohere.com/reference/chat-stream
    """
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Cohere credentials missing: set COHERE_API_KEY")

    model = provider_model_map[provider]
    messages: list[dict[str, Any] | ChatCompletionMessage] = [
        {"role": "user", "content": "Use get_weather to look up the current weather in Paris."}
    ]

    stream = await llm.acompletion(model=model, messages=messages, tools=_TOOLS, stream=True)
    first_turn = await aaccumulate_chunks(stream)

    message = first_turn.choices[0].message
    assert message.tool_calls, "Expected the model to call get_weather"
    assert first_turn.choices[0].finish_reason == "tool_calls"
    assert isinstance(message.content, str)
    assert message.content.strip(), "Expected Cohere's tool plan as the streamed content"

    messages.append(message)
    messages.extend({"role": "tool", "tool_call_id": call.id, "content": "sunny, 21 C"} for call in message.tool_calls)

    second_turn = await llm.acompletion(model=model, messages=messages, tools=_TOOLS)

    assert isinstance(second_turn, ChatCompletion)
    assert second_turn.choices[0].message.content
