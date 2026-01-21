import json
from typing import TYPE_CHECKING, Any

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
)

from any_llm import AnyLLM, Providers
from any_llm.tools import function_to_tool

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, ChatCompletionMessage


def get_current_date() -> str:
    """Get the current date and time."""
    return "2025-12-18 12:30"


def get_weather(location: str) -> str:
    """Get the weather for a location.

    Args:
        location: The city name to get weather for.
    """
    return json.dumps({"location": location, "temperature": "15C", "condition": "sunny"})


@pytest.mark.parametrize(
    ("provider", "model", "client_config"),
    [
        (Providers.OPENAI, "gpt-5-nano", {"timeout": 10}),
        (Providers.ANTHROPIC, "claude-haiku-4-5-20251001", {"timeout": 30}),
    ],
)
@pytest.mark.asyncio
async def test_agent_loop_parallel_tool_calls(
    provider: Providers,
    model: str,
    client_config: dict[str, Any],
) -> None:
    llm = AnyLLM.create(provider, **client_config)
    messages: list[dict[str, Any] | ChatCompletionMessage] = [
        {
            "role": "user",
            "content": "Get the weather for both Paris and London using the get_weather tool. Call the tool twice, once for each city.",
        }
    ]

    result: ChatCompletion = await llm.acompletion(  # type: ignore[assignment]
        model=model,
        messages=messages,
        tools=[function_to_tool(get_weather)],
    )

    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls is not None, f"Expected tool calls, got: {result.choices[0].message}"

    messages.append(result.choices[0].message.model_dump())

    for tool_call in tool_calls:
        if not isinstance(tool_call, OpenAIChatCompletionMessageFunctionToolCall):
            continue
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        tool_result = get_weather(**args)

        messages.append(
            {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
            }
        )

    second_result: ChatCompletion = await llm.acompletion(  # type: ignore[assignment]
        model=model,
        messages=messages,
        tools=[function_to_tool(get_weather)],
    )

    assert second_result.choices[0].message.content is not None or second_result.choices[0].message.tool_calls
