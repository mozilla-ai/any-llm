import inspect
import json
import re
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from openai import APIConnectionError
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
)

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS

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


def _call_tool(tool_fn: Callable[..., str], args: dict[str, Any]) -> str:
    """Call a model-selected tool without passing arguments it does not accept."""
    accepted = inspect.signature(tool_fn).parameters
    unexpected = set(args) - set(accepted)
    if unexpected:
        warnings.warn(
            f"Ignoring unexpected arguments for {tool_fn.__name__}: {', '.join(sorted(unexpected))}",
            UserWarning,
            stacklevel=2,
        )
    return tool_fn(**{name: value for name, value in args.items() if name in accepted})


def _mentions_tool_result(content: str | None) -> bool:
    """The weather tool returns 15C and sunny, so an answer built on it repeats one of them.

    ``15`` must not run into another digit, so ``150F`` does not count; ``15C``, ``15°C`` and
    ``15 degrees`` all do.
    """
    return content is not None and re.search(r"\b15(?!\d)|\bsunny\b", content, re.IGNORECASE) is not None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, False),
        ("", False),
        ("It rains in Paris.", False),
        ("It is 150F in Paris.", False),
        ("It is 15C in Paris.", True),
        ("It is 15°C in Paris.", True),
        ("Sunny in London.", True),
    ],
)
def test_mentions_tool_result(content: str | None, expected: bool) -> None:
    assert _mentions_tool_result(content) is expected


@pytest.mark.asyncio
async def test_agent_loop_parallel_tool_calls(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Execute multiple model-selected tool calls and return their results."""
    if provider in (*LOCAL_PROVIDERS, LLMProvider.PERPLEXITY):
        pytest.skip(f"{provider} does not support tools, skipping")

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")

        model_id = provider_model_map[provider]
        messages: list[dict[str, Any] | ChatCompletionMessage] = [
            {
                "role": "user",
                "content": "Get the weather for both Paris and London using the get_weather tool. Call the tool twice, once for each city.",
            }
        ]

        result: ChatCompletion = await llm.acompletion(
            model=model_id,
            messages=messages,
            tools=[get_weather],
        )

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None, f"Expected tool calls, got: {result.choices[0].message}"

        messages.append(result.choices[0].message)

        for tool_call in tool_calls:
            if not isinstance(tool_call, OpenAIChatCompletionMessageFunctionToolCall):
                continue
            assert tool_call.function.name == "get_weather"
            args = json.loads(tool_call.function.arguments)
            tool_result = _call_tool(get_weather, args)

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id,
                }
            )

        second_result: ChatCompletion = await llm.acompletion(
            model=model_id,
            messages=messages,
            tools=[get_weather],
        )

        message = second_result.choices[0].message
        assert _mentions_tool_result(message.content), f"Expected an answer from the tool results, got: {message}"

    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise


@pytest.mark.asyncio
async def test_agent_loop_sequential_tool_calls(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Execute model-selected tools over several agent-loop iterations."""
    if provider in (*LOCAL_PROVIDERS, LLMProvider.PERPLEXITY):
        pytest.skip(f"{provider} does not support tools, skipping")

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")

        model_id = provider_model_map[provider]
        messages: list[dict[str, Any] | ChatCompletionMessage] = [
            {
                "role": "user",
                "content": "First get the current date, then get the weather for Paris. Use both tools in sequence.",
            }
        ]

        tools = [get_current_date, get_weather]
        available_tools: dict[str, Callable[..., str]] = {
            "get_current_date": get_current_date,
            "get_weather": get_weather,
        }

        max_iterations = 5
        answered = False

        for _ in range(max_iterations):
            result: ChatCompletion = await llm.acompletion(
                model=model_id,
                messages=messages,
                tools=tools,
            )

            tool_calls = result.choices[0].message.tool_calls

            if tool_calls is None:
                message = result.choices[0].message
                assert _mentions_tool_result(message.content), (
                    f"Expected an answer from the tool results, got: {message}"
                )
                answered = True
                break

            messages.append(result.choices[0].message)

            for tool_call in tool_calls:
                if not isinstance(tool_call, OpenAIChatCompletionMessageFunctionToolCall):
                    continue
                tool_name = tool_call.function.name
                assert tool_name in available_tools, f"Unknown tool: {tool_name}"
                tool_fn = available_tools[tool_name]

                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                tool_result = _call_tool(tool_fn, args)

                # Callers may still send name on tool messages, so one loop keeps that shape on the wire.
                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                    }
                )

        assert answered, "Agent loop did not answer within max iterations"

    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
