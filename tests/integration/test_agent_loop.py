import inspect
import json
import re
import warnings
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
)

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletionMessage
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


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


async def _run_agent_loop(
    llm: AnyLLM,
    model_id: str,
    messages: list[dict[str, Any] | ChatCompletionMessage],
    available_tools: dict[str, Callable[..., str]],
    calls_complete: Callable[[list[tuple[str, dict[str, Any]]]], bool],
    *,
    include_tool_name: bool,
    max_iterations: int = 5,
) -> tuple[ChatCompletionMessage, list[tuple[str, dict[str, Any]]]]:
    """Execute tool calls until the required calls are complete and the model answers."""
    calls_made: list[tuple[str, dict[str, Any]]] = []

    for _ in range(max_iterations):
        result = await llm.acompletion(
            model=model_id,
            messages=messages,
            tools=list(available_tools.values()),
        )
        message = result.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            assert calls_complete(calls_made), f"Model answered before making the required tool calls: {calls_made}"
            return message, calls_made

        messages.append(message)
        for tool_call in tool_calls:
            assert isinstance(tool_call, OpenAIChatCompletionMessageFunctionToolCall), (
                f"Expected a function tool call, got: {tool_call}"
            )
            tool_name = tool_call.function.name
            assert tool_name in available_tools, f"Unknown tool: {tool_name}"
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            calls_made.append((tool_name, args))

            tool_message: dict[str, Any] = {
                "role": "tool",
                "content": _call_tool(available_tools[tool_name], args),
                "tool_call_id": tool_call.id,
            }
            if include_tool_name:
                tool_message["name"] = tool_name
            messages.append(tool_message)

    error = f"Agent loop did not answer within {max_iterations} iterations; calls: {calls_made}"
    raise AssertionError(error)


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
async def test_agent_loop_multiple_tool_calls(
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

        def called_both_locations(calls: list[tuple[str, dict[str, Any]]]) -> bool:
            locations = {args.get("location") for tool_name, args in calls if tool_name == "get_weather"}
            return locations >= {"Paris", "London"}

        message, _ = await _run_agent_loop(
            llm,
            model_id,
            messages,
            {"get_weather": get_weather},
            called_both_locations,
            include_tool_name=False,
        )
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
async def test_agent_loop_multiple_tool_types(
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

        available_tools: dict[str, Callable[..., str]] = {
            "get_current_date": get_current_date,
            "get_weather": get_weather,
        }

        def called_both_tools(calls: list[tuple[str, dict[str, Any]]]) -> bool:
            return {tool_name for tool_name, _ in calls} >= set(available_tools)

        # Callers may still send name on tool messages, so one loop keeps that shape on the wire.
        message, _ = await _run_agent_loop(
            llm,
            model_id,
            messages,
            available_tools,
            called_both_tools,
            include_tool_name=True,
        )
        assert _mentions_tool_result(message.content), f"Expected an answer from the tool results, got: {message}"

    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
