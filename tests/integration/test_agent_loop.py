import inspect
import json
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


def filter_tool_args(tool_fn: Callable[..., str], args: dict[str, Any]) -> dict[str, Any]:
    """Keep only arguments accepted by ``tool_fn`` unless it accepts ``**kwargs``."""
    sig = inspect.signature(tool_fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return args
    return {key: value for key, value in args.items() if key in sig.parameters}


def get_weather_with_extras(location: str, **kwargs: Any) -> str:
    """Get the weather for a location, forwarding extra model-supplied keys."""
    payload: dict[str, Any] = {"location": location, "temperature": "15C", "condition": "sunny"}
    payload.update(kwargs)
    return json.dumps(payload)


def test_filter_tool_args_var_keyword_forwards_extra_keys() -> None:
    args = {"location": "Paris", "unexpected_key": "ignored-by-strict-tools"}
    assert filter_tool_args(get_weather_with_extras, args) == args


def test_filter_tool_args_strict_signature_drops_extra_keys() -> None:
    args = {"location": "Paris", "unexpected_key": "drop-me"}
    assert filter_tool_args(get_weather, args) == {"location": "Paris"}


@pytest.mark.asyncio
async def test_agent_loop_parallel_tool_calls(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
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
            tool_result = get_weather(**args)

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                }
            )

        second_result: ChatCompletion = await llm.acompletion(
            model=model_id,
            messages=messages,
            tools=[get_weather],
        )

        assert second_result.choices[0].message.content is not None or second_result.choices[0].message.tool_calls

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
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            result: ChatCompletion = await llm.acompletion(
                model=model_id,
                messages=messages,
                tools=tools,
            )

            tool_calls = result.choices[0].message.tool_calls

            if tool_calls is None:
                assert result.choices[0].message.content is not None
                break

            messages.append(result.choices[0].message)

            for tool_call in tool_calls:
                if not isinstance(tool_call, OpenAIChatCompletionMessageFunctionToolCall):
                    continue
                tool_name = tool_call.function.name
                assert tool_name in available_tools, f"Unknown tool: {tool_name}"
                tool_fn = available_tools[tool_name]

                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                filtered_args = filter_tool_args(tool_fn, args)
                tool_result = tool_fn(**filtered_args)

                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                    }
                )

        assert iteration <= max_iterations, "Agent loop did not complete within max iterations"

    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
