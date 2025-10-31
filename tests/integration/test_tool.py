import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, ChatCompletionMessage


@pytest.mark.asyncio
async def test_tool(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    if provider in (LLMProvider.LLAMAFILE, LLMProvider.PERPLEXITY):
        pytest.skip(f"{provider} does not support tools, skipping")

    def echo(message: str) -> str:
        """Tool function to get the capital of a city."""
        return message

    available_tools = {"echo": echo}

    prompt = "Please call the `echo` tool with the argument `Hello, world!`. You must use the tool, do not ask any follow up questions."
    messages: list[dict[str, Any] | ChatCompletionMessage] = [{"role": "user", "content": prompt}]

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support tools, skipping")
        model_id = provider_model_map[provider]

        result: ChatCompletion = await llm.acompletion(  # type: ignore[assignment]
            model=model_id,
            messages=messages,
            tools=[echo],
        )

        messages.append(result.choices[0].message)

        completion_tool_calls = result.choices[0].message.tool_calls
        assert completion_tool_calls is not None, f"No tool calls found in response: {result.choices[0].message}"
        assert (
            len(completion_tool_calls) > 0
        )  # if the llm wants to call more than one tool that's ok for the purpose of the test
        assert hasattr(completion_tool_calls[0], "function")
        assert completion_tool_calls[0].function.name
        tool_to_call = available_tools[completion_tool_calls[0].function.name]
        args = json.loads(completion_tool_calls[0].function.arguments)
        tool_result = tool_to_call(**args)
        messages.append(
            {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": completion_tool_calls[0].id,
                "name": completion_tool_calls[0].function.name,
            }
        )
        messages.append({"role": "user", "content": "Did the tool call work?"})
        second_result: ChatCompletion = await llm.acompletion(  # type: ignore[assignment]
            model=model_id,
            messages=messages,
            tools=[echo],
        )
        assert second_result.choices[0].message
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise


@pytest.mark.asyncio
async def test_built_in_tool(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.BUILT_IN_TOOLS:
            pytest.skip(f"{provider.value} does not support built-in tools, skipping")
        model_id = provider_model_map[provider]
        if provider == LLMProvider.GEMINI:
            from google.genai import types

            tool = types.Tool(google_search=types.GoogleSearch())

        messages: list[dict[str, Any] | ChatCompletionMessage] = [
            {"role": "user", "content": "Search the web to find what the python library any-llm"}
        ]

        result: ChatCompletion = await llm.acompletion(  # type: ignore[assignment]
            model=model_id,
            messages=messages,
            tools=[tool],
        )
        assert result.choices[0].message

    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
