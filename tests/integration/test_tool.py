import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import ProviderName, acompletion
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory
from tests.constants import LOCAL_PROVIDERS

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, ChatCompletionMessage


@pytest.mark.asyncio
async def test_tool(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    if provider == ProviderName.LLAMAFILE:
        pytest.skip("Llamafile does not support tools, skipping")

    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION:
        pytest.skip(f"{provider.value} does not support tools, skipping")

    model_id = provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})

    def echo(message: str) -> str:
        """Tool function to get the capital of a city."""
        return message

    available_tools = {"echo": echo}

    prompt = "Please call the `echo` tool with the argument `Hello, world!`"
    messages: list[dict[str, Any] | ChatCompletionMessage] = [{"role": "user", "content": prompt}]

    try:
        result: ChatCompletion = await acompletion(  # type: ignore[assignment]
            model=model_id,
            provider=provider,
            messages=messages,
            tools=[echo],
            **extra_kwargs,
        )

        messages.append(result.choices[0].message)

        completion_tool_calls = result.choices[0].message.tool_calls
        assert completion_tool_calls is not None
        assert len(completion_tool_calls) == 1
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
        second_result: ChatCompletion = await acompletion(  # type: ignore[assignment]
            model=model_id,
            provider=provider,
            messages=messages,
            tools=[echo],
            **extra_kwargs,
        )
        assert second_result.choices[0].message
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
