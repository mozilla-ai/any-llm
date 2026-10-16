from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.deepseek.deepseek import DeepseekProvider
from any_llm.utils.streaming import aaccumulate_chunks
from tests.constants import EXPECTED_PROVIDERS

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletionMessage

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("provider", [LLMProvider.DEEPSEEK]),
    pytest.mark.parametrize("model", ["deepseek-v4-flash"]),
]

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


@pytest_asyncio.fixture
async def deepseek_client(
    provider: LLMProvider,
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> AsyncIterator[DeepseekProvider]:
    try:
        llm = DeepseekProvider(**provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("DeepSeek credentials missing: set DEEPSEEK_API_KEY")
    try:
        yield llm
    finally:
        await llm.client.close()


async def test_deepseek_streamed_tool_call_turn_replays_reasoning(
    deepseek_client: DeepseekProvider,
    model: str,
) -> None:
    """A streamed thinking-mode tool call can be rebuilt and replayed without a 400.

    DeepSeek requires the reasoning of an assistant turn that called a tool to be sent back verbatim.
    https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
    """
    messages: list[dict[str, Any] | ChatCompletionMessage] = [
        {"role": "user", "content": "Use get_weather to look up the current weather in Paris."}
    ]
    controls: dict[str, Any] = {
        "tools": _TOOLS,
        "max_tokens": 4096,
        "reasoning_effort": "low",
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    stream = await deepseek_client.acompletion(model=model, messages=messages, stream=True, **controls)
    first_turn = await aaccumulate_chunks(stream)

    message = first_turn.choices[0].message
    assert message.tool_calls, "Expected the model to call get_weather"
    assert message.reasoning is not None
    assert message.extra_content == {"deepseek": {"reasoning_content": message.reasoning.content}}

    messages.append(message)
    messages.extend({"role": "tool", "tool_call_id": call.id, "content": "sunny, 21 C"} for call in message.tool_calls)

    stream = await deepseek_client.acompletion(model=model, messages=messages, stream=True, **controls)
    second_turn = await aaccumulate_chunks(stream)

    assert second_turn.choices[0].message.content or second_turn.choices[0].message.tool_calls
