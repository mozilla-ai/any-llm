from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from openai import BadRequestError

from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.deepseek.deepseek import DeepseekProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("provider", [LLMProvider.DEEPSEEK]),
    pytest.mark.parametrize("model", ["deepseek-flash"]),
]

_NAMED_TOOL = {"type": "function", "function": {"name": "get_weather"}}
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
_MESSAGES: list[dict[str, Any] | ChatCompletionMessage] = [
    {"role": "user", "content": "Use get_weather to look up the current weather in Paris."}
]


@pytest_asyncio.fixture
async def deepseek_tool_choice_client(
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


@pytest.mark.parametrize(
    "tool_choice",
    [None, "auto", "none", "required", _NAMED_TOOL],
    ids=["omitted", "auto", "none", "required", "named-function"],
)
async def test_deepseek_raw_thinking_tool_choice_contract(
    deepseek_tool_choice_client: DeepseekProvider,
    model: str,
    tool_choice: str | dict[str, Any] | None,
) -> None:
    """Check the live contract without any-llm's local validation.

    Only required and named choices are prohibited in thinking mode by the API reference.
    https://api-docs.deepseek.com/api/create-chat-completion
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _MESSAGES,
        "tools": _TOOLS,
        "max_tokens": 4096,
        "reasoning_effort": "low",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if tool_choice in (None, "auto", "none"):
        response = await deepseek_tool_choice_client.client.chat.completions.create(**kwargs)
        assert response.choices
        if tool_choice == "none":
            assert not response.choices[0].message.tool_calls
        return

    with pytest.raises(BadRequestError) as exc_info:
        await deepseek_tool_choice_client.client.chat.completions.create(**kwargs)
    assert exc_info.value.status_code == 400
    assert "tool_choice" in str(exc_info.value), "Expected a tool_choice rejection, not an unrelated HTTP 400"


@pytest.mark.parametrize(
    "controls",
    [
        {},
        {"reasoning_effort": "none", "extra_body": {"thinking": {"type": "enabled"}}},
    ],
    ids=["default-thinking", "caller-enables-thinking"],
)
@pytest.mark.parametrize("tool_choice", ["auto", "none"])
async def test_deepseek_thinking_supported_tool_choice_succeeds(
    deepseek_tool_choice_client: DeepseekProvider,
    model: str,
    controls: dict[str, Any],
    tool_choice: str,
) -> None:
    """Supported tool choices must succeed, including caller thinking overrides."""
    response = await deepseek_tool_choice_client.acompletion(
        model=model,
        messages=_MESSAGES,
        tools=_TOOLS,
        tool_choice=tool_choice,
        max_tokens=4096,
        **controls,
    )
    assert isinstance(response, ChatCompletion)
    assert response.choices
    if tool_choice == "none":
        assert not response.choices[0].message.tool_calls


@pytest.mark.parametrize(
    "controls",
    [
        {"reasoning_effort": "none"},
        {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "disabled"}}},
    ],
    ids=["reasoning-none", "caller-disables-thinking"],
)
async def test_deepseek_disabled_thinking_forwards_named_tool_choice(
    deepseek_tool_choice_client: DeepseekProvider,
    model: str,
    controls: dict[str, Any],
) -> None:
    """A forced tool remains usable when thinking is explicitly disabled."""
    response = await deepseek_tool_choice_client.acompletion(
        model=model,
        messages=_MESSAGES,
        tools=_TOOLS,
        tool_choice=_NAMED_TOOL,
        max_tokens=4096,
        **controls,
    )
    assert isinstance(response, ChatCompletion)
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls
    assert all(call.type == "function" and call.function.name == "get_weather" for call in tool_calls)
