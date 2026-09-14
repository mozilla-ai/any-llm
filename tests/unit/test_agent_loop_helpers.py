import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage
from tests.integration.test_agent_loop import (
    _call_tool,
    _run_agent_loop,
    get_current_date,
    get_weather,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> ChatCompletion:
    serialized_tool_calls = [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        for index, (name, arguments) in enumerate(tool_calls or [])
    ]
    return ChatCompletion.model_validate(
        {
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if serialized_tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": serialized_tool_calls if tool_calls is not None else None,
                    },
                }
            ],
        }
    )


def test_call_tool_ignores_spurious_model_arguments_for_zero_arg_tool() -> None:
    """Ignore model-generated arguments that a zero-argument tool cannot accept."""
    with pytest.warns(UserWarning, match="Ignoring unexpected arguments for get_current_date: result"):
        assert _call_tool(get_current_date, {"result": "unexpected"}) == "2025-12-18 12:30"


def test_call_tool_preserves_declared_tool_arguments() -> None:
    """Preserve arguments declared by a parameterized tool while filtering extras."""
    with pytest.warns(UserWarning, match="Ignoring unexpected arguments for get_weather: result"):
        assert "Paris" in _call_tool(get_weather, {"location": "Paris", "result": "unexpected"})


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_tool_calls", [None, []])
async def test_run_agent_loop_continues_sequential_calls_before_answering(
    empty_tool_calls: list[tuple[str, dict[str, Any]]] | None,
) -> None:
    client = MagicMock(spec=AnyLLM)
    client.acompletion = AsyncMock(
        side_effect=[
            _completion(tool_calls=[("get_weather", {"location": "Paris"})]),
            _completion(tool_calls=[("get_weather", {"location": "London"})]),
            _completion(content="Paris and London are sunny at 15C.", tool_calls=empty_tool_calls),
        ]
    )
    messages: list[dict[str, Any] | ChatCompletionMessage] = [{"role": "user", "content": "Weather?"}]

    def called_both_locations(calls: list[tuple[str, dict[str, Any]]]) -> bool:
        return {arguments.get("location") for _, arguments in calls} >= {"Paris", "London"}

    message, calls = await _run_agent_loop(
        client,
        "test-model",
        messages,
        {"get_weather": get_weather},
        called_both_locations,
        include_tool_name=False,
    )

    assert message.content == "Paris and London are sunny at 15C."
    assert calls == [
        ("get_weather", {"location": "Paris"}),
        ("get_weather", {"location": "London"}),
    ]
    assert client.acompletion.await_count == 3
    assert all("tool_choice" not in call.kwargs for call in client.acompletion.await_args_list)
    tool_messages = [item for item in messages if isinstance(item, dict) and item.get("role") == "tool"]
    assert all("name" not in tool_message for tool_message in tool_messages)


@pytest.mark.asyncio
async def test_run_agent_loop_can_include_tool_names() -> None:
    client = MagicMock(spec=AnyLLM)
    client.acompletion = AsyncMock(
        side_effect=[
            _completion(
                tool_calls=[
                    ("get_current_date", {}),
                    ("get_weather", {"location": "Paris"}),
                ]
            ),
            _completion(content="Paris is sunny at 15C."),
        ]
    )
    messages: list[dict[str, Any] | ChatCompletionMessage] = [{"role": "user", "content": "Weather?"}]
    available_tools: dict[str, Callable[..., str]] = {
        "get_current_date": get_current_date,
        "get_weather": get_weather,
    }

    message, _ = await _run_agent_loop(
        client,
        "test-model",
        messages,
        available_tools,
        lambda calls: {name for name, _ in calls} >= set(available_tools),
        include_tool_name=True,
    )

    assert message.content == "Paris is sunny at 15C."
    assert client.acompletion.await_count == 2
    tool_messages = [item for item in messages if isinstance(item, dict) and item.get("role") == "tool"]
    assert [tool_message["name"] for tool_message in tool_messages] == ["get_current_date", "get_weather"]


@pytest.mark.asyncio
async def test_run_agent_loop_rejects_answer_before_required_calls() -> None:
    client = MagicMock(spec=AnyLLM)
    client.acompletion = AsyncMock(side_effect=[_completion(content="No tools needed.")])

    with pytest.raises(AssertionError, match="answered before making the required tool calls"):
        await _run_agent_loop(
            client,
            "test-model",
            [{"role": "user", "content": "Weather?"}],
            {"get_weather": get_weather},
            bool,
            include_tool_name=False,
        )


@pytest.mark.asyncio
async def test_run_agent_loop_rejects_repeated_calls_at_iteration_limit() -> None:
    client = MagicMock(spec=AnyLLM)
    client.acompletion = AsyncMock(
        side_effect=[
            _completion(tool_calls=[("get_weather", {"location": "Paris"})]),
            _completion(tool_calls=[("get_weather", {"location": "Paris"})]),
        ]
    )

    with pytest.raises(AssertionError, match="did not answer within 2 iterations"):
        await _run_agent_loop(
            client,
            "test-model",
            [{"role": "user", "content": "Weather?"}],
            {"get_weather": get_weather},
            lambda calls: any(arguments.get("location") == "London" for _, arguments in calls),
            include_tool_name=False,
            max_iterations=2,
        )
