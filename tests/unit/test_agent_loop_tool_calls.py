import json
from typing import Any

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    Function,
)
from tests.integration import test_agent_loop as agent_loop


def _tool_call_completion(name: str, arguments: str) -> ChatCompletion:
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id=f"call_{name}",
                type="function",
                function=Function(name=name, arguments=arguments),
            )
        ],
    )
    return _completion(message, "tool_calls")


def _text_completion(content: str) -> ChatCompletion:
    return _completion(ChatCompletionMessage(role="assistant", content=content), "stop")


def _completion(message: ChatCompletionMessage, finish_reason: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-stub",
        created=0,
        model="stub-model",
        object="chat.completion",
        choices=[Choice(finish_reason=finish_reason, index=0, message=message)],
    )


class _StubLLM:
    """Replays a fixed sequence of completions so the agent loop is deterministic."""

    SUPPORTS_COMPLETION = True

    def __init__(self, responses: list[ChatCompletion]) -> None:
        self.responses = responses
        self.calls = 0

    async def acompletion(self, **kwargs: Any) -> ChatCompletion:
        response = self.responses[self.calls]
        self.calls += 1
        return response


@pytest.mark.asyncio
async def test_sequential_agent_loop_tolerates_spurious_tool_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for #1170: Together sometimes calls the zero-parameter tool with a `result` argument."""
    llm = _StubLLM(
        [
            _tool_call_completion("get_current_date", json.dumps({"result": "2026-01-01 00:00"})),
            _tool_call_completion("get_weather", json.dumps({"location": "Paris"})),
            _text_completion("It is sunny in Paris."),
        ]
    )
    monkeypatch.setattr(AnyLLM, "create", lambda provider, **kwargs: llm)

    await agent_loop.test_agent_loop_sequential_tool_calls(
        LLMProvider.TOGETHER,
        {LLMProvider.TOGETHER: "openai/gpt-oss-20b"},
        {},
    )

    assert llm.calls == len(llm.responses)


def test_call_tool_ignores_spurious_argument_for_zero_parameter_tool() -> None:
    date = agent_loop.get_current_date()

    assert agent_loop.call_tool(agent_loop.get_current_date, json.dumps({"result": "2026-01-01 00:00"})) == date


def test_call_tool_forwards_declared_arguments() -> None:
    weather = json.loads(agent_loop.call_tool(agent_loop.get_weather, json.dumps({"location": "Paris"})))

    assert weather["location"] == "Paris"


def test_call_tool_drops_only_undeclared_arguments() -> None:
    arguments = json.dumps({"location": "London", "unit": "celsius"})
    weather = json.loads(agent_loop.call_tool(agent_loop.get_weather, arguments))

    assert weather["location"] == "London"


def test_call_tool_without_arguments() -> None:
    date = agent_loop.get_current_date()

    assert agent_loop.call_tool(agent_loop.get_current_date, None) == date
    assert agent_loop.call_tool(agent_loop.get_current_date, "") == date


def test_call_tool_ignores_non_object_arguments() -> None:
    assert agent_loop.call_tool(agent_loop.get_current_date, json.dumps(["result"])) == agent_loop.get_current_date()


def test_call_tool_forwards_everything_to_var_keyword_tool() -> None:
    def echo(**kwargs: Any) -> str:
        """Echo the arguments the tool was called with."""
        return json.dumps(kwargs, sort_keys=True)

    assert agent_loop.call_tool(echo, json.dumps({"anything": 1})) == json.dumps({"anything": 1})
