from inspect import signature
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from any_llm import AnyLLM
from any_llm.api import aresponses, responses
from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.responses import ResponsesParams


async def _response_event_stream() -> Any:
    yield "response.created"
    yield "response.completed"


@pytest.mark.asyncio
async def test_responses_invalid_model_format_no_slash() -> None:
    """Test responses raises ValueError for model without separator."""
    with pytest.raises(
        ValueError, match=r"Invalid model format. Expected 'provider:model' or 'provider/model', got 'gpt-5-nano'"
    ):
        await aresponses("gpt-5-nano", input_data=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_responses_invalid_model_format_empty_provider() -> None:
    """Test responses raises ValueError for model with empty provider."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await aresponses("/model", input_data=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_responses_invalid_model_format_empty_model() -> None:
    """Test responses raises ValueError for model with empty model name."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await aresponses("provider/", input_data=[{"role": "user", "content": "Hello"}])


def test_omitted_parallel_tool_calls_stays_none() -> None:
    """Omitted parallel_tool_calls should not appear in provider request."""
    params = ResponsesParams(model="test", input="hello")
    assert params.parallel_tool_calls is None
    assert "parallel_tool_calls" not in params.model_dump(exclude_none=True)


def test_omitted_context_management_stays_none() -> None:
    """Omitted context_management should not appear in provider request."""
    params = ResponsesParams(model="test", input="hello")

    assert params.context_management is None
    assert "context_management" not in params.model_dump(exclude_none=True)


@pytest.mark.parametrize("value", [True, False])
def test_explicit_parallel_tool_calls_preserved(value: bool) -> None:
    """Explicit True/False should be kept as-is."""
    params = ResponsesParams(model="test", input="hello", parallel_tool_calls=value)
    assert params.parallel_tool_calls is value


def test_context_management_preserved() -> None:
    """An explicit context_management value is kept as-is for the provider request."""
    context_management = [{"type": "compaction", "compact_threshold": 200_000}]
    params = ResponsesParams(model="test", input="hello", context_management=context_management)

    assert params.context_management == context_management
    assert params.model_dump(exclude_none=True)["context_management"] == context_management


def test_responses_params_preserves_codex_continuation_items() -> None:
    """Responses input accepts Codex items not yet represented by the SDK union."""
    input_data: list[dict[str, Any]] = [
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "commentary"}],
            "phase": "commentary",
        },
    ]

    params = ResponsesParams(model="test", input=input_data)

    assert params.input == input_data


def test_responses_params_rejects_non_dictionary_input_items() -> None:
    """Responses input items must be dictionaries."""
    with pytest.raises(ValidationError):
        ResponsesParams(model="test", input=cast("Any", ["hello"]))


def test_responses_params_rejects_top_level_dictionary_input() -> None:
    """Responses input must be text or a list of dictionaries."""
    with pytest.raises(ValidationError):
        ResponsesParams(model="test", input=cast("Any", {"role": "user", "content": "hello"}))


@pytest.mark.asyncio
async def test_tools_flattened_for_responses() -> None:
    """Callables and chat-format tools must reach the provider flat; built-ins pass untouched."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    chat_format = {"type": "function", "function": {"name": "sub", "parameters": {}, "strict": True}}
    flat = {"type": "function", "name": "mul", "parameters": {}}
    builtin = {"type": "web_search"}

    llm = AnyLLM.create("openai", api_key="test-key")
    with patch.object(type(llm), "_aresponses", new=AsyncMock(return_value=object())) as mock_aresponses:
        await llm.aresponses("gpt-4.1-mini", "hello", tools=[add, chat_format, flat, builtin])

    tools = mock_aresponses.call_args.args[0].tools
    assert [tool.get("name") for tool in tools[:3]] == ["add", "sub", "mul"]
    assert [tool["type"] for tool in tools[:3]] == ["function"] * 3
    assert all("function" not in tool for tool in tools[:3])
    assert tools[1]["parameters"] == {}
    assert tools[1]["strict"] is True
    assert tools[3] == {"type": "web_search"}


def test_responses_exposes_timeout_parameter() -> None:
    """The Responses entry points advertise timeout the same way the completion ones do."""
    assert "timeout" in signature(responses).parameters
    assert "timeout" in signature(aresponses).parameters
    assert "timeout" in signature(AnyLLM.aresponses).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize(("requested_timeout", "expected"), [(60, 60), (None, None)])
async def test_timeout_forwarded_to_provider_not_params(
    requested_timeout: float | None, expected: float | None
) -> None:
    """timeout is an SDK request option: it must reach the provider call, never ResponsesParams."""
    llm = AnyLLM.create("openai", api_key="test-key")
    with patch.object(type(llm), "_aresponses", new=AsyncMock(return_value=object())) as mock_aresponses:
        await llm.aresponses("gpt-4.1-mini", "hello", timeout=requested_timeout)

    assert mock_aresponses.call_args.kwargs.get("timeout") == expected
    params = mock_aresponses.call_args.args[0]
    assert "timeout" not in params.model_dump(exclude_none=True)


@pytest.mark.asyncio
async def test_aresponses_helper_forwards_timeout_to_provider() -> None:
    """The public aresponses() helper routes timeout through to the provider call."""
    provider = AnyLLM.create("openai", api_key="test-key")
    with (
        patch("any_llm.any_llm.AnyLLM.create", return_value=provider),
        patch.object(provider, "_aresponses", new=AsyncMock(return_value=object())) as mock_aresponses,
    ):
        await aresponses("gpt-4.1-mini", "hello", provider="openai", api_key="test-key", timeout=60)

    assert mock_aresponses.call_args.kwargs["timeout"] == 60


def test_responses_helper_forwards_timeout_to_provider_sync() -> None:
    """The synchronous responses() helper routes timeout through the same path."""
    provider = AnyLLM.create("openai", api_key="test-key")
    with (
        patch("any_llm.any_llm.AnyLLM.create", return_value=provider),
        patch.object(provider, "_aresponses", new=AsyncMock(return_value=object())) as mock_aresponses,
    ):
        responses("gpt-4.1-mini", "hello", provider="openai", api_key="test-key", timeout=60)

    assert mock_aresponses.call_args.kwargs["timeout"] == 60


def test_responses_helper_streaming_returns_sync_iterator() -> None:
    """The synchronous Responses API must bridge an async event stream."""
    provider = AnyLLM.create("openai", api_key="test-key")
    with (
        patch("any_llm.any_llm.AnyLLM.create", return_value=provider),
        patch.object(provider, "_aresponses", new=AsyncMock(return_value=_response_event_stream())) as mock_aresponses,
    ):
        stream = responses("gpt-4.1-mini", "hello", provider="openai", api_key="test-key", stream=True)
        assert list(stream) == ["response.created", "response.completed"]

    assert mock_aresponses.call_args.args[0].stream is True


@pytest.mark.asyncio
async def test_aresponses_rejects_timeout_for_unsupported_provider() -> None:
    """A provider declaring no per-request timeout support is rejected before the request is built."""
    llm = AnyLLM.create("openai", api_key="test-key")
    llm.TIMEOUT_SUPPORT = "unsupported"
    with (
        patch.object(llm, "_aresponses", new=AsyncMock()) as mock_aresponses,
        pytest.raises(UnsupportedParameterError, match="timeout"),
    ):
        await llm.aresponses("gpt-4.1-mini", "hello", timeout=60)

    mock_aresponses.assert_not_called()
