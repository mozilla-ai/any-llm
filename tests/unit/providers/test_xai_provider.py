import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.types.completion import ChatCompletion, CompletionParams


@contextmanager
def mock_xai_provider():  # type: ignore[no-untyped-def]
    with patch("any_llm.providers.xai.xai.XaiAsyncClient") as mock_xai:
        create_return = MagicMock()
        mock_response = MagicMock()
        mock_response.reasoning_content = None
        mock_response.content = "Test response"
        mock_response.id = "Test id"
        mock_response.proto.model = "Test model"
        mock_response.proto.created.seconds = 0
        mock_response.tool_calls = None
        create_return.sample = AsyncMock(return_value=mock_response)
        mock_xai.return_value.chat.create = MagicMock(return_value=create_return)

        yield mock_xai, mock_response


@pytest.mark.asyncio
async def test_response_function_call_id_is_preserved() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (_, mock_response):
        tool_call = MagicMock()
        tool_call.id = "expected_function_call_id"
        tool_call.function.name = "test_function"
        tool_call.function.arguments = '{"key": "value"}'
        mock_response.tool_calls = [tool_call]

        provider = XaiProvider(api_key="test-api-key")
        response = await provider._acompletion(
            CompletionParams(model_id="model", messages=[{"role": "user", "content": "Hello"}])
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices
        assert response.choices[0].message.tool_calls
        assert response.choices[0].message.tool_calls[0].id == "expected_function_call_id"


@pytest.mark.asyncio
async def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(CompletionParams(model_id="model", messages=agent_loop_messages))
        _, call_kwargs = mock_xai.return_value.chat.create.call_args

        assert len(call_kwargs["messages"]) == 3


@pytest.mark.asyncio
async def test_dataclass_response_format_uses_sample_not_parse() -> None:
    """Test that dataclass response_format uses sample() with protobuf schema, not parse()."""
    from dataclasses import dataclass

    from any_llm.providers.xai.xai import XaiProvider

    @dataclass
    class TestOutput:
        name: str

    with mock_xai_provider() as (mock_xai, _):
        create_return = mock_xai.return_value.chat.create.return_value

        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(
            CompletionParams(
                model_id="model",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=TestOutput,
            )
        )

        # Should call sample(), not parse()
        create_return.sample.assert_called_once()
        create_return.parse.assert_not_called()

        # Should pass response_format protobuf to create()
        _, call_kwargs = mock_xai.return_value.chat.create.call_args
        assert call_kwargs["response_format"] is not None


@pytest.mark.asyncio
async def test_dict_json_schema_response_format_uses_sample_not_parse() -> None:
    """Test that OpenAI dict response_format is converted to protobuf and uses sample()."""
    from any_llm.providers.xai.xai import XaiProvider

    openai_json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "TestOutput",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    with mock_xai_provider() as (mock_xai, _):
        create_return = mock_xai.return_value.chat.create.return_value

        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(
            CompletionParams(
                model_id="model",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=openai_json_schema,
            )
        )

        # Should call sample(), not parse()
        create_return.sample.assert_called_once()
        create_return.parse.assert_not_called()

        # Should pass response_format protobuf to create()
        _, call_kwargs = mock_xai.return_value.chat.create.call_args
        assert call_kwargs["response_format"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", "none"])
async def test_reasoning_effort_filtered_out(reasoning_effort: str) -> None:
    """Test that reasoning_effort 'auto' and 'none' are filtered from xAI API calls."""
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(
            CompletionParams(
                model_id="model",
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            )
        )
        _, call_kwargs = mock_xai.return_value.chat.create.call_args
        assert "reasoning_effort" not in call_kwargs


def test_stream_options_filtered_out() -> None:
    """stream_options is an OpenAI-only knob (set by the Messages bridge for
    streaming usage); the xAI SDK rejects it, so it must be dropped."""
    from any_llm.providers.xai.xai import XaiProvider

    result = XaiProvider._convert_completion_params(
        CompletionParams(
            model_id="model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert "stream_options" not in result


def test_per_request_timeout_is_declared_unsupported() -> None:
    """xAI sets timeouts on the gRPC client, so the base class rejects a per-request one."""
    from any_llm.providers.xai.xai import XaiProvider

    assert XaiProvider.TIMEOUT_SUPPORT == "unsupported"


def test_client_construction_survives_a_closed_event_loop_on_this_thread() -> None:
    """Constructing the provider synchronously, after a prior asyncio.run() has
    closed this thread's event loop, must not raise.

    grpc.aio's channel construction binds to "the current event loop" for the
    calling thread via asyncio.get_event_loop(). On Python's default policy that
    call raises RuntimeError once a previously-created loop on this thread has
    been closed and no new one has been set - exactly what happens when a sync
    caller constructs XaiProvider directly (e.g. AnyLLM.create("xai", ...)
    outside of any active event loop, as any-llm's own sync integration tests do
    after an earlier async test has run on the same worker). This deliberately
    does not mock XaiAsyncClient: the bug is in the real grpc.aio channel
    construction, and a mock would hide it.
    """
    from any_llm.providers.xai.xai import XaiProvider

    async def _noop() -> None:
        return None

    asyncio.run(_noop())

    provider = XaiProvider(api_key="test-api-key")
    assert provider.client is not None
