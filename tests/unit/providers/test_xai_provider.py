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


def test_construction_outside_a_running_loop_does_not_raise() -> None:
    """Constructing the provider synchronously, after a prior asyncio.run() closed this
    thread's loop, must not raise.

    This deliberately does not mock XaiAsyncClient. Building the real grpc.aio channel here
    is what used to fail: grpc resolves "the current event loop" for the calling thread at
    channel construction, and that raises once a previous loop has been created and closed.
    Nothing loop-bound is built at construction now, so the state is harmless.
    """
    from any_llm.providers.xai.xai import XaiProvider

    async def _noop() -> None:
        return None

    asyncio.run(_noop())

    XaiProvider(api_key="test-api-key")


def test_client_is_built_on_the_loop_that_will_drive_it() -> None:
    """The client must be constructed on the loop that will run the RPC, not on whatever
    loop happened to be around at provider construction.

    grpc binds a channel to the loop that is current when it is built, so building it
    anywhere else fails at the first call with "attached to a different loop". That is what
    broke every sync xAI request, since the sync bridge runs each one on a fresh worker loop.
    """
    from any_llm.providers.xai.xai import XaiProvider

    construction_loops: list[asyncio.AbstractEventLoop] = []

    def _record(**_: Any) -> MagicMock:
        construction_loops.append(asyncio.get_running_loop())
        return MagicMock()

    with patch("any_llm.providers.xai.xai.XaiAsyncClient", side_effect=_record):
        provider = XaiProvider(api_key="test-api-key")
        assert construction_loops == []

        async def _use() -> None:
            _ = provider.client
            assert construction_loops == [asyncio.get_running_loop()]

        asyncio.run(_use())


def test_each_event_loop_gets_its_own_client() -> None:
    """A provider reused across sync calls sees a new worker loop each time, so it needs a
    new client each time; within one loop it must reuse the channel rather than rebuild it.
    """
    from any_llm.providers.xai.xai import XaiProvider

    provider = XaiProvider(api_key="test-api-key")
    clients = []

    async def _collect() -> None:
        clients.append(provider.client)
        assert provider.client is clients[-1]

    asyncio.run(_collect())
    asyncio.run(_collect())

    assert clients[0] is not clients[1]


def test_clients_for_closed_loops_are_dropped_and_closed() -> None:
    """Sync callers get a throwaway loop per request, so the per-loop cache has to shed the
    dead ones. Left alone it would hold a live grpc channel, and its socket, per call made
    for the whole life of the provider.
    """
    from any_llm.providers.xai.xai import XaiProvider

    built: list[MagicMock] = []

    def _build(**_: Any) -> MagicMock:
        built.append(MagicMock())
        return built[-1]

    with patch("any_llm.providers.xai.xai.XaiAsyncClient", side_effect=_build):
        provider = XaiProvider(api_key="test-api-key")

        async def _use() -> None:
            _ = provider.client

        asyncio.run(_use())
        asyncio.run(_use())

        assert len(built) == 2
        assert len(provider._clients_by_loop) == 1
        built[0]._api_channel._channel.close.assert_called_once()


def test_client_outside_async_code_raises_a_clear_error() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    provider = XaiProvider(api_key="test-api-key")

    with pytest.raises(RuntimeError, match="bound to the running event loop"):
        _ = provider.client
