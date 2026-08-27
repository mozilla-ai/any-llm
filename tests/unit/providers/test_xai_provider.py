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
@pytest.mark.parametrize(
    ("xai_finish_reason", "expected"),
    [
        ("REASON_STOP", "stop"),
        ("REASON_MAX_LEN", "length"),
        ("REASON_MAX_CONTEXT", "length"),
        # No OpenAI counterpart, so the response falls back to a terminal reason.
        ("REASON_TIME_LIMIT", "stop"),
    ],
)
async def test_response_finish_reason_is_mapped(xai_finish_reason: str, expected: str) -> None:
    """A truncated response must report "length", not "stop": callers (including the
    structured-output truncation check) cannot tell a cut-off answer from a complete one
    otherwise."""
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (_, mock_response):
        mock_response.finish_reason = xai_finish_reason

        provider = XaiProvider(api_key="test-api-key")
        response = await provider._acompletion(
            CompletionParams(model_id="model", messages=[{"role": "user", "content": "Hello"}])
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].finish_reason == expected


@pytest.mark.asyncio
async def test_truncated_tool_call_response_reports_length_not_tool_calls() -> None:
    """A response cut short mid tool call must not look like a completed tool call round."""
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (_, mock_response):
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "test_function"
        tool_call.function.arguments = '{"key": "va'
        mock_response.tool_calls = [tool_call]
        mock_response.finish_reason = "REASON_MAX_LEN"

        provider = XaiProvider(api_key="test-api-key")
        response = await provider._acompletion(
            CompletionParams(model_id="model", messages=[{"role": "user", "content": "Hello"}])
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].finish_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("xai_finish_reason", ["REASON_TOOL_CALLS", "REASON_TIME_LIMIT"])
async def test_tool_call_response_reports_tool_calls_finish_reason(xai_finish_reason: str) -> None:
    """Mapped directly for REASON_TOOL_CALLS; for a reason without a counterpart, the tool calls decide."""
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (_, mock_response):
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "test_function"
        tool_call.function.arguments = "{}"
        mock_response.tool_calls = [tool_call]
        mock_response.finish_reason = xai_finish_reason

        provider = XaiProvider(api_key="test-api-key")
        response = await provider._acompletion(
            CompletionParams(model_id="model", messages=[{"role": "user", "content": "Hello"}])
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].finish_reason == "tool_calls"


@pytest.mark.parametrize(
    ("xai_finish_reason_name", "expected"),
    [
        ("REASON_INVALID", None),
        ("REASON_STOP", "stop"),
        ("REASON_MAX_LEN", "length"),
        ("REASON_TOOL_CALLS", "tool_calls"),
    ],
)
def test_stream_chunk_finish_reason_is_mapped(xai_finish_reason_name: str, expected: str | None) -> None:
    """Streamed chunks used to always report finish_reason=None, so a stream never signalled
    why it ended. Non-final chunks carry REASON_INVALID and must stay None."""
    from xai_sdk.chat import Chunk as XaiChunk
    from xai_sdk.proto import chat_pb2

    from any_llm.providers.xai.utils import _convert_xai_chunk_to_anyllm_chunk

    proto = chat_pb2.GetChatCompletionChunk(
        outputs=[
            chat_pb2.CompletionOutputChunk(
                index=0,
                delta=chat_pb2.Delta(role=chat_pb2.ROLE_ASSISTANT, content="hi"),
                finish_reason=xai_finish_reason_name,
            )
        ]
    )

    chunk = _convert_xai_chunk_to_anyllm_chunk(XaiChunk(proto, index=None))

    assert chunk.choices[0].finish_reason == expected


@pytest.mark.asyncio
async def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(CompletionParams(model_id="model", messages=agent_loop_messages))
        _, call_kwargs = mock_xai.return_value.chat.create.call_args

        assert len(call_kwargs["messages"]) == 3


@pytest.mark.asyncio
async def test_completion_replays_tool_calls_and_reversed_results_by_id() -> None:
    from xai_sdk.proto import chat_pb2

    from any_llm.providers.xai.xai import XaiProvider

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Weather in Toronto and Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_toronto",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Toronto"}'},
                },
                {
                    "id": "call_paris",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_paris", "content": '{"temp":30}'},
        {"role": "tool", "tool_call_id": "call_toronto", "content": '{"temp":-5}'},
    ]

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(CompletionParams(model_id="model", messages=messages))

        xai_messages = mock_xai.return_value.chat.create.call_args.kwargs["messages"]
        assistant_message = xai_messages[1]
        assert assistant_message.role == chat_pb2.ROLE_ASSISTANT
        assert list(assistant_message.content) == []
        assert [
            (call.id, call.type, call.function.name, call.function.arguments) for call in assistant_message.tool_calls
        ] == [
            (
                "call_toronto",
                chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                "get_weather",
                '{"city":"Toronto"}',
            ),
            (
                "call_paris",
                chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                "get_weather",
                '{"city":"Paris"}',
            ),
        ]

        tool_messages = xai_messages[2:]
        assert [message.role for message in tool_messages] == [chat_pb2.ROLE_TOOL, chat_pb2.ROLE_TOOL]
        assert [message.tool_call_id for message in tool_messages] == ["call_paris", "call_toronto"]
        assert [[part.text for part in message.content] for message in tool_messages] == [
            ['{"temp":30}'],
            ['{"temp":-5}'],
        ]


@pytest.mark.asyncio
async def test_completion_accepts_assistant_message_without_content_key() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        }
    ]

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(CompletionParams(model_id="model", messages=messages))

        xai_message = mock_xai.return_value.chat.create.call_args.kwargs["messages"][0]
        assert list(xai_message.content) == []
        assert xai_message.tool_calls[0].id == "call_weather"


@pytest.mark.asyncio
async def test_completion_serializes_parsed_tool_call_arguments() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        }
    ]

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(CompletionParams(model_id="model", messages=messages))

        xai_message = mock_xai.return_value.chat.create.call_args.kwargs["messages"][0]
        assert xai_message.tool_calls[0].function.arguments == '{"city": "Paris"}'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected_content"),
    [
        (None, []),
        ("", [""]),
        ("I will check the weather.", ["I will check the weather."]),
    ],
)
async def test_completion_preserves_assistant_content_with_tool_calls(
    content: str | None, expected_content: list[str]
) -> None:
    from any_llm.providers.xai.xai import XaiProvider

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        }
    ]

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(CompletionParams(model_id="model", messages=messages))

        xai_message = mock_xai.return_value.chat.create.call_args.kwargs["messages"][0]
        assert [part.text for part in xai_message.content] == expected_content
        assert len(xai_message.tool_calls) == 1


@pytest.mark.asyncio
async def test_completion_preserves_assistant_message_without_tool_calls() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(api_key="test-api-key")
        await provider._acompletion(
            CompletionParams(
                model_id="model",
                messages=[{"role": "assistant", "content": "Previous answer."}],
            )
        )

        xai_message = mock_xai.return_value.chat.create.call_args.kwargs["messages"][0]
        assert [part.text for part in xai_message.content] == ["Previous answer."]
        assert list(xai_message.tool_calls) == []


def test_streaming_response_preserves_tool_call_id() -> None:
    from xai_sdk.chat import Chunk
    from xai_sdk.proto import chat_pb2

    from any_llm.providers.xai.utils import _convert_xai_chunk_to_anyllm_chunk

    proto = chat_pb2.GetChatCompletionChunk(
        id="chunk",
        model="model",
        outputs=[
            chat_pb2.CompletionOutputChunk(
                index=0,
                delta=chat_pb2.Delta(
                    role=chat_pb2.ROLE_ASSISTANT,
                    tool_calls=[
                        chat_pb2.ToolCall(
                            id="call_from_xai",
                            type=chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                            function=chat_pb2.FunctionCall(name="get_weather", arguments='{"city":"Paris"}'),
                        )
                    ],
                ),
            )
        ],
    )

    converted = _convert_xai_chunk_to_anyllm_chunk(Chunk(proto, None))

    assert converted.choices[0].delta.tool_calls is not None
    assert converted.choices[0].delta.tool_calls[0].id == "call_from_xai"


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
    broke every sync xAI request, since the sync bridge runs them on a loop of its own rather
    than on whichever one was current where the provider was constructed.
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
    """A provider shared between loops, such as an async caller's and the sync API's runner,
    needs a client per loop; within one loop it must reuse the channel rather than rebuild it.
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
    """Loops still come and go, from the sync API's nested cases and from async callers that
    close their own, so the per-loop cache has to shed the dead ones. Left alone it would hold
    a live grpc channel, and its socket, per dead loop for the whole life of the provider.
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
