import json
from typing import Any
from unittest.mock import patch

import pytest

from any_llm.providers.sagemaker.sagemaker import SagemakerProvider
from any_llm.providers.sagemaker.utils import _convert_params, _create_openai_chunk_from_sagemaker_chunk
from any_llm.types.completion import CompletionParams


def test_per_request_timeout_is_declared_unsupported() -> None:
    """SageMaker serializes completion kwargs into the request body, so the base class rejects a timeout."""
    assert SagemakerProvider.TIMEOUT_SUPPORT == "unsupported"


def test_convert_params_forwards_zero_temperature() -> None:
    """temperature=0.0 asks for greedy decoding and must not be dropped as a falsy value."""
    result = _convert_params(
        CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "hi"}], temperature=0.0),
        {},
    )

    assert result["temperature"] == 0.0


def test_convert_params_forwards_zero_top_p() -> None:
    """top_p=0.0 is inside the documented range, so it must reach the request body."""
    result = _convert_params(
        CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "hi"}], top_p=0.0),
        {},
    )

    assert result["top_p"] == 0.0


def test_convert_params_omits_unset_sampling_params() -> None:
    """Params the caller never set stay absent so the model applies its own defaults."""
    result = _convert_params(
        CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "hi"}]),
        {},
    )

    assert "temperature" not in result
    assert "top_p" not in result


def _event(payload: dict[str, Any]) -> dict[str, Any]:
    return {"PayloadPart": {"Bytes": json.dumps(payload).encode("utf-8")}}


def test_stream_chunk_preserves_tool_call_delta() -> None:
    event = _event(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    tool_calls = chunk.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city": "Paris"}'


def test_stream_chunk_argument_fragment_and_tool_calls_finish_reason() -> None:
    fragment = _event({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}}]})
    chunk = _create_openai_chunk_from_sagemaker_chunk(fragment, "m")
    assert chunk is not None
    tool_calls = chunk.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].id is None
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.arguments == '"Paris"}'

    final = _event({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    last = _create_openai_chunk_from_sagemaker_chunk(final, "m")
    assert last is not None
    assert last.choices[0].finish_reason == "tool_calls"
    assert last.choices[0].delta.tool_calls is None


def test_stream_chunk_without_function_keeps_index() -> None:
    event = _event({"choices": [{"delta": {"tool_calls": [{"id": "call_2"}]}}]})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    tool_calls = chunk.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].index == 0
    assert tool_calls[0].function is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("length", "length"), ("tool_calls", "tool_calls"), ("stop", "stop"), ("content_filter", "stop"), (None, None)],
)
def test_stream_chunk_maps_choices_finish_reason(raw: str | None, expected: str | None) -> None:
    event = _event({"choices": [{"delta": {"content": "hi"}, "finish_reason": raw}]})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    assert chunk.choices[0].finish_reason == expected
    assert chunk.choices[0].delta.content == "hi"
    assert chunk.choices[0].delta.tool_calls is None


def test_stream_chunk_preserves_nonzero_tool_call_index() -> None:
    event = _event({"choices": [{"delta": {"tool_calls": [{"index": 2, "function": {"arguments": "{}"}}]}}]})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    tool_calls = chunk.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].index == 2


def test_stream_chunk_keeps_supplied_tool_call_type_without_function() -> None:
    event = _event({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_3", "type": "function"}]}}]})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    tool_calls = chunk.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function is None


def test_stream_chunk_tool_call_type_is_none_without_type_or_function() -> None:
    event = _event({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_4"}]}}]})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    tool_calls = chunk.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].type is None


def test_stream_chunk_is_finished_keeps_tool_calls_finish_reason() -> None:
    event = _event({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "is_finished": True})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    assert chunk.choices[0].finish_reason == "tool_calls"


def test_stream_chunk_is_finished_defaults_to_stop() -> None:
    event = _event({"token": {"text": "done"}, "is_finished": True})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    assert chunk.choices[0].finish_reason == "stop"


def test_stream_chunk_uses_payload_id_and_created() -> None:
    event = _event({"id": "chatcmpl-abc", "created": 1700000000, "choices": [{"delta": {"content": "hi"}}]})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    assert chunk.id == "chatcmpl-abc"
    assert chunk.created == 1700000000


def test_stream_chunk_falls_back_to_generated_id_and_created() -> None:
    event = _event({"choices": [{"delta": {"content": "hi"}}]})
    chunk = _create_openai_chunk_from_sagemaker_chunk(event, "m")
    assert chunk is not None
    assert chunk.id.startswith("chatcmpl-")
    assert chunk.created > 0


def test_convert_params_keeps_system_messages_in_messages() -> None:
    """The OpenAI chat schema served by SageMaker containers reads system prompts from `messages` only."""
    messages = [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "Be brief."},
    ]
    result = _convert_params(CompletionParams(model_id="model-id", messages=messages), {})

    assert result["messages"] == messages
    assert "system" not in result


def _stream_with_body(parts: list[bytes], **params: Any) -> tuple[list[Any], dict[str, Any]]:
    """Run a streaming completion against a mocked runtime; return the chunks and the request body."""
    with patch.object(SagemakerProvider, "_verify_and_set_api_key", return_value=None), patch("boto3.client"):
        provider = SagemakerProvider()
    provider.client.invoke_endpoint_with_response_stream.return_value = {
        "Body": [{"PayloadPart": {"Bytes": part}} for part in parts]
    }

    result = provider._completion(
        CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "hi"}], stream=True, **params)
    )
    chunks = list(result)
    body = json.loads(provider.client.invoke_endpoint_with_response_stream.call_args.kwargs["Body"])
    return chunks, body


def test_stream_request_body_asks_the_container_to_stream() -> None:
    _, body = _stream_with_body([], stream_options={"include_usage": True})

    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_stream_parses_server_sent_events_split_across_payload_parts() -> None:
    chunks, _ = _stream_with_body(
        [
            b'data: {"choices": [{"index": 0, "delta": {"content": "Hel"}}]}\n\ndata: {"choices": [{"ind',
            b'ex": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]}\n\n',
            b'data: {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
    assert chunks[1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 5


@pytest.mark.parametrize(
    "parts",
    [
        [b'{"outputs": [{"text": "a"}]}\n{"outputs": [{"text": "b"}]}\n'],
        [b'{"outputs": ', b'[{"text": "a"}]}\n{"outputs": [{"text": "b"}]}'],
        [b'{"outputs": [{"text": "a"}]}', b'{"outputs": [{"text": "b"}]}'],
    ],
)
def test_stream_parses_json_lines(parts: list[bytes]) -> None:
    chunks, _ = _stream_with_body(parts)

    assert [chunk.choices[0].delta.content for chunk in chunks] == ["a", "b"]
