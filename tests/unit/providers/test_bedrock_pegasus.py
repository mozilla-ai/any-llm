"""Tests for chat with the TwelveLabs Pegasus video models on Bedrock through InvokeModel."""

import io
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypeAlias
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.exceptions import InvalidRequestError, UnsupportedParameterError
from any_llm.providers.bedrock import BedrockProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, ParsedChatCompletion

PEGASUS = "us.twelvelabs.pegasus-1-5-v1:0"
LEGACY_PEGASUS = "us.twelvelabs.pegasus-1-2-v1:0"
VIDEO_B64 = "AAAAIGZ0eXBpc29t"
VIDEO_PART = {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{VIDEO_B64}"}}
REQUEST_ID = "req-123"

Messages: TypeAlias = list[dict[str, Any] | ChatCompletionMessage]


class Summary(BaseModel):
    summary: str


@dataclass
class Scene:
    title: str


def _messages(text: str = "Describe the video.") -> Messages:
    return [{"role": "user", "content": [{"type": "text", "text": text}, VIDEO_PART]}]


def _provider(body: dict[str, Any] | None = None) -> tuple[BedrockProvider, Mock]:
    client = Mock()
    payload = body if body is not None else {"message": "A test pattern.", "stopReason": "stop"}
    client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps(payload).encode()),
        "ResponseMetadata": {"RequestId": REQUEST_ID},
    }
    return BedrockProvider(client=client), client


def _stream_provider(events: list[dict[str, Any]]) -> tuple[BedrockProvider, Mock, Mock]:
    client = Mock()
    stream = Mock()
    stream.__iter__ = Mock(return_value=iter([{"chunk": {"bytes": json.dumps(event).encode()}} for event in events]))
    client.invoke_model_with_response_stream.return_value = {
        "body": stream,
        "ResponseMetadata": {"RequestId": REQUEST_ID},
    }
    return BedrockProvider(client=client), client, stream


def _sent_body(method: Mock) -> dict[str, Any]:
    return json.loads(method.call_args.kwargs["body"])  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_pegasus_sends_the_prompt_and_video_to_invoke_model() -> None:
    provider, client = _provider(
        {"message": "A test pattern.", "stopReason": "stop", "input_tokens": 165, "output_tokens": 20}
    )

    result = await provider.acompletion(PEGASUS, _messages(), temperature=0.1, max_tokens=512)

    request = client.invoke_model.call_args.kwargs
    assert request["modelId"] == PEGASUS
    assert request["contentType"] == "application/json"
    assert request["accept"] == "application/json"
    assert _sent_body(client.invoke_model) == {
        "inputPrompt": "Describe the video.",
        "mediaSource": {"base64String": VIDEO_B64},
        "temperature": 0.1,
        "maxOutputTokens": 512,
    }
    client.converse.assert_not_called()
    assert isinstance(result, ChatCompletion)
    assert result.id == REQUEST_ID
    assert result.model == PEGASUS
    assert result.choices[0].message.content == "A test pattern."
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage is not None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (165, 20, 185)


@pytest.mark.asyncio
async def test_pegasus_reads_the_legacy_finish_reason_and_reports_no_usage_without_counts() -> None:
    provider, _ = _provider({"message": "A test pattern...", "finishReason": "length"})

    result = await provider.acompletion(LEGACY_PEGASUS, _messages())

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].finish_reason == "length"
    assert result.usage is None


@pytest.mark.asyncio
async def test_pegasus_sends_an_s3_video_with_its_bucket_owner() -> None:
    provider, client = _provider()
    video = {"type": "video_url", "video_url": {"url": "s3://bucket/video.mp4", "bucket_owner": "123456789012"}}

    await provider.acompletion(PEGASUS, [{"role": "user", "content": [video, {"type": "text", "text": "Hi"}]}])

    assert _sent_body(client.invoke_model)["mediaSource"] == {
        "s3Location": {"uri": "s3://bucket/video.mp4", "bucketOwner": "123456789012"}
    }


@pytest.mark.asyncio
async def test_pegasus_1_2_sends_an_s3_video_without_a_bucket_owner() -> None:
    provider, client = _provider()
    video = {"type": "video_url", "video_url": {"url": "s3://bucket/video.mp4"}}

    await provider.acompletion(LEGACY_PEGASUS, [{"role": "user", "content": [video, {"type": "text", "text": "Hi"}]}])

    assert _sent_body(client.invoke_model)["mediaSource"] == {"s3Location": {"uri": "s3://bucket/video.mp4"}}


@pytest.mark.asyncio
async def test_pegasus_1_5_rejects_an_s3_video_without_a_bucket_owner() -> None:
    provider, client = _provider()
    video = {"type": "video_url", "video_url": {"url": "s3://bucket/video.mp4"}}

    with pytest.raises(InvalidRequestError, match="bucket_owner"):
        await provider.acompletion(PEGASUS, [{"role": "user", "content": [video, {"type": "text", "text": "Hi"}]}])

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_pegasus_joins_system_and_user_text_into_the_prompt() -> None:
    provider, client = _provider()
    messages: Messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "developer", "content": [{"type": "text", "text": "Use English."}]},
        {"role": "user", "content": "Look at this."},
        {"role": "user", "content": [VIDEO_PART, {"type": "text", "text": "What happens?"}]},
    ]

    await provider.acompletion(PEGASUS, messages)

    assert (
        _sent_body(client.invoke_model)["inputPrompt"]
        == "Answer briefly.\n\nUse English.\n\nLook at this.\n\nWhat happens?"
    )


@pytest.mark.asyncio
async def test_pegasus_sends_a_pydantic_response_format_as_a_json_schema() -> None:
    provider, client = _provider({"message": '{"summary": "bars"}', "stopReason": "stop"})

    result = await provider.acompletion(PEGASUS, _messages(), response_format=Summary)

    assert _sent_body(client.invoke_model)["responseFormat"] == {
        "type": "json_schema",
        "jsonSchema": Summary.model_json_schema(),
    }
    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed == Summary(summary="bars")


@pytest.mark.asyncio
async def test_pegasus_sends_a_dataclass_response_format_as_a_json_schema() -> None:
    provider, client = _provider({"message": '{"title": "bars"}', "stopReason": "stop"})

    result = await provider.acompletion(PEGASUS, _messages(), response_format=Scene)

    assert _sent_body(client.invoke_model)["responseFormat"]["jsonSchema"]["required"] == ["title"]
    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed == Scene(title="bars")


@pytest.mark.asyncio
async def test_pegasus_sends_a_json_schema_dict_response_format() -> None:
    provider, client = _provider()
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}

    await provider.acompletion(
        PEGASUS, _messages(), response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}
    )

    assert _sent_body(client.invoke_model)["responseFormat"] == {"type": "json_schema", "jsonSchema": schema}


@pytest.mark.asyncio
async def test_pegasus_1_2_takes_the_schema_without_a_type() -> None:
    provider, client = _provider({"message": '{"summary": "bars"}', "finishReason": "stop"})

    await provider.acompletion(LEGACY_PEGASUS, _messages(), response_format=Summary)

    assert _sent_body(client.invoke_model)["responseFormat"] == {"jsonSchema": Summary.model_json_schema()}


@pytest.mark.asyncio
async def test_pegasus_leaves_out_a_text_response_format() -> None:
    provider, client = _provider()

    await provider.acompletion(PEGASUS, _messages(), response_format={"type": "text"})

    assert "responseFormat" not in _sent_body(client.invoke_model)


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_object"},
        {"type": "json_schema", "json_schema": {"name": "s"}},
        {"type": "json_schema", "json_schema": "invalid"},
        str,
    ],
)
@pytest.mark.asyncio
async def test_pegasus_rejects_a_response_format_without_a_schema(response_format: Any) -> None:
    provider, client = _provider()

    with pytest.raises(UnsupportedParameterError, match="response_format"):
        await provider.acompletion(PEGASUS, _messages(), response_format=response_format)

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_pegasus_merges_additional_model_request_fields_and_forwards_other_arguments() -> None:
    provider, client = _provider()

    await provider.acompletion(
        PEGASUS,
        _messages(),
        additionalModelRequestFields={"start_time": 1.0, "end_time": 4.0},
        performanceConfigLatency="standard",
    )

    body = _sent_body(client.invoke_model)
    assert (body["start_time"], body["end_time"]) == (1.0, 4.0)
    assert client.invoke_model.call_args.kwargs["performanceConfigLatency"] == "standard"


@pytest.mark.parametrize(
    ("fields", "match"),
    [({"analysis_mode": "time_based_metadata"}, "analysis_mode"), ({"prompt_v2": {"input_text": "Hi"}}, "prompt_v2")],
)
@pytest.mark.asyncio
async def test_pegasus_rejects_request_fields_that_replace_the_chat_prompt(fields: dict[str, Any], match: str) -> None:
    provider, client = _provider()

    with pytest.raises(UnsupportedParameterError, match=match):
        await provider.acompletion(PEGASUS, _messages(), additionalModelRequestFields=fields)

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_pegasus_keeps_its_own_request_fields_over_forwarded_arguments() -> None:
    provider, client = _provider()

    await provider.acompletion(PEGASUS, _messages(), modelId="us.amazon.nova-lite-v1:0", body="{}")

    request = client.invoke_model.call_args.kwargs
    assert request["modelId"] == PEGASUS
    assert _sent_body(client.invoke_model)["inputPrompt"] == "Describe the video."


@pytest.mark.parametrize("field", ["inputPrompt", "mediaSource", "responseFormat", "temperature"])
@pytest.mark.asyncio
async def test_pegasus_rejects_request_fields_the_chat_request_already_sets(field: str) -> None:
    provider, client = _provider()

    with pytest.raises(InvalidRequestError, match=field):
        await provider.acompletion(
            PEGASUS,
            _messages(),
            temperature=0.1,
            response_format=Summary,
            additionalModelRequestFields={field: "replacement"},
        )

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_pegasus_takes_the_general_analysis_mode() -> None:
    provider, client = _provider()

    await provider.acompletion(PEGASUS, _messages(), additionalModelRequestFields={"analysis_mode": "general"})

    assert _sent_body(client.invoke_model)["analysis_mode"] == "general"


@pytest.mark.asyncio
async def test_pegasus_reports_a_guardrail_intervention_as_filtered() -> None:
    provider, _ = _provider(
        {"message": "Sorry, I cannot help.", "finishReason": "stop", "amazon-bedrock-guardrailAction": "INTERVENED"}
    )

    result = await provider.acompletion(LEGACY_PEGASUS, _messages(), guardrailIdentifier="g", guardrailVersion="1")

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_pegasus_reports_a_guardrail_intervention_without_a_stop_reason_as_filtered() -> None:
    provider, _ = _provider({"message": "Sorry.", "amazon-bedrock-guardrailAction": "INTERVENED"})

    result = await provider.acompletion(LEGACY_PEGASUS, _messages())

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_pegasus_stream_reports_a_guardrail_intervention_on_the_last_chunk() -> None:
    provider, _, _ = _stream_provider(
        [
            {"message": "Sorry.", "finishReason": "", "amazon-bedrock-guardrailAction": "INTERVENED"},
            {"message": "", "finishReason": "stop", "amazon-bedrock-guardrailAction": "INTERVENED"},
        ]
    )

    result = await provider.acompletion(LEGACY_PEGASUS, _messages(), stream=True)
    chunks = [chunk async for chunk in result]

    assert [chunk.choices[0].finish_reason for chunk in chunks] == [None, "content_filter"]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("tools", [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]),
        ("tool_choice", "required"),
        ("top_p", 0.5),
        ("stop", "END"),
        ("reasoning_effort", "high"),
        ("seed", 7),
        ("logprobs", True),
        ("top_logprobs", 2),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("logit_bias", {"50256": -100.0}),
    ],
)
@pytest.mark.asyncio
async def test_pegasus_rejects_options_it_does_not_take(option: str, value: Any) -> None:
    provider, client = _provider()

    with pytest.raises(UnsupportedParameterError, match=option):
        await provider.acompletion(PEGASUS, _messages(), **{option: value})

    client.invoke_model.assert_not_called()


@pytest.mark.parametrize("role", ["assistant", "tool"])
@pytest.mark.asyncio
async def test_pegasus_rejects_a_conversation_with_other_turns(role: str) -> None:
    provider, client = _provider()
    messages: Messages = [*_messages(), {"role": role, "content": "Earlier answer.", "tool_call_id": "t"}]

    with pytest.raises(UnsupportedParameterError, match="one prompt about one video"):
        await provider.acompletion(PEGASUS, messages)

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_pegasus_rejects_other_content_parts() -> None:
    provider, client = _provider()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

    with pytest.raises(UnsupportedParameterError, match="image_url"):
        await provider.acompletion(PEGASUS, [{"role": "user", "content": [VIDEO_PART, image]}])

    client.invoke_model.assert_not_called()


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([{"role": "user", "content": "No video here."}], "one video"),
        ([{"role": "user", "content": [VIDEO_PART, VIDEO_PART, {"type": "text", "text": "Hi"}]}], "one video"),
        ([{"role": "user", "content": [VIDEO_PART]}], "prompt"),
        ([{"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "video_url"}]}], "video_url.url"),
        (
            [{"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "video_url", "video_url": "x"}]}],
            "video_url.url",
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi"},
                        {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}},
                    ],
                }
            ],
            "base64 data URL or an s3:// URI",
        ),
        ([{"role": "user", "content": 42}], "content"),
        ([{"role": "user", "content": ["Hi", VIDEO_PART]}], "content part"),
        ([{"role": "user", "content": [{"type": "text", "text": 7}, VIDEO_PART]}], "text"),
    ],
)
@pytest.mark.asyncio
async def test_pegasus_rejects_requests_it_cannot_send(messages: Messages, match: str) -> None:
    provider, client = _provider()

    with pytest.raises(InvalidRequestError, match=match):
        await provider.acompletion(PEGASUS, messages)

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_pegasus_streams_text_then_the_finish_reason_then_usage() -> None:
    provider, client, stream = _stream_provider(
        [
            {"message": "", "stopReason": ""},
            {"message": "A test", "stopReason": ""},
            {"message": " pattern.", "stopReason": ""},
            {
                "message": "",
                "stopReason": "stop",
                "amazon-bedrock-invocationMetrics": {"inputTokenCount": 0, "outputTokenCount": 28},
            },
        ]
    )

    result = await provider.acompletion(PEGASUS, _messages(), stream=True)
    chunks = [chunk async for chunk in result]

    assert "body" in client.invoke_model_with_response_stream.call_args.kwargs
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert {chunk.id for chunk in chunks} == {REQUEST_ID}
    assert [chunk.choices[0].delta.content for chunk in chunks if chunk.choices] == ["A test", " pattern.", None]
    assert [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices] == [None, None, "stop"]
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert (chunks[-1].usage.prompt_tokens, chunks[-1].usage.completion_tokens) == (0, 28)
    stream.close.assert_called_once()


@pytest.mark.asyncio
async def test_pegasus_stream_maps_the_legacy_length_reason_without_metrics() -> None:
    provider, _, _ = _stream_provider([{"message": "A", "finishReason": ""}, {"message": "", "finishReason": "length"}])

    result = await provider.acompletion(LEGACY_PEGASUS, _messages(), stream=True)
    chunks = [chunk async for chunk in result]

    assert [chunk.choices[0].finish_reason for chunk in chunks] == [None, "length"]
    assert all(chunk.usage is None for chunk in chunks)


@pytest.mark.asyncio
async def test_pegasus_stream_closes_the_stream_when_a_read_fails() -> None:
    provider, _, stream = _stream_provider([])

    def failing_events() -> Iterator[dict[str, Any]]:
        yield {"chunk": {"bytes": json.dumps({"message": "A", "stopReason": ""}).encode()}}
        msg = "stream broke"
        raise RuntimeError(msg)

    stream.__iter__ = Mock(return_value=failing_events())

    result = await provider.acompletion(PEGASUS, _messages(), stream=True)
    with pytest.raises(RuntimeError, match="stream broke"):
        [chunk async for chunk in result]

    stream.close.assert_called_once()


@pytest.mark.asyncio
async def test_pegasus_builds_the_timeout_client_off_the_event_loop() -> None:
    provider, _ = _provider()
    on_main_thread: list[bool] = []
    build_client = provider._client_for_timeout

    def record(timeout: float | None) -> Any:
        on_main_thread.append(threading.current_thread() is threading.main_thread())
        return build_client(timeout)

    with patch.object(provider, "_client_for_timeout", side_effect=record):
        await provider.acompletion(PEGASUS, _messages())

    assert on_main_thread == [False]


@pytest.mark.asyncio
async def test_pegasus_rejects_a_timeout_on_a_custom_client() -> None:
    provider, client = _provider()

    with pytest.raises(UnsupportedParameterError, match="timeout"):
        await provider.acompletion(PEGASUS, _messages(), timeout=5)

    client.invoke_model.assert_not_called()


def test_pegasus_takes_response_format() -> None:
    assert BedrockProvider.supports_completion_response_format(PEGASUS) is True
    assert BedrockProvider.supports_completion_response_format("amazon.nova-lite-v1:0") is False


def test_pegasus_completes_synchronously() -> None:
    provider, _ = _provider()

    result = provider.completion(PEGASUS, _messages())

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "A test pattern."
