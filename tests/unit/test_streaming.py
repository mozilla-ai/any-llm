from collections.abc import AsyncIterator
from typing import Any

import pytest

from any_llm.types.completion import ChatCompletionChunk
from any_llm.utils.streaming import ChatCompletionAccumulator, aaccumulate_chunks, accumulate_chunks


def make_chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish_reason: str | None = None,
    index: int = 0,
    usage: dict[str, Any] | None = None,
    choices: bool = True,
    **fields: Any,
) -> ChatCompletionChunk:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{"index": index, "delta": delta or {}, "finish_reason": finish_reason}] if choices else [],
        **fields,
    }
    if usage is not None:
        payload["usage"] = usage
    return ChatCompletionChunk.model_validate(payload)


def tool_call_delta(index: int, **fields: Any) -> dict[str, Any]:
    function = {key: fields.pop(key) for key in ("name", "arguments") if key in fields}
    return {"tool_calls": [{"index": index, **fields, **({"function": function} if function else {})}]}


def test_text_is_concatenated_and_metadata_is_kept() -> None:
    chunks = [
        make_chunk({"role": "assistant", "content": "He"}, system_fingerprint="fp_1", service_tier="default"),
        make_chunk({"content": "llo"}),
        make_chunk({}, finish_reason="stop"),
        make_chunk(
            choices=False,
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        ),
    ]

    completion = accumulate_chunks(chunks)

    assert completion.object == "chat.completion"
    assert (completion.id, completion.model, completion.created) == ("chatcmpl-1", "test-model", 1700000000)
    assert completion.system_fingerprint == "fp_1"
    assert completion.service_tier == "default"
    assert completion.usage is not None
    assert completion.usage.total_tokens == 5
    choice = completion.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hello"
    assert choice.message.tool_calls is None
    assert choice.message.reasoning is None
    assert choice.message.extra_content is None


def test_tool_call_arguments_are_joined_per_index() -> None:
    chunks = [
        make_chunk(tool_call_delta(0, id="call_a", type="function", name="lookup", arguments="")),
        make_chunk(tool_call_delta(1, id="call_b", type="function", name="weather", arguments='{"city"')),
        make_chunk(tool_call_delta(0, arguments='{"q": ')),
        make_chunk(tool_call_delta(1, arguments=': "Paris"}')),
        make_chunk(tool_call_delta(0, arguments='"cats"}')),
        make_chunk({}, finish_reason="tool_calls"),
    ]

    completion = accumulate_chunks(chunks)

    message = completion.choices[0].message
    assert message.content is None
    assert message.tool_calls is not None
    assert [(call.id, call.function.name, call.function.arguments) for call in message.tool_calls] == [  # type: ignore[union-attr]
        ("call_a", "lookup", '{"q": "cats"}'),
        ("call_b", "weather", '{"city": "Paris"}'),
    ]


def test_tool_call_name_repeated_on_every_chunk_is_not_duplicated() -> None:
    chunks = [
        make_chunk(tool_call_delta(0, id="call_a", name="lookup", arguments='{"q"')),
        make_chunk(tool_call_delta(0, name="lookup", arguments=": 1}")),
        make_chunk({}, finish_reason="tool_calls"),
    ]

    message = accumulate_chunks(chunks).choices[0].message

    assert message.tool_calls is not None
    assert message.tool_calls[0].function.name == "lookup"  # type: ignore[union-attr]
    assert message.tool_calls[0].function.arguments == '{"q": 1}'  # type: ignore[union-attr]


def test_tool_call_name_split_across_chunks_is_joined() -> None:
    chunks = [
        make_chunk(tool_call_delta(0, id="call_a", name="get_")),
        make_chunk(tool_call_delta(0, name="weather", arguments="{}")),
        make_chunk({}, finish_reason="tool_calls"),
    ]

    message = accumulate_chunks(chunks).choices[0].message

    assert message.tool_calls is not None
    assert message.tool_calls[0].function.name == "get_weather"  # type: ignore[union-attr]


def test_tool_call_without_function_fragment_is_kept() -> None:
    chunks = [
        make_chunk(tool_call_delta(0, id="call_a", type="function")),
        make_chunk({}, finish_reason="tool_calls"),
    ]

    message = accumulate_chunks(chunks).choices[0].message

    assert message.tool_calls is not None
    call = message.tool_calls[0]
    assert (call.id, call.function.name, call.function.arguments) == ("call_a", "", "")  # type: ignore[union-attr]


def test_reasoning_and_refusal_are_concatenated() -> None:
    chunks = [
        make_chunk({"reasoning": "step one, "}),
        make_chunk({"reasoning": "step two"}),
        make_chunk({"refusal": "I can"}),
        make_chunk({"refusal": "not help"}, finish_reason="stop"),
    ]

    message = accumulate_chunks(chunks).choices[0].message

    assert message.reasoning is not None
    assert message.reasoning.content == "step one, step two"
    assert message.refusal == "I cannot help"


def test_extra_content_is_merged_on_message_and_tool_calls() -> None:
    chunks = [
        make_chunk({"extra_content": {"anthropic": {"signature": "sig-1"}, "other": {"a": 1}}}),
        make_chunk({"extra_content": {"anthropic": {"signature": "sig-2"}, "other": {"b": 2}}}),
        make_chunk(
            tool_call_delta(0, id="call_a", name="lookup", arguments="{}", extra_content={"google": {"x": "1"}}),
        ),
        make_chunk(
            {"tool_calls": [{"index": 0, "extra_content": {"google": {"thought_signature": "abc"}}}]},
            finish_reason="tool_calls",
        ),
    ]

    message = accumulate_chunks(chunks).choices[0].message

    assert message.extra_content == {"anthropic": {"signature": "sig-2"}, "other": {"a": 1, "b": 2}}
    assert message.tool_calls is not None
    assert message.tool_calls[0].extra_content == {"google": {"x": "1", "thought_signature": "abc"}}  # type: ignore[union-attr]


def test_images_are_collected() -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    chunks = [
        make_chunk({"images": [image]}),
        make_chunk({"images": [image]}, finish_reason="stop"),
    ]

    message = accumulate_chunks(chunks).choices[0].message

    assert message.images is not None
    assert [item.image_url.url for item in message.images] == ["data:image/png;base64,AAAA"] * 2


def test_logprobs_are_collected() -> None:
    def logprobs(token: str) -> dict[str, Any]:
        return {"content": [{"token": token, "logprob": -0.1, "bytes": None, "top_logprobs": []}], "refusal": None}

    chunks = [
        ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "delta": {"content": token}, "logprobs": logprobs(token), "finish_reason": stop}
                ],
            }
        )
        for token, stop in (("a", None), ("b", "stop"))
    ]

    choice = accumulate_chunks(chunks).choices[0]

    assert choice.logprobs is not None
    assert choice.logprobs.content is not None
    assert [entry.token for entry in choice.logprobs.content] == ["a", "b"]
    assert choice.logprobs.refusal is None


def test_multiple_choices_are_kept_apart_and_sorted() -> None:
    chunks = [
        make_chunk({"content": "second"}, index=1),
        make_chunk({"content": "first"}, index=0),
        make_chunk({"content": " one"}, index=0, finish_reason="stop"),
        make_chunk({"content": " two"}, index=1, finish_reason="length"),
    ]

    completion = accumulate_chunks(chunks)

    assert [(choice.index, choice.finish_reason, choice.message.content) for choice in completion.choices] == [
        (0, "stop", "first one"),
        (1, "length", "second two"),
    ]


def test_message_dump_replays_like_a_non_streamed_response() -> None:
    chunks = [
        make_chunk({"role": "assistant", "reasoning": "think"}),
        make_chunk(
            tool_call_delta(0, id="call_a", type="function", name="lookup", arguments="{}", extra_content={"g": 1}),
        ),
        make_chunk({"extra_content": {"deepseek": {"reasoning_content": "think"}}}, finish_reason="tool_calls"),
    ]

    message = accumulate_chunks(chunks).choices[0].message

    assert message.model_dump(exclude_none=True, exclude={"reasoning"}) == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
                "extra_content": {"g": 1},
            }
        ],
        "extra_content": {"deepseek": {"reasoning_content": "think"}},
    }


def test_completion_requires_chunks() -> None:
    with pytest.raises(ValueError, match="No chunks"):
        _ = ChatCompletionAccumulator().completion


def test_completion_requires_finish_reason() -> None:
    accumulator = ChatCompletionAccumulator()
    accumulator.add(make_chunk({"content": "partial"}))

    with pytest.raises(ValueError, match="Choice 0 has no finish_reason"):
        _ = accumulator.completion


@pytest.mark.asyncio
async def test_aaccumulate_chunks_consumes_an_async_stream() -> None:
    async def stream() -> AsyncIterator[ChatCompletionChunk]:
        yield make_chunk({"content": "a"})
        yield make_chunk({"content": "b"}, finish_reason="stop")

    completion = await aaccumulate_chunks(stream())

    assert completion.choices[0].message.content == "ab"
