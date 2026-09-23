import base64
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.utils import (
    CodeExecutionState,
    _convert_messages,
    _convert_response_to_response_dict,
    _create_openai_chunk_from_google_chunk,
)
from any_llm.types.completion import ChatCompletion, CompletionParams

CODE = "print(1 + 1)"


def _response(
    parts: list[types.Part], finish_reason: types.FinishReason | None = None
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=parts, role="model"), finish_reason=finish_reason)],
        model_version="gemini-2.5-flash",
    )


def _code_part(code_id: str | None = None) -> types.Part:
    return types.Part(executable_code=types.ExecutableCode(code=CODE, language=types.Language.PYTHON, id=code_id))


def _result_part(result_id: str | None = None, output: str | None = "2\n") -> types.Part:
    return types.Part(
        code_execution_result=types.CodeExecutionResult(outcome=types.Outcome.OUTCOME_OK, output=output, id=result_id)
    )


def test_convert_response_maps_code_execution_with_synthesized_ids() -> None:
    response = _response(
        [
            types.Part(text="Let me compute. "),
            _code_part(),
            _result_part(),
            _code_part(),
            _result_part(output=None),
            types.Part(text="The answer is 2."),
        ],
        types.FinishReason.STOP,
    )

    message = _convert_response_to_response_dict(response)["choices"][0]["message"]

    assert message["content"] == "Let me compute. The answer is 2."
    assert message["extra_content"] == {
        "google": {
            "code_execution": [
                {"type": "executable_code", "id": "code_exec_0", "language": "PYTHON", "code": CODE},
                {"type": "code_execution_result", "id": "code_exec_0", "outcome": "OUTCOME_OK", "output": "2\n"},
                {"type": "executable_code", "id": "code_exec_1", "language": "PYTHON", "code": CODE},
                {"type": "code_execution_result", "id": "code_exec_1", "outcome": "OUTCOME_OK", "output": ""},
            ]
        }
    }


def test_convert_response_keeps_gemini_code_execution_ids() -> None:
    response = _response([_code_part("abc"), _result_part("abc")], types.FinishReason.STOP)

    message = _convert_response_to_response_dict(response)["choices"][0]["message"]

    items = message["extra_content"]["google"]["code_execution"]
    assert [item["id"] for item in items] == ["abc", "abc"]


def test_convert_response_result_without_code_gets_a_synthesized_id() -> None:
    response = _response([_result_part()], types.FinishReason.STOP)

    message = _convert_response_to_response_dict(response)["choices"][0]["message"]

    assert message["extra_content"]["google"]["code_execution"][0]["id"] == "code_exec_0"


def test_convert_response_code_execution_coexists_with_thought_signature() -> None:
    response = _response(
        [_code_part(), _result_part(), types.Part(text="2", thought_signature=b"sig")], types.FinishReason.STOP
    )

    google = _convert_response_to_response_dict(response)["choices"][0]["message"]["extra_content"]["google"]

    assert google["thought_signature"] == base64.b64encode(b"sig").decode("utf-8")
    assert [item["type"] for item in google["code_execution"]] == ["executable_code", "code_execution_result"]


def test_convert_response_without_code_parts_leaves_extra_content_none() -> None:
    response = _response([types.Part(text="hi")], types.FinishReason.STOP)

    assert _convert_response_to_response_dict(response)["choices"][0]["message"]["extra_content"] is None


def _image_part(data: bytes = b"png") -> types.Part:
    return types.Part(inline_data=types.Blob(data=data, mime_type="image/png"))


def test_convert_response_maps_image_after_result_to_output_and_images() -> None:
    response = _response([_code_part("c"), _result_part("r"), _image_part()], types.FinishReason.STOP)

    message = _convert_response_to_response_dict(response)["choices"][0]["message"]

    encoded = base64.b64encode(b"png").decode()
    assert message["images"] == [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]
    assert message["extra_content"]["google"]["code_execution"][2] == {
        "type": "code_execution_output",
        "id": "r",
        "mime_type": "image/png",
        "data": encoded,
    }


def test_convert_response_output_before_result_takes_the_code_id() -> None:
    response = _response(
        [_code_part(), _result_part(), _code_part(), _image_part(), _result_part()], types.FinishReason.STOP
    )

    items = _convert_response_to_response_dict(response)["choices"][0]["message"]["extra_content"]["google"][
        "code_execution"
    ]

    assert [(item["type"], item["id"]) for item in items] == [
        ("executable_code", "code_exec_0"),
        ("code_execution_result", "code_exec_0"),
        ("executable_code", "code_exec_1"),
        ("code_execution_output", "code_exec_1"),
        ("code_execution_result", "code_exec_1"),
    ]


def test_convert_response_inline_image_without_preceding_code_is_not_an_output() -> None:
    response = _response([_image_part(), _result_part(), _image_part(b"b"), _code_part()], types.FinishReason.STOP)

    message = _convert_response_to_response_dict(response)["choices"][0]["message"]

    assert len(message["images"]) == 2
    assert [item["type"] for item in message["extra_content"]["google"]["code_execution"]] == [
        "code_execution_result",
        "executable_code",
    ]


def test_convert_response_without_code_keeps_images_and_no_extra_content() -> None:
    message = _convert_response_to_response_dict(_response([_image_part()], types.FinishReason.STOP))["choices"][0][
        "message"
    ]

    assert len(message["images"]) == 1
    assert message["extra_content"] is None


def test_streaming_output_in_a_later_chunk_pairs_with_its_result() -> None:
    state = CodeExecutionState()

    _create_openai_chunk_from_google_chunk(_response([_code_part(), _result_part()]), code_execution_state=state)
    chunk = _create_openai_chunk_from_google_chunk(
        _response([_image_part()], types.FinishReason.STOP), code_execution_state=state
    )

    delta = chunk.choices[0].delta
    assert delta.images is not None
    assert len(delta.images) == 1
    assert delta.extra_content == {
        "google": {
            "code_execution": [
                {
                    "type": "code_execution_output",
                    "id": "code_exec_0",
                    "mime_type": "image/png",
                    "data": base64.b64encode(b"png").decode(),
                }
            ]
        }
    }


def test_streaming_image_without_prior_code_is_not_an_output() -> None:
    chunk = _create_openai_chunk_from_google_chunk(
        _response([_image_part()]), code_execution_state=CodeExecutionState()
    )

    assert chunk.choices[0].delta.images is not None
    assert chunk.choices[0].delta.extra_content is None


def test_streaming_pairs_code_execution_ids_across_chunks() -> None:
    state = CodeExecutionState()

    first = _create_openai_chunk_from_google_chunk(_response([_code_part()]), code_execution_state=state)
    second = _create_openai_chunk_from_google_chunk(
        _response([_result_part(), types.Part(text="2")], types.FinishReason.STOP), code_execution_state=state
    )

    assert first.choices[0].delta.extra_content == {
        "google": {
            "code_execution": [{"type": "executable_code", "id": "code_exec_0", "language": "PYTHON", "code": CODE}]
        }
    }
    assert second.choices[0].delta.extra_content == {
        "google": {
            "code_execution": [
                {"type": "code_execution_result", "id": "code_exec_0", "outcome": "OUTCOME_OK", "output": "2\n"}
            ]
        }
    }
    assert second.choices[0].delta.content == "2"


def test_streaming_chunk_without_code_parts_leaves_extra_content_none() -> None:
    chunk = _create_openai_chunk_from_google_chunk(_response([types.Part(text="hi")]))

    assert chunk.choices[0].delta.extra_content is None


def test_streaming_code_execution_merges_with_thought_signature() -> None:
    chunk = _create_openai_chunk_from_google_chunk(
        _response([_code_part("x"), types.Part(text="", thought_signature=b"s")])
    )

    assert chunk.choices[0].delta.extra_content == {
        "google": {
            "thought_signature": base64.b64encode(b"s").decode("utf-8"),
            "code_execution": [{"type": "executable_code", "id": "x", "language": "PYTHON", "code": CODE}],
        }
    }


async def _aiter(items: list[types.GenerateContentResponse]) -> AsyncIterator[types.GenerateContentResponse]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_acompletion_stream_threads_code_execution_state() -> None:
    chunks = [_response([_code_part()]), _response([_result_part()], types.FinishReason.STOP)]

    with patch("any_llm.providers.gemini.gemini.genai.Client") as mock_genai:
        mock_genai.return_value.aio.models.generate_content_stream = AsyncMock(return_value=_aiter(chunks))
        provider = GeminiProvider(api_key="test-api-key")
        result = await provider._acompletion(
            CompletionParams(model_id="gemini-2.5-flash", messages=[{"role": "user", "content": "1+1"}], stream=True)
        )
        assert not isinstance(result, ChatCompletion)
        ids = [
            item["id"]
            async for chunk in result
            if chunk.choices[0].delta.extra_content
            for item in chunk.choices[0].delta.extra_content["google"]["code_execution"]
        ]

    assert ids == ["code_exec_0", "code_exec_0"]


def test_convert_messages_rebuilds_code_execution_parts_in_order() -> None:
    response = _response([_code_part("gem-1"), _result_part("gem-1"), types.Part(text="2")], types.FinishReason.STOP)
    message = _convert_response_to_response_dict(response)["choices"][0]["message"]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "1+1"},
        {
            "role": "assistant",
            "content": message["content"],
            "extra_content": message["extra_content"],
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
    ]

    contents, _ = _convert_messages(messages)

    parts = contents[1].parts
    assert parts is not None
    assert parts[0].executable_code == types.ExecutableCode(code=CODE, language=types.Language.PYTHON, id="gem-1")
    assert parts[1].code_execution_result == types.CodeExecutionResult(
        outcome=types.Outcome.OUTCOME_OK, output="2\n", id="gem-1"
    )
    assert parts[2].text == "2"
    assert parts[3].function_call is not None


def test_convert_messages_drops_synthesized_code_execution_ids() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "2",
            "extra_content": {
                "google": {
                    "code_execution": [
                        {"type": "executable_code", "id": "code_exec_0", "language": "PYTHON", "code": CODE},
                        {"type": "code_execution_result", "id": 7, "outcome": "OUTCOME_OK", "output": "2"},
                    ]
                }
            },
        }
    ]

    parts = _convert_messages(messages)[0][0].parts

    assert parts is not None
    assert parts[0].executable_code is not None
    assert parts[0].executable_code.id is None
    assert parts[1].code_execution_result is not None
    assert parts[1].code_execution_result.id is None


def test_convert_messages_does_not_replay_code_execution_outputs() -> None:
    response = _response([_code_part(), _result_part(), _image_part(), types.Part(text="2")], types.FinishReason.STOP)
    message = _convert_response_to_response_dict(response)["choices"][0]["message"]
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": message["content"], "extra_content": message["extra_content"]}
    ]

    parts = _convert_messages(messages)[0][0].parts

    assert parts is not None
    assert [part.inline_data for part in parts] == [None, None, None]
    assert parts[0].executable_code is not None
    assert parts[1].code_execution_result is not None
    assert parts[2].text == "2"


@pytest.mark.parametrize(
    "code_execution",
    [
        None,
        "not a list",
        ["not a dict", {"type": "unknown"}, {"type": "executable_code", "code": ["not", "a", "string"]}],
    ],
)
def test_convert_messages_skips_malformed_code_execution_items(code_execution: Any) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "2", "extra_content": {"google": {"code_execution": code_execution}}}
    ]

    parts = _convert_messages(messages)[0][0].parts

    assert parts is not None
    assert len(parts) == 1
    assert parts[0].text == "2"


def test_convert_messages_without_google_extra_content_adds_no_code_parts() -> None:
    messages: list[dict[str, Any]] = [{"role": "assistant", "content": "2", "extra_content": {"other": {}}}]

    parts = _convert_messages(messages)[0][0].parts

    assert parts is not None
    assert [part.text for part in parts] == ["2"]


def test_convert_response_passes_through_non_enum_language_and_missing_fields() -> None:
    code = types.ExecutableCode.model_construct(code=None, language="RUST", id=None)
    result = types.CodeExecutionResult.model_construct(outcome=None, output=None, id=None)
    response = _response(
        [types.Part(executable_code=code), types.Part(code_execution_result=result)], types.FinishReason.STOP
    )

    items = _convert_response_to_response_dict(response)["choices"][0]["message"]["extra_content"]["google"][
        "code_execution"
    ]

    assert items == [
        {"type": "executable_code", "id": "code_exec_0", "language": "RUST", "code": ""},
        {"type": "code_execution_result", "id": "code_exec_0", "outcome": "OUTCOME_UNSPECIFIED", "output": ""},
    ]
