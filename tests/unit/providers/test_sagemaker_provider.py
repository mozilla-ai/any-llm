import json

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


def _event(payload: dict) -> dict:
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
