from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.providers.mistral.utils import _patch_messages
from any_llm.types.completion import CompletionParams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def test_patch_messages_noop_when_no_tool_before_user() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    out = _patch_messages(messages)
    assert out == messages


def test_patch_messages_inserts_assistant_ok_between_tool_and_user() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "a1", "tool_calls": [{}]},
        {"role": "tool", "content": "tool-output"},
        {"role": "user", "content": "next-question"},
    ]
    out = _patch_messages(messages)
    assert out == [
        {"role": "assistant", "content": "a1", "tool_calls": [{}]},
        {"role": "tool", "content": "tool-output"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "next-question"},
    ]


def test_patch_messages_multiple_insertions() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "a1", "tool_calls": [{}]},
        {"role": "tool", "content": "t1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2", "tool_calls": [{}]},
        {"role": "tool", "content": "t2"},
        {"role": "user", "content": "u2"},
    ]
    out = _patch_messages(messages)
    assert out == [
        {"role": "assistant", "content": "a1", "tool_calls": [{}]},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2", "tool_calls": [{}]},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "u2"},
    ]


def test_patch_messages_no_insertion_when_tool_at_end() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1", "tool_calls": [{}]},
        {"role": "tool", "content": "t"},
    ]
    out = _patch_messages(messages)
    assert out == messages


def test_patch_messages_no_insertion_when_next_not_user() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "a1", "tool_calls": [{}]},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u"},
    ]
    out = _patch_messages(messages)
    assert out == messages


def test_patch_messages_with_multiple_valid_tool_calls() -> None:
    """Test patching with multiple consecutive tool calls followed by a user message."""
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "a1", "tool_calls": [{}]},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2", "tool_calls": [{}]},
        {"role": "tool", "content": "t2"},
        {"role": "user", "content": "u1"},
    ]
    out = _patch_messages(messages)
    assert out == [
        {"role": "assistant", "content": "a1", "tool_calls": [{}]},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2", "tool_calls": [{}]},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "u1"},
    ]


@pytest.mark.parametrize(
    "reasoning",
    ["thought hard", {"content": "thought hard"}],
    ids=["serialized-string", "object-form"],
)
def test_patch_messages_rebuilds_thinking_chunk_for_replay(reasoning: Any) -> None:
    """Reasoning normalized onto `reasoning` must be folded back into a Mistral ThinkChunk."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "the answer", "reasoning": reasoning},
        {"role": "user", "content": "u2"},
    ]
    out = _patch_messages(messages)
    assert out[1]["content"] == [
        {"type": "thinking", "thinking": [{"type": "text", "text": "thought hard"}]},
        {"type": "text", "text": "the answer"},
    ]


def test_patch_messages_replays_thinking_signature_from_extra_content() -> None:
    """A signature captured off the response has to be replayed with the thinking block."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "the answer",
            "reasoning": "thought hard",
            "extra_content": {"mistral": {"signature": "sig-abc"}},
        },
    ]
    out = _patch_messages(messages)
    assert out[0]["content"][0] == {
        "type": "thinking",
        "thinking": [{"type": "text", "text": "thought hard"}],
        "signature": "sig-abc",
    }


def test_patch_messages_rebuilds_thinking_chunk_without_text_when_content_empty() -> None:
    """A reasoning-only assistant turn yields a thinking chunk and no empty text chunk."""
    messages: list[dict[str, Any]] = [{"role": "assistant", "content": None, "reasoning": "thought hard"}]
    out = _patch_messages(messages)
    assert out[0]["content"] == [{"type": "thinking", "thinking": [{"type": "text", "text": "thought hard"}]}]


def test_patch_messages_leaves_prechunked_content_untouched() -> None:
    """Caller-supplied provider-native chunks are trusted rather than merged into."""
    prechunked = [{"type": "text", "text": "already chunked"}]
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": prechunked, "reasoning": "thought hard"},
    ]
    out = _patch_messages(messages)
    assert out[0]["content"] == prechunked


def test_patch_messages_ignores_unusable_reasoning_shapes() -> None:
    """An absent or malformed `reasoning` value leaves the message alone."""
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2", "reasoning": None},
        {"role": "assistant", "content": "a3", "reasoning": {"nope": "wrong-key"}},
        {"role": "assistant", "content": "a4", "reasoning": ""},
    ]
    out = _patch_messages(messages)
    assert out == messages


def test_patch_messages_does_not_mutate_input_messages() -> None:
    """Rebuilding content must not write back into the caller's message dicts."""
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "the answer", "reasoning": "thought hard"},
    ]
    _patch_messages(messages)
    assert messages == [{"role": "assistant", "content": "the answer", "reasoning": "thought hard"}]


def test_patch_messages_rebuilds_thinking_and_still_inserts_assistant_ok() -> None:
    """The thinking rebuild composes with the tool-message "OK" insertion."""
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "a1", "reasoning": "thought hard", "tool_calls": [{}]},
        {"role": "tool", "content": "t1"},
        {"role": "user", "content": "u1"},
    ]
    out = _patch_messages(messages)
    assert out[0]["content"] == [
        {"type": "thinking", "thinking": [{"type": "text", "text": "thought hard"}]},
        {"type": "text", "text": "a1"},
    ]
    assert out[0]["tool_calls"] == [{}]
    assert out[2] == {"role": "assistant", "content": "OK"}


class StructuredOutput(BaseModel):
    foo: str
    bar: int


openai_json_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "StructuredOutput",
        "schema": {**StructuredOutput.model_json_schema(), "additionalProperties": False},
        "strict": True,
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_format",
    [
        StructuredOutput,
        openai_json_schema,
    ],
    ids=["pydantic_model", "openai_json_schema"],
)
async def test_response_format(response_format: Any) -> None:
    """Test that response_format is properly converted for both Pydantic and dict formats."""
    pytest.importorskip("mistralai")
    from mistralai.client.models.responseformat import ResponseFormat

    from any_llm.providers.mistral.mistral import MistralProvider

    with (
        patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral,
        patch("any_llm.providers.mistral.mistral._create_mistral_completion_from_response") as mock_converter,
    ):
        provider = MistralProvider(api_key="test-api-key")

        mocked_mistral.return_value.chat.complete_async = AsyncMock(return_value=Mock())
        mock_converter.return_value = Mock()

        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=response_format,
            ),
        )

        completion_call_kwargs = mocked_mistral.return_value.chat.complete_async.call_args[1]
        assert "response_format" in completion_call_kwargs

        response_format_arg = completion_call_kwargs["response_format"]
        # response_format_from_pydantic_model returns a dict for Pydantic models;
        # dict inputs go through ResponseFormat.model_validate and return a model.
        if isinstance(response_format_arg, dict):
            assert response_format_arg["type"] == "json_schema"
            assert response_format_arg["json_schema"]["name"] == "StructuredOutput"
            assert response_format_arg["json_schema"]["strict"] is True
            assert response_format_arg["json_schema"]["schema"]["type"] == "object"
        else:
            assert isinstance(response_format_arg, ResponseFormat)
            assert response_format_arg.type == "json_schema"
            assert response_format_arg.json_schema.name == "StructuredOutput"  # type: ignore[union-attr]
            assert response_format_arg.json_schema.strict is True  # type: ignore[union-attr]

            expected_schema = {
                "properties": {
                    "foo": {"title": "Foo", "type": "string"},
                    "bar": {"title": "Bar", "type": "integer"},
                },
                "required": ["foo", "bar"],
                "title": "StructuredOutput",
                "type": "object",
                "additionalProperties": False,
            }
            assert response_format_arg.json_schema.schema_definition == expected_schema  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_response_format_dataclass() -> None:
    """Test that dataclass response_format is properly converted for Mistral."""
    from dataclasses import dataclass

    pytest.importorskip("mistralai")
    from mistralai.client.models.responseformat import ResponseFormat

    from any_llm.providers.mistral.mistral import MistralProvider

    @dataclass
    class DataclassOutput:
        foo: str
        bar: int

    with (
        patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral,
        patch("any_llm.providers.mistral.mistral._create_mistral_completion_from_response") as mock_converter,
    ):
        provider = MistralProvider(api_key="test-api-key")

        mocked_mistral.return_value.chat.complete_async = AsyncMock(return_value=Mock())
        mock_converter.return_value = Mock()

        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=DataclassOutput,
            ),
        )

        completion_call_kwargs = mocked_mistral.return_value.chat.complete_async.call_args[1]
        assert "response_format" in completion_call_kwargs

        response_format_arg = completion_call_kwargs["response_format"]
        if isinstance(response_format_arg, dict):
            assert response_format_arg["type"] == "json_schema"
            assert response_format_arg["json_schema"]["name"] == "DataclassOutput"
        else:
            assert isinstance(response_format_arg, ResponseFormat)
            assert response_format_arg.type == "json_schema"
            assert response_format_arg.json_schema.name == "DataclassOutput"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_user_parameter_excluded() -> None:
    """Test that the 'user' parameter is excluded when calling Mistral API."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with (
        patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral,
        patch("any_llm.providers.mistral.mistral._create_mistral_completion_from_response") as mock_converter,
    ):
        provider = MistralProvider(api_key="test-api-key")

        mocked_mistral.return_value.chat.complete_async = AsyncMock(return_value=Mock())
        mock_converter.return_value = Mock()

        # Call with 'user' parameter (OpenAI-compatible)
        await provider._acompletion(
            CompletionParams(
                model_id="mistral-small-latest",
                messages=[{"role": "user", "content": "Hello"}],
                user="user-123",  # This should be excluded
                temperature=0.7,  # This should be included
            ),
        )

        # Verify the call was made
        completion_call_kwargs = mocked_mistral.return_value.chat.complete_async.call_args[1]

        # Assert 'user' parameter is NOT passed to Mistral API
        assert "user" not in completion_call_kwargs, "The 'user' parameter should be excluded for Mistral API"

        # Assert other parameters are still passed correctly
        assert completion_call_kwargs["model"] == "mistral-small-latest"
        assert completion_call_kwargs["temperature"] == 0.7
        assert len(completion_call_kwargs["messages"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", "none"])
async def test_reasoning_effort_filtered_out(reasoning_effort: str) -> None:
    """Test that reasoning_effort 'auto' and 'none' are filtered from Mistral API calls."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with (
        patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral,
        patch("any_llm.providers.mistral.mistral._create_mistral_completion_from_response") as mock_converter,
    ):
        provider = MistralProvider(api_key="test-api-key")

        mocked_mistral.return_value.chat.complete_async = AsyncMock(return_value=Mock())
        mock_converter.return_value = Mock()

        await provider._acompletion(
            CompletionParams(
                model_id="mistral-small-latest",
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            ),
        )

        completion_call_kwargs = mocked_mistral.return_value.chat.complete_async.call_args[1]
        assert "reasoning_effort" not in completion_call_kwargs
        assert "prompt_mode" not in completion_call_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reasoning_effort",
    ["minimal", "low", "medium", "high", "xhigh", "max"],
)
@pytest.mark.parametrize("model_id", ["mistral-medium-3-5", "magistral-medium-latest"])
async def test_explicit_reasoning_effort_collapses_to_high(model_id: str, reasoning_effort: str) -> None:
    """Mistral accepts only "high" or "none", so every explicit any_llm effort sends "high".

    prompt_mode must never accompany it: Mistral rejects that param on every model, including
    the magistral ones it was originally documented for.
    """
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with (
        patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral,
        patch("any_llm.providers.mistral.mistral._create_mistral_completion_from_response") as mock_converter,
    ):
        provider = MistralProvider(api_key="test-api-key")

        mocked_mistral.return_value.chat.complete_async = AsyncMock(return_value=Mock())
        mock_converter.return_value = Mock()

        await provider._acompletion(
            CompletionParams(
                model_id=model_id,
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            ),
        )

        completion_call_kwargs = mocked_mistral.return_value.chat.complete_async.call_args[1]
        assert completion_call_kwargs["reasoning_effort"] == "high"
        assert "prompt_mode" not in completion_call_kwargs


@pytest.mark.asyncio
async def test_explicit_prompt_mode_kwarg_passes_through() -> None:
    """A caller-supplied prompt_mode still reaches the SDK, for callers on a model that takes it."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with (
        patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral,
        patch("any_llm.providers.mistral.mistral._create_mistral_completion_from_response") as mock_converter,
    ):
        provider = MistralProvider(api_key="test-api-key")

        mocked_mistral.return_value.chat.complete_async = AsyncMock(return_value=Mock())
        mock_converter.return_value = Mock()

        await provider._acompletion(
            CompletionParams(
                model_id="magistral-medium-latest",
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort="low",
            ),
            prompt_mode="reasoning",
        )

        completion_call_kwargs = mocked_mistral.return_value.chat.complete_async.call_args[1]
        assert completion_call_kwargs["prompt_mode"] == "reasoning"


@pytest.mark.asyncio
async def test_user_parameter_excluded_streaming() -> None:
    """Test that the 'user' parameter is excluded in streaming mode."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with (
        patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral,
    ):
        provider = MistralProvider(api_key="test-api-key")

        # Create a properly structured mock chunk
        mock_delta = Mock()
        mock_delta.content = "test"
        mock_delta.role = "assistant"
        mock_delta.tool_calls = None

        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_choice.index = 0
        mock_choice.finish_reason = None

        mock_chunk = Mock()
        mock_chunk.data = Mock()
        mock_chunk.data.choices = [mock_choice]
        mock_chunk.data.id = "test-id"
        mock_chunk.data.model = "mistral-small-latest"
        mock_chunk.data.created = 1234567890
        mock_chunk.data.usage = None  # No usage in streaming chunks

        # Mock streaming response
        async def mock_stream(*args: Any, **kwargs: Any) -> Any:
            async def async_iter() -> Any:
                yield mock_chunk

            return async_iter()

        mocked_mistral.return_value.chat.stream_async = AsyncMock(side_effect=mock_stream)

        # Call with 'user' parameter in streaming mode
        result = await provider._acompletion(
            CompletionParams(
                model_id="mistral-small-latest",
                messages=[{"role": "user", "content": "Hello"}],
                user="user-123",  # This should be excluded
                stream=True,
            ),
        )

        # Consume the stream to trigger the API call
        stream = cast("AsyncIterator[Any]", result)
        async for _ in stream:
            pass

        # Verify the call was made
        stream_call_kwargs = mocked_mistral.return_value.chat.stream_async.call_args[1]

        # Assert 'user' parameter is NOT passed to Mistral API
        assert "user" not in stream_call_kwargs, "The 'user' parameter should be excluded in streaming mode"


def test_convert_batch_job_to_openai_success_status() -> None:
    """Test converting a successful Mistral batch job to OpenAI format."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.utils import _convert_batch_job_to_openai

    mock_batch_job = Mock()
    mock_batch_job.id = "batch-123"
    mock_batch_job.input_files = ["file-abc"]
    mock_batch_job.endpoint = "/v1/chat/completions"
    mock_batch_job.status = Mock(value="SUCCESS")
    mock_batch_job.created_at = 1700000000
    mock_batch_job.total_requests = 10
    mock_batch_job.completed_requests = 10
    mock_batch_job.succeeded_requests = 9
    mock_batch_job.failed_requests = 1
    mock_batch_job.errors = []
    mock_batch_job.metadata = {"test": "value"}
    mock_batch_job.output_file = "output-file-123"
    mock_batch_job.error_file = None
    mock_batch_job.started_at = 1700000100
    mock_batch_job.completed_at = 1700000200

    result = _convert_batch_job_to_openai(mock_batch_job)

    assert result.id == "batch-123"
    assert result.input_file_id == "file-abc"
    assert result.endpoint == "/v1/chat/completions"
    assert result.status == "completed"
    assert result.created_at == 1700000000
    assert result.request_counts is not None
    assert result.request_counts.total == 10
    assert result.request_counts.completed == 10
    assert result.request_counts.failed == 1
    assert result.output_file_id == "output-file-123"
    assert result.in_progress_at == 1700000100
    assert result.completed_at == 1700000200
    assert result.metadata == {"test": "value"}


def test_convert_batch_job_to_openai_queued_status() -> None:
    """Test converting a queued Mistral batch job to OpenAI format."""
    pytest.importorskip("mistralai")
    from mistralai.client.types.basemodel import Unset

    from any_llm.providers.mistral.utils import _convert_batch_job_to_openai

    mock_batch_job = Mock()
    mock_batch_job.id = "batch-456"
    mock_batch_job.input_files = ["file-xyz"]
    mock_batch_job.endpoint = "/v1/embeddings"
    mock_batch_job.status = Mock(value="QUEUED")
    mock_batch_job.created_at = 1700000000
    mock_batch_job.total_requests = 5
    mock_batch_job.completed_requests = 0
    mock_batch_job.succeeded_requests = 0
    mock_batch_job.failed_requests = 0
    mock_batch_job.errors = []
    mock_batch_job.metadata = Unset()
    mock_batch_job.output_file = Unset()
    mock_batch_job.error_file = Unset()
    mock_batch_job.started_at = Unset()
    mock_batch_job.completed_at = Unset()

    result = _convert_batch_job_to_openai(mock_batch_job)

    assert result.id == "batch-456"
    assert result.status == "validating"
    assert result.output_file_id is None
    assert result.in_progress_at is None
    assert result.completed_at is None
    assert result.metadata is None


def test_convert_batch_job_status_mapping() -> None:
    """Test all status mappings from Mistral to OpenAI."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.utils import _MISTRAL_TO_OPENAI_STATUS_MAP

    expected_mappings = {
        "QUEUED": "validating",
        "RUNNING": "in_progress",
        "SUCCESS": "completed",
        "FAILED": "failed",
        "TIMEOUT_EXCEEDED": "expired",
        "CANCELLATION_REQUESTED": "cancelling",
        "CANCELLED": "cancelled",
    }

    for mistral_status, openai_status in expected_mappings.items():
        assert _MISTRAL_TO_OPENAI_STATUS_MAP[mistral_status] == openai_status


def test_convert_batch_jobs_list() -> None:
    """Test converting a list of Mistral batch jobs."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.utils import _convert_batch_jobs_list

    mock_job1 = Mock()
    mock_job1.id = "batch-1"
    mock_job1.input_files = ["file-1"]
    mock_job1.endpoint = "/v1/chat/completions"
    mock_job1.status = Mock(value="SUCCESS")
    mock_job1.created_at = 1700000000
    mock_job1.total_requests = 5
    mock_job1.completed_requests = 5
    mock_job1.succeeded_requests = 5
    mock_job1.failed_requests = 0
    mock_job1.errors = []
    mock_job1.metadata = None
    mock_job1.output_file = None
    mock_job1.error_file = None
    mock_job1.started_at = None
    mock_job1.completed_at = None

    mock_job2 = Mock()
    mock_job2.id = "batch-2"
    mock_job2.input_files = ["file-2"]
    mock_job2.endpoint = "/v1/embeddings"
    mock_job2.status = Mock(value="RUNNING")
    mock_job2.created_at = 1700001000
    mock_job2.total_requests = 10
    mock_job2.completed_requests = 3
    mock_job2.succeeded_requests = 3
    mock_job2.failed_requests = 0
    mock_job2.errors = []
    mock_job2.metadata = None
    mock_job2.output_file = None
    mock_job2.error_file = None
    mock_job2.started_at = None
    mock_job2.completed_at = None

    mock_batch_jobs = Mock()
    mock_batch_jobs.data = [mock_job1, mock_job2]

    result = _convert_batch_jobs_list(mock_batch_jobs)

    assert len(result) == 2
    assert result[0].id == "batch-1"
    assert result[0].status == "completed"
    assert result[1].id == "batch-2"
    assert result[1].status == "in_progress"


def test_convert_batch_jobs_list_empty() -> None:
    """Test converting an empty batch jobs list."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.utils import _convert_batch_jobs_list

    mock_batch_jobs = Mock()
    mock_batch_jobs.data = None

    result = _convert_batch_jobs_list(mock_batch_jobs)

    assert result == []


def test_convert_batch_job_unknown_status_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that unknown Mistral status logs a warning."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.utils import _convert_batch_job_to_openai

    mock_batch_job = Mock()
    mock_batch_job.id = "batch-unknown"
    mock_batch_job.input_files = ["file-abc"]
    mock_batch_job.endpoint = "/v1/chat/completions"
    mock_batch_job.status = Mock(value="UNKNOWN_STATUS")  # Unknown status
    mock_batch_job.created_at = 1700000000
    mock_batch_job.total_requests = 5
    mock_batch_job.completed_requests = 0
    mock_batch_job.succeeded_requests = 0
    mock_batch_job.failed_requests = 0
    mock_batch_job.errors = []
    mock_batch_job.metadata = None
    mock_batch_job.output_file = None
    mock_batch_job.error_file = None
    mock_batch_job.started_at = None
    mock_batch_job.completed_at = None

    result = _convert_batch_job_to_openai(mock_batch_job)

    assert result.status == "in_progress"  # Falls back to in_progress
    assert "Unknown Mistral batch status" in caplog.text
    assert "UNKNOWN_STATUS" in caplog.text


def test_convert_batch_job_with_timeout_hours() -> None:
    """Test that timeout_hours is converted to completion_window."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.utils import _convert_batch_job_to_openai

    mock_batch_job = Mock()
    mock_batch_job.id = "batch-timeout"
    mock_batch_job.input_files = ["file-abc"]
    mock_batch_job.endpoint = "/v1/chat/completions"
    mock_batch_job.status = Mock(value="SUCCESS")
    mock_batch_job.created_at = 1700000000
    mock_batch_job.total_requests = 5
    mock_batch_job.completed_requests = 5
    mock_batch_job.succeeded_requests = 5
    mock_batch_job.failed_requests = 0
    mock_batch_job.errors = []
    mock_batch_job.metadata = None
    mock_batch_job.output_file = None
    mock_batch_job.error_file = None
    mock_batch_job.started_at = None
    mock_batch_job.completed_at = None
    mock_batch_job.timeout_hours = 48  # Custom timeout

    result = _convert_batch_job_to_openai(mock_batch_job)

    assert result.completion_window == "48h"


@pytest.mark.asyncio
async def test_create_batch_empty_file() -> None:
    """Test creating a batch job with an empty file raises ValueError."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_upload_response = Mock()
        mock_upload_response.id = "uploaded-file-123"
        mocked_mistral.return_value.files.upload_async = AsyncMock(return_value=mock_upload_response)

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")  # Empty file
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Input file is empty"):
                await provider._acreate_batch(
                    input_file_path=tmp_path,
                    endpoint="/v1/chat/completions",
                )
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_create_batch_invalid_json() -> None:
    """Test creating a batch job with invalid JSON raises ValueError."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_upload_response = Mock()
        mock_upload_response.id = "uploaded-file-123"
        mocked_mistral.return_value.files.upload_async = AsyncMock(return_value=mock_upload_response)

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json {{{")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSONL format in first line"):
                await provider._acreate_batch(
                    input_file_path=tmp_path,
                    endpoint="/v1/chat/completions",
                )
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_create_batch_missing_model() -> None:
    """Test creating a batch job without model in JSONL raises ValueError."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_upload_response = Mock()
        mock_upload_response.id = "uploaded-file-123"
        mocked_mistral.return_value.files.upload_async = AsyncMock(return_value=mock_upload_response)

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"custom_id": "1", "body": {"messages": []}}\n')  # No model in body
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Model not found in JSONL body"):
                await provider._acreate_batch(
                    input_file_path=tmp_path,
                    endpoint="/v1/chat/completions",
                )
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_create_batch_with_explicit_model() -> None:
    """Test creating a batch job with explicit model kwarg bypasses JSONL extraction."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_upload_response = Mock()
        mock_upload_response.id = "uploaded-file-123"

        mock_batch_job = Mock()
        mock_batch_job.id = "batch-789"
        mock_batch_job.input_files = ["uploaded-file-123"]
        mock_batch_job.endpoint = "/v1/chat/completions"
        mock_batch_job.status = Mock(value="QUEUED")
        mock_batch_job.created_at = 1700000000
        mock_batch_job.total_requests = 1
        mock_batch_job.completed_requests = 0
        mock_batch_job.succeeded_requests = 0
        mock_batch_job.failed_requests = 0
        mock_batch_job.errors = []
        mock_batch_job.metadata = None
        mock_batch_job.output_file = None
        mock_batch_job.error_file = None
        mock_batch_job.started_at = None
        mock_batch_job.completed_at = None

        mocked_mistral.return_value.files.upload_async = AsyncMock(return_value=mock_upload_response)
        mocked_mistral.return_value.batch.jobs.create_async = AsyncMock(return_value=mock_batch_job)

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"custom_id": "1", "body": {"messages": []}}\n')  # No model, but we pass it explicitly
            tmp_path = f.name

        try:
            result = await provider._acreate_batch(
                input_file_path=tmp_path,
                endpoint="/v1/chat/completions",
                model="mistral-small-latest",  # Explicit model
            )

            assert result.id == "batch-789"
            # Verify model was passed to create_async
            create_call_kwargs = mocked_mistral.return_value.batch.jobs.create_async.call_args[1]
            assert create_call_kwargs["model"] == "mistral-small-latest"
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_create_batch() -> None:
    """Test creating a batch job with Mistral provider."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_upload_response = Mock()
        mock_upload_response.id = "uploaded-file-123"

        mock_batch_job = Mock()
        mock_batch_job.id = "batch-789"
        mock_batch_job.input_files = ["uploaded-file-123"]
        mock_batch_job.endpoint = "/v1/chat/completions"
        mock_batch_job.status = Mock(value="QUEUED")
        mock_batch_job.created_at = 1700000000
        mock_batch_job.total_requests = 2
        mock_batch_job.completed_requests = 0
        mock_batch_job.succeeded_requests = 0
        mock_batch_job.failed_requests = 0
        mock_batch_job.errors = []
        mock_batch_job.metadata = {"test": "batch"}
        mock_batch_job.output_file = None
        mock_batch_job.error_file = None
        mock_batch_job.started_at = None
        mock_batch_job.completed_at = None

        mocked_mistral.return_value.files.upload_async = AsyncMock(return_value=mock_upload_response)
        mocked_mistral.return_value.batch.jobs.create_async = AsyncMock(return_value=mock_batch_job)

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"custom_id": "1", "body": {"model": "mistral-small-latest", "messages": []}}\n')
            tmp_path = f.name

        try:
            result = await provider._acreate_batch(
                input_file_path=tmp_path,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"test": "batch"},
            )

            assert result.id == "batch-789"
            assert result.status == "validating"
            assert result.input_file_id == "uploaded-file-123"

            mocked_mistral.return_value.files.upload_async.assert_called_once()
            mocked_mistral.return_value.batch.jobs.create_async.assert_called_once()

            # Verify model was extracted from JSONL
            create_call_kwargs = mocked_mistral.return_value.batch.jobs.create_async.call_args[1]
            assert create_call_kwargs["model"] == "mistral-small-latest"
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_retrieve_batch() -> None:
    """Test retrieving a batch job with Mistral provider."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_batch_job = Mock()
        mock_batch_job.id = "batch-retrieve-123"
        mock_batch_job.input_files = ["file-abc"]
        mock_batch_job.endpoint = "/v1/chat/completions"
        mock_batch_job.status = Mock(value="RUNNING")
        mock_batch_job.created_at = 1700000000
        mock_batch_job.total_requests = 10
        mock_batch_job.completed_requests = 5
        mock_batch_job.succeeded_requests = 5
        mock_batch_job.failed_requests = 0
        mock_batch_job.errors = []
        mock_batch_job.metadata = None
        mock_batch_job.output_file = None
        mock_batch_job.error_file = None
        mock_batch_job.started_at = 1700000100
        mock_batch_job.completed_at = None

        mocked_mistral.return_value.batch.jobs.get_async = AsyncMock(return_value=mock_batch_job)

        result = await provider._aretrieve_batch("batch-retrieve-123")

        assert result.id == "batch-retrieve-123"
        assert result.status == "in_progress"
        mocked_mistral.return_value.batch.jobs.get_async.assert_called_once_with(job_id="batch-retrieve-123")


@pytest.mark.asyncio
async def test_cancel_batch() -> None:
    """Test cancelling a batch job with Mistral provider."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_batch_job = Mock()
        mock_batch_job.id = "batch-cancel-123"
        mock_batch_job.input_files = ["file-abc"]
        mock_batch_job.endpoint = "/v1/chat/completions"
        mock_batch_job.status = Mock(value="CANCELLATION_REQUESTED")
        mock_batch_job.created_at = 1700000000
        mock_batch_job.total_requests = 10
        mock_batch_job.completed_requests = 3
        mock_batch_job.succeeded_requests = 3
        mock_batch_job.failed_requests = 0
        mock_batch_job.errors = []
        mock_batch_job.metadata = None
        mock_batch_job.output_file = None
        mock_batch_job.error_file = None
        mock_batch_job.started_at = 1700000100
        mock_batch_job.completed_at = None

        mocked_mistral.return_value.batch.jobs.cancel_async = AsyncMock(return_value=mock_batch_job)

        result = await provider._acancel_batch("batch-cancel-123")

        assert result.id == "batch-cancel-123"
        assert result.status == "cancelling"
        mocked_mistral.return_value.batch.jobs.cancel_async.assert_called_once_with(job_id="batch-cancel-123")


@pytest.mark.asyncio
async def test_list_batches() -> None:
    """Test listing batch jobs with Mistral provider."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_job1 = Mock()
        mock_job1.id = "batch-list-1"
        mock_job1.input_files = ["file-1"]
        mock_job1.endpoint = "/v1/chat/completions"
        mock_job1.status = Mock(value="SUCCESS")
        mock_job1.created_at = 1700000000
        mock_job1.total_requests = 5
        mock_job1.completed_requests = 5
        mock_job1.succeeded_requests = 5
        mock_job1.failed_requests = 0
        mock_job1.errors = []
        mock_job1.metadata = None
        mock_job1.output_file = "output-1"
        mock_job1.error_file = None
        mock_job1.started_at = 1700000100
        mock_job1.completed_at = 1700000200

        mock_batch_jobs = Mock()
        mock_batch_jobs.data = [mock_job1]

        mocked_mistral.return_value.batch.jobs.list_async = AsyncMock(return_value=mock_batch_jobs)

        result = await provider._alist_batches(limit=10)

        assert len(result) == 1
        assert result[0].id == "batch-list-1"
        assert result[0].status == "completed"
        mocked_mistral.return_value.batch.jobs.list_async.assert_called_once_with(page=0, page_size=10)


@pytest.mark.asyncio
async def test_list_batches_after_param_raises_error() -> None:
    """Test that using 'after' parameter raises ValueError since Mistral uses page-based pagination."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral"):
        provider = MistralProvider(api_key="test-api-key")

        with pytest.raises(ValueError, match="Mistral batch API uses page-based pagination"):
            await provider._alist_batches(after="batch-123", limit=10)


@pytest.mark.asyncio
async def test_list_batches_with_page_kwarg() -> None:
    """Test that page kwarg is passed correctly for Mistral's page-based pagination."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_batch_jobs = Mock()
        mock_batch_jobs.data = []

        mocked_mistral.return_value.batch.jobs.list_async = AsyncMock(return_value=mock_batch_jobs)

        await provider._alist_batches(limit=20, page=2)

        mocked_mistral.return_value.batch.jobs.list_async.assert_called_once_with(page=2, page_size=20)


@pytest.mark.asyncio
async def test_create_batch_mixed_models_raises_error() -> None:
    """Test that batch file with different models raises MixedModelError."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider
    from any_llm.providers.mistral.utils import MixedModelError

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_upload_response = Mock()
        mock_upload_response.id = "uploaded-file-123"
        mocked_mistral.return_value.files.upload_async = AsyncMock(return_value=mock_upload_response)

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"custom_id": "1", "body": {"model": "mistral-small-latest", "messages": []}}\n')
            f.write('{"custom_id": "2", "body": {"model": "mistral-large-latest", "messages": []}}\n')
            tmp_path = f.name

        try:
            with pytest.raises(MixedModelError, match="requires all requests to use the same model"):
                await provider._acreate_batch(
                    input_file_path=tmp_path,
                    endpoint="/v1/chat/completions",
                )
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_create_batch_model_mismatch_raises_error() -> None:
    """Test that explicit model kwarg mismatching file content raises ValueError."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_upload_response = Mock()
        mock_upload_response.id = "uploaded-file-123"
        mocked_mistral.return_value.files.upload_async = AsyncMock(return_value=mock_upload_response)

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"custom_id": "1", "body": {"model": "mistral-small-latest", "messages": []}}\n')
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Model mismatch"):
                await provider._acreate_batch(
                    input_file_path=tmp_path,
                    endpoint="/v1/chat/completions",
                    model="mistral-large-latest",
                )
        finally:
            import os

            os.unlink(tmp_path)


def test_create_mistral_completion_raises_when_choice_message_is_none() -> None:
    """Guard against the mistralai v2 `choice.message: AssistantMessage | None` type.

    A real chat completion always carries a message, but the SDK types permit None.
    The converter should fail loudly rather than silently emit a truncated ChatCompletion.
    """
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.utils import _create_mistral_completion_from_response

    choice = Mock()
    choice.message = None
    response = Mock()
    response.choices = [choice]

    with pytest.raises(ValueError, match="without a message"):
        _create_mistral_completion_from_response(response, model="mistral-small-latest")


def test_create_mistral_completion_builds_chatcompletion_from_well_formed_response() -> None:
    """Happy path for `_create_mistral_completion_from_response`.

    Covers the `message_data is not None` branch the guard introduced and exercises the
    rest of the converter against a well-formed response so the function has baseline
    unit coverage beyond the error path.
    """
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.utils import _create_mistral_completion_from_response

    message = Mock()
    message.content = "Hello from Mistral!"
    message.tool_calls = None

    choice = Mock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = Mock()
    usage.prompt_tokens = 12
    usage.completion_tokens = 7
    usage.total_tokens = 19

    response = Mock()
    response.id = "chatcmpl-abc"
    response.created = 1_700_000_000
    response.choices = [choice]
    response.usage = usage

    completion = _create_mistral_completion_from_response(response, model="mistral-small-latest")

    assert completion.id == "chatcmpl-abc"
    assert completion.model == "mistral-small-latest"
    assert completion.created == 1_700_000_000
    assert completion.object == "chat.completion"
    assert len(completion.choices) == 1
    assert completion.choices[0].index == 0
    assert completion.choices[0].finish_reason == "stop"
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content == "Hello from Mistral!"
    assert completion.choices[0].message.tool_calls is None
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 12
    assert completion.usage.completion_tokens == 7
    assert completion.usage.total_tokens == 19


def test_create_mistral_completion_extracts_reasoning_and_signature() -> None:
    """A ThinkChunk's text lands on `reasoning` and its signature on `extra_content`."""
    pytest.importorskip("mistralai")
    from mistralai.client.models import TextChunk, ThinkChunk

    from any_llm.providers.mistral.utils import _create_mistral_completion_from_response

    message = Mock()
    message.content = [
        ThinkChunk(thinking=[TextChunk(text="step one"), TextChunk(text="step two")], signature="sig-abc"),
        TextChunk(text="the answer"),
    ]
    message.tool_calls = None

    choice = Mock()
    choice.message = message
    choice.finish_reason = "stop"

    response = Mock()
    response.id = "chatcmpl-abc"
    response.created = 1_700_000_000
    response.choices = [choice]
    response.usage = None

    completion = _create_mistral_completion_from_response(response, model="mistral-medium-3-5")

    assert completion.choices[0].message.content == "the answer"
    assert completion.choices[0].message.reasoning is not None
    assert completion.choices[0].message.reasoning.content == "step one\nstep two"
    assert completion.choices[0].message.extra_content == {"mistral": {"signature": "sig-abc"}}


def test_create_mistral_completion_omits_extra_content_without_signature() -> None:
    """Mistral leaves the signature unset today, so extra_content stays absent."""
    pytest.importorskip("mistralai")
    from mistralai.client.models import TextChunk, ThinkChunk

    from any_llm.providers.mistral.utils import _create_mistral_completion_from_response

    message = Mock()
    message.content = [ThinkChunk(thinking=[TextChunk(text="step one")]), TextChunk(text="the answer")]
    message.tool_calls = None

    choice = Mock()
    choice.message = message
    choice.finish_reason = "stop"

    response = Mock()
    response.id = "chatcmpl-abc"
    response.created = 1_700_000_000
    response.choices = [choice]
    response.usage = None

    completion = _create_mistral_completion_from_response(response, model="mistral-medium-3-5")

    assert completion.choices[0].message.reasoning is not None
    assert completion.choices[0].message.extra_content is None


def test_create_openai_chunk_captures_thinking_signature() -> None:
    """Streaming deltas surface the thinking signature the same way as non-streaming."""
    pytest.importorskip("mistralai")
    from mistralai.client.models import TextChunk, ThinkChunk

    from any_llm.providers.mistral.utils import _create_openai_chunk_from_mistral_chunk

    choice = Mock()
    choice.index = 0
    choice.delta.content = [ThinkChunk(thinking=[TextChunk(text="step one")], signature="sig-abc")]
    choice.delta.role = "assistant"
    choice.delta.tool_calls = None
    choice.finish_reason = None

    event = Mock()
    event.data.id = "chatcmpl-abc"
    event.data.created = 1_700_000_000
    event.data.model = "mistral-medium-3-5"
    event.data.choices = [choice]
    event.data.usage = None

    chunk = _create_openai_chunk_from_mistral_chunk(event)

    assert chunk.choices[0].delta.reasoning is not None
    assert chunk.choices[0].delta.reasoning.content == "step one"
    assert chunk.choices[0].delta.extra_content == {"mistral": {"signature": "sig-abc"}}
