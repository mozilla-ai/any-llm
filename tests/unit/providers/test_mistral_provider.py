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
    mistralai = pytest.importorskip("mistralai")
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
        assert isinstance(response_format_arg, mistralai.models.responseformat.ResponseFormat)
        assert response_format_arg.type == "json_schema"
        assert response_format_arg.json_schema.name == "StructuredOutput"
        assert response_format_arg.json_schema.strict is True

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
        assert response_format_arg.json_schema.schema_definition == expected_schema


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
