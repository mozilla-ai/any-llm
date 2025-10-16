from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.providers.mistral.utils import _patch_messages
from any_llm.types.completion import CompletionParams


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
