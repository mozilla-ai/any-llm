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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "expected_converter"),
    [
        (StructuredOutput, "response_format_from_pydantic_model"),
        ({"type": "json_object", "schema": StructuredOutput.model_json_schema()}, "ResponseFormat"),
    ],
    ids=["pydantic_model", "openai_json_schema"],
)
async def test_response_format_conversion(response_format: Any, expected_converter: str) -> None:
    """Test that response_format is properly converted for both Pydantic and dict formats."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with (
        patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral,
        patch("any_llm.providers.mistral.mistral.response_format_from_pydantic_model") as mocked_pydantic_converter,
        patch("any_llm.providers.mistral.mistral.ResponseFormat") as mocked_response_format,
        patch("any_llm.providers.mistral.mistral._create_mistral_completion_from_response") as mocked_converter,
    ):
        provider = MistralProvider(api_key="test-api-key")

        mock_response = Mock()
        mocked_mistral.return_value.chat.complete_async = AsyncMock(return_value=mock_response)
        mocked_converter.return_value = Mock()

        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=response_format,
            ),
        )

        if expected_converter == "response_format_from_pydantic_model":
            mocked_pydantic_converter.assert_called_once_with(StructuredOutput)
        elif expected_converter == "ResponseFormat":
            mocked_response_format.model_validate.assert_called_once_with(response_format)
