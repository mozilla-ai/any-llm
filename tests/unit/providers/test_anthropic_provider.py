import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC
from typing import Any, cast, get_args
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import transform_schema
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.providers.anthropic.utils import (
    DEFAULT_MAX_TOKENS,
    REASONING_EFFORT_TO_ANTHROPIC_EFFORT,
    _convert_response_format,
    _convert_tool_spec,
)
from any_llm.types.completion import ChatCompletionMessageFunctionToolCall, CompletionParams, ReasoningEffort


@contextmanager
def mock_anthropic_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic") as mock_anthropic,
        patch("any_llm.providers.anthropic.base._convert_response"),
    ):
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create = AsyncMock()
        yield mock_anthropic


@pytest.mark.asyncio
async def test_anthropic_client_created_with_api_key_and_api_base() -> None:
    api_key = "test-api-key"
    custom_endpoint = "https://custom-anthropic-endpoint"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key, api_base=custom_endpoint)
        await provider._acompletion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_anthropic.assert_called_once_with(api_key=api_key, base_url=custom_endpoint)


@pytest.mark.asyncio
async def test_anthropic_client_created_without_api_base() -> None:
    api_key = "test-api-key"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}])
        )

        mock_anthropic.assert_called_once_with(api_key=api_key, base_url=None)


@pytest.mark.asyncio
async def test_completion_with_system_message() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_with_multiple_system_messages() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [
        {"role": "system", "content": "First part."},
        {"role": "system", "content": "Second part."},
        {"role": "user", "content": "Hello"},
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            system="First part.\nSecond part.",
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_with_kwargs() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(model_id=model, messages=messages, max_tokens=100, temperature=0.5)
        )

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model, messages=messages, max_tokens=100, temperature=0.5
        )


@pytest.mark.asyncio
async def test_completion_renames_stop_to_stop_sequences_list() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages, stop=["END", "STOP"]))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            stop_sequences=["END", "STOP"],
        )


@pytest.mark.asyncio
async def test_completion_renames_stop_to_stop_sequences_string() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages, stop="END"))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            stop_sequences=["END"],
        )


@pytest.mark.asyncio
async def test_completion_with_tool_choice_required() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages, tool_choice="required"))

        expected_kwargs = {"tool_choice": {"type": "any", "disable_parallel_tool_use": False}}

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            **expected_kwargs,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "function", "function": {"name": "FOO"}},
        {"type": "custom", "custom": {"name": "FOO"}},
    ],
)
async def test_completion_with_tool_choice_specific_tool(tool_choice: dict[str, Any]) -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]
    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages, tool_choice=tool_choice))

        expected_kwargs = {"tool_choice": {"type": "tool", "name": "FOO"}}

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            **expected_kwargs,
        )


@pytest.mark.asyncio
async def test_completion_with_tool_choice_invalid_format() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]
    invalid_tool_choice = {"type": "unknown_type", "unknown": {"name": "FOO"}}
    provider = AnthropicProvider(api_key=api_key)
    with pytest.raises(ValueError, match="Unsupported tool_choice format:"):
        await provider._acompletion(
            CompletionParams(model_id=model, messages=messages, tool_choice=invalid_tool_choice)
        )


@pytest.mark.parametrize("parallel_tool_calls", [True, False])
@pytest.mark.asyncio
async def test_completion_with_tool_choice_and_parallel_tool_calls(parallel_tool_calls: bool) -> None:
    """Test that completion correctly processes tool_choice and parallel_tool_calls."""
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(
                model_id=model, messages=messages, tool_choice="auto", parallel_tool_calls=parallel_tool_calls
            ),
        )

        expected_kwargs = {"tool_choice": {"type": "auto", "disable_parallel_tool_use": not parallel_tool_calls}}

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            **expected_kwargs,
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    api_key = "test-api-key"
    model = "model-id"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=agent_loop_messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[
                {"role": "user", "content": "What is the weather like in Salvaterra?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "foo", "name": "get_weather", "input": {"location": "Salvaterra"}}
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "foo", "content": "sunny"}]},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.parametrize("reasoning_effort", [None, *get_args(ReasoningEffort)])
@pytest.mark.asyncio
async def test_completion_with_custom_reasoning_effort(reasoning_effort: ReasoningEffort | None) -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(model_id=model, messages=messages, reasoning_effort=reasoning_effort)
        )

        call_kwargs = mock_anthropic.return_value.messages.create.call_args[1]

        if reasoning_effort is None or reasoning_effort == "none":
            assert call_kwargs["thinking"] == {"type": "disabled"}
        elif reasoning_effort == "auto":
            assert "thinking" not in call_kwargs
        else:
            assert call_kwargs["thinking"] == {"type": "adaptive"}
            assert call_kwargs["output_config"] == {"effort": REASONING_EFFORT_TO_ANTHROPIC_EFFORT[reasoning_effort]}


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_effort"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "max"),
    ],
)
@pytest.mark.asyncio
async def test_reasoning_effort_maps_to_distinct_anthropic_effort(
    reasoning_effort: ReasoningEffort, expected_effort: str
) -> None:
    """Guard against mapping canonical xhigh onto Anthropic's max (see issue #1107).

    Anthropic exposes low < medium < high < xhigh < max as distinct levels, so each
    canonical effort must map to its intended Anthropic level rather than silently
    escalating xhigh to max.
    """
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key="test-api-key")
        await provider._acompletion(
            CompletionParams(model_id="model-id", messages=messages, reasoning_effort=reasoning_effort)
        )

        call_kwargs = mock_anthropic.return_value.messages.create.call_args[1]
        assert call_kwargs["output_config"] == {"effort": expected_effort}


@pytest.mark.asyncio
async def test_completion_with_images() -> None:
    api_key = "test-api-key"
    model = "model-id"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Some question about these images."},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,qwertyuiopasdfghjklzxcvbnm"}},
            ],
        }
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Some question about these images."},
                        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "qwertyuiopasdfghjklzxcvbnm",
                            },
                        },
                    ],
                }
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_with_pdf() -> None:
    api_key = "test-api-key"
    model = "model-id"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this PDF?"},
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": "data:application/pdf;base64,JVBERi0xLjQKdGVzdA==",
                    },
                },
            ],
        }
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this PDF?"},
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQKdGVzdA==",
                            },
                        },
                    ],
                }
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_with_pdf_url() -> None:
    api_key = "test-api-key"
    model = "model-id"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this PDF."},
                {
                    "type": "file",
                    "file": {
                        "filename": "report.pdf",
                        "file_data": "https://example.com/report.pdf",
                    },
                },
            ],
        }
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this PDF."},
                        {
                            "type": "document",
                            "source": {"type": "url", "url": "https://example.com/report.pdf"},
                        },
                    ],
                }
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_with_parallel_tool_calls() -> None:
    """Test that parallel tool calls are correctly converted to Anthropic format.

    When an assistant message contains multiple tool calls and their results
    come in consecutive tool messages, each tool result should correctly
    reference its corresponding tool_use_id from the tool message's tool_call_id field.
    """
    api_key = "test-api-key"
    model = "model-id"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Call first and second tool."},
        {
            "content": "",
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "toolu_014vbRyMX81AhxfBMNYRma3A",
                    "function": {"arguments": "{}", "name": "first_tool"},
                    "type": "function",
                },
                {
                    "id": "toolu_01WhTsYUFmDbsbSZbs9WDciT",
                    "function": {"arguments": '{"query": "test"}', "name": "second_tool"},
                    "type": "function",
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_014vbRyMX81AhxfBMNYRma3A",
            "content": "First Result",
            "name": "first_tool",
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01WhTsYUFmDbsbSZbs9WDciT",
            "content": "Second Result",
            "name": "second_tool",
        },
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[
                {"role": "user", "content": "Call first and second tool."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_014vbRyMX81AhxfBMNYRma3A",
                            "name": "first_tool",
                            "input": {},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_01WhTsYUFmDbsbSZbs9WDciT",
                            "name": "second_tool",
                            "input": {"query": "test"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_014vbRyMX81AhxfBMNYRma3A",
                            "content": "First Result",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01WhTsYUFmDbsbSZbs9WDciT",
                            "content": "Second Result",
                        },
                    ],
                },
            ],
            system="You are a helpful assistant.",
            max_tokens=DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_completion_with_response_format_basemodel() -> None:
    class MySchema(BaseModel):
        city_name: str

    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages, response_format=MySchema))

        call_kwargs = mock_anthropic.return_value.messages.create.call_args[1]
        expected_schema = transform_schema(MySchema.model_json_schema())
        assert call_kwargs["output_config"] == {"format": {"type": "json_schema", "schema": expected_schema}}


@pytest.mark.asyncio
async def test_completion_with_response_format_and_reasoning_effort() -> None:
    class MySchema(BaseModel):
        city_name: str

    api_key = "test-api-key"
    model = "claude-opus-4-6"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(model_id=model, messages=messages, response_format=MySchema, reasoning_effort="medium")
        )

        call_kwargs = mock_anthropic.return_value.messages.create.call_args[1]
        expected_schema = transform_schema(MySchema.model_json_schema())
        assert call_kwargs["output_config"] == {
            "format": {"type": "json_schema", "schema": expected_schema},
            "effort": "medium",
        }
        assert call_kwargs["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_completion_with_response_format_dict_json_schema() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]
    schema = {"type": "object", "properties": {"city_name": {"type": "string"}}, "required": ["city_name"]}
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {"name": "MySchema", "schema": schema},
    }

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(model_id=model, messages=messages, response_format=response_format)
        )

        call_kwargs = mock_anthropic.return_value.messages.create.call_args[1]
        assert call_kwargs["output_config"] == {"format": {"type": "json_schema", "schema": transform_schema(schema)}}


@pytest.mark.asyncio
async def test_completion_with_response_format_dict_json_object_raises() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = AnthropicProvider(api_key=api_key)

    with pytest.raises(UnsupportedParameterError, match="json_object"):
        await provider._acompletion(
            CompletionParams(
                model_id=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_stream_with_response_format_passes_output_config() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]
    schema = {"type": "object", "properties": {"city_name": {"type": "string"}}, "required": ["city_name"]}
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {"name": "Foo", "schema": schema},
    }

    async def empty_events() -> AsyncIterator[Any]:
        if False:
            yield None

    @asynccontextmanager
    async def empty_stream() -> AsyncIterator[AsyncIterator[Any]]:
        yield empty_events()

    with mock_anthropic_provider() as mock_anthropic:
        mock_anthropic.return_value.messages.stream = Mock(return_value=empty_stream())
        provider = AnthropicProvider(api_key=api_key)
        stream = cast(
            "AsyncIterator[Any]",
            await provider._acompletion(
                CompletionParams(
                    model_id=model,
                    messages=messages,
                    response_format=response_format,
                    stream=True,
                )
            ),
        )

        async for _ in stream:
            pass

        expected_output_config = {"format": {"type": "json_schema", "schema": transform_schema(schema)}}
        mock_anthropic.return_value.messages.stream.assert_called_once()
        call_kwargs = mock_anthropic.return_value.messages.stream.call_args.kwargs
        assert call_kwargs["model"] == model
        assert call_kwargs["messages"] == messages
        assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
        assert call_kwargs["output_config"] == expected_output_config


@pytest.mark.asyncio
async def test_completion_with_response_format_dict_unknown_type_raises() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = AnthropicProvider(api_key=api_key)

    with pytest.raises(ValueError, match="Unsupported response_format type"):
        await provider._acompletion(
            CompletionParams(
                model_id=model,
                messages=messages,
                response_format={"type": "unknown"},
            )
        )


def test_convert_response_format_dataclass() -> None:
    @dataclasses.dataclass
    class CityResponse:
        city_name: str

    result = _convert_response_format(CityResponse, "anthropic")
    assert "format" in result
    assert result["format"]["type"] == "json_schema"
    assert "schema" in result["format"]
    schema = result["format"]["schema"]
    assert "city_name" in schema["properties"]


def test_convert_response_format_non_dict_non_basemodel_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported response_format"):
        _convert_response_format("invalid", "anthropic")  # type: ignore[arg-type]


def test_convert_response_includes_cache_tokens_in_usage() -> None:
    """Test that prompt caching tokens are correctly included in usage calculation.

    When Anthropic returns cache_read_input_tokens or cache_creation_input_tokens,
    these should be added to prompt_tokens and total_tokens for accurate reporting.
    See: https://github.com/mozilla-ai/any-llm/issues/622
    """
    from datetime import datetime
    from unittest.mock import MagicMock

    from any_llm.providers.anthropic.utils import _convert_response

    mock_response = MagicMock()
    mock_response.id = "msg_123"
    mock_response.model = "claude-3-haiku"
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="Hello!")]
    mock_response.created_at = datetime.now(UTC)

    mock_response.usage.input_tokens = 3
    mock_response.usage.output_tokens = 122
    mock_response.usage.cache_read_input_tokens = 13332
    mock_response.usage.cache_creation_input_tokens = 0

    result = _convert_response(mock_response)

    expected_prompt_tokens = 3 + 13332 + 0
    expected_total_tokens = expected_prompt_tokens + 122

    assert result.usage is not None
    assert result.usage.prompt_tokens == expected_prompt_tokens
    assert result.usage.completion_tokens == 122
    assert result.usage.total_tokens == expected_total_tokens
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 13332


def test_convert_response_includes_cache_creation_tokens() -> None:
    """Test that cache_creation_input_tokens are included in usage when writing to cache."""
    from datetime import datetime
    from unittest.mock import MagicMock

    from any_llm.providers.anthropic.utils import _convert_response

    mock_response = MagicMock()
    mock_response.id = "msg_123"
    mock_response.model = "claude-3-haiku"
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="Hello!")]
    mock_response.created_at = datetime.now(UTC)

    mock_response.usage.input_tokens = 3
    mock_response.usage.output_tokens = 122
    mock_response.usage.cache_read_input_tokens = 0
    mock_response.usage.cache_creation_input_tokens = 13332

    result = _convert_response(mock_response)

    expected_prompt_tokens = 3 + 0 + 13332
    expected_total_tokens = expected_prompt_tokens + 122

    assert result.usage is not None
    assert result.usage.prompt_tokens == expected_prompt_tokens
    assert result.usage.total_tokens == expected_total_tokens
    assert result.usage.prompt_tokens_details is None


def test_convert_response_without_cache_tokens() -> None:
    """Test that usage is correct when no cache tokens are present."""
    from datetime import datetime
    from unittest.mock import MagicMock

    from any_llm.providers.anthropic.utils import _convert_response

    mock_response = MagicMock()
    mock_response.id = "msg_123"
    mock_response.model = "claude-3-haiku"
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="Hello!")]
    mock_response.created_at = datetime.now(UTC)

    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_response.usage.cache_read_input_tokens = None
    mock_response.usage.cache_creation_input_tokens = None

    result = _convert_response(mock_response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.prompt_tokens_details is None


def test_streaming_chunk_includes_cache_tokens_in_usage() -> None:
    """Test that streaming chunks correctly include cache tokens in usage."""
    from unittest.mock import MagicMock

    from anthropic.types import MessageStopEvent, Usage

    from any_llm.providers.anthropic.utils import _create_openai_chunk_from_anthropic_chunk

    usage = Usage(
        input_tokens=3,
        output_tokens=159,
        cache_read_input_tokens=13332,
        cache_creation_input_tokens=0,
    )

    mock_message = MagicMock()
    mock_message.usage = usage

    chunk = MessageStopEvent(type="message_stop")
    chunk.message = mock_message  # type: ignore[attr-defined]

    result = _create_openai_chunk_from_anthropic_chunk(chunk, "claude-3-haiku")

    expected_prompt_tokens = 3 + 13332 + 0
    expected_total_tokens = expected_prompt_tokens + 159

    assert result.usage is not None
    assert result.usage.prompt_tokens == expected_prompt_tokens
    assert result.usage.completion_tokens == 159
    assert result.usage.total_tokens == expected_total_tokens
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 13332


@pytest.mark.asyncio
async def test_completion_strips_openai_specific_fields() -> None:
    """Test that OpenAI-specific fields like 'refusal' are stripped from messages.

    OpenAI responses include fields like 'refusal', 'audio', 'annotations' etc.
    that are not part of Anthropic's API schema. When these messages are forwarded
    to Anthropic, these extra fields must be removed to avoid 400 errors.
    See: messages.X.refusal: Extra inputs are not permitted
    """
    api_key = "test-api-key"
    model = "model-id"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "I can help with that.",
            "refusal": None,
            "audio": None,
            "function_call": None,
            "annotations": [],
        },
        {"role": "user", "content": "Thanks"},
    ]

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_anthropic.return_value.messages.create.assert_called_once_with(
            model=model,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "I can help with that."},
                {"role": "user", "content": "Thanks"},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
        )


def test_streaming_chunk_without_cache_tokens() -> None:
    """Test that streaming chunks work correctly without cache tokens."""
    from unittest.mock import MagicMock

    from anthropic.types import MessageStopEvent, Usage

    from any_llm.providers.anthropic.utils import _create_openai_chunk_from_anthropic_chunk

    usage = Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )

    mock_message = MagicMock()
    mock_message.usage = usage

    chunk = MessageStopEvent(type="message_stop")
    chunk.message = mock_message  # type: ignore[attr-defined]

    result = _create_openai_chunk_from_anthropic_chunk(chunk, "claude-3-haiku")

    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.prompt_tokens_details is None


def test_streaming_tool_chunks_preserve_parallel_tool_index() -> None:
    from anthropic.types import ContentBlockDeltaEvent, ContentBlockStartEvent, InputJSONDelta, ToolUseBlock

    from any_llm.providers.anthropic.utils import _create_openai_chunk_from_anthropic_chunk

    start_chunk = ContentBlockStartEvent(
        type="content_block_start",
        index=1,
        content_block=ToolUseBlock(type="tool_use", id="toolu_456", name="get_weather", input={}),
    )
    start_result = _create_openai_chunk_from_anthropic_chunk(start_chunk, "claude-3-haiku")

    assert start_result.choices[0].delta.tool_calls is not None
    assert start_result.choices[0].delta.tool_calls[0].index == 1
    assert start_result.choices[0].delta.tool_calls[0].id == "toolu_456"
    assert start_result.choices[0].delta.tool_calls[0].function is not None
    assert start_result.choices[0].delta.tool_calls[0].function.name == "get_weather"

    delta_chunk = ContentBlockDeltaEvent(
        type="content_block_delta",
        index=1,
        delta=InputJSONDelta(type="input_json_delta", partial_json='{"city":"Rome"}'),
    )
    delta_result = _create_openai_chunk_from_anthropic_chunk(delta_chunk, "claude-3-haiku")

    assert delta_result.choices[0].delta.tool_calls is not None
    assert delta_result.choices[0].delta.tool_calls[0].index == 1
    assert delta_result.choices[0].delta.tool_calls[0].function is not None
    assert delta_result.choices[0].delta.tool_calls[0].function.arguments == '{"city":"Rome"}'


def test_streaming_thinking_signature_delta_sets_extra_content() -> None:
    """The encrypted thinking signature must be surfaced on the streaming delta's extra_content."""
    from anthropic.types import ContentBlockDeltaEvent, SignatureDelta

    from any_llm.providers.anthropic.utils import _create_openai_chunk_from_anthropic_chunk

    delta_chunk = ContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=SignatureDelta(type="signature_delta", signature="sig-12345"),
    )
    result = _create_openai_chunk_from_anthropic_chunk(delta_chunk, "claude-3-haiku")

    assert result.choices[0].delta.extra_content == {"anthropic": {"signature": "sig-12345"}}


def test_streaming_thinking_delta_does_not_set_extra_content() -> None:
    """A plain thinking_delta (no signature yet) should not set extra_content."""
    from anthropic.types import ContentBlockDeltaEvent, ThinkingDelta

    from any_llm.providers.anthropic.utils import _create_openai_chunk_from_anthropic_chunk

    delta_chunk = ContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=ThinkingDelta(type="thinking_delta", thinking="Let me think..."),
    )
    result = _create_openai_chunk_from_anthropic_chunk(delta_chunk, "claude-3-haiku")

    assert result.choices[0].delta.extra_content is None
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "Let me think..."


def test_non_streaming_response_preserves_multiple_tool_calls() -> None:
    from anthropic.types import Message, ToolUseBlock, Usage

    from any_llm.providers.anthropic.utils import _convert_response

    response = Message(
        id="msg_123",
        type="message",
        role="assistant",
        model="claude-3-haiku",
        stop_reason="tool_use",
        content=[
            ToolUseBlock(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Rome"}),
            ToolUseBlock(type="tool_use", id="toolu_2", name="get_time", input={"timezone": "UTC"}),
        ],
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    result = _convert_response(response)

    assert result.choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) == 2
    assert result.choices[0].message.tool_calls[0].id == "toolu_1"
    assert isinstance(result.choices[0].message.tool_calls[0], ChatCompletionMessageFunctionToolCall)
    assert result.choices[0].message.tool_calls[0].function is not None
    assert result.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert result.choices[0].message.tool_calls[1].id == "toolu_2"
    assert isinstance(result.choices[0].message.tool_calls[1], ChatCompletionMessageFunctionToolCall)
    assert result.choices[0].message.tool_calls[1].function is not None
    assert result.choices[0].message.tool_calls[1].function.name == "get_time"


def test_non_streaming_response_preserves_thinking_signature() -> None:
    """The encrypted thinking signature must be surfaced on the message's extra_content."""
    from anthropic.types import Message, ThinkingBlock, ToolUseBlock, Usage

    from any_llm.providers.anthropic.utils import _convert_response

    response = Message(
        id="msg_123",
        type="message",
        role="assistant",
        model="claude-3-haiku",
        stop_reason="tool_use",
        content=[
            ThinkingBlock(type="thinking", thinking="Let me reason...", signature="sig-12345"),
            ToolUseBlock(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Rome"}),
        ],
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    result = _convert_response(response)

    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "Let me reason..."
    assert result.choices[0].message.extra_content == {"anthropic": {"signature": "sig-12345"}}


def test_non_streaming_response_without_thinking_has_no_extra_content() -> None:
    from anthropic.types import Message, TextBlock, Usage

    from any_llm.providers.anthropic.utils import _convert_response

    response = Message(
        id="msg_123",
        type="message",
        role="assistant",
        model="claude-3-haiku",
        stop_reason="end_turn",
        content=[TextBlock(type="text", text="Hello")],
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    result = _convert_response(response)

    assert result.choices[0].message.extra_content is None


def test_non_streaming_response_empty_thinking_signature_has_no_extra_content() -> None:
    """An empty (falsy) signature, e.g. display='omitted' before the signature streams in, should not be stored."""
    from anthropic.types import Message, ThinkingBlock, Usage

    from any_llm.providers.anthropic.utils import _convert_response

    response = Message(
        id="msg_123",
        type="message",
        role="assistant",
        model="claude-3-haiku",
        stop_reason="end_turn",
        content=[ThinkingBlock(type="thinking", thinking="", signature="")],
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    result = _convert_response(response)

    assert result.choices[0].message.extra_content is None


def test_convert_messages_replays_thinking_block_with_tool_call() -> None:
    """Anthropic requires the unmodified thinking block (with signature) to be replayed
    alongside the tool_use block when continuing a turn that used extended thinking."""
    from any_llm.providers.anthropic.utils import _convert_messages_for_anthropic

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "Let me check the weather tool.",
            "extra_content": {"anthropic": {"signature": "sig-12345"}},
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": '{"temp": "20C"}'},
    ]

    _, converted = _convert_messages_for_anthropic(messages)

    assistant_message = converted[1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"][0] == {
        "type": "thinking",
        "thinking": "Let me check the weather tool.",
        "signature": "sig-12345",
    }
    assert assistant_message["content"][1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }


def test_convert_messages_replays_thinking_block_with_text() -> None:
    """A plain-text assistant message that carries a thinking signature should also replay it.

    Also covers the dict-shaped ``{"content": str}`` form of ``reasoning``, as opposed to the
    plain string form used in test_convert_messages_replays_thinking_block_with_tool_call.
    """
    from any_llm.providers.anthropic.utils import _convert_messages_for_anthropic

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is 2 + 2?"},
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning": {"content": "2 + 2 = 4."},
            "extra_content": {"anthropic": {"signature": "sig-67890"}},
        },
    ]

    _, converted = _convert_messages_for_anthropic(messages)

    assistant_message = converted[1]
    assert assistant_message["content"] == [
        {"type": "thinking", "thinking": "2 + 2 = 4.", "signature": "sig-67890"},
        {"type": "text", "text": "The answer is 4."},
    ]


def test_convert_messages_replays_thinking_block_with_list_content() -> None:
    """An assistant message whose content is already a list of blocks (not a plain string) should
    have the thinking block prepended to the existing blocks, not replace them."""
    from any_llm.providers.anthropic.utils import _convert_messages_for_anthropic

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Describe this image."},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "It's a cat."}],
            "reasoning": "Looking at the image.",
            "extra_content": {"anthropic": {"signature": "sig-list"}},
        },
    ]

    _, converted = _convert_messages_for_anthropic(messages)

    assistant_message = converted[1]
    assert assistant_message["content"] == [
        {"type": "thinking", "thinking": "Looking at the image.", "signature": "sig-list"},
        {"type": "text", "text": "It's a cat."},
    ]


def test_extract_anthropic_thinking_signature_ignores_non_string_signature() -> None:
    """A malformed signature (non-string) should be treated as absent, not crash."""
    from any_llm.providers.anthropic.utils import _extract_anthropic_thinking_signature

    message = {"extra_content": {"anthropic": {"signature": 12345}}}

    assert _extract_anthropic_thinking_signature(message) is None


def test_build_anthropic_thinking_block_defaults_to_empty_thinking_text() -> None:
    """When a signature is present but reasoning is missing or malformed, thinking text defaults to ''."""
    from any_llm.providers.anthropic.utils import _build_anthropic_thinking_block

    message = {"extra_content": {"anthropic": {"signature": "sig-no-reasoning"}}}

    assert _build_anthropic_thinking_block(message) == {
        "type": "thinking",
        "thinking": "",
        "signature": "sig-no-reasoning",
    }


def test_convert_messages_without_thinking_signature_unchanged() -> None:
    """Without a signature, assistant messages should be forwarded unchanged (no thinking block)."""
    from any_llm.providers.anthropic.utils import _convert_messages_for_anthropic

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there!"},
    ]

    _, converted = _convert_messages_for_anthropic(messages)

    assert converted[1] == {"role": "assistant", "content": "Hello there!"}


def test_convert_messages_ignores_unrelated_extra_content() -> None:
    """An extra_content dict without an 'anthropic' key (e.g. from a different provider) should be a no-op."""
    from any_llm.providers.anthropic.utils import _convert_messages_for_anthropic

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "Hello there!",
            "reasoning": "Greeting the user.",
            "extra_content": {"google": {"thought_signature": "unrelated"}},
        },
    ]

    _, converted = _convert_messages_for_anthropic(messages)

    assert converted[1] == {"role": "assistant", "content": "Hello there!"}


def test_convert_messages_replays_thinking_block_with_none_content() -> None:
    """A reasoning-only assistant turn (content=None, no tool_calls) must not crash on replay.

    Regression test: message.get("content") can be explicitly None rather than a string or
    list, e.g. for a turn that only produced reasoning and no visible text or tool call.
    """
    from any_llm.providers.anthropic.utils import _convert_messages_for_anthropic

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Just think about it, don't answer."},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "Thinking without responding.",
            "extra_content": {"anthropic": {"signature": "sig-none-content"}},
        },
    ]

    _, converted = _convert_messages_for_anthropic(messages)

    assistant_message = converted[1]
    assert assistant_message["content"] == [
        {"type": "thinking", "thinking": "Thinking without responding.", "signature": "sig-none-content"},
    ]


def test_convert_messages_replays_thinking_block_with_empty_string_content() -> None:
    """An empty string content (as opposed to None) should behave the same: only the thinking block is kept."""
    from any_llm.providers.anthropic.utils import _convert_messages_for_anthropic

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Just think about it, don't answer."},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "Thinking without responding.",
            "extra_content": {"anthropic": {"signature": "sig-empty-content"}},
        },
    ]

    _, converted = _convert_messages_for_anthropic(messages)

    assistant_message = converted[1]
    assert assistant_message["content"] == [
        {"type": "thinking", "thinking": "Thinking without responding.", "signature": "sig-empty-content"},
    ]


def test_convert_tool_spec_none_parameters() -> None:
    """Regression: parameters=None must not raise 'NoneType' object is not subscriptable."""
    tools = _convert_tool_spec([{"type": "function", "function": {"name": "ping", "parameters": None}}])
    assert len(tools) == 1
    assert tools[0]["name"] == "ping"
    assert tools[0]["input_schema"]["properties"] == {}
    assert tools[0]["input_schema"]["required"] == []


def test_convert_tool_spec_parameters_missing_properties() -> None:
    """Regression: parameters without a 'properties' key must not raise KeyError."""
    tools = _convert_tool_spec([{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}])
    assert len(tools) == 1
    assert tools[0]["input_schema"]["properties"] == {}
