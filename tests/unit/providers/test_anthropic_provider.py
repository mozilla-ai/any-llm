from contextlib import contextmanager
from datetime import UTC
from typing import Any, get_args
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.providers.anthropic.utils import DEFAULT_MAX_TOKENS, REASONING_EFFORT_TO_THINKING_BUDGETS
from any_llm.types.completion import CompletionParams, ReasoningEffort


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
            assert call_kwargs["thinking"] == {
                "type": "enabled",
                "budget_tokens": REASONING_EFFORT_TO_THINKING_BUDGETS[reasoning_effort],
            }


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
    from anthropic import transform_schema
    from pydantic import BaseModel

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
async def test_completion_with_response_format_dict_json_schema() -> None:
    from anthropic import transform_schema

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
async def test_stream_with_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = AnthropicProvider(api_key=api_key)

    with pytest.raises(UnsupportedParameterError, match="stream and response_format"):
        await provider._acompletion(
            CompletionParams(
                model_id=model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": {"name": "Foo", "schema": {}}},
                stream=True,
            )
        )


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
