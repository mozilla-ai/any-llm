from contextlib import contextmanager
from typing import Any, Literal
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.providers.google.google import REASONING_EFFORT_TO_THINKING_BUDGETS, GoogleProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_google_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.google.google.genai.Client") as mock_genai,
        patch("any_llm.providers.google.google._convert_response_to_response_dict") as mock_convert_response,
    ):
        mock_convert_response.return_value = {
            "id": "google_genai_response",
            "model": "google/genai",
            "created": 0,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok", "tool_calls": None},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        # Set up the async method properly
        mock_client = mock_genai.return_value
        mock_client.aio.models.generate_content = AsyncMock()

        yield mock_genai


@pytest.mark.asyncio
async def test_completion_with_system_instruction() -> None:
    """Test that completion works correctly with system_instruction."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]
        contents = call_kwargs["contents"]

        assert len(contents) == 1
        assert generation_config.system_instruction == "You are a helpful assistant"


@pytest.mark.parametrize(
    ("tool_choice", "expected_mode"),
    [
        ("auto", "AUTO"),
        ("required", "ANY"),
    ],
)
@pytest.mark.asyncio
async def test_completion_with_tool_choice_auto(tool_choice: str, expected_mode: str) -> None:
    """Test that completion correctly processes tool_choice='auto'."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages, tool_choice=tool_choice))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config.function_calling_config.mode.value == expected_mode


@pytest.mark.asyncio
async def test_completion_without_tool_choice() -> None:
    """Test that completion works correctly without tool_choice."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config is None


@pytest.mark.asyncio
async def test_completion_with_stream_and_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider():
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        with pytest.raises(UnsupportedParameterError):
            await provider.acompletion(
                CompletionParams(
                    model_id=model,
                    messages=messages,
                    stream=True,
                    response_format={"type": "json_object"},
                )
            )


@pytest.mark.asyncio
async def test_completion_with_parallel_tool_calls_raises() -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider():
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        with pytest.raises(UnsupportedParameterError):
            await provider.acompletion(
                CompletionParams(
                    model_id=model,
                    messages=messages,
                    parallel_tool_calls=True,
                )
            )


@pytest.mark.asyncio
async def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    api_key = "test-api-key"
    model = "gemini-pro"

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=agent_loop_messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args

        contents = call_kwargs["contents"]
        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "function"


@pytest.mark.parametrize(
    "reasoning_effort",
    [
        None,
        "low",
        "medium",
        "high",
    ],
)
@pytest.mark.asyncio
async def test_completion_with_custom_reasoning_effort(
    reasoning_effort: Literal["low", "medium", "high"] | None,
) -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(
            CompletionParams(model_id=model, messages=messages, reasoning_effort=reasoning_effort)
        )

        if reasoning_effort is None:
            expected_thinking = types.ThinkingConfig(include_thoughts=False)
        else:
            expected_thinking = types.ThinkingConfig(
                include_thoughts=True, thinking_budget=REASONING_EFFORT_TO_THINKING_BUDGETS[reasoning_effort]
            )
        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        assert call_kwargs["config"].thinking_config == expected_thinking


@pytest.mark.asyncio
async def test_completion_with_max_tokens_conversion() -> None:
    """Test that max_tokens parameter gets converted to max_output_tokens."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]
    max_tokens = 100

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages, max_tokens=max_tokens))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.max_output_tokens == max_tokens
