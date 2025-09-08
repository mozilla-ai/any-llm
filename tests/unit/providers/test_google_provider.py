from contextlib import contextmanager
from typing import Any, Literal
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ClientConfig, Provider
from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.base import REASONING_EFFORT_TO_THINKING_BUDGETS
from any_llm.providers.vertexai import VertexaiProvider
from any_llm.types.completion import CompletionParams


@pytest.fixture(params=[GeminiProvider, VertexaiProvider])
def google_provider_class(request: pytest.FixtureRequest) -> type[Provider]:
    """Parametrized fixture that provides both GeminiProvider and VertexaiProvider classes."""
    return request.param  # type: ignore[no-any-return]


@contextmanager
def mock_google_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.gemini.base.genai.Client") as mock_genai,
        patch("any_llm.providers.gemini.base._convert_response_to_response_dict") as mock_convert_response,
        patch.dict("os.environ", {"GOOGLE_PROJECT_ID": "test-project", "GOOGLE_REGION": "us-central1"}),
    ):
        mock_convert_response.return_value = {
            "id": "google_genai_response",
            "model": "gemini/genai",
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


@pytest.mark.parametrize("env_var", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_gemini_initialization_with_env_var_api_key(env_var: str) -> None:
    """Test that the provider initializes correctly with API key from environment variable."""
    with patch.dict("os.environ", {env_var: "env-api-key"}, clear=True):
        provider = GeminiProvider(ClientConfig())
        assert provider.config.api_key == "env-api-key"


def test_vertexai_initialization_with_env_var_api_key() -> None:
    """Test that the VertexaiProvider initializes correctly with GOOGLE_PROJECT_ID from environment variable."""
    with patch.dict("os.environ", {"GOOGLE_PROJECT_ID": "env-project-id"}, clear=True):
        provider = VertexaiProvider(ClientConfig())
        assert provider.config.api_key == "env-project-id"


@pytest.mark.asyncio
async def test_completion_with_system_instruction(google_provider_class: type[Provider]) -> None:
    """Test that completion works correctly with system_instruction."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = google_provider_class(ClientConfig(api_key=api_key))
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
async def test_completion_with_tool_choice_auto(
    google_provider_class: type[Provider], tool_choice: str, expected_mode: str
) -> None:
    """Test that completion correctly processes tool_choice='auto'."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = google_provider_class(ClientConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages, tool_choice=tool_choice))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config.function_calling_config.mode.value == expected_mode


@pytest.mark.asyncio
async def test_completion_without_tool_choice(google_provider_class: type[Provider]) -> None:
    """Test that completion works correctly without tool_choice."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = google_provider_class(ClientConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config is None


@pytest.mark.asyncio
async def test_completion_with_stream_and_response_format_raises(google_provider_class: type[Provider]) -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider():
        provider = google_provider_class(ClientConfig(api_key=api_key))
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
async def test_completion_with_parallel_tool_calls_raises(google_provider_class: type[Provider]) -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider():
        provider = google_provider_class(ClientConfig(api_key=api_key))
        with pytest.raises(UnsupportedParameterError):
            await provider.acompletion(
                CompletionParams(
                    model_id=model,
                    messages=messages,
                    parallel_tool_calls=True,
                )
            )


@pytest.mark.asyncio
async def test_completion_inside_agent_loop(
    google_provider_class: type[Provider], agent_loop_messages: list[dict[str, Any]]
) -> None:
    api_key = "test-api-key"
    model = "gemini-pro"

    with mock_google_provider() as mock_genai:
        provider = google_provider_class(ClientConfig(api_key=api_key))
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
    google_provider_class: type[Provider],
    reasoning_effort: Literal["low", "medium", "high"] | None,
) -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = google_provider_class(ClientConfig(api_key=api_key))
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
async def test_completion_with_max_tokens_conversion(google_provider_class: type[Provider]) -> None:
    """Test that max_tokens parameter gets converted to max_output_tokens."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]
    max_tokens = 100

    with mock_google_provider() as mock_genai:
        provider = google_provider_class(ClientConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages, max_tokens=max_tokens))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.max_output_tokens == max_tokens
