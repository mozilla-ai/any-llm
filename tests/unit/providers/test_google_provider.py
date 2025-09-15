from contextlib import contextmanager
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.genai import types

from any_llm.config import ClientConfig
from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import Provider
from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.base import REASONING_EFFORT_TO_THINKING_BUDGETS
from any_llm.providers.gemini.utils import _convert_response_to_response_dict
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


@pytest.mark.asyncio
async def test_completion_with_content_list(google_provider_class: type[Provider]) -> None:
    """Test that completion works correctly with content in list format."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

    with mock_google_provider() as mock_genai:
        provider = google_provider_class(ClientConfig(api_key=api_key))
        await provider.acompletion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        contents = call_kwargs["contents"]

        assert contents[0].parts[0].text == "Hello"


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


def test_convert_response_single_tool_call() -> None:
    """Test conversion of Google response with a single tool call to OpenAI format."""
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()
    mock_response.candidates[0].content.parts = [Mock()]

    mock_function_call = Mock()
    mock_function_call.name = "search_web"
    mock_function_call.args = {"query": "test query", "limit": 5}

    mock_response.candidates[0].content.parts[0].function_call = mock_function_call
    mock_response.candidates[0].content.parts[0].thought = None
    mock_response.candidates[0].content.parts[0].text = None

    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 15
    mock_response.usage_metadata.total_token_count = 25

    response_dict = _convert_response_to_response_dict(mock_response)

    assert len(response_dict["choices"]) == 1
    choice = response_dict["choices"][0]

    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] is None
    assert choice["finish_reason"] == "tool_calls"
    assert choice["index"] == 0

    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1

    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "search_web"
    assert tool_call["function"]["arguments"] == '{"query": "test query", "limit": 5}'
    assert tool_call["id"].startswith("call_")
    assert tool_call["id"].endswith("_0")


def test_convert_response_multiple_parallel_tool_calls() -> None:
    """Test conversion of Google response with multiple parallel tool calls to OpenAI format."""
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()

    mock_function_call_1 = Mock()
    mock_function_call_1.name = "search_web"
    mock_function_call_1.args = {"query": "test query"}

    mock_function_call_2 = Mock()
    mock_function_call_2.name = "get_weather"
    mock_function_call_2.args = {"location": "New York"}

    mock_function_call_3 = Mock()
    mock_function_call_3.name = "calculate"
    mock_function_call_3.args = {"expression": "2+2"}

    mock_part_1 = Mock()
    mock_part_1.function_call = mock_function_call_1
    mock_part_1.thought = None
    mock_part_1.text = None

    mock_part_2 = Mock()
    mock_part_2.function_call = mock_function_call_2
    mock_part_2.thought = None
    mock_part_2.text = None

    mock_part_3 = Mock()
    mock_part_3.function_call = mock_function_call_3
    mock_part_3.thought = None
    mock_part_3.text = None

    mock_response.candidates[0].content.parts = [mock_part_1, mock_part_2, mock_part_3]

    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 20
    mock_response.usage_metadata.candidates_token_count = 30
    mock_response.usage_metadata.total_token_count = 50

    response_dict = _convert_response_to_response_dict(mock_response)

    assert len(response_dict["choices"]) == 1
    choice = response_dict["choices"][0]

    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] is None
    assert choice["finish_reason"] == "tool_calls"
    assert choice["index"] == 0

    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 3

    assert tool_calls[0]["function"]["name"] == "search_web"
    assert tool_calls[0]["function"]["arguments"] == '{"query": "test query"}'
    assert tool_calls[0]["id"].endswith("_0")

    assert tool_calls[1]["function"]["name"] == "get_weather"
    assert tool_calls[1]["function"]["arguments"] == '{"location": "New York"}'
    assert tool_calls[1]["id"].endswith("_1")

    assert tool_calls[2]["function"]["name"] == "calculate"
    assert tool_calls[2]["function"]["arguments"] == '{"expression": "2+2"}'
    assert tool_calls[2]["id"].endswith("_2")

    tool_call_ids = [tc["id"] for tc in tool_calls]
    assert len(set(tool_call_ids)) == 3
