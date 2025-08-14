import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig, ProviderFactory
from any_llm.providers.google.google import GoogleProvider
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
        yield mock_genai


def test_completion_with_system_instruction() -> None:
    """Test that completion works correctly with system_instruction."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        provider.completion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.models.generate_content.call_args
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
def test_completion_with_tool_choice_auto(tool_choice: str, expected_mode: str) -> None:
    """Test that completion correctly processes tool_choice='auto'."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]
    # pass tool_choice explicitly to avoid ambiguous **kwargs typing issues

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        provider.completion(CompletionParams(model_id=model, messages=messages, tool_choice=tool_choice))

        _, call_kwargs = mock_genai.return_value.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config.function_calling_config.mode.value == expected_mode


def test_completion_without_tool_choice() -> None:
    """Test that completion works correctly without tool_choice."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        provider.completion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config is None


def test_completion_with_stream_and_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider():
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        with pytest.raises(UnsupportedParameterError):
            provider.completion(
                CompletionParams(
                    model_id=model,
                    messages=messages,
                    stream=True,
                    response_format={"type": "json_object"},
                )
            )


def test_completion_with_parallel_tool_calls_raises() -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider():
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        with pytest.raises(UnsupportedParameterError):
            provider.completion(
                CompletionParams(
                    model_id=model,
                    messages=messages,
                    parallel_tool_calls=True,
                )
            )


def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    api_key = "test-api-key"
    model = "gemini-pro"

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        provider.completion(CompletionParams(model_id=model, messages=agent_loop_messages))

        _, call_kwargs = mock_genai.return_value.models.generate_content.call_args

        contents = call_kwargs["contents"]
        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "function"


def test_provider_with_no_packages_installed() -> None:
    with patch.dict(sys.modules, dict.fromkeys(["google"])):
        try:
            import any_llm.providers.google  # noqa: F401
        except ImportError:
            pytest.fail("Import raised an unexpected ImportError")


def test_call_to_provider_with_no_packages_installed() -> None:
    packages = ["google"]
    with patch.dict(sys.modules, dict.fromkeys(packages)):
        # Ensure a fresh import under the patched environment so PACKAGES_INSTALLED is recalculated
        for mod in list(sys.modules):
            if mod.startswith("any_llm.providers.google"):
                sys.modules.pop(mod)
        with pytest.raises(ImportError, match="google required packages are not installed"):
            ProviderFactory.create_provider("google", ApiConfig())
