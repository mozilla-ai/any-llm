import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig, ProviderFactory
from any_llm.providers.google.google import GoogleProvider


@contextmanager
def mock_google_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.google.google.genai.Client") as mock_genai,
        patch("any_llm.providers.google.google._convert_messages") as mock_convert_messages,
        patch("any_llm.providers.google.google._convert_response_to_response_dict") as mock_convert_response,
    ):
        mock_convert_messages.return_value = [SimpleNamespace(role="user", parts=[SimpleNamespace(text="Hello")])]
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


@pytest.mark.parametrize(
    ("tool_choice", "expected_mode"),
    [
        ("auto", "AUTO"),
        ("required", "ANY"),
    ],
)
def testcompletion_with_tool_choice_auto(tool_choice: str, expected_mode: str) -> None:
    """Test that completion correctly processes tool_choice='auto'."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]
    kwargs = {"tool_choice": tool_choice}

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        provider.completion(model, messages, **kwargs)

        _, call_kwargs = mock_genai.return_value.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config.function_calling_config.mode.value == expected_mode


def testcompletion_without_tool_choice() -> None:
    """Test that completion works correctly without tool_choice."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        provider.completion(model, messages)

        _, call_kwargs = mock_genai.return_value.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config is None


def testcompletion_with_stream_and_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider():
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        with pytest.raises(UnsupportedParameterError):
            provider.completion(
                model,
                messages,
                stream=True,
                response_format={"type": "json_object"},
            )


def testcompletion_with_parallel_tool_calls_raises() -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider():
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        with pytest.raises(UnsupportedParameterError):
            provider.completion(
                model,
                messages,
                parallel_tool_calls=True,
            )


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
