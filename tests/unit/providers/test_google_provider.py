from contextlib import contextmanager
from unittest.mock import patch

import pytest

from any_llm.provider import ApiConfig
from any_llm.providers.google.google import GoogleProvider


@contextmanager
def mock_google_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.google.google.genai.Client") as mock_genai,
        patch("any_llm.providers.google.google._convert_messages"),
        patch("any_llm.providers.google.google._convert_response_to_response_dict"),
        patch("any_llm.providers.google.google.create_completion_from_response"),
    ):
        yield mock_genai


@pytest.mark.parametrize(
    "tool_choice,expected_mode",
    [
        ("auto", "AUTO"),
        ("required", "ANY"),
    ],
)
def test_make_api_call_with_tool_choice_auto(tool_choice: str, expected_mode: str) -> None:
    """Test that _make_api_call correctly processes tool_choice='auto'."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]
    kwargs = {"tool_choice": tool_choice}

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        provider._make_api_call(model, messages, **kwargs)

        _, call_kwargs = mock_genai.return_value.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config.function_calling_config.mode.value == expected_mode


def test_make_api_call_without_tool_choice() -> None:
    """Test that _make_api_call works correctly without tool_choice."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_google_provider() as mock_genai:
        provider = GoogleProvider(ApiConfig(api_key=api_key))
        provider._make_api_call(model, messages)

        _, call_kwargs = mock_genai.return_value.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config is None
