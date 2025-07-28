from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from any_llm.provider import ApiConfig
from any_llm.providers.huggingface.huggingface import HuggingfaceProvider


@contextmanager
def mock_huggingface_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.huggingface.huggingface.InferenceClient") as mock_huggingface,
        patch("any_llm.providers.huggingface.huggingface.create_completion_from_response"),
    ):
        yield mock_huggingface


def test_huggingface_with_api_key() -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_huggingface_provider() as mock_huggingface:
        provider = HuggingfaceProvider(ApiConfig(api_key=api_key))
        provider._make_api_call("model-id", messages)

        mock_huggingface.assert_called_with(token=api_key, timeout=None)

        mock_huggingface.return_value.chat_completion.assert_called_with(model="model-id", messages=messages)


def test_huggingface_with_tools(tools: list[dict[str, Any]]) -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_huggingface_provider() as mock_huggingface:
        provider = HuggingfaceProvider(ApiConfig(api_key=api_key))
        provider._make_api_call("model-id", messages, tools=tools)

        mock_huggingface.assert_called_with(token=api_key, timeout=None)

        mock_huggingface.return_value.chat_completion.assert_called_with(
            model="model-id", messages=messages, tools=tools
        )


def test_huggingface_with_max_tokens() -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_huggingface_provider() as mock_huggingface:
        provider = HuggingfaceProvider(ApiConfig(api_key=api_key))
        provider._make_api_call("model-id", messages, max_tokens=100)

        mock_huggingface.assert_called_with(token=api_key, timeout=None)

        mock_huggingface.return_value.chat_completion.assert_called_with(
            model="model-id", messages=messages, max_new_tokens=100
        )
