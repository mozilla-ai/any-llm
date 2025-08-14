import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from any_llm.provider import ApiConfig, ProviderFactory
from any_llm.types.completion import ChatCompletion, CompletionParams


@contextmanager
def mock_xai_provider():  # type: ignore[no-untyped-def]
    with patch("any_llm.providers.xai.xai.XaiClient") as mock_xai:
        create_return = MagicMock()
        mock_response = MagicMock()
        mock_response.reasoning_content = None
        mock_response.content = "Test response"
        mock_response.id = "Test id"
        mock_response.proto.model = "Test model"
        mock_response.proto.created.seconds = 0
        mock_response.tool_calls = None
        create_return.sample.return_value = mock_response
        mock_xai.return_value.chat.create.return_value = create_return

        yield mock_xai, mock_response


def test_provider_with_no_packages_installed() -> None:
    with patch.dict(sys.modules, dict.fromkeys(["xai_sdk"])):
        try:
            import any_llm.providers.xai  # noqa: F401
        except ImportError:
            pytest.fail("Import raised an unexpected ImportError")


def test_call_to_provider_with_no_packages_installed() -> None:
    packages = ["xai_sdk"]
    with patch.dict(sys.modules, dict.fromkeys(packages)):
        with pytest.raises(ImportError, match="xai required packages are not installed"):
            ProviderFactory.create_provider("xai", ApiConfig())


def test_response_function_call_id_is_preserved() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (_, mock_response):
        tool_call = MagicMock()
        tool_call.id = "expected_function_call_id"
        tool_call.function.name = "test_function"
        tool_call.function.arguments = '{"key": "value"}'
        mock_response.tool_calls = [tool_call]

        provider = XaiProvider(ApiConfig(api_key="test-api-key"))
        response = provider.completion(
            CompletionParams(model_id="model", messages=[{"role": "user", "content": "Hello"}])
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices
        assert response.choices[0].message.tool_calls
        assert response.choices[0].message.tool_calls[0].id == "expected_function_call_id"


def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(ApiConfig(api_key="test-api-key"))
        provider.completion(CompletionParams(model_id="model", messages=agent_loop_messages))
        _, call_kwargs = mock_xai.return_value.chat.create.call_args

        assert len(call_kwargs["messages"]) == 3
