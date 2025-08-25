from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.provider import ApiConfig
from any_llm.types.completion import ChatCompletion, CompletionParams


@contextmanager
def mock_xai_provider():  # type: ignore[no-untyped-def]
    with patch("any_llm.providers.xai.xai.XaiAsyncClient") as mock_xai:
        create_return = MagicMock()
        mock_response = MagicMock()
        mock_response.reasoning_content = None
        mock_response.content = "Test response"
        mock_response.id = "Test id"
        mock_response.proto.model = "Test model"
        mock_response.proto.created.seconds = 0
        mock_response.tool_calls = None
        create_return.sample = AsyncMock(return_value=mock_response)
        mock_xai.return_value.chat.create = MagicMock(return_value=create_return)

        yield mock_xai, mock_response


@pytest.mark.asyncio
async def test_response_function_call_id_is_preserved() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (_, mock_response):
        tool_call = MagicMock()
        tool_call.id = "expected_function_call_id"
        tool_call.function.name = "test_function"
        tool_call.function.arguments = '{"key": "value"}'
        mock_response.tool_calls = [tool_call]

        provider = XaiProvider(ApiConfig(api_key="test-api-key"))
        response = await provider.acompletion(
            CompletionParams(model_id="model", messages=[{"role": "user", "content": "Hello"}])
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices
        assert response.choices[0].message.tool_calls
        assert response.choices[0].message.tool_calls[0].id == "expected_function_call_id"


@pytest.mark.asyncio
async def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with mock_xai_provider() as (mock_xai, _):
        provider = XaiProvider(ApiConfig(api_key="test-api-key"))
        await provider.acompletion(CompletionParams(model_id="model", messages=agent_loop_messages))
        _, call_kwargs = mock_xai.return_value.chat.create.call_args

        assert len(call_kwargs["messages"]) == 3
