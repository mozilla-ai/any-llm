from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
)
from any_llm.types.model import Model


class MockPortkeyCompletion(BaseModel):
    """Mock Pydantic model to simulate portkey SDK's ChatCompletions type."""

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int] | None = None


class MockPortkeyChunk(BaseModel):
    """Mock Pydantic model to simulate portkey SDK's ChatCompletionChunk type."""

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]


class MockPortkeyModel(BaseModel):
    """Mock Pydantic model to simulate portkey SDK's Model type."""

    id: str
    object: str
    created: int | None = None
    owned_by: str | None = None


@pytest.mark.asyncio
async def test_init_client_with_api_key() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    with patch("any_llm.providers.portkey.portkey.AsyncPortkey") as mocked_portkey:
        provider = PortkeyProvider(api_key="test-api-key")
        mocked_portkey.assert_called_once_with(
            api_key="test-api-key",
            base_url="https://api.portkey.ai/v1",
        )
        assert provider.client == mocked_portkey.return_value


@pytest.mark.asyncio
async def test_init_client_with_custom_base_url() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    with patch("any_llm.providers.portkey.portkey.AsyncPortkey") as mocked_portkey:
        provider = PortkeyProvider(api_key="test-api-key", api_base="https://custom.portkey.ai/v1")
        mocked_portkey.assert_called_once_with(
            api_key="test-api-key",
            base_url="https://custom.portkey.ai/v1",
        )
        assert provider.client == mocked_portkey.return_value


@pytest.mark.asyncio
async def test_convert_completion_params_with_pydantic_model() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    class TestOutput(BaseModel):
        name: str
        value: int

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        response_format=TestOutput,
    )

    converted = PortkeyProvider._convert_completion_params(params)

    assert converted["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response_schema",
            "schema": {
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "value": {"title": "Value", "type": "integer"},
                },
                "required": ["name", "value"],
                "title": "TestOutput",
                "type": "object",
            },
        },
    }


@pytest.mark.asyncio
async def test_convert_completion_params_with_dict_response_format() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        response_format={"type": "json_object"},
    )

    converted = PortkeyProvider._convert_completion_params(params)

    assert converted["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_convert_completion_params_without_response_format() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    converted = PortkeyProvider._convert_completion_params(params)

    assert "response_format" not in converted


@pytest.mark.asyncio
async def test_acompletion_calls_client() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    with patch("any_llm.providers.portkey.portkey.AsyncPortkey") as mocked_portkey:
        provider = PortkeyProvider(api_key="test-api-key")

        mock_response = MockPortkeyCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )

        mocked_portkey.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            ),
        )

        mocked_portkey.return_value.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_provider_supports_flags() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    assert PortkeyProvider.SUPPORTS_COMPLETION is True
    assert PortkeyProvider.SUPPORTS_COMPLETION_STREAMING is True
    assert PortkeyProvider.SUPPORTS_COMPLETION_REASONING is True
    assert PortkeyProvider.SUPPORTS_EMBEDDING is False
    assert PortkeyProvider.SUPPORTS_RESPONSES is False
    assert PortkeyProvider.SUPPORTS_LIST_MODELS is True


@pytest.mark.asyncio
async def test_provider_config() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    assert PortkeyProvider.API_BASE == "https://api.portkey.ai/v1"
    assert PortkeyProvider.ENV_API_KEY_NAME == "PORTKEY_API_KEY"
    assert PortkeyProvider.PROVIDER_NAME == "portkey"


def test_convert_completion_response_with_portkey_sdk_type() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    mock_portkey_response = MockPortkeyCompletion(
        id="chatcmpl-123",
        object="chat.completion",
        created=1234567890,
        model="test-model",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from portkey!"},
                "finish_reason": "stop",
            }
        ],
    )

    result = PortkeyProvider._convert_completion_response(cast("Any", mock_portkey_response))

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello from portkey!"


def test_convert_completion_chunk_response_with_portkey_sdk_type() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    mock_portkey_chunk = MockPortkeyChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="test-model",
        choices=[
            {
                "index": 0,
                "delta": {"content": "Hello from portkey!"},
                "finish_reason": None,
            }
        ],
    )

    result = PortkeyProvider._convert_completion_chunk_response(cast("Any", mock_portkey_chunk))

    assert isinstance(result, ChatCompletionChunk)
    assert result.choices[0].delta.content == "Hello from portkey!"


def test_convert_list_models_response_with_model_type() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    mock_model = MockPortkeyModel(
        id="test-model",
        object="model",
        created=1234567890,
        owned_by="test",
    )
    mock_response = Mock()
    mock_response.data = [mock_model]

    result = PortkeyProvider._convert_list_models_response(mock_response)

    assert len(result) == 1
    assert result[0].id == "test-model"


def test_convert_list_models_response_with_portkey_sdk_type() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    mock_model = MockPortkeyModel(
        id="portkey-model",
        object="model",
        created=1234567890,
        owned_by="portkey",
    )
    mock_response = Mock()
    mock_response.data = [mock_model]

    result = PortkeyProvider._convert_list_models_response(mock_response)

    assert len(result) == 1
    assert isinstance(result[0], Model)
    assert result[0].id == "portkey-model"


def test_convert_list_models_response_with_none_fields() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    mock_model = MockPortkeyModel(
        id="portkey-model",
        object="model",
        created=None,
        owned_by=None,
    )
    mock_response = Mock()
    mock_response.data = [mock_model]

    result = PortkeyProvider._convert_list_models_response(mock_response)

    assert len(result) == 1
    assert isinstance(result[0], Model)
    assert result[0].id == "portkey-model"
    assert result[0].created == 0
    assert result[0].owned_by == "portkey"


def test_convert_list_models_response_with_none_data() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    mock_response = Mock()
    mock_response.data = None

    result = PortkeyProvider._convert_list_models_response(mock_response)

    assert len(result) == 0


@pytest.mark.asyncio
async def test_convert_completion_response_async_non_streaming() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    with patch("any_llm.providers.portkey.portkey.AsyncPortkey"):
        provider = PortkeyProvider(api_key="test-api-key")

        mock_response = MockPortkeyCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
        )

        result = provider._convert_completion_response_async(cast("Any", mock_response))

        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content == "Hello!"


@pytest.mark.asyncio
async def test_convert_completion_response_async_streaming() -> None:
    pytest.importorskip("portkey_ai")
    from any_llm.providers.portkey.portkey import PortkeyProvider

    with patch("any_llm.providers.portkey.portkey.AsyncPortkey"):
        provider = PortkeyProvider(api_key="test-api-key")

        async def mock_stream() -> AsyncIterator[MockPortkeyChunk]:
            chunks = [
                MockPortkeyChunk(
                    id="chatcmpl-123",
                    object="chat.completion.chunk",
                    created=1234567890,
                    model="test-model",
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": "Hello"},
                            "finish_reason": None,
                        }
                    ],
                ),
                MockPortkeyChunk(
                    id="chatcmpl-123",
                    object="chat.completion.chunk",
                    created=1234567890,
                    model="test-model",
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": " World"},
                            "finish_reason": "stop",
                        }
                    ],
                ),
            ]
            for chunk in chunks:
                yield chunk

        mock_response = mock_stream()

        result = provider._convert_completion_response_async(cast("Any", mock_response))

        # Result should be an async iterator
        result_iter = cast("AsyncIterator[ChatCompletionChunk]", result)

        chunks = []
        async for chunk in result_iter:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " World"
