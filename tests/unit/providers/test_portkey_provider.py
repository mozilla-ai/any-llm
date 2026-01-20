from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.types.completion import CompletionParams


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

        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

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
