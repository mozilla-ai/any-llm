from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.types.completion import CompletionParams


@pytest.mark.asyncio
async def test_stream_with_response_format_raises() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(ApiConfig(api_key="test-api-key"))

    with pytest.raises(UnsupportedParameterError, match="stream and response_format"):
        await provider.acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_unsupported_max_tool_calls_parameter() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(ApiConfig(api_key="test-api-key"))

    with pytest.raises(UnsupportedParameterError):
        await provider.aresponses("test_model", "test_data", max_tool_calls=3)


@pytest.mark.asyncio
async def test_groq_accepts_http_client_async() -> None:
    """Test that AsyncGroq client accepts and passes through http_client."""
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    api_key = "test-api-key"
    mock_http_client = Mock(spec=httpx.AsyncClient)

    with (
        patch("groq.AsyncGroq") as mock_async_groq,
        patch("any_llm.providers.groq.utils.to_chat_completion", return_value=Mock()),
    ):
        mock_client = Mock()
        mock_async_groq.return_value = mock_client
        mock_client.chat.completions.create = Mock(return_value=Mock())

        provider = GroqProvider(ApiConfig(api_key=api_key))
        await provider.acompletion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]),
            http_client=mock_http_client,
        )

        # Verify AsyncGroq client was instantiated with http_client
        mock_async_groq.assert_called_once_with(api_key=api_key, base_url=None, http_client=mock_http_client)


@pytest.mark.asyncio
async def test_completion_with_response_format_basemodel() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    class TestOutput(BaseModel):
        foo: str

    provider = GroqProvider(ApiConfig(api_key="test-api-key"))

    with (
        patch("any_llm.providers.groq.groq.groq.AsyncGroq") as mocked_groq,
        patch("any_llm.providers.groq.groq.to_chat_completion") as mocked_to_chat_completion,
    ):
        mock_response = Mock()
        mocked_groq.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        mocked_to_chat_completion.return_value = mock_response

        await provider.acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=TestOutput,
            ),
        )
        call_kwargs = mocked_groq.return_value.chat.completions.create.call_args[1]

        assert call_kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "TestOutput",
                "schema": {
                    "properties": {"foo": {"title": "Foo", "type": "string"}},
                    "required": ["foo"],
                    "title": "TestOutput",
                    "type": "object",
                },
            },
        }
