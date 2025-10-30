from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from any_llm.providers.sambanova.sambanova import SambanovaProvider
from any_llm.types.completion import CompletionParams


class PersonSchema(BaseModel):
    name: str
    age: int


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_converts_pydantic_response_format(mock_openai_class: MagicMock) -> None:
    """Test that Pydantic BaseModel response_format is converted to JSON schema format."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_client.chat.completions.parse = AsyncMock(return_value=mock_response)

    provider = SambanovaProvider(api_key="test-key")

    messages = [{"role": "user", "content": "Hello"}]
    params = CompletionParams(model_id="test-model", messages=messages, response_format=PersonSchema)

    await provider._acompletion(params)

    # Verify the client was called with the converted response_format
    mock_client.chat.completions.parse.assert_called_once()
    call_args = mock_client.chat.completions.parse.call_args

    assert call_args is not None
    kwargs = call_args.kwargs

    expected_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "response_schema",
            "schema": PersonSchema.model_json_schema(),
        },
    }

    assert kwargs["response_format"] == expected_response_format
    assert kwargs["model"] == "test-model"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_preserves_dict_response_format(mock_openai_class: MagicMock) -> None:
    """Test that dict response_format is passed through unchanged."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_client.chat.completions.parse = AsyncMock(return_value=mock_response)

    provider = SambanovaProvider(api_key="test-key")

    messages = [{"role": "user", "content": "Hello"}]
    dict_response_format = {"type": "json_object"}
    params = CompletionParams(model_id="test-model", messages=messages, response_format=dict_response_format)

    await provider._acompletion(params)

    # Verify the client was called with the original dict response_format
    mock_client.chat.completions.parse.assert_called_once()
    call_args = mock_client.chat.completions.parse.call_args

    assert call_args is not None
    kwargs = call_args.kwargs

    assert kwargs["response_format"] == dict_response_format
    assert kwargs["model"] == "test-model"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@pytest.mark.asyncio
async def test_sambanova_without_response_format(mock_openai_class: MagicMock) -> None:
    """Test normal completion without response_format."""
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = SambanovaProvider(api_key="test-key")

    messages = [{"role": "user", "content": "Hello"}]
    params = CompletionParams(model_id="test-model", messages=messages)

    await provider._acompletion(params)

    # Verify the normal create method was called
    mock_client.chat.completions.create.assert_called_once()
    mock_client.chat.completions.parse.assert_not_called()
