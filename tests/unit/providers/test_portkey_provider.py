from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from portkey_ai.api_resources.types.chat_complete_type import (
    ChatCompletions as PortkeyChatCompletions,
)
from portkey_ai.api_resources.types.models_type import (
    Model as PortkeyModel,
)
from portkey_ai.api_resources.types.models_type import (
    ModelList as PortkeyModelList,
)

from any_llm.providers.portkey.portkey import PortkeyProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


def test_convert_completion_params_with_dataclass_response_format() -> None:
    """Test that dataclass response_format is converted to JSON schema format."""

    @dataclass
    class TestOutput:
        name: str
        value: int

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        response_format=TestOutput,
    )

    result = PortkeyProvider._convert_completion_params(params)

    assert "response_format" in result
    assert result["response_format"]["type"] == "json_schema"
    assert result["response_format"]["json_schema"]["name"] == "response_schema"
    assert "properties" in result["response_format"]["json_schema"]["schema"]
    assert "name" in result["response_format"]["json_schema"]["schema"]["properties"]
    assert "value" in result["response_format"]["json_schema"]["schema"]["properties"]


def test_portkey_convert_completion_response_creates_anyllm_chatcompletion() -> None:
    """
    Test that the _convert_completion_response_async converts a portkey ChatCompletion Object
    into an anyllm ChatCompletion Object
    """

    provider = PortkeyProvider(api_key="test")

    response = PortkeyChatCompletions(
        id="test",
        choices=[],
        created=123,
        model="test-model",
        object="chat.completion",
    )

    result = provider._convert_completion_response(response)

    assert isinstance(result, ChatCompletion)
    assert result.id == "test"
    assert result.model == "test-model"


@pytest.mark.asyncio
async def test_acompletion_with_async_portkey() -> None:
    """Test acompletion works with AsyncPortkey Provider"""

    with patch("any_llm.providers.portkey.portkey.AsyncPortkey") as mocked_portkey:
        mock_client = AsyncMock()
        mocked_portkey.return_value = mock_client

        response = PortkeyChatCompletions(
            id="test",
            choices=[],
            created=123,
            model="test-model",
            object="chat.completion",
        )

        mock_client.chat.completions.create = AsyncMock(return_value=response)

        provider = PortkeyProvider(api_key="test")

        result = await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

    mock_client.chat.completions.create.assert_called_once()
    assert isinstance(result, ChatCompletion)
    assert result.id == "test"
    assert result.model == "test-model"


def test_portkey_convert_completion_chunk_response_creates_anyllm_chatcompletion() -> None:
    """
    Test that the _convert_completion_chunk_response converts a portkey ChatCompletion Object
    into an anyllm ChatCompletion Object
    """

    provider = PortkeyProvider(api_key="test")

    class FakePortkeyChunk:
        def model_dump(self) -> dict[str, Any]:
            return {
                "id": "test",
                "choices": [],
                "created": 123,
                "model": "test-model",
                "object": "chat.completion.chunk",
            }

    response = FakePortkeyChunk()

    result = provider._convert_completion_chunk_response(response)

    assert isinstance(result, ChatCompletionChunk)
    assert result.id == "test"
    assert result.model == "test-model"


@pytest.mark.asyncio
async def test_acompletion_with_async_chunk_portkey() -> None:
    """Test acompletion works with AsyncPortkey Provider"""

    with patch("any_llm.providers.portkey.portkey.AsyncPortkey") as mocked_portkey:
        mock_client = AsyncMock()
        mocked_portkey.return_value = mock_client

        class FakePortkeyChunk:
            def model_dump(self) -> dict[str, Any]:
                return {
                    "id": "test",
                    "choices": [],
                    "created": 123,
                    "model": "test-model",
                    "object": "chat.completion.chunk",
                }

        async def fake_stream() -> AsyncIterator[FakePortkeyChunk]:
            yield FakePortkeyChunk()

        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream())

        provider = PortkeyProvider(api_key="test")

        result = cast(
            "AsyncIterator[ChatCompletionChunk]",
            await provider._acompletion(
                CompletionParams(
                    model_id="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )
            ),
        )

    chunks = []

    async for chunk in result:
        chunks.append(chunk)

    mock_client.chat.completions.create.assert_called_once()
    assert len(chunks) == 1
    assert isinstance(chunks[0], ChatCompletionChunk)
    assert chunks[0].id == "test"
    assert chunks[0].model == "test-model"


@pytest.mark.asyncio
async def test_alist_models_with_async_portkey() -> None:
    """Test that Portkey model responses are converted into any-llm models."""

    with patch("any_llm.providers.portkey.portkey.AsyncPortkey") as mocked_portkey:
        mock_client = AsyncMock()
        mocked_portkey.return_value = mock_client

        response = PortkeyModelList(
            object="list",
            data=[
                PortkeyModel(
                    id="test-model",
                    object="model",
                    created=123,
                    owned_by="test-provider",
                )
            ],
        )

        mock_client.models.list = AsyncMock(return_value=response)

        provider = PortkeyProvider(api_key="test")
        result = await provider._alist_models()

    assert len(result) == 1
    assert result[0].id == "test-model"
    assert result[0].object == "model"
    assert result[0].created == 123
    assert result[0].owned_by == "test-provider"
