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
from portkey_ai.api_resources.types.moderations_type import (
    Categories as PortkeyCategories,
)
from portkey_ai.api_resources.types.moderations_type import (
    CategoryScores as PortkeyCategoryScores,
)
from portkey_ai.api_resources.types.moderations_type import (
    Moderation as PortkeyModeration,
)
from portkey_ai.api_resources.types.moderations_type import (
    ModerationCreateResponse as PortkeyModerationCreateResponse,
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


def test_convert_completion_params_with_non_structured_response_format() -> None:

    response_format = {"type": "json_object"}

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        response_format=response_format,
    )

    result = PortkeyProvider._convert_completion_params(params)

    assert result["response_format"] == response_format


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


def test_convert_embedding_params_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Portkey does not support embeddings"):
        PortkeyProvider._convert_embedding_params(None)


def test_convert_embedding_response_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Portkey does not support embeddings"):
        PortkeyProvider._convert_embedding_response(None)


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


@pytest.mark.asyncio
async def test_amoderation_with_async_portkey() -> None:
    with patch("any_llm.providers.portkey.portkey.AsyncPortkey") as mocked_portkey:
        mock_client = AsyncMock()
        mocked_portkey.return_value = mock_client

        response = PortkeyModerationCreateResponse(
            id="mod-test",
            model="test-model",
            results=[
                PortkeyModeration(
                    flagged=True,
                    categories=PortkeyCategories(
                        violence=True,
                    ),
                    category_scores=PortkeyCategoryScores(
                        violence=0.95,
                    ),
                )
            ],
        )

        mock_client.moderations.create = AsyncMock(return_value=response)

        provider = PortkeyProvider(api_key="test")
        result = await provider._amoderation(
            model="test-model",
            input="test input",
        )

    assert result.id == "mod-test"
    assert result.model == "test-model"
    assert len(result.results) == 1
    assert result.results[0].flagged is True
    assert result.results[0].categories["violence"] is True
    assert result.results[0].category_scores["violence"] == 0.95


def test_portkey_convert_completion_response_normalizes_reasoning() -> None:
    provider = PortkeyProvider(api_key="test")

    class FakePortkeyCompletion:
        def model_dump(self) -> dict[str, Any]:
            return {
                "id": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "<think>xml reasoning</think>Final answer",
                            "reasoning_content": "provider reasoning",
                        },
                    }
                ],
                "created": 123,
                "model": "test-model",
                "object": "chat.completion",
            }

    result = provider._convert_completion_response(FakePortkeyCompletion())

    assert result.choices[0].message.content == "Final answer"
    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "provider reasoning\nxml reasoning"


def test_portkey_convert_completion_chunk_response_normalizes_reasoning() -> None:
    provider = PortkeyProvider(api_key="test")

    class FakePortkeyChunk:
        def model_dump(self) -> dict[str, Any]:
            return {
                "id": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "<think>xml reasoning</think>Final answer",
                            "reasoning_content": "provider reasoning",
                        },
                        "finish_reason": None,
                    }
                ],
                "created": 123,
                "model": "test-model",
                "object": "chat.completion.chunk",
            }

    result = provider._convert_completion_chunk_response(FakePortkeyChunk())

    assert result.choices[0].delta.content == "Final answer"
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "provider reasoning\nxml reasoning"


@pytest.mark.asyncio
async def test_acompletion_with_async_chunk_portkey_preserves_reasoning() -> None:
    with patch("any_llm.providers.portkey.portkey.AsyncPortkey") as mocked_portkey:
        mock_client = AsyncMock()
        mocked_portkey.return_value = mock_client

        class FakePortkeyChunk:
            def model_dump(self) -> dict[str, Any]:
                return {
                    "id": "test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "<think>xml reasoning</think>Final answer",
                                "reasoning_content": "provider reasoning",
                            },
                            "finish_reason": None,
                        }
                    ],
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

    chunks = [chunk async for chunk in result]

    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "Final answer"
    assert chunks[0].choices[0].delta.reasoning is not None
    assert chunks[0].choices[0].delta.reasoning.content == "provider reasoning\nxml reasoning"


def test_convert_list_models_response_returns_empty_list_when_data_is_none() -> None:
    response = PortkeyModelList.model_construct(
        object="list",
        data=None,
    )

    result = PortkeyProvider._convert_list_models_response(response)

    assert result == []
