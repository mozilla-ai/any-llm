from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from any_llm import AnyLLM, ParsedChatCompletion
from any_llm.types.completion import ChatCompletion


class CityResponse(BaseModel):
    city_name: str


def _make_chat_completion(finish_reason: str = "stop", **message_overrides: Any) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": '{"city_name": "Paris"}'}
    message.update(message_overrides)
    return ChatCompletion.model_validate(
        {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        }
    )


@pytest.fixture
def provider() -> AnyLLM:
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        return AnyLLM.create("openai", api_key="test-key")


@pytest.mark.asyncio
async def test_parsed_completion_from_content(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(return_value=_make_chat_completion())  # type: ignore[method-assign]

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_format=CityResponse,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert isinstance(result.choices[0].message.parsed, CityResponse)
    assert result.choices[0].message.parsed.city_name == "Paris"


@pytest.mark.asyncio
async def test_parsed_completion_refusal(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(content=None, refusal="I cannot help with that"),
    )

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        response_format=CityResponse,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed is None
    assert result.choices[0].message.refusal == "I cannot help with that"


@pytest.mark.asyncio
async def test_parsed_completion_length_finish_reason(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(finish_reason="length"),
    )

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        response_format=CityResponse,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed is None
    assert result.choices[0].finish_reason == "length"
    assert result.choices[0].message.content == '{"city_name": "Paris"}'


@pytest.mark.asyncio
async def test_parsed_completion_content_filter_finish_reason(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(finish_reason="content_filter", content=None),
    )

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        response_format=CityResponse,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed is None
    assert result.choices[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_no_parsed_completion_without_response_format(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(return_value=_make_chat_completion())  # type: ignore[method-assign]

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
    )

    assert isinstance(result, ChatCompletion)
    assert not isinstance(result, ParsedChatCompletion)


@pytest.mark.asyncio
async def test_parsed_completion_no_content_no_refusal(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(content=None),
    )

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        response_format=CityResponse,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed is None
    assert result.choices[0].message.content is None


@pytest.mark.asyncio
async def test_no_parsed_completion_with_dict_response_format(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(return_value=_make_chat_completion())  # type: ignore[method-assign]

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        response_format={"type": "json_object"},
    )

    assert isinstance(result, ChatCompletion)
    assert not isinstance(result, ParsedChatCompletion)
