import dataclasses
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from any_llm import AnyLLM, AnyLLMError, ParsedChatCompletion
from any_llm.exceptions import ContentFilterFinishReasonError, LengthFinishReasonError
from any_llm.types.completion import ChatCompletion


class CityResponse(BaseModel):
    city_name: str


@dataclasses.dataclass
class CityResponseDataclass:
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

    with pytest.raises(LengthFinishReasonError) as exc_info:
        await provider.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=CityResponse,
        )

    assert isinstance(exc_info.value.completion, ParsedChatCompletion)
    assert exc_info.value.completion.choices[0].finish_reason == "length"
    assert exc_info.value.completion.choices[0].message.content == '{"city_name": "Paris"}'


@pytest.mark.asyncio
async def test_parsed_completion_content_filter_finish_reason(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(finish_reason="content_filter", content=None),
    )

    with pytest.raises(ContentFilterFinishReasonError) as exc_info:
        await provider.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=CityResponse,
        )

    assert isinstance(exc_info.value.completion, ParsedChatCompletion)
    assert exc_info.value.completion.choices[0].finish_reason == "content_filter"


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
async def test_parsed_completion_already_parsed(provider: AnyLLM) -> None:
    """When a provider returns a ParsedChatCompletion with .parsed already set, skip re-parsing."""
    completion = _make_chat_completion()
    parsed_completion: ParsedChatCompletion[Any] = ParsedChatCompletion.model_validate(completion, from_attributes=True)
    parsed_completion.choices[0].message.parsed = CityResponse(city_name="Paris")
    provider._acompletion = AsyncMock(return_value=parsed_completion)  # type: ignore[method-assign]

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_format=CityResponse,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert isinstance(result.choices[0].message.parsed, CityResponse)
    assert result.choices[0].message.parsed.city_name == "Paris"


@pytest.mark.asyncio
async def test_parsed_completion_with_parsed_none_and_valid_content(provider: AnyLLM) -> None:
    """When a provider returns ParsedChatCompletion with .parsed=None but valid content, parse it."""
    completion = _make_chat_completion()
    parsed_completion: ParsedChatCompletion[Any] = ParsedChatCompletion.model_validate(completion, from_attributes=True)
    # Simulate providers (e.g. deepseek, sambanova) where the OpenAI SDK's .parse()
    # returns ParsedChatCompletion but doesn't populate .parsed
    assert parsed_completion.choices[0].message.parsed is None
    provider._acompletion = AsyncMock(return_value=parsed_completion)  # type: ignore[method-assign]

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_format=CityResponse,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert isinstance(result.choices[0].message.parsed, CityResponse)
    assert result.choices[0].message.parsed.city_name == "Paris"


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
async def test_parsed_completion_invalid_json_raises_validation_error(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(content='{"wrong_field": "value"}'),
    )

    with pytest.raises(ValidationError):
        await provider.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=CityResponse,
        )


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


@pytest.mark.asyncio
async def test_parsed_completion_dataclass_from_content(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(return_value=_make_chat_completion())  # type: ignore[method-assign]

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_format=CityResponseDataclass,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert isinstance(result.choices[0].message.parsed, CityResponseDataclass)
    assert result.choices[0].message.parsed.city_name == "Paris"


@pytest.mark.asyncio
async def test_parsed_completion_dataclass_refusal(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(content=None, refusal="I cannot help with that"),
    )

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        response_format=CityResponseDataclass,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed is None
    assert result.choices[0].message.refusal == "I cannot help with that"


@pytest.mark.asyncio
async def test_parsed_completion_dataclass_length_finish_reason(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(finish_reason="length"),
    )

    with pytest.raises(LengthFinishReasonError) as exc_info:
        await provider.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=CityResponseDataclass,
        )

    assert isinstance(exc_info.value.completion, ParsedChatCompletion)
    assert exc_info.value.completion.choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_parsed_completion_dataclass_content_filter_finish_reason(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(finish_reason="content_filter", content=None),
    )

    with pytest.raises(ContentFilterFinishReasonError) as exc_info:
        await provider.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=CityResponseDataclass,
        )

    assert isinstance(exc_info.value.completion, ParsedChatCompletion)
    assert exc_info.value.completion.choices[0].finish_reason == "content_filter"


def test_finish_reason_errors_are_any_llm_errors() -> None:
    assert issubclass(LengthFinishReasonError, AnyLLMError)
    assert issubclass(ContentFilterFinishReasonError, AnyLLMError)


@pytest.mark.asyncio
async def test_parsed_completion_dataclass_invalid_json_raises_validation_error(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(content='{"wrong_field": "value"}'),
    )

    with pytest.raises(ValidationError):
        await provider.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            response_format=CityResponseDataclass,
        )


@pytest.mark.asyncio
async def test_parsed_completion_dataclass_no_content_no_refusal(provider: AnyLLM) -> None:
    provider._acompletion = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_chat_completion(content=None),
    )

    result = await provider.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        response_format=CityResponseDataclass,
    )

    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed is None
    assert result.choices[0].message.content is None
