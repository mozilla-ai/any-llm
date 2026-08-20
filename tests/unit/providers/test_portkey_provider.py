from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

import pytest

from any_llm.providers.portkey.portkey import PortkeyProvider
from any_llm.types.completion import ChatCompletion, CompletionParams


def _native_portkey_model(data: dict[str, Any]) -> Any:
    """Build the actual vendored Pydantic model returned by portkey-ai."""
    if data["object"] == "chat.completion":
        from portkey_ai._vendor.openai.types.chat.chat_completion import ChatCompletion as NativeChatCompletion

        return NativeChatCompletion.model_validate(data)
    if data["object"] == "chat.completion.chunk":
        from portkey_ai._vendor.openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk as NativeChatCompletionChunk,
        )

        return NativeChatCompletionChunk.model_validate(data)

    from portkey_ai._vendor.openai.types.model import Model as NativeModel

    return NativeModel.model_validate(data)


class _NativePortkeyStream:
    """Async iterator matching the stream returned by AsyncPortkey."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class _NativePortkeyClient:
    """Minimal AsyncPortkey-shaped client with deterministic native SDK responses."""

    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.models = self

    async def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _NativePortkeyStream(
                [
                    _native_portkey_model(
                        {
                            "id": "chunk-1",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": "test-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": "<think>because</think>answer"},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    )
                ]
            )
        return _native_portkey_model(
            {
                "id": "completion-1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "<think>because</think>answer"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def list(self, **kwargs: Any) -> Any:
        return type(
            "NativeModelList",
            (),
            {
                "data": [
                    _native_portkey_model({"id": "test-model", "object": "model", "created": 0, "owned_by": "portkey"})
                ]
            },
        )()


def test_portkey_uses_native_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider initializes Portkey's native SDK instead of OpenAI's client."""
    calls: dict[str, Any] = {}

    class FakeAsyncPortkey:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    monkeypatch.setattr("any_llm.providers.portkey.portkey.AsyncPortkey", FakeAsyncPortkey)
    PortkeyProvider(api_key="test-key", api_base="https://example.test/v1")

    assert calls == {"api_key": "test-key", "base_url": "https://example.test/v1"}


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


@pytest.mark.asyncio
async def test_native_portkey_completion_converts_vendored_model_and_xml_reasoning() -> None:
    provider = PortkeyProvider(api_key="test-key")
    provider.client = _NativePortkeyClient()

    result = await provider._acompletion(
        CompletionParams(model_id="test-model", messages=[{"role": "user", "content": "Hello"}])
    )

    completion = cast("ChatCompletion", result)
    assert completion.choices[0].message.content == "answer"
    assert completion.choices[0].message.reasoning is not None
    assert completion.choices[0].message.reasoning.content == "because"


@pytest.mark.asyncio
async def test_native_portkey_stream_converts_vendored_chunks_and_xml_reasoning() -> None:
    provider = PortkeyProvider(api_key="test-key")
    provider.client = _NativePortkeyClient()

    result = await provider._acompletion(
        CompletionParams(
            model_id="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
    )

    chunks = [chunk async for chunk in cast("AsyncIterator[Any]", result)]

    assert chunks[0].choices[0].delta.content == "answer"
    assert chunks[0].choices[0].delta.reasoning is not None
    assert chunks[0].choices[0].delta.reasoning.content == "because"


@pytest.mark.asyncio
async def test_native_portkey_list_models_converts_vendored_models() -> None:
    provider = PortkeyProvider(api_key="test-key")
    provider.client = _NativePortkeyClient()

    models = await provider._alist_models()

    assert [(model.id, model.owned_by) for model in models] == [("test-model", "portkey")]
