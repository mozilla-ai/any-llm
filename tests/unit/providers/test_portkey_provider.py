import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from any_llm.providers.portkey.portkey import PortkeyProvider
from any_llm.types.completion import ChatCompletion, CompletionParams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _mock_portkey_http_client() -> httpx.AsyncClient:
    """Create a real Portkey SDK client backed by deterministic HTTP responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "test-model", "object": "model", "created": 0, "owned_by": "portkey"}],
                },
            )

        if json.loads(request.content).get("stream"):
            chunk = {
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
            body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

        return httpx.Response(
            200,
            json={
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
            },
        )

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=httpx.Timeout(600.0, connect=5.0),
    )


@pytest.mark.asyncio
async def test_portkey_uses_native_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider initializes Portkey's native SDK instead of OpenAI's client."""
    calls: dict[str, Any] = {}

    class FakeAsyncPortkey:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    monkeypatch.setattr("any_llm.providers.portkey.portkey.AsyncPortkey", FakeAsyncPortkey)
    PortkeyProvider(api_key="test-key", api_base="https://example.test/v1")

    assert calls["api_key"] == "test-key"
    assert calls["base_url"] == "https://example.test/v1"
    assert calls["request_timeout"] == httpx.Timeout(600.0, connect=5.0)
    assert calls["http_client"].timeout == httpx.Timeout(600.0, connect=5.0)
    await calls["http_client"].aclose()


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
    from portkey_ai import AsyncPortkey

    provider = PortkeyProvider(api_key="test-key")
    http_client = _mock_portkey_http_client()
    provider.client = cast("Any", AsyncPortkey(api_key="test-key", http_client=http_client))

    try:
        result = await provider._acompletion(
            CompletionParams(model_id="test-model", messages=[{"role": "user", "content": "Hello"}])
        )
    finally:
        await http_client.aclose()

    completion = cast("ChatCompletion", result)
    assert completion.choices[0].message.content == "answer"
    assert completion.choices[0].message.reasoning is not None
    assert completion.choices[0].message.reasoning.content == "because"


@pytest.mark.asyncio
async def test_native_portkey_stream_converts_vendored_chunks_and_xml_reasoning() -> None:
    from portkey_ai import AsyncPortkey

    provider = PortkeyProvider(api_key="test-key")
    http_client = _mock_portkey_http_client()
    provider.client = cast("Any", AsyncPortkey(api_key="test-key", http_client=http_client))

    try:
        result = await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
        )
        chunks = [chunk async for chunk in cast("AsyncIterator[Any]", result)]
    finally:
        await http_client.aclose()

    assert chunks[0].choices[0].delta.content == "answer"
    assert chunks[0].choices[0].delta.reasoning is not None
    assert chunks[0].choices[0].delta.reasoning.content == "because"


@pytest.mark.asyncio
async def test_native_portkey_list_models_converts_vendored_models() -> None:
    from portkey_ai import AsyncPortkey

    provider = PortkeyProvider(api_key="test-key")
    http_client = _mock_portkey_http_client()
    provider.client = cast("Any", AsyncPortkey(api_key="test-key", http_client=http_client))

    try:
        models = await provider._alist_models()
    finally:
        await http_client.aclose()

    assert [(model.id, model.owned_by) for model in models] == [("test-model", "portkey")]
