from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from any_llm.exceptions import ProviderError
from any_llm.types.completion import ChatCompletionChunk, CompletionParams

PROVIDERS = ["groq", "mistral", "cerebras"]


def _chunk(finish_reason: str | None, *, choices: bool = True) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": finish_reason}] if choices else [],
        }
    )


async def _stream(provider_name: str, chunks: list[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
    """Run the provider's streaming path over an SDK stream that yields ``chunks`` and then closes."""
    pytest.importorskip({"groq": "groq", "mistral": "mistralai", "cerebras": "cerebras.cloud.sdk"}[provider_name])

    async def sdk_stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in chunks:
            yield chunk

    sdk_client, create = {
        "groq": ("any_llm.providers.groq.groq.AsyncGroq", "chat.completions.create"),
        "mistral": ("any_llm.providers.mistral.mistral.Mistral", "chat.stream_async"),
        "cerebras": ("any_llm.providers.cerebras.cerebras.cerebras.AsyncCerebras", "chat.completions.create"),
    }[provider_name]
    module = __import__(f"any_llm.providers.{provider_name}", fromlist=["*"])
    provider_class = getattr(module, f"{provider_name.capitalize()}Provider")

    with patch(sdk_client) as client_class:
        target: Any = client_class.return_value
        *parents, method = create.split(".")
        for name in parents:
            target = getattr(target, name)
        setattr(target, method, AsyncMock(return_value=sdk_stream()))
        provider = provider_class(api_key="test-api-key")

    with patch.object(provider_class, "_convert_completion_chunk_response", side_effect=lambda chunk: chunk):
        result = await provider._acompletion(
            CompletionParams(model_id="m", messages=[{"role": "user", "content": "Hi"}], stream=True)
        )
        async for chunk in result:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", PROVIDERS)
async def test_stream_raises_when_it_ends_before_finish_reason(provider_name: str) -> None:
    chunks = [_chunk(None), _chunk(None)]
    stream = _stream(provider_name, chunks)
    assert [await anext(stream), await anext(stream)] == chunks
    with pytest.raises(ProviderError, match=r"finish_reason for choice\(s\) \[0\]"):
        await anext(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize(
    "chunks",
    [
        [_chunk(None), _chunk("stop")],
        [_chunk("stop"), _chunk(None)],
        [_chunk("stop"), _chunk(None, choices=False)],
        [],
    ],
    ids=["finished", "choice-repeated-after-finish", "trailing-usage-only", "empty"],
)
async def test_stream_ends_normally(provider_name: str, chunks: list[ChatCompletionChunk]) -> None:
    assert [chunk async for chunk in _stream(provider_name, chunks)] == chunks
