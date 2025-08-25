import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.providers.cerebras.cerebras import CerebrasProvider


@pytest.mark.asyncio
async def test_stream_with_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = CerebrasProvider(ApiConfig(api_key=api_key))

    chunks = provider._stream_completion_async(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    with pytest.raises(UnsupportedParameterError):
        async for _ in chunks:
            pass
