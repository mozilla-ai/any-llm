from collections.abc import AsyncIterator
from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.BEDROCK])]


async def test_bedrock_stream_reports_usage_on_a_chunk_with_no_choices(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """The stream ends with the usage on a chunk of its own with no choices, as OpenAI's include_usage chunk."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Bedrock credentials missing")

    stream = await llm.acompletion(
        model=provider_model_map[provider],
        max_tokens=20,
        messages=[{"role": "user", "content": "Say hello in exactly one word."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    assert isinstance(stream, AsyncIterator)
    chunks = [chunk async for chunk in stream]

    with_usage = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(with_usage) == 1
    assert with_usage[0] is chunks[-1]
    usage_chunk = chunks[-1]
    assert usage_chunk.choices == []
    assert usage_chunk.usage is not None
    assert usage_chunk.usage.prompt_tokens > 0
    assert usage_chunk.usage.completion_tokens > 0
    assert any(choice.finish_reason is not None for chunk in chunks[:-1] for choice in chunk.choices)
