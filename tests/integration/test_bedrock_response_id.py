from collections.abc import AsyncIterator
from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.BEDROCK])]

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "Say hello in exactly one word."}]


def _create(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> AnyLLM:
    try:
        return AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Bedrock credentials missing")


async def test_bedrock_completion_has_an_id_and_a_creation_time(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Converse puts neither in its body, so they used to come back as "" and 0."""
    llm = _create(provider, provider_client_config)

    result = await llm.acompletion(model=provider_model_map[provider], max_tokens=20, messages=MESSAGES)
    assert isinstance(result, ChatCompletion)

    assert result.id
    assert result.created > 0


async def test_bedrock_stream_chunks_share_one_id(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Every chunk used to mint its own id, so one response had as many ids as chunks."""
    llm = _create(provider, provider_client_config)

    stream = await llm.acompletion(
        model=provider_model_map[provider],
        max_tokens=20,
        messages=MESSAGES,
        stream=True,
        stream_options={"include_usage": True},
    )
    assert isinstance(stream, AsyncIterator)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) > 1
    assert len({chunk.id for chunk in chunks}) == 1
    assert chunks[0].id
    assert len({chunk.created for chunk in chunks}) == 1
