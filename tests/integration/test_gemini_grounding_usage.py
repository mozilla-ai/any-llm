from collections.abc import AsyncIterator
from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.GEMINI])]

_GROUNDED_REQUEST: dict[str, Any] = {
    "messages": [{"role": "user", "content": "Use Google Search: what is the latest stable Python release?"}],
    "tools": [{"google_search": {}}],
}


def _create(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> AnyLLM:
    try:
        return AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Gemini API key missing")


async def test_grounded_completion_reports_search_queries(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """A completion grounded with Google Search counts its queries as ``server_tool_use.web_search_requests``."""
    llm = _create(provider, provider_client_config)

    result = await llm.acompletion(model=provider_model_map[provider], **_GROUNDED_REQUEST)
    assert isinstance(result, ChatCompletion)

    assert result.usage is not None
    assert result.usage.server_tool_use is not None
    assert (result.usage.server_tool_use.web_search_requests or 0) >= 1


async def test_grounded_stream_reports_search_queries_on_its_usage(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """The stream's final usage carries the query count, as a caller reading only that usage bills it."""
    llm = _create(provider, provider_client_config)

    stream = await llm.acompletion(
        model=provider_model_map[provider],
        stream=True,
        stream_options={"include_usage": True},
        **_GROUNDED_REQUEST,
    )
    assert isinstance(stream, AsyncIterator)
    usages = [chunk.usage async for chunk in stream if chunk.usage is not None]

    assert usages
    assert usages[-1].server_tool_use is not None
    assert (usages[-1].server_tool_use.web_search_requests or 0) >= 1
