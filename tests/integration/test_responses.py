from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from openresponses_types import ResponseResource

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_responses_async(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_RESPONSES:
            pytest.skip(f"{provider.value} does not support responses, skipping")
        model_id = provider_reasoning_model_map[provider]
        result = await llm.aresponses(
            model_id,
            input_data="What's the capital of France? Please think step by step.",
            instructions="Talk like a pirate.",
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, ResponseResource)
    # make sure it contains the reasoning
    assert result.reasoning is not None
    # if it's openai, there should be a summary. If it's not openai, expect content
    if provider == LLMProvider.OPENAI:
        assert result.reasoning.summary is not None
