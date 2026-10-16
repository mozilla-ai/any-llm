from typing import Any

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion
from tests.constants import EXPECTED_PROVIDERS


@pytest.mark.asyncio
async def test_xai_reasoning_tokens_count_as_completion_tokens(
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """xAI reports reasoning tokens apart from completion tokens but includes them in the total."""
    try:
        llm = AnyLLM.create(LLMProvider.XAI, **provider_client_config.get(LLMProvider.XAI, {}))
    except MissingApiKeyError:
        if LLMProvider.XAI in EXPECTED_PROVIDERS:
            raise
        pytest.skip("xai API key not provided, skipping")

    result = await llm.acompletion(
        model=provider_reasoning_model_map[LLMProvider.XAI],
        messages=[{"role": "user", "content": "Please say hello! Think before you respond."}],
    )

    assert isinstance(result, ChatCompletion)
    assert result.usage is not None
    assert result.usage.completion_tokens_details is not None
    reasoning_tokens = result.usage.completion_tokens_details.reasoning_tokens
    assert reasoning_tokens is not None
    assert 0 < reasoning_tokens < result.usage.completion_tokens
    assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens
