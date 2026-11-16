from collections.abc import AsyncIterator
from typing import Any

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.BEDROCK])]


@pytest.mark.parametrize("stream", [False, True])
async def test_bedrock_developer_instructions(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
    stream: bool,
) -> None:
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        result = await llm.acompletion(
            model=provider_model_map[provider],
            messages=[
                {"role": "developer", "content": [{"type": "text", "text": "The secret word is capybara."}]},
                {"role": "user", "content": "What is the secret word? Answer with only that word."},
                {"role": "developer", "content": "Use lowercase letters."},
            ],
            stream=stream,
            max_tokens=32,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Bedrock credentials not provided")

    if isinstance(result, AsyncIterator):
        output = "".join([chunk.choices[0].delta.content or "" async for chunk in result if chunk.choices])
    else:
        assert isinstance(result, ChatCompletion)
        output = result.choices[0].message.content or ""
    assert "capybara" in output.lower()
