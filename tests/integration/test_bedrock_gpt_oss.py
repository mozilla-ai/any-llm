from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.BEDROCK])]

GPT_OSS_MODELS = ["openai.gpt-oss-20b-1:0", "openai.gpt-oss-safeguard-120b"]


def _client(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> AnyLLM:
    try:
        return AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Bedrock credentials missing")


@pytest.mark.parametrize("model", GPT_OSS_MODELS)
async def test_gpt_oss_takes_reasoning_effort(
    provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]], model: str
) -> None:
    result = await _client(provider, provider_client_config).acompletion(
        model=model,
        messages=[{"role": "user", "content": "What is 2 + 2? Answer with the number only."}],
        reasoning_effort="low",
        max_tokens=500,
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content


async def test_gpt_oss_streams_with_reasoning_effort(
    provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]
) -> None:
    stream = await _client(provider, provider_client_config).acompletion(
        model=GPT_OSS_MODELS[0],
        messages=[{"role": "user", "content": "What is 2 + 2? Answer with the number only."}],
        reasoning_effort="low",
        max_tokens=500,
        stream=True,
    )

    content = "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])
    assert "4" in content
