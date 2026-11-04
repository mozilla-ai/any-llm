from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.decide import Choice, DecideResponse, Noul, Score
from tests.constants import EXPECTED_PROVIDERS

# llama-server answers /v1/systemone only with a decision model loaded, and CI's llama-server
# loads the chat model in LLAMACPP_MODEL_FILE (.github/workflows/tests-integration.yaml).
# llama.cpp decide is covered by the request-shape unit tests in tests/unit/providers/test_llamacpp.py.
DECIDE_INFRA_NOT_CONFIGURED_IN_CI = [LLMProvider.LLAMACPP]


@pytest.mark.asyncio
async def test_decide_providers_async(
    provider: LLMProvider,
    decide_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that every decide provider answers each question type."""
    if not AnyLLM.get_provider_class(provider).SUPPORTS_DECIDE:
        pytest.skip(f"{provider.value} does not support decide, skipping")
    if provider in DECIDE_INFRA_NOT_CONFIGURED_IN_CI:
        pytest.skip(f"{provider.value} decide needs a decision model, which is not served in CI, skipping")
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        result = await llm.adecide(
            decide_provider_model_map[provider],
            "I was charged twice. Please fix this ASAP.",
            {
                "billing": Noul(instructions="Is this ticket about billing?"),
                "tone": Choice(instructions="What is the customer's tone?", criteria={"calm": None, "angry": None}),
                "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
            },
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} connection failed, skipping")

    assert isinstance(result, DecideResponse)
    assert result.model
    assert result.nouls["billing"].noul > 0.5
    assert result.choices["tone"].choice in {"calm", "angry"}
    assert 0.0 <= result.scores["urgency"].score <= 2.0
