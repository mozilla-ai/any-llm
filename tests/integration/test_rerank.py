from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.rerank import RerankResponse
from tests.constants import EXPECTED_PROVIDERS

# Together supports rerank in its SDK, but serves no rerank model on the serverless tier
# ("There are currently no rerank models offered via serverless",
# https://docs.together.ai/docs/serverless-models). Reaching Salesforce/Llama-Rank-v1 needs a
# dedicated endpoint, which CI does not provision, so this test cannot exercise it with the
# plain TOGETHER_API_KEY the rest of the suite uses. Together rerank is covered by the
# SDK-object unit tests in tests/unit/test_rerank.py.
RERANK_INFRA_NOT_CONFIGURED_IN_CI = [LLMProvider.TOGETHER]

_DOCUMENTS = [
    "The Great Wall of China is visible from certain low Earth orbits.",
    "Carbon dioxide levels are measured in parts per million.",
    "Reranking reorders candidate documents by relevance to a query.",
]


@pytest.mark.asyncio
async def test_rerank_providers_async(
    provider: LLMProvider,
    rerank_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all rerank-supported providers reorder documents successfully."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_RERANK:
            pytest.skip(f"{provider.value} does not support rerank, skipping")
        if provider in RERANK_INFRA_NOT_CONFIGURED_IN_CI:
            pytest.skip(f"{provider.value} rerank needs a dedicated endpoint, which is not configured in CI, skipping")

        model_id = rerank_provider_model_map[provider]
        result = await llm.arerank(
            model=model_id,
            query="What does a reranker do?",
            documents=_DOCUMENTS,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} connection failed, skipping")

    assert isinstance(result, RerankResponse)
    assert len(result.results) == len(_DOCUMENTS)
    assert {r.index for r in result.results} == set(range(len(_DOCUMENTS)))
    scores = [r.relevance_score for r in result.results]
    assert scores == sorted(scores, reverse=True)
    # The third document is the only one that answers the query.
    assert result.results[0].index == 2


@pytest.mark.asyncio
async def test_rerank_top_n_async(
    provider: LLMProvider,
    rerank_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that top_n truncates the result set, including Voyage's top_n -> top_k rename."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_RERANK:
            pytest.skip(f"{provider.value} does not support rerank, skipping")
        if provider in RERANK_INFRA_NOT_CONFIGURED_IN_CI:
            pytest.skip(f"{provider.value} rerank needs a dedicated endpoint, which is not configured in CI, skipping")

        model_id = rerank_provider_model_map[provider]
        result = await llm.arerank(
            model=model_id,
            query="What does a reranker do?",
            documents=_DOCUMENTS,
            top_n=1,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} connection failed, skipping")

    assert len(result.results) == 1
    assert result.results[0].index == 2
