from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.moderation import ModerationResponse


@pytest.fixture
def moderation_provider_model_map() -> dict[LLMProvider, str]:
    return {
        LLMProvider.OPENAI: "omni-moderation-latest",
        LLMProvider.MISTRAL: "mistral-moderation-latest",
    }


@pytest.mark.asyncio
async def test_moderation_providers_async(
    provider: LLMProvider,
    moderation_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Happy-path moderation call for every provider that opts in."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except ImportError:
        pytest.skip(f"{provider.value} optional dependency missing, skipping")
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")

    if not llm.SUPPORTS_MODERATION:
        pytest.skip(f"{provider.value} does not support moderation, skipping")

    model_id = moderation_provider_model_map.get(provider)
    if model_id is None:
        pytest.skip(f"No moderation model mapped for {provider.value}, skipping")

    try:
        result = await llm.amoderation(model=model_id, input="I want to hurt someone")
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        pytest.skip(f"{provider.value} connection failed, skipping")
    except Exception as exc:
        if "model" in str(exc).lower() or "moderation" in str(exc).lower():
            pytest.skip(f"{provider.value} moderation model not available: {exc}")
        raise

    assert isinstance(result, ModerationResponse)
    assert result.results, "at least one moderation result expected"
    first = result.results[0]
    assert isinstance(first.flagged, bool)
    assert isinstance(first.categories, dict)
    assert isinstance(first.category_scores, dict)


@pytest.mark.asyncio
async def test_moderation_openai_multimodal_input(
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """OpenAI omni-moderation accepts multimodal content-part lists."""
    try:
        llm = AnyLLM.create(LLMProvider.OPENAI, **provider_client_config.get(LLMProvider.OPENAI, {}))
    except MissingApiKeyError:
        pytest.skip("OpenAI API key not provided, skipping")

    multimodal_input = [
        {"type": "text", "text": "I want to hurt someone"},
    ]
    try:
        result = await llm.amoderation(model="omni-moderation-latest", input=multimodal_input)
    except MissingApiKeyError:
        pytest.skip("OpenAI API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        pytest.skip("OpenAI connection failed, skipping")

    assert isinstance(result, ModerationResponse)
    assert result.results
