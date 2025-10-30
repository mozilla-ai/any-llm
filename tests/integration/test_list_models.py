from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.model import Model
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_list_models(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> None:
    """Test that all supported providers can be loaded successfully."""
    try:
        config = provider_client_config.get(provider, {})
        if provider == "huggingface":
            # We don't want to use the custom endpoint for listing models
            config.pop("api_base")
        llm = AnyLLM.create(provider, **config)
        if not llm.SUPPORTS_LIST_MODELS:
            pytest.skip(f"{provider.value} does not support listing models, skipping")

        available_models = await llm.alist_models()
        assert len(available_models) > 0
        assert isinstance(available_models, list)
        assert all(isinstance(model, Model) for model in available_models)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
