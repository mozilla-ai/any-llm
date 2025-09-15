from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, list_models
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.model import Model
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


def test_list_models(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> None:
    """Test that all supported providers can be loaded successfully."""
    cls = AnyLLM.get_provider_class(provider)
    if not cls.SUPPORTS_LIST_MODELS:
        pytest.skip(f"{provider.value} does not support listing models, skipping")
    extra_kwargs = provider_client_config.get(provider, {})
    try:
        available_models = list_models(provider=provider, **extra_kwargs)
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
