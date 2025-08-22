from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import list_models
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory, ProviderName
from any_llm.types.model import Model
from tests.constants import LOCAL_PROVIDERS


def test_list_models(provider: ProviderName, provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]]) -> None:
    """Test that all supported providers can be loaded successfully."""
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_LIST_MODELS:
        pytest.skip(f"{provider.value} does not support listing models, skipping")
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    try:
        available_models = list_models(provider=provider, **extra_kwargs)
        assert len(available_models) > 0
        assert isinstance(available_models, list)
        assert all(isinstance(model, Model) for model in available_models)
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
