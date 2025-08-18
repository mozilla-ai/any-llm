from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import ProviderName, completion
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory
from any_llm.types.completion import ChatCompletion
from tests.constants import LOCAL_PROVIDERS


def test_completion_reasoning(
    provider: ProviderName,
    provider_reasoning_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION_REASONING:
        pytest.skip(f"{provider.value} does not support completion reasoning, skipping")
    model_id = provider_reasoning_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    try:
        result = completion(
            model=model_id,
            provider=provider,
            **extra_kwargs,
            messages=[{"role": "user", "content": "Please say hello! Think very briefly before you respond."}],
        )
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content is not None
