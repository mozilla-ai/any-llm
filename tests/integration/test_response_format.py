from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from any_llm import ProviderName, completion
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory
from any_llm.types.completion import ChatCompletion


def test_response_format(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    # first check if the provider supports response_format
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION:
        pytest.skip(f"{provider.value} does not support response_format, skipping")
    """Test that all supported providers can be loaded successfully."""
    if provider in [ProviderName.COHERE]:
        pytest.skip(f"{provider.value} does not support response_format")
        return
    model_id = provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})

    # From https://github.com/mozilla-ai/any-llm/issues/150, should be ok to set stream=False
    extra_kwargs["stream"] = False

    class ResponseFormat(BaseModel):
        city_name: str

    prompt = "What is the capital of France?"
    try:
        result = completion(
            f"{provider.value}/{model_id}",
            **extra_kwargs,
            messages=[{"role": "user", "content": prompt}],
            response_format=ResponseFormat,
        )
        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content is not None
        output = ResponseFormat.model_validate_json(result.choices[0].message.content)
        assert "paris" in output.city_name.lower()
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in [ProviderName.OLLAMA, ProviderName.LMSTUDIO]:
            pytest.skip("Local Model host is not set up, skipping")
        raise
