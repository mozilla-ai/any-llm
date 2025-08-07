from typing import Any
import httpx
from pydantic import BaseModel
import pytest
from any_llm import completion, ProviderName
from any_llm.exceptions import MissingApiKeyError
from openai import APIConnectionError
from any_llm.types.completion import ChatCompletion


def test_response_format(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    if provider in [ProviderName.COHERE]:
        pytest.skip(f"{provider.value} does not support response_format")
        return
    model_id = provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})

    class ResponseFormat(BaseModel):
        city_name: str

    prompt = "What is the capital of France?"
    try:
        result = completion(
            f"{provider.value}/{model_id}",
            **extra_kwargs,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
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
