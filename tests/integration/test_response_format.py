from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from any_llm import ProviderName, acompletion
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.provider import ProviderFactory
from any_llm.types.completion import ChatCompletion
from tests.constants import LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_response_format(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""

    if provider == ProviderName.LLAMAFILE:
        pytest.skip("Llamafile does not support response_format, skipping")
    cls = ProviderFactory.get_provider_class(provider)
    if not cls.SUPPORTS_COMPLETION:
        pytest.skip(f"{provider.value} does not support response_format, skipping")
    model_id = provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})

    # From https://github.com/mozilla-ai/any-llm/issues/150, should be ok to set stream=False
    extra_kwargs["stream"] = False

    class ResponseFormat(BaseModel):
        city_name: str

    prompt = "What is the capital of France?"
    try:
        result = await acompletion(
            model=model_id,
            provider=provider,
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
    except UnsupportedParameterError:
        pytest.skip(f"{provider.value} does not support response_format, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
