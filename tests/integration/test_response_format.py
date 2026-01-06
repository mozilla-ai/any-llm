from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.completion import ChatCompletion
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_response_format(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""

    if provider == LLMProvider.LLAMAFILE:
        pytest.skip("Llamafile does not support response_format, skipping")

    if provider == LLMProvider.HUGGINGFACE:
        pytest.skip("HuggingFace TGI does not support response_format, skipping")

    class ResponseFormat(BaseModel):
        city_name: str

    prompt = "What is the capital of France?"
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support response_format, skipping")

        model_id = provider_model_map[provider]

        result = await llm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format=ResponseFormat,
            # From https://github.com/mozilla-ai/any-llm/issues/150, should be ok to set stream=False
            stream=False,
        )
        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content is not None
        output = ResponseFormat.model_validate_json(result.choices[0].message.content)
        assert "paris" in output.city_name.lower()
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"{provider.value} does not support response_format, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
