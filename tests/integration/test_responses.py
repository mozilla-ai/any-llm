import dataclasses
from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from openresponses_types import ResponseResource
from pydantic import BaseModel

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.responses import ParsedResponse, Response
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS

# HuggingFace routes the Responses API through a community OpenResponses Space that
# returns a server-side 400 for strict json_schema requests, so structured output is
# unreliable there regardless of the model.
STRUCTURED_RESPONSES_UNSUPPORTED_PROVIDERS = [LLMProvider.HUGGINGFACE]

# Fireworks' Responses API echoes the input back instead of returning a conformant
# structured response on the create()+text-format path used for non-Pydantic schemas
# (e.g. dataclasses); the native parse() path used for Pydantic models works.
NONPYDANTIC_RESPONSES_UNSUPPORTED_PROVIDERS = [*STRUCTURED_RESPONSES_UNSUPPORTED_PROVIDERS, LLMProvider.FIREWORKS]


@pytest.mark.asyncio
async def test_responses_async(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_RESPONSES:
            pytest.skip(f"{provider.value} does not support responses, skipping")
        model_id = provider_reasoning_model_map[provider]
        result = await llm.aresponses(
            model_id,
            input_data="What's the capital of France? Please think step by step.",
            instructions="Talk like a pirate.",
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, (ResponseResource, Response))


@pytest.mark.asyncio
async def test_responses_format_basemodel(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Structured output via the Responses API returns a ParsedResponse with output_parsed."""

    class CityResponse(BaseModel):
        city_name: str

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_RESPONSES:
            pytest.skip(f"{provider.value} does not support responses, skipping")
        if provider in STRUCTURED_RESPONSES_UNSUPPORTED_PROVIDERS:
            pytest.skip(f"{provider.value} does not reliably support structured responses, skipping")
        model_id = provider_model_map[provider]
        result = await llm.aresponses(
            model_id,
            input_data="What is the capital of France?",
            response_format=CityResponse,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"{provider.value} does not support structured responses, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise

    assert isinstance(result, ParsedResponse)
    assert isinstance(result.output_parsed, CityResponse)
    assert "paris" in result.output_parsed.city_name.lower()


@pytest.mark.asyncio
async def test_responses_format_dataclass(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Structured output via the Responses API also works with a dataclass type."""

    @dataclasses.dataclass
    class CityResponse:
        city_name: str

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_RESPONSES:
            pytest.skip(f"{provider.value} does not support responses, skipping")
        if provider in NONPYDANTIC_RESPONSES_UNSUPPORTED_PROVIDERS:
            pytest.skip(f"{provider.value} does not reliably support non-Pydantic structured responses, skipping")
        model_id = provider_model_map[provider]
        result = await llm.aresponses(
            model_id,
            input_data="What is the capital of France?",
            response_format=CityResponse,
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"{provider.value} does not support structured responses, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise

    assert isinstance(result, ParsedResponse)
    assert isinstance(result.output_parsed, CityResponse)
    assert "paris" in result.output_parsed.city_name.lower()
