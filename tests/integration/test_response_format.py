import dataclasses
import json
from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from any_llm import AnyLLM, LLMProvider, ParsedChatCompletion
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessageFunctionToolCall
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
        assert isinstance(result, ParsedChatCompletion)
        assert result.choices[0].message.content is not None
        output = ResponseFormat.model_validate_json(result.choices[0].message.content)
        assert "paris" in output.city_name.lower()
        parsed = result.choices[0].message.parsed
        assert isinstance(parsed, ResponseFormat)
        assert "paris" in parsed.city_name.lower()
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


@pytest.mark.asyncio
async def test_response_format_bedrock_claude(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Regression test for https://github.com/mozilla-ai/any-llm/issues/1184.

    `test_response_format` above exercises Bedrock with an Amazon Nova model (via
    `provider_model_map`), which still correctly raises `UnsupportedParameterError`.
    This test targets a Claude model on Bedrock specifically, since that's the model
    family #1184 adds `response_format` support for (emulated via a forced tool call,
    see `any_llm.providers.bedrock.utils._convert_response_format_to_tool_spec`).
    """
    if provider != LLMProvider.BEDROCK:
        pytest.skip("This test only targets Claude models on Bedrock, see issue #1184")

    class ResponseFormat(BaseModel):
        city_name: str

    prompt = "What is the capital of France?"
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        model_id = provider_reasoning_model_map[provider]

        result = await llm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format=ResponseFormat,
            stream=False,
        )
        assert isinstance(result, ParsedChatCompletion)
        assert result.choices[0].message.content is not None
        output = ResponseFormat.model_validate_json(result.choices[0].message.content)
        assert "paris" in output.city_name.lower()
        parsed = result.choices[0].message.parsed
        assert isinstance(parsed, ResponseFormat)
        assert "paris" in parsed.city_name.lower()
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        raise
    except Exception as exc:
        # boto3 can raise these directly (rather than any-llm's MissingApiKeyError) when
        # ambient AWS credentials are missing/incomplete, e.g. in CI without the
        # run-integration-tests label. Matched by name to avoid a hard botocore import.
        if type(exc).__name__ in {"NoCredentialsError", "PartialCredentialsError", "NoRegionError"}:
            pytest.skip(f"{provider.value} AWS credentials not available, skipping: {exc}")
        raise


@pytest.mark.asyncio
async def test_response_format_dataclass(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that providers work with dataclass response_format (not just BaseModel)."""

    if provider == LLMProvider.LLAMAFILE:
        pytest.skip("Llamafile does not support response_format, skipping")

    if provider == LLMProvider.HUGGINGFACE:
        pytest.skip("HuggingFace TGI does not support response_format, skipping")

    @dataclasses.dataclass
    class CityResponse:
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
            response_format=CityResponse,
            stream=False,
        )
        assert isinstance(result, ParsedChatCompletion)
        assert result.choices[0].message.content is not None
        content = json.loads(result.choices[0].message.content)
        assert "paris" in content["city_name"].lower()
        parsed = result.choices[0].message.parsed
        assert isinstance(parsed, CityResponse)
        assert "paris" in parsed.city_name.lower()
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


@pytest.mark.asyncio
async def test_together_response_format_on_strict_model() -> None:
    """Pin Together's documented nested json_schema shape against a strict model.

    Together validates response_format per model. The model used by
    ``provider_model_map`` accepts the undocumented flat ``{"type", "schema"}`` shape as
    well as the documented nested one, so it cannot tell a correct provider from a
    broken one. Kimi rejects anything but the nested shape with a 400.
    """

    class CityResponse(BaseModel):
        city_name: str

    try:
        llm = AnyLLM.create(LLMProvider.TOGETHER)
        result = await llm.acompletion(
            model="moonshotai/Kimi-K3",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format=CityResponse,
            stream=False,
        )
    except MissingApiKeyError:
        if LLMProvider.TOGETHER in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Together API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        raise

    assert isinstance(result, ParsedChatCompletion)
    parsed = result.choices[0].message.parsed
    assert isinstance(parsed, CityResponse)
    assert "paris" in parsed.city_name.lower()


_COHERE_CITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city_name": {"type": "string"}},
    "required": ["city_name"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_cohere_openai_json_schema_response_format(provider_model_map: dict[LLMProvider, str]) -> None:
    """Cohere answers 400 to the OpenAI json_schema shape unless the provider unwraps it."""
    try:
        llm = AnyLLM.create(LLMProvider.COHERE)
        result = await llm.acompletion(
            model=provider_model_map[LLMProvider.COHERE],
            messages=[{"role": "user", "content": "What is the capital of France? Answer in JSON."}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "City", "strict": True, "schema": _COHERE_CITY_SCHEMA},
            },
            stream=False,
        )
    except MissingApiKeyError:
        if LLMProvider.COHERE in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Cohere API key not provided, skipping")

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert "paris" in json.loads(result.choices[0].message.content)["city_name"].lower()


@pytest.mark.asyncio
async def test_cohere_strict_function_tool(provider_model_map: dict[LLMProvider, str]) -> None:
    """Cohere answers 400 to `strict` inside a function definition; it takes `strict_tools` instead."""
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "strict": True,
            "parameters": _COHERE_CITY_SCHEMA,
        },
    }
    try:
        llm = AnyLLM.create(LLMProvider.COHERE)
        result = await llm.acompletion(
            model=provider_model_map[LLMProvider.COHERE],
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[tool],
            stream=False,
        )
    except MissingApiKeyError:
        if LLMProvider.COHERE in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Cohere API key not provided, skipping")

    assert isinstance(result, ChatCompletion)
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls
    tool_call = tool_calls[0]
    assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
    assert tool_call.function.name == "get_weather"
    assert "paris" in json.loads(tool_call.function.arguments)["city_name"].lower()
