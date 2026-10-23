from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.completion import ChatCompletion
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel_tool_calls", [True, False])
async def test_parallel_tool_calls_is_accepted(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
    tools: list[dict[str, Any]],
    parallel_tool_calls: bool,
) -> None:
    """A provider that takes ``parallel_tool_calls`` gets the request through.

    Whether the model then keeps to one call is the model's behavior, not any-llm's:
    Together forwards ``False`` to every model, and some still make several calls.
    """
    if provider in (*LOCAL_PROVIDERS, LLMProvider.PERPLEXITY):
        pytest.skip(f"{provider} does not support tools, skipping")

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")
        result = await llm.acompletion(
            model=provider_model_map[provider],
            messages=[{"role": "user", "content": "What is the weather in Paris, France and in Rome, Italy?"}],
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )
        assert isinstance(result, ChatCompletion)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"parallel_tool_calls is not supported for {provider.value}")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
