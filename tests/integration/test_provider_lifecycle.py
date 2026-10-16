from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_provider_async_context_with_live_completion(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    provider_class = AnyLLM.get_provider_class(provider)
    if provider_class.aclose is AnyLLM.aclose:
        pytest.skip(f"{provider.value} holds no persistent async transport")
    if not provider_class.SUPPORTS_COMPLETION:
        pytest.skip(f"{provider.value} does not support completion")

    credential_errors: tuple[type[Exception], ...] = ()
    if provider == LLMProvider.VERTEXAI:
        from google.auth.exceptions import DefaultCredentialsError

        credential_errors = (DefaultCredentialsError,)

    try:
        async with AnyLLM.create(provider, **provider_client_config.get(provider, {})) as llm:
            result = await llm.acompletion(
                model=provider_model_map[provider],
                messages=[{"role": "user", "content": "Hello"}],
            )
            assert isinstance(result, ChatCompletion)
            assert result.choices
        await llm.aclose()
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided")
    except credential_errors:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} Google Application Default Credentials not configured")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError, ConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip(f"{provider.value} local model host is not configured")
        raise
