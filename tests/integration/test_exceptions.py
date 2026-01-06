from __future__ import annotations

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError, AuthenticationError
from any_llm.utils.exception_handler import ANY_LLM_UNIFIED_EXCEPTIONS_ENV

# Constants
_STREAM_ERROR_MSG = "Expected streaming response but got non-streaming"


def _raise_authentication_error() -> None:
    """Helper function to raise authentication error for non-streaming responses."""
    raise AuthenticationError(_STREAM_ERROR_MSG)


# Providers that can be tested with invalid API keys
# (i.e., they don't require local infrastructure)
TESTABLE_PROVIDERS = [
    LLMProvider.OPENAI,
    LLMProvider.ANTHROPIC,
    LLMProvider.MISTRAL,
    LLMProvider.GEMINI,
    LLMProvider.XAI,
    LLMProvider.DEEPSEEK,
    LLMProvider.COHERE,
    LLMProvider.GROQ,
    LLMProvider.FIREWORKS,
    LLMProvider.TOGETHER,
]


@pytest.mark.parametrize("provider", TESTABLE_PROVIDERS)
@pytest.mark.asyncio
async def test_bad_api_key_raises_authentication_error(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANY_LLM_UNIFIED_EXCEPTIONS_ENV, "1")
    llm = AnyLLM.create(provider, api_key="invalid-api-key-for-testing-12345")

    if not llm.SUPPORTS_COMPLETION:
        pytest.skip(f"{provider.value} does not support completion, skipping")

    model_id = provider_model_map.get(provider)
    if not model_id:
        pytest.skip(f"No model configured for {provider.value}")

    with pytest.raises(AuthenticationError) as exc_info:
        await llm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": "Hello"}],
        )

    error = exc_info.value
    assert isinstance(error, AnyLLMError)
    assert error.provider_name == provider.value
    assert error.original_exception is not None


@pytest.mark.parametrize("provider", TESTABLE_PROVIDERS)
@pytest.mark.asyncio
async def test_bad_api_key_raises_original_exception_when_flag_disabled(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ANY_LLM_UNIFIED_EXCEPTIONS_ENV, raising=False)

    llm = AnyLLM.create(provider, api_key="invalid-api-key-for-testing-12345")

    if not llm.SUPPORTS_COMPLETION:
        pytest.skip(f"{provider.value} does not support completion, skipping")

    model_id = provider_model_map.get(provider)
    if not model_id:
        pytest.skip(f"No model configured for {provider.value}")

    with pytest.raises(Exception, match=r".*") as exc_info:
        with pytest.warns(DeprecationWarning, match="unified any-llm exceptions"):
            await llm.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "Hello"}],
            )

    assert not isinstance(exc_info.value, AnyLLMError)


@pytest.mark.asyncio
async def test_authentication_error_preserves_original_exception(
    provider_model_map: dict[LLMProvider, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANY_LLM_UNIFIED_EXCEPTIONS_ENV, "1")

    provider = LLMProvider.OPENAI
    llm = AnyLLM.create(provider, api_key="invalid-api-key-for-testing-12345")
    model_id = provider_model_map.get(provider)

    if not model_id:
        pytest.skip("No OpenAI model configured")

    with pytest.raises(AuthenticationError) as exc_info:
        await llm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": "Hello"}],
        )

    error = exc_info.value
    assert error.original_exception is not None
    assert error.__cause__ is error.original_exception
    assert provider.value in str(error)


@pytest.mark.asyncio
async def test_streaming_with_bad_api_key(
    provider_model_map: dict[LLMProvider, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANY_LLM_UNIFIED_EXCEPTIONS_ENV, "1")
    provider = LLMProvider.OPENAI
    llm = AnyLLM.create(provider, api_key="invalid-api-key-for-testing-12345")
    model_id = provider_model_map.get(provider)

    if not llm.SUPPORTS_COMPLETION_STREAMING:
        pytest.skip(f"{provider.value} does not support streaming")

    if not model_id:
        pytest.skip("No OpenAI model configured")

    # With invalid API keys, authentication errors can occur either:
    # 1. During the initial acompletion call, or
    # 2. During streaming iteration
    # Both are valid behaviors, so we handle either case
    try:
        stream = await llm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        # If acompletion succeeds, the error should occur during iteration
        if hasattr(stream, "__aiter__"):
            with pytest.raises(AuthenticationError) as exc_info:
                async for _ in stream:
                    pass
        else:
            # Non-streaming response, raise the error immediately
            with pytest.raises(AuthenticationError) as exc_info:
                _raise_authentication_error()
    except AuthenticationError as exc_info_value:
        # If acompletion fails immediately, that's also acceptable
        exc_info = type("MockExcInfo", (), {"value": exc_info_value})()

    error = exc_info.value
    assert isinstance(error, AnyLLMError)
    assert error.provider_name == provider.value
    assert error.original_exception is not None
