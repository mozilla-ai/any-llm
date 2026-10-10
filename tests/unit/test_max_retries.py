import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from openai import AsyncOpenAI

from any_llm import AnyLLM, completion
from any_llm.constants import LLMProvider
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.vertexai import VertexaiProvider
from any_llm.types.completion import CompletionParams


def _create_kwargs(provider: LLMProvider) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": "test_key", "api_base": "https://test.example.com"}
    if provider == LLMProvider.BEDROCK:
        kwargs["region_name"] = "us-east-1"
    if provider == LLMProvider.VERTEXAI:
        kwargs["project"] = "test-project"
        kwargs["location"] = "test-location"
    return kwargs


def _capture_init_client(provider: LLMProvider, **create_kwargs: Any) -> dict[str, Any]:
    captured_kwargs: dict[str, Any] = {}

    def capture_init_client(
        _self: Any, _api_key: str | None = None, _api_base: str | None = None, **kwargs: Any
    ) -> None:
        captured_kwargs.update(kwargs)

    provider_class = AnyLLM.get_provider_class(provider)
    with patch.object(provider_class, "_init_client", capture_init_client):
        AnyLLM.create(provider.value, **_create_kwargs(provider), **create_kwargs)
    return captured_kwargs


def _skip_unavailable(provider: LLMProvider) -> None:
    if provider == LLMProvider.SAGEMAKER:
        pytest.skip("sagemaker requires AWS credentials on instantiation")
    if sys.version_info >= (3, 14) and provider.value in ("voyage", "watsonx"):
        pytest.skip(f"{provider.value} is not compatible with Python 3.14+")


def test_max_retries_reaches_client_or_is_rejected(provider: LLMProvider) -> None:
    """Every provider either forwards max_retries in its SDK's form or rejects it, never drops it."""
    _skip_unavailable(provider)
    support = AnyLLM.get_provider_class(provider).MAX_RETRIES_SUPPORT

    if support == "unsupported":
        with pytest.raises(UnsupportedParameterError, match="max_retries"):
            _capture_init_client(provider, max_retries=3)
        return

    captured_kwargs = _capture_init_client(provider, max_retries=3)
    if support == "native":
        assert captured_kwargs["max_retries"] == 3
    elif provider == LLMProvider.VOYAGE:
        assert captured_kwargs["max_retries"] == 4
    else:
        assert "max_retries" not in captured_kwargs
        assert captured_kwargs["http_options"].retry_options.attempts == 4


def test_omitted_max_retries_leaves_client_kwargs_unchanged(provider: LLMProvider) -> None:
    _skip_unavailable(provider)

    captured_kwargs = _capture_init_client(provider)

    assert "max_retries" not in captured_kwargs
    assert "http_options" not in captured_kwargs


@pytest.mark.parametrize("max_retries", [-1, True, 1.5, "2"])
def test_invalid_max_retries_is_rejected(max_retries: Any) -> None:
    with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
        AnyLLM.create("openai", api_key="test_key", max_retries=max_retries)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "deepseek", "groq", "cerebras", "together"])
@pytest.mark.parametrize("max_retries", [0, 5])
def test_native_sdk_client_receives_max_retries(provider: str, max_retries: int) -> None:
    llm = AnyLLM.create(provider, api_key="test_key", max_retries=max_retries)

    assert llm.client.max_retries == max_retries  # type: ignore[attr-defined]


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="voyage is not compatible with Python 3.14+")
@pytest.mark.parametrize("max_retries", [0, 5])
def test_voyage_client_counts_the_first_request_as_an_attempt(max_retries: int) -> None:
    llm = AnyLLM.create("voyage", api_key="test_key", max_retries=max_retries)

    assert llm.client.max_retries == max_retries + 1  # type: ignore[attr-defined]


def test_cohere_client_receives_max_retries() -> None:
    llm = AnyLLM.create("cohere", api_key="test_key", max_retries=0)

    assert llm.client._client_wrapper.get_max_retries() == 0  # type: ignore[attr-defined]


def test_meta_applies_max_retries_to_both_clients() -> None:
    from any_llm.providers.meta.meta import MetaProvider

    llm = AnyLLM.create("meta", api_key="test_key", max_retries=0)

    assert isinstance(llm, MetaProvider)
    assert llm.client.max_retries == 0
    assert llm._anthropic_client.max_retries == 0


def test_openai_compatible_endpoint_accepts_max_retries() -> None:
    llm = AnyLLM.create_openai_compatible("mygateway", api_base="https://gw.example/v1", max_retries=0)

    assert llm.client.max_retries == 0  # type: ignore[attr-defined]


def test_gemini_client_gets_retry_attempts_alongside_api_base() -> None:
    llm = AnyLLM.create("gemini", api_key="test_key", api_base="https://gemini.example", max_retries=2)

    http_options = llm.client._api_client._http_options  # type: ignore[attr-defined]
    assert http_options.base_url == "https://gemini.example"
    assert http_options.retry_options.attempts == 3


def test_gemini_max_retries_merges_into_dict_http_options() -> None:
    llm = AnyLLM.create("gemini", api_key="test_key", max_retries=0, http_options={"api_version": "v1"})

    http_options = llm.client._api_client._http_options  # type: ignore[attr-defined]
    assert http_options.api_version == "v1"
    assert http_options.retry_options.attempts == 1


def test_gemini_explicit_retry_options_take_precedence_over_max_retries() -> None:
    from google.genai import types

    dict_options: dict[str, Any] = {"retry_options": {"attempts": 7}}
    model_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=7))

    for http_options in (dict_options, model_options):
        llm = AnyLLM.create("gemini", api_key="test_key", max_retries=0, http_options=http_options)

        assert llm.client._api_client._http_options.retry_options.attempts == 7  # type: ignore[attr-defined]


def test_gemini_max_retries_does_not_mutate_callers_http_options() -> None:
    from google.genai import types

    model_options = types.HttpOptions(api_version="v1")
    dict_options: dict[str, Any] = {"api_version": "v1"}

    for http_options in (model_options, dict_options):
        llm = AnyLLM.create("gemini", api_key="test_key", max_retries=1, http_options=http_options)

        assert llm.client._api_client._http_options.retry_options.attempts == 2  # type: ignore[attr-defined]
    assert model_options.retry_options is None
    assert dict_options == {"api_version": "v1"}


def test_vertexai_client_gets_retry_attempts() -> None:
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as mock_client:
        AnyLLM.create("vertexai", max_retries=0)

    assert mock_client.call_args.kwargs["http_options"].retry_options.attempts == 1


def _create_vertexai(**create_kwargs: Any) -> VertexaiProvider:
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as mock_client:
        mock_client.return_value._api_client.project = "test-project"
        mock_client.return_value._api_client.location = "us-central1"
        mock_client.return_value._api_client._async_access_token = AsyncMock(return_value="token")
        llm = AnyLLM.create("vertexai", **create_kwargs)
    assert isinstance(llm, VertexaiProvider)
    return llm


async def _mistral_client_of(llm: VertexaiProvider) -> AsyncOpenAI:
    params = CompletionParams(model_id="mistral-small-2503", messages=[{"role": "user", "content": "Hi"}])
    with patch("any_llm.providers.vertexai.vertexai.acompletion_mistral", AsyncMock()):
        await llm._acompletion(params)
    assert llm._mistral_client is not None
    return llm._mistral_client


@pytest.mark.asyncio
async def test_vertexai_openai_sdk_clients_get_max_retries() -> None:
    llm = _create_vertexai(max_retries=0)

    assert llm._get_partner_provider().client.max_retries == 0
    assert (await _mistral_client_of(llm)).max_retries == 0


@pytest.mark.asyncio
async def test_vertexai_openai_sdk_clients_keep_sdk_default_without_max_retries() -> None:
    llm = _create_vertexai()
    sdk_default = AsyncOpenAI(api_key="x").max_retries

    assert llm._get_partner_provider().client.max_retries == sdk_default
    assert (await _mistral_client_of(llm)).max_retries == sdk_default


def test_unsupported_provider_rejects_max_retries_before_building_client() -> None:
    with (
        patch("any_llm.providers.mistral.mistral.MistralProvider._init_client") as init_client,
        pytest.raises(UnsupportedParameterError, match="max_retries"),
    ):
        AnyLLM.create("mistral", api_key="test_key", max_retries=0)

    init_client.assert_not_called()


def test_client_args_max_retries_reaches_provider_from_top_level_api() -> None:
    with patch("any_llm.any_llm.AnyLLM._create_provider") as create_provider:
        create_provider.return_value.completion.return_value = "ok"
        completion(
            model="gpt-4o-mini",
            provider="openai",
            messages=[{"role": "user", "content": "Hello"}],
            client_args={"max_retries": 0},
        )

    assert create_provider.call_args.kwargs["max_retries"] == 0
