import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider

_AzureADTokenProvider = Callable[[], str | Awaitable[str]]
_PROVIDER_NAME = "azureopenai"
_API_KEY_ENV_NAME = "AZURE_OPENAI_API_KEY"
_AD_TOKEN_ENV_NAME = "AZURE_OPENAI_AD_TOKEN"  # noqa: S105, environment variable name, not a credential


def _resolve_credential(
    api_key: str | None,
    azure_ad_token: str | None,
    azure_ad_token_provider: _AzureADTokenProvider | None,
) -> str | Callable[[], Awaitable[str]]:
    # Match the official Azure client's explicit-credential and Entra-first
    # environment precedence, while treating empty environment values as absent.
    # https://github.com/openai/openai-python/blob/88391abf981df3ea395ca1b5bf55ec6a4011ea93/src/openai/lib/azure.py
    explicit_credential = any(value is not None for value in (api_key, azure_ad_token, azure_ad_token_provider))
    api_key = api_key or None
    azure_ad_token = azure_ad_token or None
    if not explicit_credential:
        azure_ad_token = os.getenv(_AD_TOKEN_ENV_NAME) or None
        if azure_ad_token is None:
            api_key = os.getenv(_API_KEY_ENV_NAME) or None

    configured_credentials = sum(value is not None for value in (api_key, azure_ad_token, azure_ad_token_provider))
    if configured_credentials > 1:
        message = (
            "The `api_key`, `azure_ad_token` and `azure_ad_token_provider` arguments are mutually exclusive; "
            "only one can be passed at a time."
        )
        raise OpenAIError(message)

    if azure_ad_token_provider is not None:

        async def get_token() -> str:
            token = await asyncio.to_thread(azure_ad_token_provider)
            resolved_token = token if isinstance(token, str) else await token
            if not resolved_token:
                message = "Expected `azure_ad_token_provider` to return a non-empty string."
                raise ValueError(message)
            return resolved_token

        return get_token

    credential = api_key or azure_ad_token
    if credential is None:
        env_var_name = f"{_API_KEY_ENV_NAME} or {_AD_TOKEN_ENV_NAME}"
        raise MissingApiKeyError(_PROVIDER_NAME, env_var_name)

    return credential


class AzureopenaiProvider(BaseOpenAIProvider):
    """Azure OpenAI provider using Azure's GA v1 OpenAI-compatible API."""

    ENV_API_KEY_NAME = _API_KEY_ENV_NAME
    ENV_API_BASE_NAME = "AZURE_OPENAI_ENDPOINT"
    PROVIDER_NAME = _PROVIDER_NAME
    PROVIDER_DOCUMENTATION_URL = "https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle"
    SUPPORTS_RESPONSES = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_COMPLETION_PDF = False
    # Azure media remains a dated preview API. Inheriting the false media
    # capability defaults prevents GA v1 requests from being misrouted there.
    SUPPORTS_MODERATION = False

    client: AsyncOpenAI

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # Resolve Azure's three credential forms together in _init_client so an
        # ambient API key cannot override an explicit Microsoft Entra credential.
        return api_key

    @override
    def _resolve_api_base(self, api_base: str | None = None) -> str | None:
        # Defer the environment fallback until _init_client can apply precedence
        # between api_base and the Azure-specific azure_endpoint argument.
        return api_base

    @override
    def _init_client(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        *,
        azure_endpoint: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: _AzureADTokenProvider | None = None,
        api_version: str | None = None,
        azure_deployment: str | None = None,
        default_query: Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> None:
        # Microsoft v1 defaults to an implicit `v1` API version and uses the
        # deployment name in `model`. Rejecting legacy routing options prevents
        # a dated-route configuration from appearing to work while being ignored.
        # https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle
        if api_version is not None:
            parameter_name = "api_version"
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)
        if os.getenv("OPENAI_API_VERSION"):
            parameter_name = "OPENAI_API_VERSION"
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)
        if azure_deployment is not None:
            parameter_name = "azure_deployment"
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)
        # The GA schema still permits an explicit `api-version=v1`, even though
        # the lifecycle guide recommends omitting it. Dated values belong to the
        # retired route family, and preview features now use headers or paths.
        # https://learn.microsoft.com/rest/api/microsoft-foundry/azureopenai/chat
        if default_query is not None and default_query.get("api-version", "v1") != "v1":
            parameter_name = "default_query['api-version']"
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)

        client_api_key = _resolve_credential(api_key, azure_ad_token, azure_ad_token_provider)

        endpoint = api_base or azure_endpoint or os.getenv(self.ENV_API_BASE_NAME)
        if not endpoint:
            message = (
                "Azure OpenAI endpoint is required. Pass `api_base` or `azure_endpoint`, "
                f"or set {self.ENV_API_BASE_NAME}."
            )
            raise ValueError(message)

        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/openai/v1"):
            endpoint = f"{endpoint}/openai/v1"

        # Current Microsoft Python examples use the generic OpenAI client for
        # both API keys and Entra token providers. This also leaves retries,
        # refresh timing and sensitive Authorization redirects with the SDK.
        self.client = AsyncOpenAI(
            api_key=client_api_key,
            base_url=f"{endpoint}/",
            default_query=default_query,
            **kwargs,
        )
