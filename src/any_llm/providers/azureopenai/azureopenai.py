import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from openai import AsyncOpenAI, AsyncStream, OpenAIError
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from openai.types.chat import ChatCompletionChunk as OpenAIChatCompletionChunk
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.audio import AudioSpeechParams, AudioTranscriptionParams, Transcription
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.types.image import ImageGenerationParams, ImagesResponse

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
    explicit_credentials = sum(value is not None for value in (api_key, azure_ad_token, azure_ad_token_provider))
    if explicit_credentials > 1:
        message = (
            "The `api_key`, `azure_ad_token` and `azure_ad_token_provider` arguments are mutually exclusive; "
            "only one can be passed at a time."
        )
        raise OpenAIError(message)

    api_key = api_key or None
    azure_ad_token = azure_ad_token or None
    if not explicit_credentials:
        azure_ad_token = os.getenv(_AD_TOKEN_ENV_NAME) or None
        if azure_ad_token is None:
            api_key = os.getenv(_API_KEY_ENV_NAME) or None

    if azure_ad_token_provider is not None:

        async def get_token() -> str:
            token = await asyncio.to_thread(azure_ad_token_provider)
            resolved_token = await token if isinstance(token, Awaitable) else token
            if not isinstance(resolved_token, str) or not resolved_token:
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
    """Azure OpenAI v1 with GA core routes and operation-scoped preview media.

    Supply deployment names as request models, not as client routing options.
    Explicit credentials and endpoint arguments take precedence over Azure
    environment settings. Dated API versions are not supported.
    """

    ENV_API_KEY_NAME = _API_KEY_ENV_NAME
    ENV_API_BASE_NAME = "AZURE_OPENAI_ENDPOINT"
    PROVIDER_NAME = _PROVIDER_NAME
    PROVIDER_DOCUMENTATION_URL = "https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle"
    SUPPORTS_RESPONSES = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_IMAGE_GENERATION = True
    SUPPORTS_AUDIO_TRANSCRIPTION = True
    SUPPORTS_AUDIO_SPEECH = True
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
        selected_version = api_version if api_version is not None else os.getenv("OPENAI_API_VERSION") or None
        if selected_version not in (None, "v1"):
            parameter_name = "api_version" if api_version is not None else "OPENAI_API_VERSION"
            raise UnsupportedParameterError(
                parameter_name,
                self.PROVIDER_NAME,
                'Azure OpenAI now uses /openai/v1/. Remove the dated version or set api_version="v1".',
            )
        if azure_deployment is not None:
            parameter_name = "azure_deployment"
            raise UnsupportedParameterError(
                parameter_name,
                self.PROVIDER_NAME,
                "Pass your Azure deployment name as `model` on each request instead of `azure_deployment`.",
            )
        # The GA schema still permits an explicit `api-version=v1`, even though
        # the lifecycle guide recommends omitting it. Dated values belong to the
        # retired route family, and preview features now use headers or paths.
        # https://learn.microsoft.com/rest/api/microsoft-foundry/azureopenai/chat
        if default_query is not None and default_query.get("api-version", "v1") != "v1":
            parameter_name = "default_query['api-version']"
            raise UnsupportedParameterError(
                parameter_name,
                self.PROVIDER_NAME,
                "Remove this query entry or use 'v1'. Media preview options are scoped to individual requests.",
            )

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

    @override
    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if isinstance(response, OpenAIChatCompletion):
            return self._convert_completion_response(response)

        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            try:
                async for chunk in response:
                    yield self._convert_completion_chunk_response(chunk)
            finally:
                await response.close()

        return chunks()

    def _media_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        # Azure documents these routes under the v1 preview reference, not the
        # dated deployment API. Keep preview local to media on the same client.
        # https://learn.microsoft.com/azure/ai-foundry/openai/reference-preview-latest
        query = {"api-version": "preview", **(kwargs.get("extra_query") or {})}
        if query["api-version"] not in ("v1", "preview"):
            parameter_name = "extra_query['api-version']"
            raise UnsupportedParameterError(
                parameter_name,
                self.PROVIDER_NAME,
                "Media uses /openai/v1/. Use 'preview' (default) or 'v1', not a dated API version.",
            )
        return {**kwargs, "extra_query": query}

    @override
    async def _aimage_generation(self, params: ImageGenerationParams, **kwargs: Any) -> ImagesResponse:
        return await super()._aimage_generation(params, **self._media_options(kwargs))

    @override
    async def _atranscription(self, params: AudioTranscriptionParams, **kwargs: Any) -> Transcription:
        return await super()._atranscription(params, **self._media_options(kwargs))

    @override
    async def _aspeech(self, params: AudioSpeechParams, **kwargs: Any) -> bytes:
        return await super()._aspeech(params, **self._media_options(kwargs))
