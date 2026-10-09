from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from google import genai
from google.genai import types
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from typing_extensions import override

from any_llm.exceptions import InvalidRequestError, MissingApiKeyError
from any_llm.providers.gemini.base import GoogleProvider
from any_llm.providers.openai.base import BaseOpenAIProvider

from .mistral import acompletion_mistral, create_mistral_client, is_mistral_model

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams

# Model Garden partner (MaaS) models that Vertex serves only through its OpenAI-compatible
# endpoint; `generateContent` rejects some of their requests (for example a streamed request
# with tools on Qwen answers 400 "Expected a valid JSON object in the request").
_PARTNER_MODEL_PREFIXES = (
    "qwen",
    "openai/gpt-oss-",
    "deepseek-ai",
    "llama",
    "meta/llama",
    "minimaxai/",
    "moonshotai/",
    "zai-org/",
)


def _is_partner_model(model_id: str) -> bool:
    return model_id.lower().startswith(_PARTNER_MODEL_PREFIXES)


def _partner_api_base(project: str, location: str, base_url: str | None = None) -> str:
    """Build the base URL of Vertex's OpenAI-compatible endpoint.

    A ``base_url`` configured on the genai client (a private endpoint or a gateway) replaces the public host.
    See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/call-open-model-apis
    """
    if base_url:
        root = base_url.rstrip("/")
    else:
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        root = f"https://{host}"
    return f"{root}/v1/projects/{project}/locations/{location}/endpoints/openapi"


def _partner_client_kwargs(http_options: types.HttpOptions | None) -> dict[str, Any]:
    """Carry the genai client's HTTP settings over to the OpenAI SDK client used for partner models."""
    if http_options is None:
        return {}
    client_kwargs: dict[str, Any] = {}
    if http_options.timeout is not None:
        client_kwargs["timeout"] = http_options.timeout / 1000
    if http_options.headers:
        # The partner client sends its own refreshed OAuth token.
        client_kwargs["default_headers"] = {
            name: value for name, value in http_options.headers.items() if name.lower() != "authorization"
        }
    if http_options.httpx_async_client is not None:
        client_kwargs["http_client"] = http_options.httpx_async_client
    elif http_options.async_client_args:
        # genai reads these as aiohttp arguments when aiohttp is installed; the OpenAI SDK only speaks httpx, so
        # arguments it cannot take fail here instead of silently dropping a proxy or CA from partner requests.
        try:
            client_kwargs["http_client"] = DefaultAsyncHttpxClient(**http_options.async_client_args)
        except TypeError as exc:
            msg = "http_options.async_client_args must be httpx.AsyncClient arguments to call Vertex AI partner models"
            raise InvalidRequestError(msg, exc, "vertexai") from exc
    return client_kwargs


class _VertexaiPartnerProvider(BaseOpenAIProvider):
    """OpenAI-compatible client for Vertex AI partner models, authenticated with a Vertex access token."""

    PROVIDER_NAME = "vertexai"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/generative-ai/docs/maas/call-open-model-apis"
    ENV_API_KEY_NAME = ""
    ENV_API_BASE_NAME = ""

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Vertex documents ``max_tokens`` for MaaS models; ``max_completion_tokens`` is only an alias for Gemini."""
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if "max_completion_tokens" in converted_params:
            converted_params["max_tokens"] = converted_params.pop("max_completion_tokens")
        return converted_params

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        # The bearer is a short-lived OAuth token, so the SDK asks for a fresh one on every request.
        self.client = AsyncOpenAI(base_url=api_base, api_key=kwargs.pop("token_provider"), **kwargs)


class VertexaiProvider(GoogleProvider):
    """Vertex AI Provider using Google Cloud Vertex AI.

    Gemini models use `generateContent` through `google-genai`. Model Garden partner models
    (Qwen, gpt-oss, DeepSeek, Llama, MiniMax, Kimi, and GLM) use Vertex's OpenAI-compatible
    chat completions endpoint. Mistral models (IDs starting with `mistral` or `codestral`, such
    as `mistral-small-2503`) go to the `mistralai` publisher's `rawPredict` endpoints. All of
    them authenticate with the same Google Cloud credentials.
    """

    PROVIDER_NAME = "vertexai"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"
    ENV_API_KEY_NAME = ""
    ENV_API_BASE_NAME = "VERTEXAI_API_BASE"

    _partner_provider: _VertexaiPartnerProvider | None = None
    _mistral_client: AsyncOpenAI | None = None
    # Seconds, taken from the genai client's http_options so the OpenAI-SDK routes time out the same way.
    _http_timeout: float | None = None
    _http_options: types.HttpOptions | None = None
    # Kept for the OpenAI-SDK clients of partner and Mistral models; genai gets it through http_options.
    _max_retries: int | None = None

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return api_key

    @override
    def _apply_max_retries(self, max_retries: int, client_kwargs: dict[str, Any]) -> None:
        super()._apply_max_retries(max_retries, client_kwargs)
        self._max_retries = max_retries

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        """Get Vertex AI client."""

        # Ensure timeout is correctly configured if present.
        if (timeout := kwargs.pop("timeout", None)) is not None:
            GoogleProvider._merge_timeout_into_http_options(timeout, kwargs)

        http_options = kwargs.get("http_options")
        if isinstance(http_options, dict):
            http_options = types.HttpOptions.model_validate(http_options)
        if isinstance(http_options, types.HttpOptions):
            self._http_options = http_options
            if http_options.timeout is not None:
                self._http_timeout = http_options.timeout / 1000

        self.client = genai.Client(
            vertexai=True,
            **kwargs,
        )
        if self.client._api_client.project is None:
            msg = "vertexai"
            raise MissingApiKeyError(msg, "GOOGLE_CLOUD_PROJECT")
        if self.client._api_client.location is None:
            msg = "vertexai"
            raise MissingApiKeyError(msg, "GOOGLE_CLOUD_LOCATION")

    async def _access_token(self) -> str:
        """Return a valid OAuth access token from the credentials the genai client uses, refreshing it if expired."""
        return cast("str", await self.client._api_client._async_access_token())

    def _get_partner_provider(self) -> _VertexaiPartnerProvider:
        if self._partner_provider is None:
            api_client = self.client._api_client
            self._partner_provider = _VertexaiPartnerProvider(
                api_base=_partner_api_base(
                    cast("str", api_client.project),
                    cast("str", api_client.location),
                    self._http_options.base_url if self._http_options else None,
                ),
                token_provider=self._access_token,
                max_retries=self._max_retries,
                **_partner_client_kwargs(self._http_options),
            )
        return self._partner_provider

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if _is_partner_model(params.model_id):
            return await self._get_partner_provider()._acompletion(params, **kwargs)
        if not is_mistral_model(params.model_id):
            return await super()._acompletion(params, **kwargs)

        api_client = self.client._api_client
        if self._mistral_client is None:
            self._mistral_client = create_mistral_client(
                project=str(api_client.project),
                location=str(api_client.location),
                timeout=self._http_timeout,
                max_retries=self._max_retries,
            )
        # The genai client owns the credentials and refreshes the token when it expires.
        access_token = await self._access_token()
        return await acompletion_mistral(self._mistral_client, access_token, params, **kwargs)
