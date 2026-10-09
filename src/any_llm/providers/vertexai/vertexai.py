from collections.abc import AsyncIterator
from typing import Any, cast

from google import genai
from openai import AsyncOpenAI
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.gemini.base import GoogleProvider
from any_llm.providers.openai.base import BaseOpenAIProvider
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


def _partner_api_base(project: str, location: str) -> str:
    """Build the base URL of Vertex's OpenAI-compatible endpoint.

    See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/call-open-model-apis
    """
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


class _VertexaiPartnerProvider(BaseOpenAIProvider):
    """OpenAI-compatible client for Vertex AI partner models, authenticated with a Vertex access token."""

    PROVIDER_NAME = "vertexai"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/generative-ai/docs/maas/call-open-model-apis"
    ENV_API_KEY_NAME = ""
    ENV_API_BASE_NAME = ""

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
    chat completions endpoint, authenticated with the same Google Cloud credentials.
    """

    PROVIDER_NAME = "vertexai"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"
    ENV_API_KEY_NAME = ""
    ENV_API_BASE_NAME = "VERTEXAI_API_BASE"

    _partner_provider: _VertexaiPartnerProvider | None = None

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        """Get Vertex AI client."""

        # Ensure timeout is correctly configured if present.
        if (timeout := kwargs.pop("timeout", None)) is not None:
            GoogleProvider._merge_timeout_into_http_options(timeout, kwargs)

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
                api_base=_partner_api_base(cast("str", api_client.project), cast("str", api_client.location)),
                token_provider=self._access_token,
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
        return await super()._acompletion(params, **kwargs)
