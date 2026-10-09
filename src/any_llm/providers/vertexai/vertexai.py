from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google import genai
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.gemini.base import GoogleProvider

from .mistral import acompletion_mistral, create_mistral_client, is_mistral_model

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai import AsyncOpenAI

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class VertexaiProvider(GoogleProvider):
    """Vertex AI Provider using Google Cloud Vertex AI.

    Gemini models go through the genai SDK. Mistral models (IDs starting with `mistral` or
    `codestral`, such as `mistral-small-2503`) go to the `mistralai` publisher's `rawPredict`
    endpoints instead, authenticated with the same Google Cloud credentials.
    """

    PROVIDER_NAME = "vertexai"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"
    ENV_API_KEY_NAME = ""
    ENV_API_BASE_NAME = "VERTEXAI_API_BASE"

    _mistral_client: AsyncOpenAI | None = None
    _mistral_timeout: float | None = None

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        """Get Vertex AI client."""

        # Ensure timeout is correctly configured if present.
        if (timeout := kwargs.pop("timeout", None)) is not None:
            GoogleProvider._merge_timeout_into_http_options(timeout, kwargs)
            self._mistral_timeout = timeout

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

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if not is_mistral_model(params.model_id):
            return await super()._acompletion(params, **kwargs)

        api_client = self.client._api_client
        if self._mistral_client is None:
            self._mistral_client = create_mistral_client(
                project=str(api_client.project), location=str(api_client.location), timeout=self._mistral_timeout
            )
        # The genai client owns the credentials and refreshes the token when it expires.
        access_token = await api_client._async_access_token()
        return await acompletion_mistral(self._mistral_client, str(access_token), params, **kwargs)
