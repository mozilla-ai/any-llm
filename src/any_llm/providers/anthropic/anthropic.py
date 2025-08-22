from collections.abc import AsyncIterator, Sequence
from typing import Any

try:
    from anthropic import Anthropic, AsyncAnthropic

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.provider import Provider
from any_llm.providers.anthropic.utils import (
    _convert_models_list,
    _convert_params,
    _convert_response,
    _create_openai_chunk_from_anthropic_chunk,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.model import Model


class AnthropicProvider(Provider):
    """
    Anthropic Provider using enhanced Provider framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_LABEL = "Anthropic"
    PROVIDER_NAME = "anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.anthropic.com/en/home"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    async def _stream_completion_async(
        self, client: "AsyncAnthropic", **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        async with client.messages.stream(
            **kwargs,
        ) as anthropic_stream:
            async for event in anthropic_stream:
                yield _create_openai_chunk_from_anthropic_chunk(event, kwargs.get("model", "unknown"))

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Anthropic with instructor support."""
        client = AsyncAnthropic(api_key=self.config.api_key, base_url=self.config.api_base)

        kwargs["provider_name"] = self.PROVIDER_NAME
        converted_kwargs = _convert_params(params, **kwargs)

        if converted_kwargs.pop("stream", False):
            return self._stream_completion_async(client, **converted_kwargs)

        message = await client.messages.create(**converted_kwargs)

        return _convert_response(message)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """Return a list of Model for Anthropic models.
        Supports query params: before_id, after_id, limit.
        """
        client = Anthropic(api_key=self.config.api_key, base_url=self.config.api_base)
        query_params = {
            "limit": kwargs.get("limit", 100),
            **{k: v for k in ["before_id", "after_id"] if (v := kwargs.get(k)) is not None},
        }
        # documentation - https://docs.anthropic.com/en/api/models-list
        try:
            anthropic_models = client.models.list(**query_params)
            results = _convert_models_list(anthropic_models, provider_name=self.PROVIDER_NAME)
        except Exception as e:
            from any_llm.exceptions import UnsupportedModelResponseError

            raise UnsupportedModelResponseError(
                message="Failed to parse Anthropic model response.", original_exception=e
            ) from e
        else:
            return results
