from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from typing_extensions import override

from any_llm.providers.openai.xml_reasoning import XMLReasoningOpenAIProvider
from any_llm.providers.openai.xml_reasoning_utils import (
    convert_chat_completion_chunk_with_xml_reasoning,
    convert_chat_completion_with_xml_reasoning,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.model import Model
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

try:
    import portkey_ai

    AsyncPortkey: Any = portkey_ai.AsyncPortkey
except ImportError as e:  # pragma: no cover - exercised by the package guard
    AsyncPortkey = None
    MISSING_PACKAGES_ERROR: ImportError | None = e
else:
    MISSING_PACKAGES_ERROR = None


class PortkeyProvider(XMLReasoningOpenAIProvider):
    """Portkey provider for accessing 200+ LLMs through Portkey's AI Gateway."""

    API_BASE = "https://api.portkey.ai/v1"
    ENV_API_KEY_NAME = "PORTKEY_API_KEY"
    ENV_API_BASE_NAME = "PORTKEY_API_BASE"
    PROVIDER_NAME = "portkey"
    PROVIDER_DOCUMENTATION_URL = "https://portkey.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    _DEFAULT_REASONING_EFFORT = None
    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR
    client: Any

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        """Initialize Portkey's native async client."""
        # Preserve the timeout behavior of the former OpenAI client: a bounded read
        # timeout with a shorter connection timeout, unless explicitly overridden.
        timeout = kwargs.pop("timeout", httpx.Timeout(600.0, connect=5.0))
        kwargs.setdefault("http_client", httpx.AsyncClient(timeout=timeout))
        kwargs.setdefault("request_timeout", timeout)
        self.client = AsyncPortkey(
            api_key=api_key,
            base_url=api_base or self.API_BASE,
            **kwargs,
        )

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert a Portkey completion response and extract XML reasoning."""
        if isinstance(response, OpenAIChatCompletion):
            return convert_chat_completion_with_xml_reasoning(response)
        if isinstance(response, ChatCompletion):
            return response
        if hasattr(response, "model_dump"):
            return convert_chat_completion_with_xml_reasoning(response)
        return ChatCompletion.model_validate(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert a Portkey completion chunk and extract XML reasoning."""
        if isinstance(response, OpenAIChatCompletionChunk):
            return convert_chat_completion_chunk_with_xml_reasoning(response)
        if isinstance(response, ChatCompletionChunk):
            return response
        if hasattr(response, "model_dump"):
            return convert_chat_completion_chunk_with_xml_reasoning(response)
        return ChatCompletionChunk.model_validate(response)

    @override
    def _convert_completion_response_async(self, response: Any) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert native Portkey completion objects and streams.

        Portkey returns Pydantic models from its vendored OpenAI SDK, so they do
        not satisfy the OpenAI SDK ``isinstance`` checks in the shared base.
        """
        if not hasattr(response, "__aiter__"):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                yield self._convert_completion_chunk_response(chunk)

        from any_llm.providers.openai.xml_reasoning import wrap_chunks_with_xml_reasoning

        return wrap_chunks_with_xml_reasoning(chunk_iterator())

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Portkey's vendored OpenAI model objects to AnyLLM models."""
        models = response.data if hasattr(response, "data") else response
        return [Model.model_validate(model.model_dump() if hasattr(model, "model_dump") else model) for model in models]

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenAI API."""
        if is_structured_output_type(params.response_format):
            params.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": get_json_schema(params.response_format),
                },
            }
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.setdefault("timeout", 600.0)
        converted_params.update(kwargs)
        return converted_params

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        """List models with the legacy bounded read timeout."""
        kwargs.setdefault("timeout", 600.0)
        response = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(response)
