from __future__ import annotations

from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.portkey.utils import _convert_chat_completion, _convert_chat_completion_chunk
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, Reasoning
from any_llm.types.model import Model
from any_llm.utils.reasoning import (
    process_streaming_reasoning_chunks,
)

MISSING_PACKAGES_ERROR = None
try:
    from portkey_ai import AsyncPortkey
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from openai._streaming import AsyncStream
    from portkey_ai import AsyncPortkey  # noqa: TC004


class PortkeyProvider(BaseOpenAIProvider):
    """Portkey provider for accessing 200+ LLMs through Portkey's AI Gateway."""

    API_BASE = "https://api.portkey.ai/v1"
    ENV_API_KEY_NAME = "PORTKEY_API_KEY"
    PROVIDER_NAME = "portkey"
    PROVIDER_DOCUMENTATION_URL = "https://portkey.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    _DEFAULT_REASONING_EFFORT = None

    client: AsyncPortkey  # type: ignore[assignment]

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncPortkey(
            api_key=api_key,
            base_url=api_base or self.API_BASE,
            **kwargs,
        )

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        if isinstance(response, OpenAIChatCompletion):
            return _convert_chat_completion(response)
        if isinstance(response, ChatCompletion):
            return response
        # Handle portkey SDK's ChatCompletions type (a Pydantic BaseModel)
        if isinstance(response, BaseModel):
            return _convert_chat_completion(OpenAIChatCompletion.model_validate(response.model_dump()))
        return ChatCompletion.model_validate(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        if isinstance(response, OpenAIChatCompletionChunk):
            return _convert_chat_completion_chunk(response)
        if isinstance(response, ChatCompletionChunk):
            return response
        # Handle portkey SDK's ChatCompletionChunk type (a Pydantic BaseModel)
        if isinstance(response, BaseModel):
            return _convert_chat_completion_chunk(OpenAIChatCompletionChunk.model_validate(response.model_dump()))
        return ChatCompletionChunk.model_validate(response)

    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert an OpenAI completion response with streaming reasoning support."""
        # Check for non-streaming response: either OpenAI type or portkey SDK type (which is not async iterable)
        if isinstance(response, OpenAIChatCompletion) or not isinstance(response, AsyncIterable):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                yield self._convert_completion_chunk_response(chunk)

        def get_content(chunk: ChatCompletionChunk) -> str | None:
            return chunk.choices[0].delta.content if len(chunk.choices) > 0 else None

        def set_content(chunk: ChatCompletionChunk, content: str | None) -> ChatCompletionChunk:
            chunk.choices[0].delta.content = content
            return chunk

        def set_reasoning(chunk: ChatCompletionChunk, reasoning: str) -> ChatCompletionChunk:
            chunk.choices[0].delta.reasoning = Reasoning(content=reasoning)
            return chunk

        return process_streaming_reasoning_chunks(
            chunk_iterator(),
            get_content=get_content,
            set_content=set_content,
            set_reasoning=set_reasoning,
        )

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert portkey list models response to any-llm format."""
        data = getattr(response, "data", None)
        if data is not None:
            models = []
            for model in data:
                if isinstance(model, Model):
                    models.append(model)
                elif isinstance(model, BaseModel):
                    model_dict = model.model_dump()
                    # Portkey SDK may return None for required fields
                    if model_dict.get("created") is None:
                        model_dict["created"] = 0
                    if model_dict.get("owned_by") is None:
                        model_dict["owned_by"] = "portkey"
                    models.append(Model.model_validate(model_dict))
                else:
                    models.append(Model.model_validate(model))
            return models
        return [Model.model_validate(item) if not isinstance(item, Model) else item for item in response]

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenAI API."""
        if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
            params.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": params.response_format.model_json_schema(),
                },
            }
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params
