from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_llm.provider import ApiConfig, Provider

MISSING_PACKAGES_ERROR = None
try:
    from ollama import AsyncClient, Client

    from .utils import (
        _convert_models_list,
        _create_chat_completion_from_ollama_response,
        _create_openai_chunk_from_ollama_chunk,
        _create_openai_embedding_response_from_ollama,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from ollama import AsyncClient, Client  # noqa: TC004
    from ollama import ChatResponse as OllamaChatResponse

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class OllamaProvider(Provider):
    """
    Ollama Provider using the new response conversion utilities.

    It uses the ollama sdk.
    Read more here - https://github.com/ollama/ollama-python
    """

    PROVIDER_NAME = "ollama"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/ollama/ollama"
    ENV_API_KEY_NAME = "None"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    def __init__(self, config: ApiConfig) -> None:
        """We don't use the Provider init because by default we don't require an API key."""
        self._verify_no_missing_packages()

        self.url = config.api_base or os.getenv("OLLAMA_API_URL")

    async def _stream_completion_async(
        self,
        client: AsyncClient,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        kwargs.pop("stream", None)
        response: AsyncIterator[OllamaChatResponse] = await client.chat(
            model=model,
            messages=messages,
            think=kwargs.pop("think", None),
            stream=True,
            options=kwargs,
        )
        async for chunk in response:
            yield _create_openai_chunk_from_ollama_chunk(chunk)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Ollama."""

        if params.response_format is not None:
            response_format = params.response_format
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # response_format is a Pydantic model class
                output_format = response_format.model_json_schema()
            else:
                # response_format is already a dict/schema
                output_format = response_format
        else:
            output_format = None

        # (https://www.reddit.com/r/ollama/comments/1ked8x2/feeding_tool_output_back_to_llm/)
        cleaned_messages = []
        for input_message in params.messages:
            if input_message["role"] == "tool":
                cleaned_message = {
                    "role": "user",
                    "content": json.dumps(input_message["content"]),
                }
            elif input_message["role"] == "assistant" and "tool_calls" in input_message:
                content = input_message["content"] + "\n" + json.dumps(input_message["tool_calls"])
                cleaned_message = {
                    "role": "assistant",
                    "content": content,
                }
            else:
                cleaned_message = input_message.copy()

            cleaned_messages.append(cleaned_message)

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        kwargs = {
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}),
            **kwargs,
        }

        kwargs["num_ctx"] = kwargs.get("num_ctx", 32000)

        if params.reasoning_effort is not None:
            kwargs["think"] = True

        client = AsyncClient(host=self.url, timeout=kwargs.pop("timeout", None))

        if params.stream:
            return self._stream_completion_async(client, params.model_id, cleaned_messages, **kwargs)

        response: OllamaChatResponse = await client.chat(
            model=params.model_id,
            tools=kwargs.pop("tools", None),
            think=kwargs.pop("think", None),
            messages=cleaned_messages,
            format=output_format,
            options=kwargs,
        )
        return _create_chat_completion_from_ollama_response(response)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Generate embeddings using Ollama."""
        client = AsyncClient(host=self.url, timeout=kwargs.pop("timeout", None))

        response = await client.embed(
            model=model,
            input=inputs,
            **kwargs,
        )
        return _create_openai_embedding_response_from_ollama(response)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        client = Client(host=self.url, timeout=kwargs.pop("timeout", None))
        models_list = client.list(**kwargs)
        return _convert_models_list(models_list)
