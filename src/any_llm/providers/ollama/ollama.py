import json
import os
from collections.abc import Iterator
from typing import Any

try:
    from ollama import ChatResponse as OllamaChatResponse
    from ollama import Client

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from pydantic import BaseModel

from any_llm.provider import ApiConfig, Provider
from any_llm.providers.ollama.utils import (
    _create_chat_completion_from_ollama_response,
    _create_openai_chunk_from_ollama_chunk,
    _create_openai_embedding_response_from_ollama,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse


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

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def __init__(self, config: ApiConfig) -> None:
        """We don't use the Provider init because by default we don't require an API key."""

        self.url = config.api_base or os.getenv("OLLAMA_API_URL")

    def _stream_completion(
        self,
        client: Client,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        kwargs.pop("stream", None)
        response: Iterator[OllamaChatResponse] = client.chat(
            model=model,
            messages=messages,
            think=kwargs.pop("think", None),
            stream=True,
            options=kwargs,
        )
        for chunk in response:
            yield _create_openai_chunk_from_ollama_chunk(chunk)

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
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

        kwargs = {
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}),
            **kwargs,
        }

        kwargs["num_ctx"] = kwargs.get("num_ctx", 32000)

        client = Client(host=self.url, timeout=kwargs.pop("timeout", None))

        if params.stream:
            return self._stream_completion(client, params.model_id, cleaned_messages, **kwargs)

        response: OllamaChatResponse = client.chat(
            model=params.model_id,
            tools=kwargs.pop("tools", None),
            think=kwargs.pop("think", None),
            messages=cleaned_messages,
            format=output_format,
            options=kwargs,
        )
        return _create_chat_completion_from_ollama_response(response)

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Generate embeddings using Ollama."""
        client = Client(host=self.url, timeout=kwargs.pop("timeout", None))

        response = client.embed(
            model=model,
            input=inputs,
            **kwargs,
        )
        return _create_openai_embedding_response_from_ollama(response)
