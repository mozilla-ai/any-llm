import os
from typing import Any, Iterator
import json

try:
    from ollama import ChatResponse as OllamaChatResponse
    from ollama import Client
except ImportError:
    msg = "ollama is not installed. Please install it with `pip install any-llm-sdk[ollama]`"
    raise ImportError(msg)

from pydantic import BaseModel
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse
from any_llm.provider import ApiConfig, Provider
from any_llm.providers.helpers import create_completion_from_response


from any_llm.providers.ollama.utils import (
    _create_openai_embedding_response_from_ollama,
    _create_openai_chunk_from_ollama_chunk,
    _create_response_dict_from_ollama_response,
)


class OllamaProvider(Provider):
    """
    Ollama Provider using the new response conversion utilities.

    It uses the ollama sdk.
    Read more here - https://github.com/ollama/ollama-python
    """

    PROVIDER_NAME = "Ollama"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/ollama/ollama"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = True
    SUPPORTS_EMBEDDING = True

    def __init__(self, config: ApiConfig) -> None:
        """We don't use the Provider init because by default we don't require an API key."""

        self.url = config.api_base or os.getenv("OLLAMA_API_URL")

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Ollama provider."""
        pass

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

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Ollama."""

        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # response_format is a Pydantic model class
                format = response_format.model_json_schema()
            else:
                # response_format is already a dict/schema
                format = response_format
        else:
            format = None

        # (https://www.reddit.com/r/ollama/comments/1ked8x2/feeding_tool_output_back_to_llm/)
        cleaned_messages = []
        for message in messages:
            if message["role"] == "tool":
                cleaned_message = {
                    "role": "user",
                    "content": json.dumps(message["content"]),
                }
            elif message["role"] == "assistant" and "tool_calls" in message:
                content = message["content"] + "\n" + json.dumps(message["tool_calls"])
                cleaned_message = {
                    "role": "assistant",
                    "content": content,
                }
            else:
                cleaned_message = message.copy()

            cleaned_messages.append(cleaned_message)

        kwargs["num_ctx"] = kwargs.get("num_ctx", 32000)

        client = Client(host=self.url, timeout=kwargs.pop("timeout", None))

        if kwargs.get("stream", False):
            return self._stream_completion(client, model, cleaned_messages, **kwargs)

        response: OllamaChatResponse = client.chat(
            model=model,
            tools=kwargs.pop("tools", None),
            think=kwargs.pop("think", None),
            messages=cleaned_messages,
            format=format,
            options=kwargs,
        )

        response_dict = _create_response_dict_from_ollama_response(response)
        return create_completion_from_response(
            response_data=response_dict,
            model=model,
            provider_name=self.PROVIDER_NAME,
        )

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
