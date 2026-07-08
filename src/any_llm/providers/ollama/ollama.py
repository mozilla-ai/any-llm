from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

MISSING_PACKAGES_ERROR = None
try:
    from ollama import AsyncClient

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

    from ollama import AsyncClient  # noqa: TC004
    from ollama import ChatResponse as OllamaChatResponse

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


# Ollama's `think` parameter accepts only low/medium/high levels (plus booleans)
# Reasoning_effort scale is collapsed at the extremes.
REASONING_EFFORT_TO_OLLAMA_THINK: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class OllamaProvider(AnyLLM):
    """
    Ollama Provider using the new response conversion utilities.

    It uses the ollama sdk.
    Read more here - https://github.com/ollama/ollama-python
    """

    PROVIDER_NAME = "ollama"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/ollama/ollama"
    ENV_API_KEY_NAME = "None"
    ENV_API_BASE_NAME = "OLLAMA_HOST"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False
    SUPPORTS_RERANK = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncClient

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Ollama API."""
        converted_params = params.model_dump(
            exclude_none=True,
            exclude={"model_id", "messages", "reasoning_effort", "response_format", "stream", "stream_options"},
        )
        if params.reasoning_effort == "none":
            converted_params["think"] = False
        elif params.reasoning_effort is not None and params.reasoning_effort != "auto":
            converted_params["think"] = REASONING_EFFORT_TO_OLLAMA_THINK[params.reasoning_effort]
        converted_params.update(kwargs)
        converted_params["num_ctx"] = converted_params.get("num_ctx", 32000)
        return converted_params

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Ollama response to OpenAI format."""
        return _create_chat_completion_from_ollama_response(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Ollama chunk response to OpenAI format."""
        return _create_openai_chunk_from_ollama_chunk(response)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Ollama."""
        converted_params = {"input": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Ollama embedding response to OpenAI format."""
        return _create_openai_embedding_response_from_ollama(response)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Ollama list models response to OpenAI format."""
        return _convert_models_list(response)

    @staticmethod
    def _convert_response_format(
        response_format: dict[str, Any] | type | None,
    ) -> Literal["json"] | dict[str, Any] | None:
        """Convert a `response_format` into Ollama's `format` argument.

        Ollama's `format` accepts either the string "json" (unconstrained JSON mode) or a raw
        JSON schema dict (https://docs.ollama.com/capabilities/structured-outputs). OpenAI-style
        dicts (`json_object`, `json_schema`, `text`) are translated accordingly; any other dict
        is assumed to already be a raw schema and passed through unchanged.
        """
        if response_format is None:
            return None
        if is_structured_output_type(response_format):
            return get_json_schema(response_format)
        if isinstance(response_format, dict):
            response_type = response_format.get("type")
            if response_type == "json_schema":
                json_schema = response_format.get("json_schema")
                if json_schema is None or "schema" not in json_schema:
                    msg = "json_schema response_format must include 'json_schema.schema'"
                    raise ValueError(msg)
                schema: dict[str, Any] = json_schema["schema"]
                return schema
            if response_type == "json_object":
                return "json"
            if response_type == "text":
                return None
            return response_format
        return None

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncClient(host=api_base, **kwargs)

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return api_key

    def _extract_images_from_message(self, message: dict[str, Any]) -> tuple[str, list[str]]:
        """
        Extract images from OpenAI format message and return text content + base64 image strings.

        Args:
            message: OpenAI format message that may contain image_url content

        Returns:
            tuple of (text_content, list_of_base64_image_strings)
        """
        content = message.get("content", "")
        images = []

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image/"):
                        base64_data = image_url.split(",", 1)[1] if "," in image_url else ""
                        if base64_data:
                            # Ollama expects base64 strings directly
                            images.append(base64_data)
            content = " ".join(text_parts)

        return content, images

    async def _stream_completion_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        output_format: Literal["json"] | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        kwargs.pop("stream", None)
        response: AsyncIterator[OllamaChatResponse] = await self.client.chat(
            model=model,
            messages=messages,
            tools=kwargs.pop("tools", None),
            think=kwargs.pop("think", None),
            format=output_format,
            stream=True,
            options=kwargs,
        )
        # Ollama streams each tool call in its own chunk, and the chunk
        # converter (_create_openai_chunk_from_ollama_chunk) stamps every
        # tool-call delta with index=0. A consumer that accumulates streaming
        # deltas by index then concatenates the arguments of distinct tool calls
        # into a single, invalid JSON string. Assign a stream-global,
        # monotonically increasing index so each tool call stays in its own slot.
        tool_call_index = 0
        async for chunk in response:
            converted = self._convert_completion_chunk_response(chunk)
            for choice in converted.choices:
                delta = choice.delta
                if delta is not None and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        tool_call.index = tool_call_index
                        tool_call_index += 1
            yield converted

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Ollama."""

        output_format = self._convert_response_format(params.response_format)

        # (https://www.reddit.com/r/ollama/comments/1ked8x2/feeding_tool_output_back_to_llm/)
        cleaned_messages = []
        for input_message in params.messages:
            if input_message["role"] == "tool":
                cleaned_message: dict[str, Any] = {
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

                if input_message["role"] == "user":
                    text_content, image_base64_list = self._extract_images_from_message(input_message)
                    cleaned_message["content"] = text_content

                    if image_base64_list:
                        cleaned_message["images"] = image_base64_list

            cleaned_messages.append(cleaned_message)

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                params.model_id, cleaned_messages, output_format=output_format, **completion_kwargs
            )

        response: OllamaChatResponse = await self.client.chat(
            model=params.model_id,
            tools=completion_kwargs.pop("tools", None),
            think=completion_kwargs.pop("think", None),
            messages=cleaned_messages,
            format=output_format,
            options=completion_kwargs,
        )
        return self._convert_completion_response(response)

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Generate embeddings using Ollama."""
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        response = await self.client.embed(
            model=model,
            **embedding_kwargs,
        )
        return self._convert_embedding_response(response)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.list(**kwargs)
        return self._convert_list_models_response(models_list)
