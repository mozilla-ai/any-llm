import os
from abc import ABC
from collections.abc import Iterator
from typing import Any, cast

from openai import OpenAI
from openai._streaming import Stream
from openai._types import NOT_GIVEN
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

from any_llm.logging import logger
from any_llm.provider import Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.responses import Response, ResponseStreamEvent


class BaseOpenAIProvider(Provider, ABC):
    """
    Base provider for OpenAI-compatible services.

    This class provides a common foundation for providers that use OpenAI-compatible APIs.
    Subclasses only need to override configuration defaults and client initialization
    if needed.
    """

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = True

    PACKAGES_INSTALLED = True

    def _normalize_reasoning_on_message(self, message_dict: dict[str, Any]) -> None:
        """Mutate a message dict to move provider-specific reasoning fields to our Reasoning type."""
        if isinstance(message_dict.get("reasoning"), dict) and "content" in message_dict["reasoning"]:
            return

        possible_fields = [
            "reasoning_content",
            "thinking",
            "chain_of_thought",
        ]
        value: Any | None = None
        for field_name in possible_fields:
            if field_name in message_dict and message_dict[field_name] is not None:
                value = message_dict[field_name]
                break

        if value is None and isinstance(message_dict.get("reasoning"), str):
            value = message_dict["reasoning"]

        if value is not None:
            message_dict["reasoning"] = {"content": str(value)}

    def _normalize_openai_dict_response(self, response_dict: dict[str, Any]) -> dict[str, Any]:
        """Return a dict where non-standard reasoning fields are normalized.

        - For non-streaming: response.choices[*].message
        - For streaming: chunk.choices[*].delta
        """
        choices = response_dict.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict):
                    self._normalize_reasoning_on_message(message)

                delta = choice.get("delta") if isinstance(choice, dict) else None
                if isinstance(delta, dict):
                    self._normalize_reasoning_on_message(delta)

        return response_dict

    def _convert_completion_response(
        self, response: OpenAIChatCompletion | Stream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Convert an OpenAI completion response to an AnyLLM completion response."""
        if isinstance(response, OpenAIChatCompletion):
            if response.object != "chat.completion":
                # Force setting this here because it's a requirement Literal in the OpenAI API, but the Llama API has
                # a typo where they set it to "chat.completions". I filed a ticket with them to fix it. No harm in setting it here
                # Because this is the only accepted value anyways.
                logger.warning(
                    "API returned an unexpected object type: %s. Setting to 'chat.completion'.",
                    response.object,
                )
                response.object = "chat.completion"
            if not isinstance(response.created, int):
                # Sambanova returns a float instead of an int.
                logger.warning(
                    "API returned an unexpected created type: %s. Setting to int.",
                    type(response.created),
                )
                response.created = int(response.created)
            normalized = self._normalize_openai_dict_response(response.model_dump())
            return ChatCompletion.model_validate(normalized)

        def _convert_chunk(chunk: OpenAIChatCompletionChunk) -> ChatCompletionChunk:
            if not isinstance(chunk.created, int):
                logger.warning(
                    "API returned an unexpected created type: %s. Setting to int.",
                    type(chunk.created),
                )
                chunk.created = int(chunk.created)
            normalized_chunk = self._normalize_openai_dict_response(chunk.model_dump())
            return ChatCompletionChunk.model_validate(normalized_chunk)

        return (_convert_chunk(chunk) for chunk in response)

    def completion(self, params: CompletionParams, **kwargs: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Make the API call to OpenAI-compatible service."""
        client = OpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )

        if params.response_format:
            if params.stream:
                msg = "stream is not supported for response_format"
                raise ValueError(msg)

            response = client.chat.completions.parse(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
                **kwargs,
            )
        else:
            response = client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages"}),
                **kwargs,
            )
        return self._convert_completion_response(response)

    def responses(self, model: str, input_data: Any, **kwargs: Any) -> Response | Iterator[ResponseStreamEvent]:
        """Call OpenAI Responses API and normalize into ChatCompletion/Chunks.

        For now we only return a non-streaming ChatCompletion, or streaming chunks
        mapped to ChatCompletionChunk using the same converter.
        """
        client = OpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )
        response = client.responses.create(
            model=model,
            input=input_data,
            **kwargs,
        )
        if not isinstance(response, Response | Stream):
            msg = f"Responses API returned an unexpected type: {type(response)}"
            raise ValueError(msg)
        return response

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        # Classes that inherit from BaseOpenAIProvider may override SUPPORTS_EMBEDDING
        if not self.SUPPORTS_EMBEDDING:
            msg = "This provider does not support embeddings."
            raise NotImplementedError(msg)

        client = OpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )
        return client.embeddings.create(
            model=model,
            input=inputs,
            dimensions=kwargs.get("dimensions", NOT_GIVEN),
            **kwargs,
        )
