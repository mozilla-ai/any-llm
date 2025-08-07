import os
from typing import Any, Iterator
from abc import ABC

from openai import OpenAI
from any_llm.types.completion import ChatCompletion, CreateEmbeddingResponse
from openai._streaming import Stream
from any_llm.types.completion import ChatCompletionChunk
from openai._types import NOT_GIVEN
from any_llm.provider import Provider

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from any_llm.logging import logger


class BaseOpenAIProvider(Provider, ABC):
    """
    Base provider for OpenAI-compatible services.

    This class provides a common foundation for providers that use OpenAI-compatible APIs.
    Subclasses only need to override configuration defaults and client initialization
    if needed.
    """

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
    SUPPORTS_EMBEDDING = True

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Default is that all kwargs are supported."""
        pass

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
                    f"API returned an unexpected object type: {response.object}. Setting to 'chat.completion'."
                )
                response.object = "chat.completion"
            if not isinstance(response.created, int):
                # Sambanova returns a float instead of an int.
                logger.warning(f"API returned an unexpected created type: {type(response.created)}. Setting to int.")
                response.created = int(response.created)
            return ChatCompletion.model_validate(response.model_dump())
        else:
            # Handle streaming response - return a generator
            def _convert_chunk(chunk: OpenAIChatCompletionChunk) -> ChatCompletionChunk:
                if not isinstance(chunk.created, int):
                    # Sambanova returns a float instead of an int.
                    logger.warning(f"API returned an unexpected created type: {type(chunk.created)}. Setting to int.")
                    chunk.created = int(chunk.created)
                return ChatCompletionChunk.model_validate(chunk.model_dump())

            return (_convert_chunk(chunk) for chunk in response)

    def _make_api_call(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Make the API call to OpenAI-compatible service."""
        client = OpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )

        if "response_format" in kwargs:
            response = client.chat.completions.parse(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
        else:
            response = client.chat.completions.create(  # type: ignore[assignment]
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
        return self._convert_completion_response(response)

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        # Classes that inherit from BaseOpenAIProvider may override SUPPORTS_EMBEDDING
        if not self.SUPPORTS_EMBEDDING:
            raise NotImplementedError("This provider does not support embeddings.")

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
