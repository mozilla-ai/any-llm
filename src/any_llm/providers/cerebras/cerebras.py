from collections.abc import Iterator
from typing import Any, cast

try:
    import cerebras.cloud.sdk as cerebras
    import instructor
    from cerebras.cloud.sdk.types.chat.chat_completion import ChatChunkResponse

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig, Provider
from any_llm.providers.cerebras.utils import (
    _convert_response,
    _create_openai_chunk_from_cerebras_chunk,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.utils.instructor import _convert_instructor_response


class CerebrasProvider(Provider):
    """Cerebras Provider using the official Cerebras SDK with instructor support for structured outputs."""

    PROVIDER_NAME = "cerebras"
    ENV_API_KEY_NAME = "CEREBRAS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.cerebras.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cerebras provider."""
        super().__init__(config)
        self.client = cerebras.Cerebras(api_key=config.api_key)
        self.instructor_client = instructor.from_cerebras(self.client)

    def _stream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        if kwargs.get("response_format", None) is not None:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        cerebras_stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        for chunk in cerebras_stream:
            if isinstance(chunk, ChatChunkResponse):
                yield _create_openai_chunk_from_cerebras_chunk(chunk)
            else:
                msg = f"Unsupported chunk type: {type(chunk)}"
                raise ValueError(msg)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Cerebras with instructor support for structured outputs."""
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            instructor_response = self.instructor_client.chat.completions.create(
                model=model,
                messages=cast("Any", messages),
                response_model=response_format,
                **kwargs,
            )

            return _convert_instructor_response(instructor_response, model, self.PROVIDER_NAME)

        if kwargs.get("stream", False):
            kwargs.pop("stream")
            return self._stream_completion(model, messages, **kwargs)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

        if hasattr(response, "model_dump"):
            response_data = response.model_dump()
        else:
            msg = "Streaming responses are not supported in this context"
            raise ValueError(msg)

        return _convert_response(response_data)
