from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig, Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.model import Model
from any_llm.utils.instructor import _convert_instructor_response

MISSING_PACKAGES_ERROR = None
try:
    import cerebras.cloud.sdk as cerebras
    import instructor
    from cerebras.cloud.sdk.types.chat.chat_completion import ChatChunkResponse

    from .utils import (
        _convert_models_list,
        _convert_response,
        _create_openai_chunk_from_cerebras_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


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
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cerebras provider."""
        super().__init__(config)
        self.client = cerebras.Cerebras(api_key=config.api_key)
        self.instructor_client = instructor.from_cerebras(self.client)

    async def _stream_completion_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""

        client = cerebras.AsyncCerebras(api_key=self.config.api_key)

        if kwargs.get("response_format", None) is not None:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        cerebras_stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in cast("cerebras.AsyncStream[ChatCompletion]", cerebras_stream):
            if isinstance(chunk, ChatChunkResponse):
                yield _create_openai_chunk_from_cerebras_chunk(chunk)
            else:
                msg = f"Unsupported chunk type: {type(chunk)}"
                raise ValueError(msg)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Cerebras with instructor support for structured outputs."""

        # Cerebras does not support providing reasoning effort
        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                params.messages,
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
                **kwargs,
            )

        client = cerebras.AsyncCerebras(api_key=self.config.api_key)
        instructor_client = instructor.from_cerebras(client)

        if params.response_format:
            if not isinstance(params.response_format, type) or not issubclass(params.response_format, BaseModel):
                msg = "response_format must be a pydantic model"
                raise ValueError(msg)

            instructor_response = await instructor_client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                response_model=params.response_format,
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format"}),
                **kwargs,
            )

            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        response = await client.chat.completions.create(
            model=params.model_id,
            messages=params.messages,
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
            **kwargs,
        )

        if hasattr(response, "model_dump"):
            response_data = response.model_dump()
        else:
            msg = "Streaming responses are not supported in this context"
            raise ValueError(msg)

        return _convert_response(response_data)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        models_list = self.client.models.list(**kwargs)
        return _convert_models_list(models_list)
