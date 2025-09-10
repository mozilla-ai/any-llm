from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from any_llm.provider import Provider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CreateEmbeddingResponse,
)
from any_llm.types.model import Model
from any_llm.utils.instructor import _convert_instructor_response

MISSING_PACKAGES_ERROR = None
try:
    import instructor
    import together

    from .utils import (
        _convert_together_response_to_chat_completion,
        _create_openai_chunk_from_together_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from together.types import (
        ChatCompletionResponse,
    )
    from together.types.chat_completions import ChatCompletionChunk as TogetherChatCompletionChunk


class TogetherProvider(Provider):
    PROVIDER_NAME = "together"
    ENV_API_KEY_NAME = "TOGETHER_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://together.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Together API."""
        # Together does not support providing reasoning effort
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format"})
        if converted_params.get("reasoning_effort") == "auto":
            converted_params.pop("reasoning_effort")
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Together response to OpenAI format."""
        # We need the model parameter for conversion
        model = response.get("model", "together-model")
        return _convert_together_response_to_chat_completion(response, model)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Together chunk response to OpenAI format."""
        return _create_openai_chunk_from_together_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Together."""
        msg = "Together does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Together embedding response to OpenAI format."""
        msg = "Together does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Together list models response to OpenAI format."""
        msg = "Together does not support listing models"
        raise NotImplementedError(msg)

    async def _stream_completion_async(
        self,
        client: "together.AsyncTogether",
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        from typing import cast

        response = cast(
            "AsyncIterator[TogetherChatCompletionChunk]",
            await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            ),
        )
        async for chunk in response:
            yield self._convert_completion_chunk_response(chunk)

    def _stream_completion(
        self,
        client: "together.Together",
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        from typing import cast

        response = cast(
            "Iterator[TogetherChatCompletionChunk]",
            client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            ),
        )
        for chunk in response:
            yield self._convert_completion_chunk_response(chunk)

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Make the API call to Together AI with instructor support for structured outputs."""
        client = together.Together(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            **(self.config.client_args if self.config.client_args else {}),
        )

        if params.response_format:
            instructor_client = instructor.patch(client, mode=instructor.Mode.JSON)  # type: ignore [call-overload]

            instructor_response = instructor_client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                response_model=params.response_format,
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format"}),
                **kwargs,
            )

            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion(
                client,
                params.model_id,
                params.messages,
                **completion_kwargs,
            )

        response = cast(
            "ChatCompletionResponse",
            client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **completion_kwargs,
            ),
        )

        return self._convert_completion_response(response.model_dump())

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Make the API call to Together AI with instructor support for structured outputs."""
        client = together.AsyncTogether(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            **(self.config.client_args if self.config.client_args else {}),
        )

        if params.response_format:
            instructor_client = instructor.patch(client, mode=instructor.Mode.JSON)  # type: ignore [call-overload]

            instructor_response = await instructor_client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                response_model=params.response_format,
                **params.model_dump(
                    exclude_none=True,
                    exclude={"model_id", "messages", "reasoning_effort", "response_format"},
                ),
                **kwargs,
            )

            return _convert_instructor_response(instructor_response, params.model_id, self.PROVIDER_NAME)

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                client,
                params.model_id,
                params.messages,
                **completion_kwargs,
            )

        response = cast(
            "ChatCompletionResponse",
            await client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **completion_kwargs,
            ),
        )

        return self._convert_completion_response(response.model_dump())
