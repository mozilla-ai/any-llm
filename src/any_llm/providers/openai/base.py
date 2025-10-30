import asyncio
from collections.abc import AsyncIterator, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN, Omit
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

from any_llm.any_llm import AnyLLM
from any_llm.logging import logger
from any_llm.providers.openai.utils import _convert_chat_completion, _normalize_openai_dict_response
from any_llm.types.batch import Batch
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model
from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent


class BaseOpenAIProvider(AnyLLM):
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
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    PACKAGES_INSTALLED = True

    _DEFAULT_REASONING_EFFORT: Literal["minimal", "low", "medium", "high", "auto"] | None = None

    client: AsyncOpenAI

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenAI API."""
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert OpenAI response to OpenAI format (passthrough)."""
        if isinstance(response, OpenAIChatCompletion):
            return _convert_chat_completion(response)
        # If it's already our ChatCompletion type, return it
        if isinstance(response, ChatCompletion):
            return response
        # Otherwise, validate it as our type
        return ChatCompletion.model_validate(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert OpenAI chunk response to OpenAI format (passthrough)."""
        if isinstance(response, OpenAIChatCompletionChunk):
            if not isinstance(response.created, int):
                logger.warning(
                    "API returned an unexpected created type: %s. Setting to int.",
                    type(response.created),
                )
                response.created = int(response.created)
            normalized_chunk = _normalize_openai_dict_response(response.model_dump())
            # Some APIs (i.e. Perplexity) return `chat.completion` without the chunk
            # We can hardcode it as openai expects a literal
            normalized_chunk["object"] = "chat.completion.chunk"
            return ChatCompletionChunk.model_validate(normalized_chunk)
        # If it's already our ChatCompletionChunk type, return it
        if isinstance(response, ChatCompletionChunk):
            return response
        # Otherwise, validate it as our type
        return ChatCompletionChunk.model_validate(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for OpenAI API."""
        converted_params = {"input": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert OpenAI embedding response to OpenAI format (passthrough)."""
        if isinstance(response, CreateEmbeddingResponse):
            return response
        return CreateEmbeddingResponse.model_validate(response)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert OpenAI list models response to OpenAI format (passthrough)."""
        if hasattr(response, "data"):
            # Validate each model in the data
            return [Model.model_validate(model) if not isinstance(model, Model) else model for model in response.data]
        # If it's already a sequence of our Model type, return it
        if isinstance(response, (list, tuple)) and all(isinstance(item, Model) for item in response):
            return response
        # Otherwise, validate each item
        return [Model.model_validate(item) if not isinstance(item, Model) else item for item in response]

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncOpenAI(
            base_url=api_base or self.API_BASE,
            api_key=api_key,
            **kwargs,
        )

    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert an OpenAI completion response to an AnyLLM completion response."""
        if isinstance(response, OpenAIChatCompletion):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                yield self._convert_completion_chunk_response(chunk)

        return chunk_iterator()

    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.reasoning_effort == "auto":
            params.reasoning_effort = self._DEFAULT_REASONING_EFFORT

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.response_format:
            if params.stream:
                msg = "stream is not supported for response_format"
                raise ValueError(msg)
            completion_kwargs.pop("stream", None)
            response = await self.client.chat.completions.parse(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **completion_kwargs,
            )
        else:
            response = await self.client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **completion_kwargs,
            )
        return self._convert_completion_response_async(response)

    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Call OpenAI Responses API"""
        response = await self.client.responses.create(**params.model_dump(exclude_none=True), **kwargs)

        if not isinstance(response, Response | AsyncStream):
            msg = f"Responses API returned an unexpected type: {type(response)}"
            raise ValueError(msg)
        return response

    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        # Classes that inherit from BaseOpenAIProvider may override SUPPORTS_EMBEDDING
        if not self.SUPPORTS_EMBEDDING:
            msg = "This provider does not support embeddings."
            raise NotImplementedError(msg)

        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        return self._convert_embedding_response(
            await self.client.embeddings.create(
                model=model,
                dimensions=kwargs.get("dimensions", NOT_GIVEN),
                **embedding_kwargs,
            )
        )

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        if not self.SUPPORTS_LIST_MODELS:
            message = f"{self.PROVIDER_NAME} does not support listing models."
            raise NotImplementedError(message)
        response = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(response)

    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Create a batch job using the OpenAI Batch API.

        This method automatically uploads the file before creating the batch.
        """
        if not self.SUPPORTS_BATCH:
            message = f"{self.PROVIDER_NAME} does not support batch completions."
            raise NotImplementedError(message)

        file_path = Path(input_file_path)
        file_content = await asyncio.to_thread(file_path.read_bytes)

        file_obj = BytesIO(file_content)
        file_obj.name = file_path.name

        uploaded_file = await self.client.files.create(file=file_obj, purpose="batch")

        valid_endpoint = cast(
            "Literal['/v1/chat/completions', '/v1/embeddings', '/v1/completions']",
            endpoint,
        )
        valid_completion_window = cast("Literal['24h']", completion_window)

        return await self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint=valid_endpoint,
            completion_window=valid_completion_window,
            metadata=metadata or {},
            **kwargs,
        )

    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch job using the OpenAI Batch API."""
        if not self.SUPPORTS_BATCH:
            message = f"{self.PROVIDER_NAME} does not support batch completions."
            raise NotImplementedError(message)

        return await self.client.batches.retrieve(batch_id, **kwargs)

    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Cancel a batch job using the OpenAI Batch API."""
        if not self.SUPPORTS_BATCH:
            message = f"{self.PROVIDER_NAME} does not support batch completions."
            raise NotImplementedError(message)

        return await self.client.batches.cancel(batch_id, **kwargs)

    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batch jobs using the OpenAI Batch API."""
        if not self.SUPPORTS_BATCH:
            message = f"{self.PROVIDER_NAME} does not support batch completions."
            raise NotImplementedError(message)

        after_param: str | Omit = after if after is not None else Omit()
        limit_param: int | Omit = limit if limit is not None else Omit()

        response = await self.client.batches.list(
            after=after_param,
            limit=limit_param,
            **kwargs,
        )
        return response.data
