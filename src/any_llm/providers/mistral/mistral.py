from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_llm.any_llm import AnyLLM

MISSING_PACKAGES_ERROR = None
try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model
    from mistralai.models.responseformat import ResponseFormat

    from .utils import (
        _convert_batch_job_to_openai,
        _convert_batch_jobs_list,
        _convert_models_list,
        _create_mistral_completion_from_response,
        _create_openai_chunk_from_mistral_chunk,
        _create_openai_embedding_response_from_mistral,
        _parse_completion_window_to_hours,
        _patch_messages,
        _validate_batch_file_models,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from mistralai import Mistral  # noqa: TC004
    from mistralai.models.embeddingresponse import EmbeddingResponse

    from any_llm.types.batch import Batch
    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class MistralProvider(AnyLLM):
    """Mistral Provider using the new response conversion utilities."""

    PROVIDER_NAME = "mistral"
    ENV_API_KEY_NAME = "MISTRAL_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.mistral.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: Mistral

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Mistral API."""
        # Mistral does not support providing reasoning effort
        converted_params = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "stream", "user"}
        )
        converted_params["messages"] = _patch_messages(params.messages)

        if params.response_format is not None:
            if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
                converted_params["response_format"] = response_format_from_pydantic_model(params.response_format)
            elif isinstance(params.response_format, dict):
                converted_params["response_format"] = ResponseFormat.model_validate(params.response_format)

        if converted_params.get("reasoning_effort") in ("auto", "none"):
            converted_params.pop("reasoning_effort")

        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Mistral response to OpenAI format."""
        # We need the model parameter for conversion
        model = getattr(response, "model", "mistral-model")
        return _create_mistral_completion_from_response(response_data=response, model=model)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Mistral chunk response to OpenAI format."""
        return _create_openai_chunk_from_mistral_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Mistral."""
        converted_params = {"inputs": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Mistral embedding response to OpenAI format."""
        return _create_openai_embedding_response_from_mistral(response)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Mistral list models response to OpenAI format."""
        return _convert_models_list(response)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = Mistral(
            api_key=api_key,
            server_url=api_base,
            **kwargs,
        )

    async def _stream_completion_async(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        mistral_stream = await self.client.chat.stream_async(model=model, messages=messages, **kwargs)  # type: ignore[arg-type]

        async for event in mistral_stream:
            yield self._convert_completion_chunk_response(event)

    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                **completion_kwargs,
            )

        response = await self.client.chat.complete_async(
            model=params.model_id,
            **completion_kwargs,
        )

        return self._convert_completion_response(response)

    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        result: EmbeddingResponse = await self.client.embeddings.create_async(
            model=model,
            **embedding_kwargs,
        )
        return self._convert_embedding_response(result)

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.models.list_async(**kwargs)
        return self._convert_list_models_response(models_list)

    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Create a batch job using the Mistral Batch API."""
        file_path = Path(input_file_path)
        file_content = await asyncio.to_thread(file_path.read_bytes)

        uploaded_file = await self.client.files.upload_async(
            file={"file_name": file_path.name, "content": file_content},
            purpose="batch",
        )

        timeout_hours = _parse_completion_window_to_hours(completion_window)

        model = kwargs.pop("model", None)
        file_text = file_content.decode("utf-8")

        # Mistral requires model at job level and all requests must use the same model.
        # Validate consistency and extract model from JSONL if not provided.
        validated_model = _validate_batch_file_models(file_text)

        if model is None:
            if validated_model is None:
                msg = (
                    "Model not found in JSONL body and not provided via 'model' kwarg. "
                    "Mistral batch API requires model at job level."
                )
                raise ValueError(msg)
            model = validated_model
        elif validated_model is not None and model != validated_model:
            msg = (
                f"Model mismatch: 'model' kwarg is '{model}' but batch file "
                f"contains requests for model '{validated_model}'"
            )
            raise ValueError(msg)

        batch_job = await self.client.batch.jobs.create_async(
            input_files=[uploaded_file.id],
            endpoint=endpoint,
            model=model,
            metadata=metadata,
            timeout_hours=timeout_hours,
            **kwargs,
        )

        return _convert_batch_job_to_openai(batch_job)

    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch job using the Mistral Batch API."""
        batch_job = await self.client.batch.jobs.get_async(job_id=batch_id, **kwargs)
        return _convert_batch_job_to_openai(batch_job)

    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Cancel a batch job using the Mistral Batch API."""
        batch_job = await self.client.batch.jobs.cancel_async(job_id=batch_id, **kwargs)
        return _convert_batch_job_to_openai(batch_job)

    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batch jobs using the Mistral Batch API.

        Note: Mistral uses page-based pagination, not cursor-based.
        Use `page` kwarg (0-indexed) instead of `after`, and `limit` for page size.

        Args:
            after: Not supported for Mistral. Raises ValueError if provided.
            limit: Page size (number of results per page). Defaults to 100.
            **kwargs: Additional arguments, including `page` (0-indexed page number).

        Returns:
            Sequence of Batch objects for the requested page.

        Raises:
            ValueError: If `after` parameter is provided.
        """
        if after is not None:
            msg = (
                "Mistral batch API uses page-based pagination, not cursor-based. "
                "The 'after' parameter is not supported. "
                "Use 'page' kwarg for pagination and 'limit' for page size. "
            )
            raise ValueError(msg)

        page = kwargs.pop("page", 0)
        page_size = limit if limit is not None else 100

        batch_jobs = await self.client.batch.jobs.list_async(
            page=page,
            page_size=page_size,
            **kwargs,
        )

        return _convert_batch_jobs_list(batch_jobs)
