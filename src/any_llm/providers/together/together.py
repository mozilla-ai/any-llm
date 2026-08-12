from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import BatchNotCompleteError, ProviderError, UnsupportedParameterError
from any_llm.logging import logger
from any_llm.types.batch import BatchResult, BatchResultError, BatchResultItem
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

MISSING_PACKAGES_ERROR = None
try:
    import together

    from .utils import (
        _convert_batch_job_to_openai,
        _convert_models_list,
        _convert_together_response_to_chat_completion,
        _create_openai_chunk_from_together_chunk,
        _create_openai_embedding_response_from_together,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from together.types import (
        ChatCompletionChunk as TogetherChatCompletionChunk,
    )
    from together.types import (
        ChatCompletionResponse as TogetherChatCompletion,
    )

    from any_llm.types.batch import Batch
    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionChunk,
        CompletionParams,
        CreateEmbeddingResponse,
    )
    from any_llm.types.model import Model

DEFAULT_SCHEMA_NAME = "response_schema"
_SCHEMA_METADATA_KEYS = ("name", "description", "strict")


def _convert_response_format(response_format: dict[str, Any]) -> dict[str, Any]:
    """Convert a dict response_format to one of the two shapes Together documents.

    Together accepts `{"type": "json_object"}` or
    `{"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}`, and rejects a
    `json_schema` payload that has no `name`. Any other dict is passed through untouched.

    See https://docs.together.ai/reference/chat-completions#body-response-format
    """
    if response_format.get("type") != "json_schema":
        return response_format

    json_schema: dict[str, Any] = dict(response_format.get("json_schema") or {})
    if "schema" not in json_schema:
        # Tolerate a bare schema supplied under "json_schema" or next to it at the top level,
        # keeping any name/description/strict the caller left beside it.
        metadata = {key: response_format[key] for key in _SCHEMA_METADATA_KEYS if key in response_format}
        metadata.update({key: json_schema.pop(key) for key in _SCHEMA_METADATA_KEYS if key in json_schema})
        json_schema = {**metadata, "schema": json_schema or response_format.get("schema") or {}}
    json_schema.setdefault("name", DEFAULT_SCHEMA_NAME)
    return {"type": "json_schema", "json_schema": json_schema}


class TogetherProvider(AnyLLM):
    PROVIDER_NAME = "together"
    ENV_API_KEY_NAME = "TOGETHER_API_KEY"
    ENV_API_BASE_NAME = "TOGETHER_API_BASE"
    PROVIDER_DOCUMENTATION_URL = "https://together.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True
    SUPPORTS_RERANK = False

    # The Together SDK accepts a per-request `timeout` on its client calls, so it forwards unchanged.
    TIMEOUT_SUPPORT = "native"

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: together.AsyncTogether

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Together API."""
        converted_params = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "stream_options"}
        )
        if converted_params.get("reasoning_effort") in ("auto", "none"):
            converted_params.pop("reasoning_effort")
        if params.response_format is not None:
            if is_structured_output_type(params.response_format):
                converted_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": params.response_format.__name__,
                        "schema": get_json_schema(params.response_format),
                    },
                }
            elif isinstance(params.response_format, dict):
                converted_params["response_format"] = _convert_response_format(params.response_format)

        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Together response to OpenAI format."""
        # We need the model parameter for conversion
        model = response.get("model", "together-model")
        return _convert_together_response_to_chat_completion(response, model)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Together chunk response to OpenAI format."""
        return _create_openai_chunk_from_together_chunk(response)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Together."""
        converted_params = {"input": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Together embedding response to OpenAI format."""
        return _create_openai_embedding_response_from_together(response)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Together list models response to OpenAI format."""
        return _convert_models_list(response)

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = together.AsyncTogether(
            api_key=api_key,
            base_url=api_base,
            **kwargs,
        )

    async def _stream_completion_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion with reasoning support."""
        from typing import cast

        response = cast(
            "AsyncIterator[TogetherChatCompletionChunk]",
            await self.client.chat.completions.create(
                model=model,
                messages=cast("Any", messages),
                **kwargs,
            ),
        )

        async for chunk in response:
            yield self._convert_completion_chunk_response(chunk)

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        completion_kwargs = self._convert_completion_params(params, **kwargs)
        # Together API rejects empty tool_calls arrays
        cleaned_messages = [{k: v for k, v in msg.items() if k != "tool_calls" or v} for msg in params.messages]

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                cleaned_messages,
                **completion_kwargs,
            )

        response = cast(
            "TogetherChatCompletion",
            await self.client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", cleaned_messages),
                **completion_kwargs,
            ),
        )

        return self._convert_completion_response(response.model_dump())

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        response = await self.client.embeddings.create(model=model, **embedding_kwargs)
        return self._convert_embedding_response(response)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(models_list)

    @override
    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Create a batch job using the Together Batch API.

        This method uploads the input file before creating the batch.

        Together does not accept batch metadata, so a ``metadata`` argument is ignored.

        Optional keyword arguments:
            model_id: Model to use for all requests in the batch.
            priority: Scheduling priority for the batch.
        """
        if metadata:
            logger.warning("Together batch API does not support metadata, ignoring it.")

        # Together validates uploads against the fine-tuning format unless the check is
        # disabled, and batch JSONL does not satisfy it.
        uploaded_file = await self.client.files.upload(
            file=Path(input_file_path),
            purpose="batch-api",
            check=False,
        )

        created = await self.client.batches.create(
            endpoint=cast("Any", endpoint),
            input_file_id=uploaded_file.id,
            completion_window=completion_window,
            **kwargs,
        )

        if created.job is None:
            msg = f"Together did not return a batch job. Warning: {created.warning or 'none'}"
            raise ProviderError(msg, provider_name=self.PROVIDER_NAME)

        if created.warning:
            logger.warning("Together returned a warning for the created batch: %s", created.warning)

        return _convert_batch_job_to_openai(created.job)

    @override
    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch job using the Together Batch API."""
        batch_job = await self.client.batches.retrieve(batch_id, **kwargs)
        return _convert_batch_job_to_openai(batch_job)

    @override
    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Cancel a batch job using the Together Batch API."""
        batch_job = await self.client.batches.cancel(batch_id, **kwargs)
        return _convert_batch_job_to_openai(batch_job)

    @override
    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batch jobs using the Together Batch API.

        Together returns every batch in one unpaginated response.

        Args:
            after: Not supported for Together. Raises UnsupportedParameterError if provided.
            limit: Applied client-side to the full response, so it truncates rather than pages.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Sequence of Batch objects, newest first as returned by Together.

        Raises:
            UnsupportedParameterError: If `after` is provided.
        """
        if after is not None:
            msg = "after"
            raise UnsupportedParameterError(
                msg,
                self.PROVIDER_NAME,
                "Together's batch listing is not paginated, so there is no cursor to resume from.",
            )

        batch_jobs = await self.client.batches.list(**kwargs)
        if not batch_jobs:
            return []

        batches = [_convert_batch_job_to_openai(batch_job) for batch_job in batch_jobs]
        return batches[:limit] if limit is not None else batches

    @override
    async def _aretrieve_batch_results(self, batch_id: str, **kwargs: Any) -> BatchResult:
        """Retrieve the results of a completed batch job using the Together Batch API."""
        batch_job = await self.client.batches.retrieve(batch_id, **kwargs)
        converted = _convert_batch_job_to_openai(batch_job)
        if converted.status != "completed":
            raise BatchNotCompleteError(
                batch_id=batch_id,
                status=converted.status or "unknown",
                provider_name=self.PROVIDER_NAME,
            )

        if not batch_job.output_file_id:
            return BatchResult(results=[])

        content = await self.client.files.content(batch_job.output_file_id)
        text = (await content.read()).decode("utf-8")

        results: list[BatchResultItem] = []
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            item = BatchResultItem(custom_id=entry["custom_id"])
            if entry.get("response") and entry["response"].get("status_code") == 200:
                item.result = self._convert_completion_response(entry["response"]["body"])
            elif entry.get("error"):
                item.error = BatchResultError(
                    code=entry["error"].get("code", "unknown"),
                    message=entry["error"].get("message", "Unknown error"),
                )
            else:
                item.error = BatchResultError(code="unknown", message="Unexpected response format")
            results.append(item)
        return BatchResult(results=results)
