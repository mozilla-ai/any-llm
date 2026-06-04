from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, cast

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import BatchNotCompleteError, InvalidRequestError, UnsupportedParameterError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Choice,
    CompletionParams,
    CompletionUsage,
    CreateEmbeddingResponse,
    Function,
    Reasoning,
)
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

MISSING_PACKAGES_ERROR = None
try:
    from google.genai import types

    from .utils import (
        _convert_google_batch_job_to_openai_batch,
        _convert_google_batch_output_to_result,
        _convert_messages,
        _convert_models_list,
        _convert_openai_request_to_inlined_request,
        _convert_response_to_response_dict,
        _convert_tool_choice,
        _convert_tool_spec,
        _create_openai_chunk_from_google_chunk,
        _create_openai_embedding_response_from_google,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from google import genai
    from openai.types.chat.chat_completion_message_custom_tool_call import (
        ChatCompletionMessageCustomToolCall,
    )
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
    )

    from any_llm.types.batch import Batch, BatchResult
    from any_llm.types.model import Model

    ChatCompletionMessageToolCallType = (
        OpenAIChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall
    )

REASONING_EFFORT_TO_THINKING_BUDGETS = {
    "minimal": 256,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
    "xhigh": 32768,
    "max": 32768,
}
_SUPPORTED_BATCH_ENDPOINTS = frozenset({"/v1/chat/completions"})


class GoogleProvider(AnyLLM):
    """Base Google Provider class with common functionality for Gemini and Vertex AI."""

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True
    SUPPORTS_RERANK = False

    BUILT_IN_TOOLS: ClassVar[list[Any] | None] = [types.Tool]

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: genai.Client

    @staticmethod
    def _merge_timeout_into_http_options(timeout: float, kwargs: dict[str, Any]) -> None:
        """Apply a timeout (seconds) to kwargs["http_options"] as milliseconds.

        Creates http_options if missing and only sets the timeout if one is not already
        configured.
        """
        timeout_ms = int(timeout * 1000)
        http_options = kwargs.get("http_options")

        if http_options is None:
            kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms)
            return

        if isinstance(http_options, dict):
            http_options.setdefault("timeout", timeout_ms)
            return

        if isinstance(http_options, types.HttpOptions) and http_options.timeout is None:
            http_options.timeout = timeout_ms

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Google API."""
        provider_name = kwargs.pop("provider_name")

        # Ensure timeout is correctly configured if present.
        if (timeout := kwargs.pop("timeout", None)) is not None:
            GoogleProvider._merge_timeout_into_http_options(timeout, kwargs)

        if params.parallel_tool_calls is not None:
            error_message = "parallel_tool_calls"
            raise UnsupportedParameterError(error_message, provider_name)
        if params.stream and params.response_format is not None:
            error_message = "stream and response_format"
            raise UnsupportedParameterError(error_message, provider_name)

        if params.frequency_penalty is not None:
            kwargs["frequency_penalty"] = params.frequency_penalty
        if params.max_tokens is not None:
            kwargs["max_output_tokens"] = params.max_tokens
        if params.presence_penalty is not None:
            kwargs["presence_penalty"] = params.presence_penalty
        if params.reasoning_effort != "auto":
            if params.reasoning_effort is None or params.reasoning_effort == "none":
                kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=False)
            else:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True, thinking_budget=REASONING_EFFORT_TO_THINKING_BUDGETS[params.reasoning_effort]
                )
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.tools is not None:
            kwargs["tools"] = _convert_tool_spec(params.tools)
        if isinstance(params.tool_choice, str):
            kwargs["tool_config"] = _convert_tool_choice(params.tool_choice)
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.stop is not None:
            if isinstance(params.stop, str):
                kwargs["stop_sequences"] = [params.stop]
            else:
                kwargs["stop_sequences"] = params.stop

        response_format = params.response_format
        if is_structured_output_type(response_format):
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = get_json_schema(response_format)
        elif isinstance(response_format, dict):
            response_type = response_format.get("type")
            if response_type == "json_schema":
                kwargs["response_mime_type"] = "application/json"
                kwargs["response_schema"] = response_format["json_schema"]["schema"]
            elif response_type == "json_object":
                kwargs["response_mime_type"] = "application/json"
            elif response_type == "text":
                pass
            else:
                msg = f"Unsupported response_format type: {response_type}"
                raise ValueError(msg)

        formatted_messages, system_instruction = _convert_messages(params.messages, provider_name=provider_name)
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        result_kwargs: dict[str, Any] = {
            "config": types.GenerateContentConfig(**kwargs),
            "contents": formatted_messages,
            "model": params.model_id,
        }

        return result_kwargs

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Google response data to OpenAI ChatCompletion format."""
        # Expect response to be a tuple of (response_dict, model_id)
        response_dict, model_id = response
        choices_out: list[Choice] = []
        for i, choice_item in enumerate(response_dict.get("choices", [])):
            message_dict: dict[str, Any] = choice_item.get("message", {})
            tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] | None = None
            if message_dict.get("tool_calls"):
                tool_calls_list: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
                for tc in message_dict["tool_calls"]:
                    tool_calls_list.append(
                        ChatCompletionMessageFunctionToolCall(
                            id=tc.get("id"),
                            type="function",
                            function=Function(
                                name=tc["function"]["name"],
                                arguments=tc["function"]["arguments"],
                            ),
                            extra_content=tc.get("extra_content"),
                        )
                    )
                tool_calls = tool_calls_list

            reasoning_content = message_dict.get("reasoning")
            message = ChatCompletionMessage(
                role="assistant",
                content=message_dict.get("content"),
                tool_calls=cast("list[ChatCompletionMessageToolCallType] | None", tool_calls),
                reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
            )
            from typing import Literal

            choices_out.append(
                Choice(
                    index=i,
                    finish_reason=cast(
                        "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
                        choice_item.get("finish_reason", "stop"),
                    ),
                    message=message,
                )
            )

        usage_dict = response_dict.get("usage", {})
        usage = CompletionUsage(
            prompt_tokens=usage_dict.get("prompt_tokens", 0),
            completion_tokens=usage_dict.get("completion_tokens", 0),
            total_tokens=usage_dict.get("total_tokens", 0),
            prompt_tokens_details=usage_dict.get("prompt_tokens_details"),
        )

        return ChatCompletion(
            id=response_dict.get("id", ""),
            model=model_id,
            created=response_dict.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Google chunk response to OpenAI format."""
        return _create_openai_chunk_from_google_chunk(response)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Google API."""
        converted_params = {"contents": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Google embedding response to OpenAI format."""
        # We need the model parameter for conversion
        model = response.get("model", "google-model")
        return _create_openai_embedding_response_from_google(model, response["result"])

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Google list models response to OpenAI format."""
        return _convert_models_list(response)

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        result = await self.client.aio.models.embed_content(
            model=model,
            **embedding_kwargs,
        )

        response_data = {"model": model, "result": result}
        return self._convert_embedding_response(response_data)

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        kwargs["provider_name"] = self.PROVIDER_NAME
        converted_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            response_stream = await self.client.aio.models.generate_content_stream(**converted_kwargs)

            async def _stream() -> AsyncIterator[ChatCompletionChunk]:
                async for chunk in response_stream:
                    yield self._convert_completion_chunk_response(chunk)

            return _stream()

        response: types.GenerateContentResponse = await self.client.aio.models.generate_content(**converted_kwargs)

        response_dict = _convert_response_to_response_dict(response)
        return self._convert_completion_response((response_dict, params.model_id))

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.aio.models.list(**kwargs)
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
        """Create a batch job using the Google GenAI Batch API.

        Reads a local JSONL file, converts each request from OpenAI format to
        Google ``InlinedRequest`` objects, and submits them as a batch.

        Optional keyword arguments:
            dest: GCS or BigQuery URI for output (e.g. ``gs://bucket/output``).
            display_name: Human-readable name for the batch job.
            model: Model to use for all requests (overrides per-request model).
        """
        import asyncio

        if endpoint not in _SUPPORTED_BATCH_ENDPOINTS:
            msg = f"Google batch API only supports endpoints: {sorted(_SUPPORTED_BATCH_ENDPOINTS)}, got: '{endpoint}'"
            raise InvalidRequestError(msg, provider_name=self.PROVIDER_NAME)

        dest: str | None = kwargs.pop("dest", None)
        display_name: str | None = kwargs.pop("display_name", None)
        model_override: str | None = kwargs.pop("model", None)

        file_content = await asyncio.to_thread(self._read_file, input_file_path)

        inlined_requests: list[types.InlinedRequest] = []
        first_model = model_override or ""
        for line in file_content.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            req = _convert_openai_request_to_inlined_request(entry, provider_name=self.PROVIDER_NAME)
            if model_override:
                req = types.InlinedRequest(
                    model=model_override,
                    contents=req.contents,
                    config=req.config,
                    metadata=req.metadata,
                )
            inlined_requests.append(req)
            if not first_model and req.model:
                first_model = req.model

        if not first_model:
            msg = "No model specified: provide a 'model' kwarg or include 'model' in the JSONL request bodies."
            raise ValueError(msg)

        config_kwargs: dict[str, Any] = {}
        if display_name:
            config_kwargs["display_name"] = display_name
        if dest:
            config_kwargs["dest"] = dest

        config = types.CreateBatchJobConfig(**config_kwargs) if config_kwargs else None

        result = await self.client.aio.batches.create(
            model=first_model,
            src=inlined_requests,
            config=config,
        )
        return _convert_google_batch_job_to_openai_batch(result)

    @staticmethod
    def _read_file(path: str) -> str:
        """Read file content synchronously (called via ``asyncio.to_thread``)."""
        from pathlib import Path

        return Path(path).read_text()

    @override
    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch job from the Google GenAI Batch API."""
        result = await self.client.aio.batches.get(name=batch_id)
        return _convert_google_batch_job_to_openai_batch(result)

    @override
    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Cancel a batch job using the Google GenAI Batch API."""
        await self.client.aio.batches.cancel(name=batch_id)
        result = await self.client.aio.batches.get(name=batch_id)
        return _convert_google_batch_job_to_openai_batch(result)

    @override
    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batch jobs using the Google GenAI Batch API."""
        config_kwargs: dict[str, Any] = {}
        if limit:
            config_kwargs["page_size"] = limit
        if after:
            config_kwargs["page_token"] = after

        config = types.ListBatchJobsConfig(**config_kwargs) if config_kwargs else None
        pager = await self.client.aio.batches.list(config=config)
        batches: list[Batch] = []
        async for job in pager:
            batches.append(_convert_google_batch_job_to_openai_batch(job))
        return batches

    @override
    async def _aretrieve_batch_results(self, batch_id: str, **kwargs: Any) -> BatchResult:
        """Retrieve the results of a completed batch job.

        Reads the output JSONL from the GCS location specified in the batch
        job's ``output_info``.  Requires ``google-cloud-storage`` to be
        installed.
        """
        import asyncio

        job = await self.client.aio.batches.get(name=batch_id)

        state_str = job.state.value if job.state else "JOB_STATE_UNSPECIFIED"
        if state_str not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            openai_batch = _convert_google_batch_job_to_openai_batch(job)
            raise BatchNotCompleteError(
                batch_id=batch_id,
                status=openai_batch.status or "unknown",
                provider_name=self.PROVIDER_NAME,
            )

        gcs_dir = job.output_info.gcs_output_directory if job.output_info else None
        if not gcs_dir:
            msg = (
                f"Batch '{batch_id}' has no GCS output directory. "
                "Ensure a destination was configured when creating the batch."
            )
            raise ValueError(msg)

        output_lines = await asyncio.to_thread(self._read_gcs_output, gcs_dir)
        return _convert_google_batch_output_to_result(output_lines)

    @staticmethod
    def _read_gcs_output(gcs_dir: str) -> list[str]:
        """Read all JSONL output files from a GCS directory.

        Requires the ``google-cloud-storage`` package.
        """
        try:
            from google.cloud import storage
        except ImportError:
            msg = (
                "google-cloud-storage is required to retrieve batch results from GCS. "
                "Install it with: pip install google-cloud-storage"
            )
            raise ImportError(msg)  # noqa: B904

        if not gcs_dir.startswith("gs://"):
            msg = f"Expected a GCS URI starting with 'gs://', got: {gcs_dir}"
            raise ValueError(msg)

        without_scheme = gcs_dir[len("gs://") :]
        slash_idx = without_scheme.find("/")
        if slash_idx == -1:
            bucket_name = without_scheme
            prefix = ""
        else:
            bucket_name = without_scheme[:slash_idx]
            prefix = without_scheme[slash_idx + 1 :]

        client = storage.Client()  # type: ignore[no-untyped-call]
        bucket = client.bucket(bucket_name)  # type: ignore[no-untyped-call]
        blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda b: b.name)

        all_lines: list[str] = []
        for blob in blobs:
            if blob.name.endswith(".jsonl") or blob.name.endswith(".json"):
                content = blob.download_as_text()
                all_lines.extend(content.strip().split("\n"))

        return all_lines
