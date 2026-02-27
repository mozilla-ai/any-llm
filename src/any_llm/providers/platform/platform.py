from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any, cast

from any_llm_platform_client import AnyLLMPlatformClient
from httpx import AsyncClient
from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.logging import logger
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CreateEmbeddingResponse,
)

from .utils import export_completion_trace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from openresponses_types import ResponseResource

    from any_llm.types.batch import Batch
    from any_llm.types.model import Model
    from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent


class PlatformProvider(AnyLLM):
    PROVIDER_NAME = "platform"
    ENV_API_KEY_NAME = "ANY_LLM_KEY"
    ENV_API_BASE_NAME = "ANY_LLM_PLATFORM_URL"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/mozilla-ai/any-llm"

    # All features are marked as supported by default. When a provider class is
    # assigned via the `provider` setter, these flags are updated to reflect the
    # wrapped provider's actual capabilities.
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        client_name: str | None = None,
        **kwargs: Any,
    ):
        self.any_llm_key = self._verify_and_set_api_key(api_key)
        self.api_base = api_base
        self.client_name = client_name
        self.kwargs = kwargs
        self.provider_key_id: str | None = None
        self.project_id: str | None = None
        self._provider_class: type[AnyLLM] | None = None
        self._provider: AnyLLM | None = None
        self._provider_initialized = False

        self._init_client(api_key=api_key, api_base=api_base, **kwargs)

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncClient(**kwargs)
        # Initialize the platform client for authentication and trace export
        from .utils import ANY_LLM_PLATFORM_API_URL

        self.platform_client = AnyLLMPlatformClient(
            any_llm_platform_url=ANY_LLM_PLATFORM_API_URL,
            client_name=self.client_name,
        )

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        raise NotImplementedError

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        raise NotImplementedError

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        raise NotImplementedError

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        raise NotImplementedError

    async def _ensure_provider_initialized(self) -> None:
        """Lazily initialize the wrapped provider using async HTTP on first use."""
        if self._provider_initialized:
            return
        if self._provider_class is None:
            msg = "No provider class has been set. Use the provider setter first."
            raise RuntimeError(msg)
        if self.any_llm_key is None:
            msg = "any_llm_key is required for platform provider"
            raise ValueError(msg)
        if self._provider_class.PROVIDER_NAME == LLMProvider.MZAI.value:
            # For mzai, use JWT token directly as API key
            token = await self.platform_client._aensure_valid_token(self.any_llm_key)
            self.provider_key_id = None
            self.project_id = None
            self._provider = self._provider_class(api_key=token, api_base=self.api_base, **self.kwargs)
        else:
            provider_key_result = await self.platform_client.aget_decrypted_provider_key(
                any_llm_key=self.any_llm_key, provider=self._provider_class.PROVIDER_NAME
            )
            self.provider_key_id = str(provider_key_result.provider_key_id)
            self.project_id = str(provider_key_result.project_id)
            self._provider = self._provider_class(
                api_key=provider_key_result.api_key, api_base=self.api_base, **self.kwargs
            )
        self._provider_initialized = True

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        client_name = kwargs.pop("client_name", None)
        session_label = kwargs.pop("session_label", None)
        if client_name is not None:
            msg = (
                "Passing client_name at request time is not supported for PlatformProvider. "
                "Set client_name when creating the provider (for example, AnyLLM.create(..., client_name=...))."
            )
            raise ValueError(msg)

        await self._ensure_provider_initialized()
        start_time = time.perf_counter()
        start_time_ns = time.time_ns()

        if params.stream and params.stream_options is None:
            # Auto-inject stream_options to request usage data for tracking.
            # Providers that don't support stream_options strip it in their
            # _convert_completion_params and include usage data automatically.
            params_copy = params.model_copy()
            params_copy.stream_options = {"include_usage": True}
            completion = await self.provider._acompletion(params=params_copy, **kwargs)
        else:
            completion = await self.provider._acompletion(params=params, **kwargs)

        if not params.stream:
            end_time = time.perf_counter()
            end_time_ns = time.time_ns()
            total_duration_ms = (end_time - start_time) * 1000

            await export_completion_trace(
                platform_client=self.platform_client,
                client=self.client,
                any_llm_key=self.any_llm_key,  # type: ignore[arg-type]
                provider=self.provider.PROVIDER_NAME,
                request_model=params.model_id,
                completion=cast("ChatCompletion", completion),
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                client_name=self.client_name,
                session_label=session_label,
                conversation_id=params.user,
                total_duration_ms=total_duration_ms,
            )
            return completion

        # For streaming, wrap the iterator to collect usage info
        return self._stream_with_usage_tracking(
            cast("AsyncIterator[ChatCompletionChunk]", completion),
            start_time,
            start_time_ns,
            params.model_id,
            params.user,
            session_label,
        )

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]:
        await self._ensure_provider_initialized()
        return await self.provider._aresponses(params, **kwargs)

    @override
    async def _aembedding(self, model: str, inputs: str | list[str], **kwargs: Any) -> CreateEmbeddingResponse:
        await self._ensure_provider_initialized()
        return await self.provider._aembedding(model, inputs, **kwargs)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        await self._ensure_provider_initialized()
        return await self.provider._alist_models(**kwargs)

    @override
    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        await self._ensure_provider_initialized()
        return await self.provider._acreate_batch(
            input_file_path=input_file_path,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
            **kwargs,
        )

    @override
    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        await self._ensure_provider_initialized()
        return await self.provider._aretrieve_batch(batch_id, **kwargs)

    @override
    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        await self._ensure_provider_initialized()
        return await self.provider._acancel_batch(batch_id, **kwargs)

    @override
    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        await self._ensure_provider_initialized()
        return await self.provider._alist_batches(after=after, limit=limit, **kwargs)

    async def _stream_with_usage_tracking(
        self,
        stream: AsyncIterator[ChatCompletionChunk],
        start_time: float,
        start_time_ns: int,
        request_model: str,
        conversation_id: str | None,
        session_label: str | None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Wrap the stream to track usage after completion."""
        chunks: list[ChatCompletionChunk] = []
        time_to_first_token_ms: float | None = None
        time_to_last_content_token_ms: float | None = None
        chunk_latencies: list[float] = []
        previous_chunk_time: float | None = None

        async for chunk in stream:
            current_time = time.perf_counter()

            # Capture time to first token (first chunk with content)
            if time_to_first_token_ms is None and chunk.choices and chunk.choices[0].delta.content:
                time_to_first_token_ms = (current_time - start_time) * 1000

            # Track inter-chunk latency
            if previous_chunk_time is not None:
                inter_chunk_latency = (current_time - previous_chunk_time) * 1000
                chunk_latencies.append(inter_chunk_latency)
            previous_chunk_time = current_time

            chunks.append(chunk)

            # Count tokens as we stream and track last content token time
            if chunk.choices and chunk.choices[0].delta.content:
                time_to_last_content_token_ms = (current_time - start_time) * 1000

            yield chunk

        # After stream completes, reconstruct completion for usage tracking
        if chunks:
            end_time = time.perf_counter()
            end_time_ns = time.time_ns()
            total_duration_ms = (end_time - start_time) * 1000

            # Use time_to_last_content_token_ms if available, otherwise use total_duration_ms
            time_to_last_token_ms = time_to_last_content_token_ms or total_duration_ms

            # Calculate tokens per second based on actual output tokens from usage
            tokens_per_second: float | None = None
            chunks_received = len(chunks)
            avg_chunk_size: float | None = None
            inter_chunk_latency_variance_ms: float | None = None

            # Combine chunks into a single ChatCompletion-like object (do this once)
            final_completion = self._combine_chunks(chunks)

            # Get actual token count from usage data
            last_chunk = chunks[-1]

            # Try to get token count from last chunk's usage data, fallback to combined completion
            actual_output_tokens: int | None = None
            if last_chunk.usage and last_chunk.usage.completion_tokens:
                actual_output_tokens = last_chunk.usage.completion_tokens
            elif final_completion.usage and final_completion.usage.completion_tokens:
                actual_output_tokens = final_completion.usage.completion_tokens

            # Calculate metrics if we have token count
            if actual_output_tokens is not None and actual_output_tokens > 0:
                if time_to_last_token_ms > 0:
                    tokens_per_second = (actual_output_tokens * 1000) / time_to_last_token_ms

                # Calculate average chunk size
                if chunks_received > 0:
                    avg_chunk_size = actual_output_tokens / chunks_received

            # Calculate inter-chunk latency variance
            if len(chunk_latencies) > 1:
                inter_chunk_latency_variance_ms = statistics.variance(chunk_latencies)
            await export_completion_trace(
                platform_client=self.platform_client,
                client=self.client,
                any_llm_key=self.any_llm_key,  # type: ignore [arg-type]
                provider=self.provider.PROVIDER_NAME,
                request_model=request_model,
                completion=final_completion,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                client_name=self.client_name,
                session_label=session_label,
                conversation_id=conversation_id,
                time_to_first_token_ms=time_to_first_token_ms,
                time_to_last_token_ms=time_to_last_token_ms,
                total_duration_ms=total_duration_ms,
                tokens_per_second=tokens_per_second,
                chunks_received=chunks_received,
                avg_chunk_size=avg_chunk_size,
                inter_chunk_latency_variance_ms=inter_chunk_latency_variance_ms,
            )

    def _combine_chunks(self, chunks: list[ChatCompletionChunk]) -> ChatCompletion:
        """Combine streaming chunks into a ChatCompletion for usage tracking."""
        # Get the last chunk which typically has the full usage info
        last_chunk = chunks[-1]

        if not last_chunk.usage:
            msg = (
                "The last chunk of your streaming response does not contain usage data. "
                "Performance metrics requiring token counts will not be available. "
                "Consult your provider documentation on how to enable usage data in streaming responses."
            )
            logger.warning(msg)

            return ChatCompletion(
                id=last_chunk.id,
                model=last_chunk.model,
                created=last_chunk.created,
                object="chat.completion",
                usage=None,  # Set to None instead of zeros to distinguish from actual zero tokens
                choices=[],
            )

        # Create a minimal ChatCompletion object with the data needed for usage tracking
        # We only need id, model, created, usage, and object type
        return ChatCompletion(
            id=last_chunk.id,
            model=last_chunk.model,
            created=last_chunk.created,
            object="chat.completion",
            usage=last_chunk.usage if hasattr(last_chunk, "usage") and last_chunk.usage else None,
            choices=[],
        )

    @property
    def provider(self) -> AnyLLM:
        if self._provider is None:
            msg = "Provider not yet initialized. Call an async method (e.g. acompletion, alist_models) first."
            raise RuntimeError(msg)
        return self._provider

    @provider.setter
    def provider(self, provider_class: type[AnyLLM]) -> None:
        """Store the provider class for lazy async initialization.

        The actual provider instance is created on the first async method call
        via _ensure_provider_initialized(), which uses async HTTP to fetch the
        decrypted provider key without blocking the event loop.
        """
        if self.any_llm_key is None:
            msg = "any_llm_key is required for platform provider"
            raise ValueError(msg)
        self._provider_class = provider_class
        self._provider_initialized = False
        self._provider = None

        # Sync capability flags to match the wrapped provider
        self.SUPPORTS_COMPLETION_STREAMING = provider_class.SUPPORTS_COMPLETION_STREAMING
        self.SUPPORTS_COMPLETION = provider_class.SUPPORTS_COMPLETION
        self.SUPPORTS_RESPONSES = provider_class.SUPPORTS_RESPONSES
        self.SUPPORTS_COMPLETION_REASONING = provider_class.SUPPORTS_COMPLETION_REASONING
        self.SUPPORTS_COMPLETION_IMAGE = provider_class.SUPPORTS_COMPLETION_IMAGE
        self.SUPPORTS_COMPLETION_PDF = provider_class.SUPPORTS_COMPLETION_PDF
        self.SUPPORTS_EMBEDDING = provider_class.SUPPORTS_EMBEDDING
        self.SUPPORTS_LIST_MODELS = provider_class.SUPPORTS_LIST_MODELS
        self.SUPPORTS_BATCH = provider_class.SUPPORTS_BATCH
