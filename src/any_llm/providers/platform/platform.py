from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from any_llm_platform_client import AnyLLMPlatformClient
from httpx import AsyncClient
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from typing_extensions import override

from any_llm import __version__
from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.logging import logger
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CreateEmbeddingResponse,
)

from .utils import (
    _get_or_create_tracer_provider,
    activate_trace_export,
    deactivate_trace_export,
    export_completion_trace,
    export_responses_trace,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from openresponses_types import ResponseResource

    from any_llm.types.batch import Batch, BatchResult
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
        self.client = AsyncClient()
        # Initialize the platform client for authentication and trace export
        from .utils import ANY_LLM_PLATFORM_API_URL

        self.platform_client = AnyLLMPlatformClient(
            any_llm_platform_url=ANY_LLM_PLATFORM_API_URL,
        )

    @staticmethod
    def _provider_requires_api_key(provider_class: type[AnyLLM]) -> bool:
        env_api_key_name = provider_class.ENV_API_KEY_NAME
        if env_api_key_name is None:
            return False
        normalized = env_api_key_name.strip().lower()
        return normalized not in {"", "none", "null"}

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
        elif not self._provider_requires_api_key(self._provider_class):
            self.provider_key_id = None
            self.project_id = None
            self._provider = self._provider_class(api_key=None, api_base=self.api_base, **self.kwargs)
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
        user_session_label = kwargs.pop("session_label", None)
        if client_name is not None:
            msg = (
                "Passing client_name at request time is not supported for PlatformProvider. "
                "Set client_name when creating the provider (for example, AnyLLM.create(..., client_name=...))."
            )
            raise ValueError(msg)

        await self._ensure_provider_initialized()
        start_time_ns = time.time_ns()
        start_perf_counter_ns = time.perf_counter_ns()
        any_llm_key = self.any_llm_key
        if any_llm_key is None:
            msg = "any_llm_key is required for platform provider"
            raise ValueError(msg)

        # Obtain the access token *before* creating the span so we can use the
        # per-token TracerProvider that has the OTLP exporter attached.
        access_token: str | None = None
        trace_export_activated = False
        try:
            access_token = await self.platform_client._aensure_valid_token(any_llm_key)
        except Exception as exc:
            logger.debug("Unable to obtain access token for trace export: %s", exc)

        if access_token is not None:
            provider_tp = _get_or_create_tracer_provider(access_token)
            tracer = provider_tp.get_tracer("any-llm", __version__)
        else:
            tracer = trace.get_tracer("any-llm", __version__)

        llm_span = tracer.start_span(
            "llm.request",
            kind=SpanKind.CLIENT,
            start_time=start_time_ns,
        )
        llm_span.set_attribute("gen_ai.provider.name", self.provider.PROVIDER_NAME)
        llm_span.set_attribute("gen_ai.request.model", params.model_id)
        if params.user is not None:
            llm_span.set_attribute("gen_ai.conversation.id", params.user)
        if self.client_name is not None:
            llm_span.set_attribute("anyllm.client_name", self.client_name)
        session_trace_label = f"{llm_span.get_span_context().trace_id:032x}"
        llm_span.set_attribute("anyllm.session_label", session_trace_label)
        if user_session_label is not None:
            llm_span.set_attribute("anyllm.user_session_label", user_session_label)

        trace_id = llm_span.get_span_context().trace_id
        try:
            if access_token is not None:
                activate_trace_export(trace_id, access_token)
                trace_export_activated = True
        except Exception as exc:
            logger.debug("Unable to activate scoped OpenTelemetry export: %s", exc)

        try:
            with trace.use_span(llm_span, end_on_exit=False):
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
                end_time_ns = time.time_ns()

                await export_completion_trace(
                    platform_client=self.platform_client,
                    client=self.client,
                    any_llm_key=any_llm_key,
                    provider=self.provider.PROVIDER_NAME,
                    request_model=params.model_id,
                    completion=cast("ChatCompletion", completion),
                    start_time_ns=start_time_ns,
                    end_time_ns=end_time_ns,
                    client_name=self.client_name,
                    session_label=session_trace_label,
                    user_session_label=user_session_label,
                    conversation_id=params.user,
                    access_token=access_token,
                    existing_span=llm_span,
                )
                if trace_export_activated:
                    deactivate_trace_export(trace_id)
                return completion

            # For streaming, wrap the iterator to collect final usage info
            return self._stream_with_usage_tracking(
                cast("AsyncIterator[ChatCompletionChunk]", completion),
                start_time_ns,
                params.model_id,
                params.user,
                session_trace_label,
                user_session_label,
                start_perf_counter_ns,
                any_llm_key,
                llm_span,
                trace_id,
                access_token,
                trace_export_activated,
            )
        except Exception as exc:
            llm_span.set_attribute("error.type", exc.__class__.__name__)
            llm_span.set_status(Status(StatusCode.ERROR, "llm request failed"))
            llm_span.end(end_time=time.time_ns())
            if trace_export_activated:
                deactivate_trace_export(trace_id)
            raise

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]:
        user_session_label = kwargs.pop("session_label", None)

        await self._ensure_provider_initialized()
        start_time_ns = time.time_ns()
        start_perf_counter_ns = time.perf_counter_ns()
        any_llm_key = self.any_llm_key
        if any_llm_key is None:
            msg = "any_llm_key is required for platform provider"
            raise ValueError(msg)

        access_token: str | None = None
        trace_export_activated = False
        try:
            access_token = await self.platform_client._aensure_valid_token(any_llm_key)
        except Exception as exc:
            logger.debug("Unable to obtain access token for trace export: %s", exc)

        if access_token is not None:
            provider_tp = _get_or_create_tracer_provider(access_token)
            tracer = provider_tp.get_tracer("any-llm", __version__)
        else:
            tracer = trace.get_tracer("any-llm", __version__)

        llm_span = tracer.start_span(
            "llm.request",
            kind=SpanKind.CLIENT,
            start_time=start_time_ns,
        )
        llm_span.set_attribute("gen_ai.provider.name", self.provider.PROVIDER_NAME)
        llm_span.set_attribute("gen_ai.request.model", params.model)
        if params.user is not None:
            llm_span.set_attribute("gen_ai.conversation.id", params.user)
        if self.client_name is not None:
            llm_span.set_attribute("anyllm.client_name", self.client_name)
        session_trace_label = f"{llm_span.get_span_context().trace_id:032x}"
        llm_span.set_attribute("anyllm.session_label", session_trace_label)
        if user_session_label is not None:
            llm_span.set_attribute("anyllm.user_session_label", user_session_label)

        trace_id = llm_span.get_span_context().trace_id
        try:
            if access_token is not None:
                activate_trace_export(trace_id, access_token)
                trace_export_activated = True
        except Exception as exc:
            logger.debug("Unable to activate scoped OpenTelemetry export: %s", exc)

        try:
            with trace.use_span(llm_span, end_on_exit=False):
                result = await self.provider._aresponses(params, **kwargs)

            if not params.stream:
                end_time_ns = time.time_ns()
                response_model: str | None = None
                input_tokens: int | None = None
                output_tokens: int | None = None

                usage = getattr(result, "usage", None)
                if usage is not None:
                    input_tokens = usage.input_tokens
                    output_tokens = usage.output_tokens
                response_model = getattr(result, "model", None)

                await export_responses_trace(
                    platform_client=self.platform_client,
                    client=self.client,
                    any_llm_key=any_llm_key,
                    provider=self.provider.PROVIDER_NAME,
                    request_model=params.model,
                    response_model=response_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    start_time_ns=start_time_ns,
                    end_time_ns=end_time_ns,
                    client_name=self.client_name,
                    session_label=session_trace_label,
                    user_session_label=user_session_label,
                    conversation_id=params.user,
                    access_token=access_token,
                    existing_span=llm_span,
                )
                if trace_export_activated:
                    deactivate_trace_export(trace_id)
                return result

            return self._stream_responses_with_usage_tracking(
                cast("AsyncIterator[ResponseStreamEvent]", result),
                start_time_ns,
                params.model,
                params.user,
                session_trace_label,
                user_session_label,
                start_perf_counter_ns,
                any_llm_key,
                llm_span,
                trace_id,
                access_token,
                trace_export_activated,
            )
        except Exception as exc:
            llm_span.set_attribute("error.type", exc.__class__.__name__)
            llm_span.set_status(Status(StatusCode.ERROR, "llm request failed"))
            llm_span.end(end_time=time.time_ns())
            if trace_export_activated:
                deactivate_trace_export(trace_id)
            raise

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

    @override
    async def _aretrieve_batch_results(self, batch_id: str, **kwargs: Any) -> BatchResult:
        await self._ensure_provider_initialized()
        return await self.provider._aretrieve_batch_results(batch_id, **kwargs)

    async def _stream_with_usage_tracking(
        self,
        stream: AsyncIterator[ChatCompletionChunk],
        start_time_ns: int,
        request_model: str,
        conversation_id: str | None,
        session_label: str,
        user_session_label: str | None,
        start_perf_counter_ns: int,
        any_llm_key: str,
        llm_span: trace.Span,
        trace_id: int,
        access_token: str | None,
        trace_export_activated: bool,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Wrap the stream to export a trace after completion."""
        chunks: list[ChatCompletionChunk] = []
        first_chunk_received = False

        span_ended = False
        try:
            with trace.use_span(llm_span, end_on_exit=False):
                async for chunk in stream:
                    if not first_chunk_received:
                        first_chunk_received = True
                        ttft_ms = (time.perf_counter_ns() - start_perf_counter_ns) / 1_000_000
                        llm_span.set_attribute("anyllm.performance.ttft_ms", ttft_ms)
                        llm_span.add_event("llm.first_token", {"anyllm.performance.ttft_ms": ttft_ms})
                    chunks.append(chunk)
                    yield chunk

            if chunks:
                end_time_ns = time.time_ns()

                # Combine chunks into a single ChatCompletion-like object (do this once)
                final_completion = self._combine_chunks(chunks)
                await export_completion_trace(
                    platform_client=self.platform_client,
                    client=self.client,
                    any_llm_key=any_llm_key,
                    provider=self.provider.PROVIDER_NAME,
                    request_model=request_model,
                    completion=final_completion,
                    start_time_ns=start_time_ns,
                    end_time_ns=end_time_ns,
                    client_name=self.client_name,
                    session_label=session_label,
                    user_session_label=user_session_label,
                    conversation_id=conversation_id,
                    access_token=access_token,
                    existing_span=llm_span,
                )
            else:
                llm_span.end(end_time=time.time_ns())
            span_ended = True
        except Exception as exc:
            llm_span.set_attribute("error.type", exc.__class__.__name__)
            llm_span.set_status(Status(StatusCode.ERROR, "llm request failed"))
            llm_span.end(end_time=time.time_ns())
            span_ended = True
            raise
        finally:
            if not span_ended:
                end_time_ns = time.time_ns()
                llm_span.set_attribute("anyllm.stream.cancelled", True)
                if chunks:
                    final_completion = self._combine_chunks(chunks)
                    if final_completion.model:
                        llm_span.set_attribute("gen_ai.response.model", final_completion.model)
                    if final_completion.usage:
                        llm_span.set_attribute("gen_ai.usage.input_tokens", final_completion.usage.prompt_tokens)
                        llm_span.set_attribute("gen_ai.usage.output_tokens", final_completion.usage.completion_tokens)
                llm_span.end(end_time=end_time_ns)
            if trace_export_activated:
                deactivate_trace_export(trace_id)

    def _combine_chunks(self, chunks: list[ChatCompletionChunk]) -> ChatCompletion:
        """Combine streaming chunks into a ChatCompletion for usage tracking."""
        # Get the last chunk which typically has the full usage info
        last_chunk = chunks[-1]

        if not last_chunk.usage:
            msg = (
                "The last chunk of your streaming response does not contain usage data. "
                "Usage attributes will not be available in exported traces. "
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
            usage=last_chunk.usage or None,
            choices=[],
        )

    async def _stream_responses_with_usage_tracking(
        self,
        stream: AsyncIterator[ResponseStreamEvent],
        start_time_ns: int,
        request_model: str,
        conversation_id: str | None,
        session_label: str,
        user_session_label: str | None,
        start_perf_counter_ns: int,
        any_llm_key: str,
        llm_span: trace.Span,
        trace_id: int,
        access_token: str | None,
        trace_export_activated: bool,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """Wrap a Responses API stream to export a trace after completion."""
        first_event_received = False
        response_model: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        span_ended = False
        try:
            with trace.use_span(llm_span, end_on_exit=False):
                async for event in stream:
                    if not first_event_received:
                        first_event_received = True
                        ttft_ms = (time.perf_counter_ns() - start_perf_counter_ns) / 1_000_000
                        llm_span.set_attribute("anyllm.performance.ttft_ms", ttft_ms)
                        llm_span.add_event("llm.first_token", {"anyllm.performance.ttft_ms": ttft_ms})

                    if event.type == "response.completed":
                        usage = getattr(event.response, "usage", None)
                        if usage is not None:
                            input_tokens = usage.input_tokens
                            output_tokens = usage.output_tokens
                        response_model = getattr(event.response, "model", None)

                    yield event

            end_time_ns = time.time_ns()
            await export_responses_trace(
                platform_client=self.platform_client,
                client=self.client,
                any_llm_key=any_llm_key,
                provider=self.provider.PROVIDER_NAME,
                request_model=request_model,
                response_model=response_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                client_name=self.client_name,
                session_label=session_label,
                user_session_label=user_session_label,
                conversation_id=conversation_id,
                access_token=access_token,
                existing_span=llm_span,
            )
            span_ended = True
        except Exception as exc:
            llm_span.set_attribute("error.type", exc.__class__.__name__)
            llm_span.set_status(Status(StatusCode.ERROR, "llm request failed"))
            llm_span.end(end_time=time.time_ns())
            span_ended = True
            raise
        finally:
            if not span_ended:
                end_time_ns = time.time_ns()
                llm_span.set_attribute("anyllm.stream.cancelled", True)
                if response_model is not None:
                    llm_span.set_attribute("gen_ai.response.model", response_model)
                if input_tokens is not None:
                    llm_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                if output_tokens is not None:
                    llm_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                llm_span.end(end_time=end_time_ns)
            if trace_export_activated:
                deactivate_trace_export(trace_id)

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
