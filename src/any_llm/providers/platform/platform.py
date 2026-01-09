from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any, cast

from any_llm_platform_client import AnyLLMPlatformClient
from httpx import AsyncClient

from any_llm.any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.logging import logger
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CreateEmbeddingResponse,
)

from .utils import post_completion_usage_event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.model import Model


class PlatformProvider(AnyLLM):
    PROVIDER_NAME = "platform"
    ENV_API_KEY_NAME = "ANY_LLM_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/mozilla-ai/any-llm"

    # All features are marked as supported, but depending on which provider you call inside the gateway, they may not all work.
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

        self._init_client(api_key=api_key, api_base=api_base, **kwargs)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncClient(**kwargs)
        # Initialize the platform client for authentication and usage tracking
        from .utils import ANY_LLM_PLATFORM_API_URL

        self.platform_client = AnyLLMPlatformClient(any_llm_platform_url=ANY_LLM_PLATFORM_API_URL)

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        raise NotImplementedError

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        raise NotImplementedError

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        raise NotImplementedError

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        raise NotImplementedError

    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        start_time = time.perf_counter()

        # List of providers that don't support stream_options and automatically return token usage.
        # This list may need to be updated if a provider updates its usage of stream options.
        providers_without_stream_options = {
            LLMProvider.ANTHROPIC,
            LLMProvider.CEREBRAS,
            LLMProvider.COHERE,
            LLMProvider.GEMINI,
            LLMProvider.MISTRAL,
            LLMProvider.OLLAMA,
            LLMProvider.TOGETHER,
        }

        if params.stream:
            if self.provider.PROVIDER_NAME in providers_without_stream_options:
                if params.stream_options is not None:
                    logger.warning(
                        f"stream_options was set but {self.provider.PROVIDER_NAME} does not support it. "
                        "The parameter will be ignored for this request."
                    )
                params_copy = params.model_copy()
                params_copy.stream_options = None
                completion = await self.provider._acompletion(params=params_copy, **kwargs)
            else:
                if params.stream_options is None:
                    params_copy = params.model_copy()
                    params_copy.stream_options = {"include_usage": True}
                    completion = await self.provider._acompletion(params=params_copy, **kwargs)
                else:
                    completion = await self.provider._acompletion(params=params, **kwargs)
        else:
            completion = await self.provider._acompletion(params=params, **kwargs)

        if not params.stream:
            end_time = time.perf_counter()
            total_duration_ms = (end_time - start_time) * 1000

            await post_completion_usage_event(
                platform_client=self.platform_client,
                client=self.client,
                any_llm_key=self.any_llm_key,  # type: ignore[arg-type]
                provider=self.provider.PROVIDER_NAME,
                completion=cast("ChatCompletion", completion),
                provider_key_id=self.provider_key_id,  # type: ignore[arg-type]
                client_name=self.client_name,
                total_duration_ms=total_duration_ms,
            )
            return completion

        # For streaming, wrap the iterator to collect usage info
        return self._stream_with_usage_tracking(cast("AsyncIterator[ChatCompletionChunk]", completion), start_time)

    async def _stream_with_usage_tracking(
        self, stream: AsyncIterator[ChatCompletionChunk], start_time: float
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
            await post_completion_usage_event(
                platform_client=self.platform_client,
                client=self.client,
                any_llm_key=self.any_llm_key,  # type: ignore [arg-type]
                provider=self.provider.PROVIDER_NAME,
                completion=final_completion,
                provider_key_id=self.provider_key_id,  # type: ignore[arg-type]
                client_name=self.client_name,
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
        return self._provider

    @provider.setter
    def provider(self, provider_class: type[AnyLLM]) -> None:
        if self.any_llm_key is None:
            msg = "any_llm_key is required for platform provider"
            raise ValueError(msg)
        provider_key_result = self.platform_client.get_decrypted_provider_key(
            any_llm_key=self.any_llm_key, provider=provider_class.PROVIDER_NAME
        )
        self.provider_key_id = str(provider_key_result.provider_key_id)
        self.project_id = str(provider_key_result.project_id)
        self._provider = provider_class(api_key=provider_key_result.api_key, api_base=self.api_base, **self.kwargs)
