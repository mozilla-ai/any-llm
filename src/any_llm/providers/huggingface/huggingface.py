from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseStreamEvent
from openresponses_types import ResponseResource
from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionParams,
    CompletionUsage,
    CreateEmbeddingResponse,
    Reasoning,
)

MISSING_PACKAGES_ERROR = None
try:
    from huggingface_hub import AsyncInferenceClient, HfApi

    from any_llm.utils.reasoning import (
        normalize_reasoning_from_provider_fields_and_xml_tags,
        process_streaming_reasoning_chunks,
    )

    from .utils import (
        _convert_models_list,
        _convert_params,
        _create_openai_chunk_from_huggingface_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
        ChatCompletionStreamOutput as HuggingFaceChatCompletionStreamOutput,
    )

    from any_llm.types.model import Model
    from any_llm.types.responses import Response, ResponsesParams


class HuggingfaceProvider(AnyLLM):
    """HuggingFace Provider using the new response conversion utilities."""

    PROVIDER_NAME = "huggingface"
    ENV_API_KEY_NAME = "HF_TOKEN"
    ENV_API_BASE_NAME = "HUGGINGFACE_API_BASE"
    PROVIDER_DOCUMENTATION_URL = "https://huggingface.co/docs/huggingface_hub/package_reference/inference_client"

    # OpenResponses router endpoint for the OpenResponses API
    OPENRESPONSES_ROUTER_URL = "https://evalstate-openresponses.hf.space/v1"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncInferenceClient
    responses_client: AsyncOpenAI

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for HuggingFace API."""
        return _convert_params(params, **kwargs)

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert HuggingFace response to OpenAI format."""
        # If it's already our ChatCompletion type, return it
        if isinstance(response, ChatCompletion):
            return response
        # Otherwise, validate it as our type
        return ChatCompletion.model_validate(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert HuggingFace chunk response to OpenAI format."""
        return _create_openai_chunk_from_huggingface_chunk(response)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for HuggingFace."""
        msg = "HuggingFace does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert HuggingFace embedding response to OpenAI format."""
        msg = "HuggingFace does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert HuggingFace list models response to OpenAI format."""
        return _convert_models_list(response)

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self.kwargs = kwargs
        self.client = AsyncInferenceClient(
            base_url=api_base,
            token=api_key,
            **kwargs,
        )
        # Initialize OpenAI-compatible client for OpenResponses API
        self.responses_client = AsyncOpenAI(
            base_url=self.OPENRESPONSES_ROUTER_URL,
            api_key=api_key,
            **kwargs,
        )

    async def _stream_completion_async(
        self,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        response: AsyncIterator[HuggingFaceChatCompletionStreamOutput] = await self.client.chat_completion(**kwargs)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                yield self._convert_completion_chunk_response(chunk)

        def get_content(chunk: ChatCompletionChunk) -> str | None:
            return chunk.choices[0].delta.content if len(chunk.choices) > 0 else None

        def set_content(chunk: ChatCompletionChunk, content: str | None) -> ChatCompletionChunk:
            chunk.choices[0].delta.content = content
            return chunk

        def set_reasoning(chunk: ChatCompletionChunk, reasoning: str) -> ChatCompletionChunk:
            chunk.choices[0].delta.reasoning = Reasoning(content=reasoning)
            return chunk

        async for chunk in process_streaming_reasoning_chunks(
            chunk_iterator(),
            get_content=get_content,
            set_content=set_content,
            set_reasoning=set_reasoning,
        ):
            yield chunk

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        converted_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            converted_kwargs["stream"] = True
            return self._stream_completion_async(**converted_kwargs)

        response = await self.client.chat_completion(**converted_kwargs)

        data = response
        choices_out: list[Choice] = []
        for i, ch in enumerate(data.get("choices", [])):
            msg = ch.get("message", {})

            normalize_reasoning_from_provider_fields_and_xml_tags(msg)

            reasoning_obj = None
            if msg.get("reasoning") and isinstance(msg["reasoning"], dict):
                if "content" in msg["reasoning"]:
                    reasoning_obj = Reasoning(content=msg["reasoning"]["content"])

            message = ChatCompletionMessage(
                role="assistant",
                content=msg.get("content"),
                tool_calls=msg.get("tool_calls"),
                reasoning=reasoning_obj,
            )
            choices_out.append(Choice(index=i, finish_reason=ch.get("finish_reason"), message=message))

        usage = None
        if data.get("usage"):
            u = data["usage"]
            usage = CompletionUsage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )

        return ChatCompletion(
            id=data.get("id", ""),
            model=params.model_id,
            created=data.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        client = HfApi(endpoint=self.api_base, token=self.api_key, **self.kwargs)
        if kwargs.get("inference") is None and kwargs.get("inference_provider") is None:
            kwargs["inference"] = "warm"
        if kwargs.get("limit") is None:
            kwargs["limit"] = 20
        models_list = client.list_models(**kwargs)
        return self._convert_list_models_response(models_list)

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]:
        """Call OpenResponses API via HuggingFace router.

        See: https://huggingface.co/docs/inference-providers/guides/responses-api
        """
        response: Response | AsyncStream[ResponseStreamEvent] = await self.responses_client.responses.create(
            **params.model_dump(exclude_none=True), **kwargs
        )

        if isinstance(response, OpenAIResponse):
            return ResponseResource.model_validate(response.model_dump(warnings=False))

        if isinstance(response, AsyncStream):

            async def stream_iterator() -> AsyncIterator[ResponseStreamEvent]:
                async for event in response:
                    yield event

            return stream_iterator()

        msg = f"Responses API returned an unexpected type: {type(response)}"
        raise ValueError(msg)
