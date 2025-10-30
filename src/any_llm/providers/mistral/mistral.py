from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_llm.any_llm import AnyLLM

MISSING_PACKAGES_ERROR = None
try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model
    from mistralai.models.responseformat import ResponseFormat

    from .utils import (
        _convert_models_list,
        _create_mistral_completion_from_response,
        _create_openai_chunk_from_mistral_chunk,
        _create_openai_embedding_response_from_mistral,
        _patch_messages,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from mistralai import Mistral  # noqa: TC004
    from mistralai.models.embeddingresponse import EmbeddingResponse

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
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: Mistral

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Mistral API."""
        # Mistral does not support providing reasoning effort
        converted_params = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}
        )
        converted_params["messages"] = _patch_messages(params.messages)

        if params.response_format is not None:
            if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
                converted_params["response_format"] = response_format_from_pydantic_model(params.response_format)
            elif isinstance(params.response_format, dict):
                converted_params["response_format"] = ResponseFormat.model_validate(params.response_format)

        if converted_params.get("reasoning_effort") == "auto":
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
