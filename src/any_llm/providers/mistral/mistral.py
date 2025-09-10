from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from any_llm.provider import Provider

MISSING_PACKAGES_ERROR = None
try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model

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


class MistralProvider(Provider):
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

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Mistral API."""
        # Mistral does not support providing reasoning effort
        converted_params = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}
        )
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

    async def _stream_completion_async(
        self, client: Mistral, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        mistral_stream = await client.chat.stream_async(model=model, messages=messages, **kwargs)  # type: ignore[arg-type]

        async for event in mistral_stream:
            yield self._convert_completion_chunk_response(event)

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        patched_messages = _patch_messages(params.messages)

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        if (
            params.response_format is not None
            and isinstance(params.response_format, type)
            and issubclass(params.response_format, BaseModel)
        ):
            kwargs["response_format"] = response_format_from_pydantic_model(params.response_format)

        client = Mistral(
            api_key=self.config.api_key,
            server_url=self.config.api_base,
            **(self.config.client_args if self.config.client_args else {}),
        )

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                client,
                params.model_id,
                patched_messages,
                **completion_kwargs,
            )

        response = await client.chat.complete_async(
            model=params.model_id,
            messages=patched_messages,  # type: ignore[arg-type]
            **completion_kwargs,
        )

        return self._convert_completion_response(response)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        client = Mistral(
            api_key=self.config.api_key,
            server_url=self.config.api_base,
            **(self.config.client_args if self.config.client_args else {}),
        )
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        result: EmbeddingResponse = await client.embeddings.create_async(
            model=model,
            **embedding_kwargs,
        )

        return self._convert_embedding_response(result)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        client = Mistral(
            api_key=self.config.api_key,
            server_url=self.config.api_base,
            **(self.config.client_args if self.config.client_args else {}),
        )
        models_list = client.models.list(**kwargs)
        return self._convert_list_models_response(models_list)
