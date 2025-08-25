from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig, Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.model import Model

MISSING_PACKAGES_ERROR = None
try:
    import cohere

    from .utils import (
        _convert_models_list,
        _convert_response,
        _create_openai_chunk_from_cohere_chunk,
        _patch_messages,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class CohereProvider(Provider):
    """Cohere Provider using the new response conversion utilities."""

    PROVIDER_NAME = "cohere"
    ENV_API_KEY_NAME = "CO_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://cohere.com/api"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cohere provider."""
        super().__init__(config)
        self.client = cohere.ClientV2(api_key=config.api_key)

    async def _stream_completion_async(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        client = cohere.AsyncClientV2(api_key=self.config.api_key)

        cohere_stream = client.chat_stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        async for chunk in cohere_stream:
            yield _create_openai_chunk_from_cohere_chunk(chunk)

    @staticmethod
    def _preprocess_response_format(response_format: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
        # if response format is a BaseModel, generate model json schema
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return {"type": "json_object", "schema": response_format.model_json_schema()}
        # can either be json schema already in dict
        # or {"type": "json_object"} to just generate *a* JSON (JSON mode)
        # see docs here: https://docs.cohere.com/docs/structured-outputs#json-mode
        if isinstance(response_format, dict):
            return response_format
        # For now, let Cohere API handle invalid schemas.
        # Note that Cohere has a bunch of limitations on JSON schemas (e.g., no oneOf, numeric/str ranges, weird regex limitations)
        # see docs here: https://docs.cohere.com/docs/structured-outputs#unsupported-schema-features
        # Validation logic could/would eventually go here
        return response_format

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        if params.response_format is not None:
            kwargs["response_format"] = self._preprocess_response_format(params.response_format)
        if params.stream and params.response_format is not None:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        if params.parallel_tool_calls is not None:
            msg = "parallel_tool_calls"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)

        patched_messages = _patch_messages(params.messages)

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                patched_messages,
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}),
                **kwargs,
            )

        client = cohere.AsyncClientV2(api_key=self.config.api_key)

        # note: ClientV2.chat does not have a `stream` parameter
        response = await client.chat(
            model=params.model_id,
            messages=patched_messages,  # type: ignore[arg-type]
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "", "stream", "response_format"}),
            **kwargs,
        )

        return _convert_response(response, params.model_id)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        model_list = self.client.models.list(**kwargs)
        return _convert_models_list(model_list)
