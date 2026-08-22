from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.providers.openai.utils import (
    _convert_moderation_response_from_openai,
    _normalize_openai_dict_response,
)
from any_llm.providers.openai.xml_reasoning import (
    wrap_chunks_with_xml_reasoning,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.model import Model
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

MISSING_PACKAGES_ERROR = None
try:
    from portkey_ai import AsyncPortkey
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from portkey_ai.api_resources.types.chat_complete_type import (
        ChatCompletions as PortkeyChatCompletions,
    )
    from portkey_ai.api_resources.types.models_type import (
        ModelList as PortkeyModelList,
    )

    from any_llm.types.completion import CreateEmbeddingResponse
    from any_llm.types.moderation import ModerationResponse


class PortkeyProvider(AnyLLM):
    """Portkey provider for accessing 200+ LLMs through Portkey's AI Gateway."""

    API_BASE = "https://api.portkey.ai/v1"
    ENV_API_KEY_NAME = "PORTKEY_API_KEY"
    ENV_API_BASE_NAME = "PORTKEY_API_BASE"
    PROVIDER_NAME = "portkey"
    PROVIDER_DOCUMENTATION_URL = "https://portkey.ai/docs"

    TIMEOUT_SUPPORT = "native"
    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    SUPPORTS_MODERATION = True

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    _DEFAULT_REASONING_EFFORT = None

    client: AsyncPortkey

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncPortkey(
            base_url=api_base or self.API_BASE,
            api_key=api_key,
            **kwargs,
        )

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:

        response_dict = _normalize_openai_dict_response(response.model_dump())

        choices = response_dict.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict):
                    normalize_reasoning_from_provider_fields_and_xml_tags(message)

                delta = choice.get("delta") if isinstance(choice, dict) else None
                if isinstance(delta, dict):
                    normalize_reasoning_from_provider_fields_and_xml_tags(delta)

        return ChatCompletion.model_validate(response_dict)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        response_dict = _normalize_openai_dict_response(response.model_dump())

        choices = response_dict.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                delta = choice.get("delta") if isinstance(choice, dict) else None

                if isinstance(delta, dict):
                    normalize_reasoning_from_provider_fields_and_xml_tags(delta)

        response_dict["object"] = "chat.completion.chunk"
        return ChatCompletionChunk.model_validate(response_dict)

    async def _stream_async_completion(
        self, params: CompletionParams, **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:

        completion_kwargs = self._convert_completion_params(params, **kwargs)
        stream = cast(
            "AsyncIterator[Any]",
            await self.client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                stream=True,
                **completion_kwargs,
            ),
        )

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in stream:
                yield self._convert_completion_chunk_response(chunk)

        return wrap_chunks_with_xml_reasoning(chunk_iterator())

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Portkey SDK."""
        if params.response_format:
            if is_structured_output_type(params.response_format):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": get_json_schema(params.response_format),
                    },
                }
            else:
                kwargs["response_format"] = params.response_format

        converted_params = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "stream", "response_format"}
        )
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Portkey."""
        msg = "Portkey does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Portkey embedding response to OpenAI format."""
        msg = "Portkey does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(models_list: PortkeyModelList) -> list[Model]:

        data = models_list.data or []

        return [Model.model_validate(model.model_dump()) for model in data]

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:

        if params.reasoning_effort == "auto":
            params.reasoning_effort = self._DEFAULT_REASONING_EFFORT

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return await self._stream_async_completion(
                params,
                **kwargs,
            )
        response = cast(
            "PortkeyChatCompletions",
            await self.client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **completion_kwargs,
            ),
        )

        return self._convert_completion_response(response)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(models_list)

    @override
    async def _amoderation(
        self,
        model: str,
        input: str | list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModerationResponse:
        include_raw = kwargs.pop("include_raw", False)
        model_name = model or "omni-moderation-latest"

        raw = await self.client.moderations.create(
            model=model_name,
            input=cast("Any", input),
            **kwargs,
        )

        return _convert_moderation_response_from_openai(
            raw,
            include_raw=include_raw,
        )
