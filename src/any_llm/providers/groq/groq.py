from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError

if TYPE_CHECKING:
    from any_llm.types.completion import CreateEmbeddingResponse

MISSING_PACKAGES_ERROR = None
try:
    from groq import AsyncGroq

    from .utils import (
        _convert_models_list,
        _create_openai_chunk_from_groq_chunk,
        to_chat_completion,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from groq import AsyncStream as GroqAsyncStream
    from groq.types.chat import ChatCompletion as GroqChatCompletion
    from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
    from any_llm.types.model import Model


class GroqProvider(AnyLLM):
    """Groq Provider."""

    PROVIDER_NAME = "groq"
    ENV_API_KEY_NAME = "GROQ_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://groq.com/api"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncGroq

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Groq API."""
        # Groq does not support providing reasoning effort
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        if converted_params.get("reasoning_effort") in ("auto", "none"):
            converted_params.pop("reasoning_effort")
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Groq response to OpenAI format."""
        return to_chat_completion(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Groq chunk response to OpenAI format."""
        return _create_openai_chunk_from_groq_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Groq."""
        msg = "Groq does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Groq embedding response to OpenAI format."""
        msg = "Groq does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Groq list models response to OpenAI format."""
        return _convert_models_list(response)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.kwargs = kwargs
        self.client = AsyncGroq(api_key=api_key, **kwargs)

    async def _stream_async_completion(
        self, params: CompletionParams, **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        if params.stream and params.response_format:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)

        completion_kwargs = self._convert_completion_params(params, **kwargs)
        stream: GroqAsyncStream[GroqChatCompletionChunk] = await self.client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **completion_kwargs,
        )

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in stream:
                yield self._convert_completion_chunk_response(chunk)

        return _stream()

    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.response_format:
            if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": params.response_format.__name__,
                        "schema": params.response_format.model_json_schema(),
                    },
                }
            else:
                kwargs["response_format"] = params.response_format

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return await self._stream_async_completion(
                params,
                **kwargs,
            )
        response: GroqChatCompletion = await self.client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **completion_kwargs,
        )

        return self._convert_completion_response(response)

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(models_list)

    # Groq is not compliant with OpenResponses spec


"""
>           return ResponseResource.model_validate(response.model_dump())
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           pydantic_core._pydantic_core.ValidationError: 36 validation errors for ResponseResource
E           output.1.Message.content.0.InputTextContent.type
E             Input should be 'input_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.OutputTextContent.logprobs
E             Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/list_type
E           output.1.Message.content.0.TextContent.type
E             Input should be 'text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.SummaryTextContent.type
E             Input should be 'summary_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.ReasoningTextContent.type
E             Input should be 'reasoning_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.RefusalContent.type
E             Input should be 'refusal' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.RefusalContent.refusal
E             Field required [type=missing, input_value={'annotations': [], 'text...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.Message.content.0.InputImageContent.type
E             Input should be 'input_image' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.InputImageContent.image_url
E             Field required [type=missing, input_value={'annotations': [], 'text...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.Message.content.0.InputImageContent.detail
E             Field required [type=missing, input_value={'annotations': [], 'text...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.Message.content.0.InputFileContent.type
E             Input should be 'input_file' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.InputVideoContent.type
E             Input should be 'input_video' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.InputVideoContent.video_url
E             Field required [type=missing, input_value={'annotations': [], 'text...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCall.type
E             Input should be 'function_call' [type=literal_error, input_value='message', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.FunctionCall.call_id
E             Field required [type=missing, input_value={'id': 'msg_01kfktfn0tefy...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCall.name
E             Field required [type=missing, input_value={'id': 'msg_01kfktfn0tefy...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCall.arguments
E             Field required [type=missing, input_value={'id': 'msg_01kfktfn0tefy...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCallOutput.type
E             Input should be 'function_call_output' [type=literal_error, input_value='message', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.FunctionCallOutput.call_id
E             Field required [type=missing, input_value={'id': 'msg_01kfktfn0tefy...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCallOutput.output
E             Field required [type=missing, input_value={'id': 'msg_01kfktfn0tefy...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.ReasoningBody.type
E             Input should be 'reasoning' [type=literal_error, input_value='message', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.InputTextContent.type
E             Input should be 'input_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.OutputTextContent.logprobs
E             Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/list_type
E           output.1.ReasoningBody.content.0.TextContent.type
E             Input should be 'text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.SummaryTextContent.type
E             Input should be 'summary_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.ReasoningTextContent.type
E             Input should be 'reasoning_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.RefusalContent.type
E             Input should be 'refusal' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.RefusalContent.refusal
E             Field required [type=missing, input_value={'annotations': [], 'text...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.ReasoningBody.content.0.InputImageContent.type
E             Input should be 'input_image' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.InputImageContent.image_url
E             Field required [type=missing, input_value={'annotations': [], 'text...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.ReasoningBody.content.0.InputImageContent.detail
E             Field required [type=missing, input_value={'annotations': [], 'text...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.ReasoningBody.content.0.InputFileContent.type
E             Input should be 'input_file' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.summary
E             Field required [type=missing, input_value={'id': 'msg_01kfktfn0tefy...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           presence_penalty
E             Field required [type=missing, input_value={'id': 'resp_01kfktfn0tef...': None, 'store': False}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           frequency_penalty
E             Field required [type=missing, input_value={'id': 'resp_01kfktfn0tef...': None, 'store': False}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           top_logprobs
E             Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/int_type

src/any_llm/providers/groq/groq.py:172: ValidationError
"""
