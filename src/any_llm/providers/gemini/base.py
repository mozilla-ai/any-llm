from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import BaseModel

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError
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

MISSING_PACKAGES_ERROR = None
try:
    from google.genai import types

    from .utils import (
        _convert_messages,
        _convert_models_list,
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

    from any_llm.types.model import Model

REASONING_EFFORT_TO_THINKING_BUDGETS = {"minimal": 256, "low": 1024, "medium": 8192, "high": 24576}


class GoogleProvider(AnyLLM):
    """Base Google Provider class with common functionality for Gemini and Vertex AI."""

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False  # TODO: Add image support https://github.com/mozilla-ai/any-llm/issues/415
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    BUILT_IN_TOOLS: ClassVar[list[Any] | None] = [types.Tool]

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: genai.Client

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Google API."""
        provider_name = kwargs.pop("provider_name")

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
            if params.reasoning_effort is None:
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
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_format

        formatted_messages, system_instruction = _convert_messages(params.messages)
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        result_kwargs: dict[str, Any] = {
            "config": types.GenerateContentConfig(**kwargs),
            "contents": formatted_messages,
            "model": params.model_id,
        }

        return result_kwargs

    @staticmethod
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
                        )
                    )
                tool_calls = tool_calls_list

            reasoning_content = message_dict.get("reasoning")
            message = ChatCompletionMessage(
                role="assistant",
                content=message_dict.get("content"),
                tool_calls=tool_calls,
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
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Google chunk response to OpenAI format."""
        return _create_openai_chunk_from_google_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Google API."""
        converted_params = {"contents": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Google embedding response to OpenAI format."""
        # We need the model parameter for conversion
        model = response.get("model", "google-model")
        return _create_openai_embedding_response_from_google(model, response["result"])

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Google list models response to OpenAI format."""
        return _convert_models_list(response)

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

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.aio.models.list(**kwargs)
        return self._convert_list_models_response(models_list)
