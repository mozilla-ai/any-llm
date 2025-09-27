from abc import abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from any_llm.config import ClientConfig
from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import Provider
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
from any_llm.types.model import Model

MISSING_PACKAGES_ERROR = None
try:
    from google import genai
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
    from google import genai

REASONING_EFFORT_TO_THINKING_BUDGETS = {"minimal": 256, "low": 1024, "medium": 8192, "high": 24576}


class GoogleProvider(Provider):
    """Base Google Provider class with common functionality for Gemini and Vertex AI."""

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False  # TODO: Add image support https://github.com/mozilla-ai/any-llm/issues/415
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Google API."""
        base_kwargs = params.model_dump(
            exclude_none=True,
            exclude={
                "model_id",
                "messages",
                "response_format",
                "stream",
                "tools",
                "tool_choice",
                "reasoning_effort",
                "max_tokens",
            },
        )
        if params.max_tokens is not None:
            base_kwargs["max_output_tokens"] = params.max_tokens
        base_kwargs.update(kwargs)
        return base_kwargs

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

    @abstractmethod
    def _get_client(self, config: ClientConfig) -> "genai.Client":
        """Get the appropriate client for this provider implementation."""

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        client = self._get_client(self.config)
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        result = await client.aio.models.embed_content(
            model=model,
            **embedding_kwargs,
        )

        response_data = {"model": model, "result": result}
        return self._convert_embedding_response(response_data)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.stream and params.response_format is not None:
            error_message = "stream and response_format"
            raise UnsupportedParameterError(error_message, self.PROVIDER_NAME)

        if params.parallel_tool_calls is not None:
            error_message = "parallel_tool_calls"
            raise UnsupportedParameterError(error_message, self.PROVIDER_NAME)
        tools = None
        if params.tools is not None:
            tools = _convert_tool_spec(params.tools)
            kwargs["tools"] = tools

        if isinstance(params.tool_choice, str):
            kwargs["tool_config"] = _convert_tool_choice(params.tool_choice)

        if params.reasoning_effort is None:
            kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=False)
        elif params.reasoning_effort != "auto":
            kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True, thinking_budget=REASONING_EFFORT_TO_THINKING_BUDGETS[params.reasoning_effort]
            )

        stream = bool(params.stream)
        response_format = params.response_format
        base_kwargs = self._convert_completion_params(params, **kwargs)
        generation_config = types.GenerateContentConfig(**base_kwargs)
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = response_format

        formatted_messages, system_instruction = _convert_messages(params.messages)
        if system_instruction:
            generation_config.system_instruction = system_instruction

        client = self._get_client(self.config)
        if stream:
            response_stream = await client.aio.models.generate_content_stream(
                model=params.model_id,
                contents=formatted_messages,  # type: ignore[arg-type]
                config=generation_config,
            )

            async def _stream() -> AsyncIterator[ChatCompletionChunk]:
                async for chunk in response_stream:
                    yield self._convert_completion_chunk_response(chunk)

            return _stream()

        response: types.GenerateContentResponse = await client.aio.models.generate_content(
            model=params.model_id,
            contents=formatted_messages,  # type: ignore[arg-type]
            config=generation_config,
        )

        response_dict = _convert_response_to_response_dict(response)
        return self._convert_completion_response((response_dict, params.model_id))

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """Fetch available models from the /v1/models endpoint."""
        client = self._get_client(self.config)
        models_list = client.models.list(**kwargs)
        return self._convert_list_models_response(models_list)
