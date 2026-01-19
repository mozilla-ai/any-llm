from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from any_llm.any_llm import AnyLLM
from any_llm.providers.llama.utils import (
    _convert_tool_call_arguments,
    _extract_content_text,
    _map_stop_reason_to_finish_reason,
    _patch_json_schema,
)
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
)

MISSING_PACKAGES_ERROR = None
try:
    from llama_api_client import AsyncLlamaAPIClient
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from openai.types.chat.chat_completion_message_custom_tool_call import (
        ChatCompletionMessageCustomToolCall,
    )
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
    )

    from any_llm.types.model import Model

    ChatCompletionMessageToolCallType = (
        OpenAIChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall
    )


class LlamaProvider(AnyLLM):
    """Llama provider for accessing multiple LLMs through Llama's API."""

    API_BASE = "https://api.llama.com/compat/v1/"
    ENV_API_KEY_NAME = "LLAMA_API_KEY"
    PROVIDER_NAME = "llama"
    PROVIDER_DOCUMENTATION_URL = "https://www.llama.com/products/llama-api/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncLlamaAPIClient

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Llama API."""
        converted_params = params.model_dump(
            exclude_none=True, exclude={"model_id", "messages", "stream", "reasoning_effort"}
        )

        # Handle tools - apply JSON schema patch for oneOf properties
        if converted_params.get("tools"):
            converted_params["tools"] = [
                _patch_json_schema(tool) if tool.get("type") == "function" else tool
                for tool in converted_params["tools"]
            ]

        # Handle response_format for structured outputs
        if params.response_format:
            if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
                converted_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": params.response_format.model_json_schema(),
                        "strict": True,
                    },
                }

        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any, model_id: str = "") -> ChatCompletion:
        """Convert Llama response to OpenAI ChatCompletion format."""
        completion_message = getattr(response, "completion_message", None)
        choices_out: list[Choice] = []

        if completion_message is not None:
            content = _extract_content_text(getattr(completion_message, "content", None))
            tool_calls = LlamaProvider._build_tool_calls(getattr(completion_message, "tool_calls", None))

            message = ChatCompletionMessage(
                role=getattr(completion_message, "role", "assistant"),
                content=content,
                tool_calls=cast("list[ChatCompletionMessageToolCallType] | None", tool_calls),
            )

            finish_reason = _map_stop_reason_to_finish_reason(getattr(response, "stop_reason", None)) or "stop"
            choices_out.append(Choice(index=0, finish_reason=finish_reason, message=message))

        usage = LlamaProvider._build_usage(getattr(response, "metrics", None))

        return ChatCompletion(
            id=f"chatcmpl-{id(response)}",
            model=model_id,
            created=0,
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )

    @staticmethod
    def _build_tool_calls(
        tool_calls_raw: Any,
    ) -> list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] | None:
        """Build tool calls list from raw Llama tool calls."""
        if not tool_calls_raw:
            return None

        tool_calls_list: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
        for i, tc in enumerate(tool_calls_raw):
            call_id = getattr(tc, "id", None) or getattr(tc, "call_id", f"call_{i}")
            tool_name = getattr(tc, "tool_name", None) or getattr(tc, "name", "")
            arguments = _convert_tool_call_arguments(getattr(tc, "arguments", None))

            tool_calls_list.append(
                ChatCompletionMessageFunctionToolCall(
                    id=call_id,
                    type="function",
                    function=Function(name=tool_name, arguments=arguments),
                )
            )
        return tool_calls_list

    @staticmethod
    def _build_usage(metrics: Any) -> CompletionUsage | None:
        """Build usage from Llama metrics."""
        if not metrics:
            return None

        prompt_tokens = getattr(metrics, "prompt_token_count", 0) or 0
        completion_tokens = getattr(metrics, "completion_token_count", 0) or 0
        return CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Llama streaming chunk to OpenAI ChatCompletionChunk format."""
        delta, finish_reason = LlamaProvider._extract_chunk_delta(response)

        chunk_dict: dict[str, Any] = {
            "id": f"chatcmpl-{id(response)}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": kwargs.get("model", "llama-model"),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}],
            "usage": None,
        }

        return ChatCompletionChunk.model_validate(chunk_dict)

    @staticmethod
    def _extract_chunk_delta(response: Any) -> tuple[dict[str, Any], str | None]:
        """Extract delta and finish_reason from Llama streaming chunk."""
        delta: dict[str, Any] = {}
        finish_reason = None

        event = getattr(response, "event", None)
        if event is None:
            return delta, finish_reason

        event_delta = getattr(event, "delta", None)
        if event_delta is not None:
            text = getattr(event_delta, "text", None)
            if text is not None:
                delta["content"] = text

            role = getattr(event_delta, "role", None)
            if role:
                delta["role"] = role

            tool_call = getattr(event_delta, "tool_call", None)
            if tool_call is not None:
                delta["tool_calls"] = [LlamaProvider._build_streaming_tool_call(tool_call)]

        finish_reason = _map_stop_reason_to_finish_reason(getattr(event, "stop_reason", None))
        return delta, finish_reason

    @staticmethod
    def _build_streaming_tool_call(tool_call: Any) -> dict[str, Any]:
        """Build a streaming tool call dict from Llama tool call object."""
        tool_call_dict: dict[str, Any] = {
            "index": getattr(tool_call, "index", 0) or 0,
            "id": getattr(tool_call, "id", None),
            "type": "function",
        }
        tool_name = getattr(tool_call, "tool_name", None)
        arguments = getattr(tool_call, "arguments", None)
        if tool_name or arguments:
            tool_call_dict["function"] = {
                "name": tool_name,
                "arguments": _convert_tool_call_arguments(arguments),
            }
        return tool_call_dict

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        msg = "Llama does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        msg = "Llama does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        msg = "Llama does not support listing models"
        raise NotImplementedError(msg)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncLlamaAPIClient(api_key=api_key, **kwargs)

    async def _stream_completion_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            yield self._convert_completion_chunk_response(chunk, model=model)

    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        completion_kwargs = self._convert_completion_params(params, **kwargs)

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                params.messages,
                **completion_kwargs,
            )

        response = await self.client.chat.completions.create(
            model=params.model_id,
            messages=params.messages,
            **completion_kwargs,
        )

        return self._convert_completion_response(response, model_id=params.model_id)
