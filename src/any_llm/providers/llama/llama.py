from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

from llama_api_client.types.system_message_param import SystemMessageParam
from llama_api_client.types.tool_response_message_param import ToolResponseMessageParam
from llama_api_client.types.user_message_param import UserMessageParam
from pydantic import BaseModel

from any_llm.any_llm import AnyLLM
from any_llm.providers.llama.utils import _patch_json_schema
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
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
    from collections.abc import AsyncIterator, Iterable, Sequence

    from llama_api_client.types import (
        CreateChatCompletionResponse,
        CreateChatCompletionResponseStreamChunk,
        MessageParam,
    )
    from llama_api_client.types.completion_message_param import CompletionMessageParam
    from llama_api_client.types.create_chat_completion_response import Metric
    from llama_api_client.types.create_chat_completion_response_stream_chunk import EventDeltaToolCallDelta
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
    def _convert_messages(messages: list[dict[str, Any]]) -> Iterable[MessageParam]:
        """Convert message dictionaries to Llama SDK MessageParam types."""
        result: list[MessageParam] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                result.append(UserMessageParam(role="user", content=content))
            elif role == "system":
                result.append(SystemMessageParam(role="system", content=content))
            elif role == "assistant":
                assistant_msg: CompletionMessageParam = {"role": "assistant", "content": content}
                if msg.get("tool_calls"):
                    assistant_msg["tool_calls"] = msg["tool_calls"]
                if msg.get("stop_reason"):
                    assistant_msg["stop_reason"] = msg["stop_reason"]
                result.append(assistant_msg)
            elif role == "tool":
                result.append(
                    ToolResponseMessageParam(
                        role="tool",
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )

        return result

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
    def _convert_completion_response(response: CreateChatCompletionResponse, model_id: str = "") -> ChatCompletion:
        """Convert Llama response to OpenAI ChatCompletion format."""
        cm = response.completion_message
        choices_out: list[Choice] = []

        if cm is not None:
            # Extract content (str | MessageTextContentItem | None)
            content = cm.content if isinstance(cm.content, str) else (cm.content.text if cm.content else None)

            # Build tool calls using SDK's ToolCall structure
            tool_calls = None
            if cm.tool_calls:
                tool_calls = [
                    ChatCompletionMessageFunctionToolCall(
                        id=tc.id,
                        type="function",
                        function=Function(name=tc.function.name, arguments=tc.function.arguments),
                    )
                    for tc in cm.tool_calls
                ]

            message = ChatCompletionMessage(
                role=cm.role or "assistant",
                content=content,
                tool_calls=cast("list[ChatCompletionMessageToolCallType] | None", tool_calls),
            )

            # stop_reason is already OpenAI-compatible
            finish_reason = cast(
                "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
                cm.stop_reason or "stop",
            )
            choices_out.append(Choice(index=0, finish_reason=finish_reason, message=message))

        usage = LlamaProvider._build_usage(response.metrics)

        return ChatCompletion(
            id=response.id or f"chatcmpl-{id(response)}",
            model=model_id,
            created=0,
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )

    @staticmethod
    def _build_usage(metrics: list[Metric] | None) -> CompletionUsage | None:
        """Build usage from Llama metrics list."""
        if not metrics:
            return None

        prompt_tokens = 0
        completion_tokens = 0
        for m in metrics:
            if m.metric == "prompt_token_count":
                prompt_tokens = int(m.value)
            elif m.metric == "completion_token_count":
                completion_tokens = int(m.value)

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
    def _extract_chunk_delta(
        response: CreateChatCompletionResponseStreamChunk,
    ) -> tuple[dict[str, Any], str | None]:
        """Extract delta and finish_reason from Llama streaming chunk."""
        delta: dict[str, Any] = {}
        event = response.event

        if event is None:
            return delta, None

        event_delta = event.delta
        if event_delta is not None:
            # Check for text content
            if hasattr(event_delta, "text") and event_delta.text is not None:
                delta["content"] = event_delta.text

            # Check for role
            if hasattr(event_delta, "role") and event_delta.role:
                delta["role"] = event_delta.role

            # Check for tool call
            if hasattr(event_delta, "tool_call") and event_delta.tool_call is not None:
                delta["tool_calls"] = [LlamaProvider._build_streaming_tool_call(event_delta.tool_call)]

        # stop_reason is already OpenAI-compatible
        return delta, event.stop_reason

    @staticmethod
    def _build_streaming_tool_call(tool_call: EventDeltaToolCallDelta) -> dict[str, Any]:
        """Build a streaming tool call dict from Llama tool call object."""
        result: dict[str, Any] = {
            "index": 0,
            "id": tool_call.id,
            "type": "function",
        }
        if tool_call.function and (tool_call.function.name or tool_call.function.arguments):
            result["function"] = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments or "",
            }
        return result

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
        messages: Iterable[MessageParam],
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
        converted_messages = self._convert_messages(params.messages)

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                converted_messages,
                **completion_kwargs,
            )

        response = await self.client.chat.completions.create(
            model=params.model_id,
            messages=converted_messages,
            **completion_kwargs,
        )

        return self._convert_completion_response(response, model_id=params.model_id)
