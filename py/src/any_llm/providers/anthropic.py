import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal, cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    RawContentBlockDeltaEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolResultBlockParam,
    ToolUseBlock,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.chat.chat_completion_message_tool_call import Function

from any_llm.any_llm import AnyLLM
from any_llm.errors import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceededError,
    InvalidRequestError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from any_llm.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionParams,
    CompletionUsage,
)

FinishReasonType = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]

PROVIDER_NAME = "Anthropic"
ENV_VAR_NAME = "ANTHROPIC_API_KEY"
DEFAULT_MAX_TOKENS = 8192

# Reasoning effort to budget_tokens mapping
REASONING_EFFORT_MAP: dict[str, int | dict[str, str]] = {
    "none": {"type": "disabled"},
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 24576,
    "xhigh": 32768,
}


class AnthropicProvider(AnyLLM):
    """Anthropic provider implementation using the official Anthropic SDK."""

    _client: AsyncAnthropic

    def _init_client(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic async client.

        Args:
            api_key: API key for Anthropic. If None, checks ANTHROPIC_API_KEY env var.
            api_base: Base URL for the API. If None, uses Anthropic default.
            **kwargs: Additional arguments passed to AsyncAnthropic.

        Raises:
            MissingApiKeyError: If no API key is provided or found in environment.

        """
        resolved_key = api_key or os.environ.get(ENV_VAR_NAME)
        if not resolved_key:
            raise MissingApiKeyError(PROVIDER_NAME, ENV_VAR_NAME)

        self._client = AsyncAnthropic(
            api_key=resolved_key,
            base_url=api_base,
            **kwargs,
        )

    async def acompletion(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Perform a chat completion request using Anthropic.

        Args:
            model: Model ID to use for the completion.
            messages: List of messages comprising the conversation.
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0).
            logit_bias: Map of token IDs to bias values.
            logprobs: Whether to return log probabilities.
            max_completion_tokens: Maximum tokens to generate.
            max_tokens: Legacy parameter for maximum tokens.
            n: Number of completions to generate.
            parallel_tool_calls: Enable parallel function calling.
            presence_penalty: Penalize new topics (-2.0 to 2.0).
            response_format: Format specification for structured outputs.
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            stream_options: Options for streaming.
            temperature: Sampling temperature (0 to 2).
            tool_choice: Controls tool calling behavior.
            tools: List of tools the model may call.
            top_logprobs: Number of top tokens to return.
            top_p: Nucleus sampling threshold.
            user: Unique user identifier.
            **kwargs: Additional arguments passed to the API.

        Returns:
            ChatCompletion response object.

        """
        if response_format is not None:
            msg = "response_format is not supported by Anthropic. Use tool-based JSON extraction instead."
            raise InvalidRequestError(message=msg, provider_name=PROVIDER_NAME)

        params = self._convert_completion_params(
            CompletionParams(
                model=model,
                messages=list(messages),
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                n=n,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                response_format=response_format,
                seed=seed,
                stop=stop,
                stream_options=stream_options,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=list(tools) if tools else None,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
            ),
            **kwargs,
        )

        try:
            response = await self._client.messages.create(**params)
            return self._convert_completion_response(response, model=model)
        except anthropic.APIError as e:
            raise self._map_anthropic_error(e) from e

    async def acompletion_stream(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Perform a streaming chat completion request using Anthropic.

        Args:
            model: Model ID to use for the completion.
            messages: List of messages comprising the conversation.
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0).
            logit_bias: Map of token IDs to bias values.
            logprobs: Whether to return log probabilities.
            max_completion_tokens: Maximum tokens to generate.
            max_tokens: Legacy parameter for maximum tokens.
            n: Number of completions to generate.
            parallel_tool_calls: Enable parallel function calling.
            presence_penalty: Penalize new topics (-2.0 to 2.0).
            response_format: Format specification for structured outputs.
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            stream_options: Options for streaming.
            temperature: Sampling temperature (0 to 2).
            tool_choice: Controls tool calling behavior.
            tools: List of tools the model may call.
            top_logprobs: Number of top tokens to return.
            top_p: Nucleus sampling threshold.
            user: Unique user identifier.
            **kwargs: Additional arguments passed to the API.

        Yields:
            ChatCompletionChunk objects.

        """
        if response_format is not None:
            msg = "response_format is not supported by Anthropic. Use tool-based JSON extraction instead."
            raise InvalidRequestError(message=msg, provider_name=PROVIDER_NAME)

        params = self._convert_completion_params(
            CompletionParams(
                model=model,
                messages=list(messages),
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                n=n,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                response_format=response_format,
                seed=seed,
                stop=stop,
                stream=True,
                stream_options=stream_options,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=list(tools) if tools else None,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
            ),
            **kwargs,
        )

        try:
            # Remove stream from params since we pass it directly to create()
            if "stream" in params:
                del params["stream"]
            stream = await self._client.messages.create(**params, stream=True)

            # State tracking for streaming
            message_id: str = ""
            current_content_blocks: list[dict[str, Any]] = []
            current_block_index: int | None = None
            created_at: int = int(time.time())

            async for event in stream:
                chunk = self._convert_stream_event(
                    event,
                    model=model,
                    message_id=message_id,
                    current_content_blocks=current_content_blocks,
                    current_block_index=current_block_index,
                    created_at=created_at,
                )
                if chunk:
                    # Update state based on event type
                    if isinstance(event, MessageStartEvent):
                        message_id = event.message.id
                        created_at = int(time.time())
                    elif isinstance(event, ContentBlockStartEvent):
                        current_block_index = event.index
                        content_block = event.content_block
                        if isinstance(content_block, ToolUseBlock):
                            current_content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": content_block.id,
                                    "name": content_block.name,
                                    "arguments": "",
                                }
                            )
                        elif isinstance(content_block, TextBlock):
                            current_content_blocks.append(
                                {
                                    "type": "text",
                                    "text": "",
                                }
                            )
                        elif isinstance(content_block, ThinkingBlock):
                            current_content_blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": "",
                                }
                            )
                        else:
                            # Handle other block types by type attribute
                            block_type = getattr(content_block, "type", "")
                            if block_type == "tool_use":
                                current_content_blocks.append(
                                    {
                                        "type": "tool_use",
                                        "id": getattr(content_block, "id", ""),
                                        "name": getattr(content_block, "name", ""),
                                        "arguments": "",
                                    }
                                )
                            elif block_type == "text":
                                current_content_blocks.append(
                                    {
                                        "type": "text",
                                        "text": "",
                                    }
                                )
                            elif block_type == "thinking":
                                current_content_blocks.append(
                                    {
                                        "type": "thinking",
                                        "thinking": "",
                                    }
                                )
                    elif isinstance(event, RawContentBlockDeltaEvent):
                        if current_block_index is not None and current_block_index < len(current_content_blocks):
                            block = current_content_blocks[current_block_index]
                            if hasattr(event.delta, "text"):
                                block["text"] = block.get("text", "") + event.delta.text
                            elif hasattr(event.delta, "partial_json"):
                                block["arguments"] = block.get("arguments", "") + event.delta.partial_json
                            elif hasattr(event.delta, "thinking"):
                                block["thinking"] = block.get("thinking", "") + event.delta.thinking
                    yield chunk
        except anthropic.APIError as e:
            raise self._map_anthropic_error(e) from e

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert completion parameters to Anthropic API format.

        Args:
            params: The completion parameters.
            **kwargs: Additional arguments to include.

        Returns:
            Anthropic API-compatible parameters dictionary.

        """
        api_params: dict[str, Any] = {}
        api_params["model"] = params.model

        # Convert messages (extract system messages, convert tool messages)
        messages_list = list(params.messages)
        system_messages, anthropic_messages = AnthropicProvider._convert_messages(messages_list)

        if system_messages:
            api_params["system"] = "\n".join(system_messages)

        api_params["messages"] = anthropic_messages

        # max_tokens is required for Anthropic
        max_tokens = params.max_tokens or params.max_completion_tokens or DEFAULT_MAX_TOKENS
        api_params["max_tokens"] = max_tokens

        # Optional parameters
        if params.temperature is not None:
            api_params["temperature"] = params.temperature

        if params.top_p is not None:
            api_params["top_p"] = params.top_p

        if params.stop is not None:
            api_params["stop_sequences"] = params.stop if isinstance(params.stop, list) else [params.stop]

        # Convert tools
        if params.tools:
            api_params["tools"] = [AnthropicProvider._convert_tool(tool) for tool in params.tools]

        # Convert tool_choice
        if params.tool_choice is not None:
            api_params["tool_choice"] = AnthropicProvider._convert_tool_choice(params.tool_choice)

        # Convert parallel_tool_calls (inverted logic)
        if params.parallel_tool_calls is not None:
            # OpenAI's parallel_tool_calls: true -> Anthropic's disable_parallel_tool_use: false
            api_params["tool_choice"] = api_params.get("tool_choice", {"type": "auto"})
            if isinstance(api_params["tool_choice"], dict):
                api_params["tool_choice"]["disable_parallel_tool_use"] = not params.parallel_tool_calls

        # Handle reasoning effort from kwargs
        if "reasoning_effort" in kwargs:
            effort = kwargs.pop("reasoning_effort")
            if effort in REASONING_EFFORT_MAP:
                budget = REASONING_EFFORT_MAP[effort]
                if isinstance(budget, dict):
                    api_params["thinking"] = budget
                else:
                    api_params["thinking"] = {"type": "enabled", "budget_tokens": budget}

        # Stream handling
        if params.stream:
            api_params["stream"] = True

        # Add any remaining kwargs
        api_params.update(kwargs)

        return api_params

    @staticmethod
    def _convert_messages(messages: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
        """Convert OpenAI-style messages to Anthropic format.

        Args:
            messages: List of OpenAI-style messages.

        Returns:
            Tuple of (system_messages, anthropic_messages).

        """
        system_messages: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        # Group consecutive tool results
        pending_tool_results: list[ToolResultBlockParam] = []

        for msg in messages:
            # Convert Pydantic models to dicts if needed
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()

            role = msg.get("role")
            content = msg.get("content")

            # Handle system messages
            if role == "system":
                if isinstance(content, str):
                    system_messages.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_messages.append(part.get("text", ""))
                        elif isinstance(part, str):
                            system_messages.append(part)
                continue

            # Handle tool messages (convert to user message with tool_result)
            if role == "tool":
                tool_result: ToolResultBlockParam = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else json.dumps(content),
                }
                pending_tool_results.append(tool_result)
                continue

            # Flush pending tool results before adding non-tool message
            if pending_tool_results:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": list(pending_tool_results),
                    }
                )
                pending_tool_results = []

            # Handle assistant messages
            if role == "assistant":
                anthropic_msg = AnthropicProvider._convert_assistant_message(msg)
                anthropic_messages.append(anthropic_msg)
                continue

            # Handle user messages
            if role == "user":
                anthropic_msg = AnthropicProvider._convert_user_message(msg)
                anthropic_messages.append(anthropic_msg)
                continue

        # Flush any remaining tool results
        if pending_tool_results:
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": list(pending_tool_results),
                }
            )

        return system_messages, anthropic_messages

    @staticmethod
    def _convert_assistant_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenAI assistant message to Anthropic format.

        Args:
            msg: OpenAI-style assistant message.

        Returns:
            Anthropic-style assistant message.

        """
        content_blocks: list[dict[str, Any]] = []

        # Handle text content
        text_content = msg.get("content")
        if text_content:
            if isinstance(text_content, str):
                content_blocks.append({"type": "text", "text": text_content})
            elif isinstance(text_content, list):
                for part in text_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_blocks.append({"type": "text", "text": part.get("text", "")})

        # Handle tool calls (convert to tool_use blocks)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                if hasattr(tc, "model_dump"):
                    tc = tc.model_dump()

                func = tc.get("function", {})
                arguments = func.get("arguments", "{}")

                # Parse arguments if it's a string
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": func.get("name", ""),
                        "input": arguments,
                    }
                )

        return {
            "role": "assistant",
            "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
        }

    @staticmethod
    def _convert_user_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenAI user message to Anthropic format.

        Args:
            msg: OpenAI-style user message.

        Returns:
            Anthropic-style user message.

        """
        content = msg.get("content")

        if isinstance(content, str):
            return {"role": "user", "content": content}

        # Handle multimodal content
        content_blocks: list[dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    content_blocks.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        content_blocks.append({"type": "text", "text": part.get("text", "")})
                    elif part_type == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                        content_blocks.append(AnthropicProvider._convert_image_url(url))

        return {"role": "user", "content": content_blocks}

    @staticmethod
    def _convert_image_url(url: str) -> dict[str, Any]:
        """Convert an image URL to Anthropic format.

        Args:
            url: Image URL (can be regular URL or data URL).

        Returns:
            Anthropic-style image content block.

        """
        # Handle base64 data URLs
        if url.startswith("data:"):
            # Parse data:<media_type>;base64,<data>
            try:
                prefix, data = url.split(",", 1)
                media_type = prefix.split(":")[1].split(";")[0]
            except (ValueError, IndexError):
                pass
            else:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }

        # Regular URL
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    @staticmethod
    def _convert_tool(tool: ChatCompletionToolParam | dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI tool spec to Anthropic format.

        Args:
            tool: OpenAI-style tool specification.

        Returns:
            Anthropic-style tool specification.

        """
        tool_dict: dict[str, Any]
        if hasattr(tool, "model_dump"):
            tool_dict = tool.model_dump()  # type: ignore[union-attr]
        else:
            tool_dict = tool  # type: ignore[assignment]

        func = tool_dict.get("function", {})
        return {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}, "required": []}),
        }

    @staticmethod
    def _convert_tool_choice(tool_choice: str | dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI tool_choice to Anthropic format.

        Args:
            tool_choice: OpenAI-style tool choice.

        Returns:
            Anthropic-style tool choice.

        """
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                return {"type": "any"}
            if tool_choice == "auto":
                return {"type": "auto"}
            if tool_choice == "none":
                return {"type": "auto"}  # Anthropic doesn't have "none", use auto
            return {"type": "auto"}

        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                func = tool_choice.get("function", {})
                return {"type": "tool", "name": func.get("name", "")}

        return {"type": "auto"}

    @staticmethod
    def _convert_completion_response(response: Message, **kwargs: Any) -> ChatCompletion:
        """Convert Anthropic response to ChatCompletion format.

        Args:
            response: The Anthropic API response.
            **kwargs: Additional arguments (e.g., model).

        Returns:
            ChatCompletion object.

        """
        model = kwargs.get("model", response.model)

        # Convert stop reason
        finish_reason = AnthropicProvider._convert_stop_reason(response.stop_reason)

        # Process content blocks
        content_text = ""
        tool_calls: list[ChatCompletionMessageToolCall] = []

        for block in response.content:
            if isinstance(block, TextBlock):
                content_text += block.text
            elif isinstance(block, ToolUseBlock):
                tool_call = ChatCompletionMessageToolCall(
                    id=block.id,
                    type="function",
                    function=Function(
                        name=block.name,
                        arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                    ),
                )
                tool_calls.append(tool_call)
            # Note: ThinkingBlock content is not included in OpenAI-compatible format

        # Build message
        message = ChatCompletionMessage(
            role="assistant",
            content=content_text if content_text else None,
            tool_calls=tool_calls if tool_calls else None,  # type: ignore[arg-type]
        )

        # Build usage
        usage = CompletionUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        # Build choice
        choice = Choice(
            index=0,
            message=message,
            finish_reason=cast("FinishReasonType", finish_reason),
            logprobs=None,
        )

        return ChatCompletion(
            id=response.id,
            choices=[choice],
            created=int(time.time()),
            model=model,
            object="chat.completion",
            usage=usage,
        )

    @staticmethod
    def _convert_stop_reason(stop_reason: str | None) -> str:
        """Convert Anthropic stop reason to OpenAI format.

        Args:
            stop_reason: Anthropic stop reason.

        Returns:
            OpenAI-style finish reason.

        """
        if stop_reason == "end_turn":
            return "stop"
        if stop_reason == "max_tokens":
            return "length"
        if stop_reason == "tool_use":
            return "tool_calls"
        if stop_reason == "stop_sequence":
            return "stop"
        return "stop"

    @staticmethod
    def _convert_stream_event(
        event: Any,
        model: str,
        message_id: str,
        current_content_blocks: list[dict[str, Any]],
        current_block_index: int | None,
        created_at: int,
    ) -> ChatCompletionChunk | None:
        """Convert Anthropic streaming event to ChatCompletionChunk.

        Args:
            event: Anthropic streaming event.
            model: Model name.
            message_id: Current message ID.
            current_content_blocks: List of current content blocks being built.
            current_block_index: Index of current block being processed.
            created_at: Timestamp of message creation.

        Returns:
            ChatCompletionChunk or None if event doesn't produce a chunk.

        """
        chunk_id = message_id or f"chatcmpl-{uuid.uuid4().hex[:8]}"

        if isinstance(event, MessageStartEvent):
            # Initial chunk with role
            delta = ChoiceDelta(role="assistant", content=None)
            choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
            return ChatCompletionChunk(
                id=event.message.id,
                choices=[choice],
                created=created_at,
                model=model,
                object="chat.completion.chunk",
            )

        if isinstance(event, ContentBlockStartEvent):
            block = event.content_block
            if isinstance(block, TextBlock):
                delta = ChoiceDelta(content="")
                choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
                return ChatCompletionChunk(
                    id=chunk_id,
                    choices=[choice],
                    created=created_at,
                    model=model,
                    object="chat.completion.chunk",
                )
            if isinstance(block, ToolUseBlock):
                # Start of tool call
                tool_call = ChoiceDeltaToolCall(
                    index=event.index,
                    id=block.id,
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name=block.name, arguments=""),
                )
                delta = ChoiceDelta(tool_calls=[tool_call])
                choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
                return ChatCompletionChunk(
                    id=chunk_id,
                    choices=[choice],
                    created=created_at,
                    model=model,
                    object="chat.completion.chunk",
                )
            # Handle other block types by checking the type attribute
            block_type = getattr(block, "type", "")
            if block_type == "text":
                delta = ChoiceDelta(content="")
                choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
                return ChatCompletionChunk(
                    id=chunk_id,
                    choices=[choice],
                    created=created_at,
                    model=model,
                    object="chat.completion.chunk",
                )
            if block_type == "tool_use":
                tool_call = ChoiceDeltaToolCall(
                    index=event.index,
                    id=getattr(block, "id", ""),
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name=getattr(block, "name", ""), arguments=""),
                )
                delta = ChoiceDelta(tool_calls=[tool_call])
                choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
                return ChatCompletionChunk(
                    id=chunk_id,
                    choices=[choice],
                    created=created_at,
                    model=model,
                    object="chat.completion.chunk",
                )

        if isinstance(event, RawContentBlockDeltaEvent):
            delta_obj = event.delta
            if isinstance(delta_obj, TextDelta):
                delta = ChoiceDelta(content=delta_obj.text)
                choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
                return ChatCompletionChunk(
                    id=chunk_id,
                    choices=[choice],
                    created=created_at,
                    model=model,
                    object="chat.completion.chunk",
                )
            # Handle other delta types by checking attributes
            delta_type = getattr(delta_obj, "type", "")
            if delta_type == "text_delta" and hasattr(delta_obj, "text"):
                delta = ChoiceDelta(content=getattr(delta_obj, "text", ""))
                choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
                return ChatCompletionChunk(
                    id=chunk_id,
                    choices=[choice],
                    created=created_at,
                    model=model,
                    object="chat.completion.chunk",
                )
            if delta_type == "input_json_delta":
                # Tool argument delta
                tool_call = ChoiceDeltaToolCall(
                    index=current_block_index or 0,
                    function=ChoiceDeltaToolCallFunction(arguments=getattr(delta_obj, "partial_json", "")),
                )
                delta = ChoiceDelta(tool_calls=[tool_call])
                choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
                return ChatCompletionChunk(
                    id=chunk_id,
                    choices=[choice],
                    created=created_at,
                    model=model,
                    object="chat.completion.chunk",
                )

        if isinstance(event, ContentBlockStopEvent):
            # Check if this was a tool_use block
            if current_block_index is not None and current_block_index < len(current_content_blocks):
                block_dict = current_content_blocks[current_block_index]
                if block_dict.get("type") == "tool_use":
                    delta = ChoiceDelta()
                    choice = ChunkChoice(index=0, delta=delta, finish_reason="tool_calls")
                    return ChatCompletionChunk(
                        id=chunk_id,
                        choices=[choice],
                        created=created_at,
                        model=model,
                        object="chat.completion.chunk",
                    )

        if isinstance(event, MessageStopEvent):
            delta = ChoiceDelta()
            choice = ChunkChoice(index=0, delta=delta, finish_reason="stop")
            return ChatCompletionChunk(
                id=chunk_id,
                choices=[choice],
                created=created_at,
                model=model,
                object="chat.completion.chunk",
            )

        if isinstance(event, MessageDeltaEvent):
            # Final message delta with usage
            finish_reason = AnthropicProvider._convert_stop_reason(event.delta.stop_reason)
            delta = ChoiceDelta()
            choice = ChunkChoice(index=0, delta=delta, finish_reason=cast("FinishReasonType", finish_reason))
            usage = None
            if hasattr(event, "usage") and event.usage:
                usage = CompletionUsage(
                    prompt_tokens=getattr(event.usage, "input_tokens", 0),
                    completion_tokens=event.usage.output_tokens,
                    total_tokens=getattr(event.usage, "input_tokens", 0) + event.usage.output_tokens,
                )
            return ChatCompletionChunk(
                id=chunk_id,
                choices=[choice],
                created=created_at,
                model=model,
                object="chat.completion.chunk",
                usage=usage,
            )

        return None

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Anthropic streaming chunk to ChatCompletionChunk format.

        This method is called for individual chunks during streaming.

        Args:
            response: The Anthropic streaming event.
            **kwargs: Additional arguments.

        Returns:
            ChatCompletionChunk object.

        """
        # This is handled by _convert_stream_event in acompletion_stream
        # Keeping this for interface compatibility
        model = kwargs.get("model", "")
        delta = ChoiceDelta(content="")
        choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
        return ChatCompletionChunk(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            choices=[choice],
            created=int(time.time()),
            model=model,
            object="chat.completion.chunk",
        )

    @staticmethod
    def _map_anthropic_error(error: anthropic.APIError) -> AnyLLMError:
        """Map Anthropic SDK errors to AnyLLM error types.

        Args:
            error: The Anthropic API error.

        Returns:
            Corresponding AnyLLMError subclass.

        """
        error_message = str(error)

        if isinstance(error, anthropic.AuthenticationError):
            return AuthenticationError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        if isinstance(error, anthropic.RateLimitError):
            return RateLimitError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        if isinstance(error, anthropic.BadRequestError):
            lower_message = error_message.lower()
            if "context length" in lower_message or "maximum context" in lower_message or "too long" in lower_message:
                return ContextLengthExceededError(
                    message=error_message,
                    original_exception=error,
                    provider_name=PROVIDER_NAME,
                )
            if "content filter" in lower_message or "content_filter" in lower_message or "safety" in lower_message:
                return ContentFilterError(
                    message=error_message,
                    original_exception=error,
                    provider_name=PROVIDER_NAME,
                )
            return InvalidRequestError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        if isinstance(error, anthropic.NotFoundError):
            return ModelNotFoundError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        if isinstance(error, anthropic.InternalServerError):
            return ProviderError(
                message=error_message,
                original_exception=error,
                provider_name=PROVIDER_NAME,
            )

        # Default to generic provider error
        return ProviderError(
            message=error_message,
            original_exception=error,
            provider_name=PROVIDER_NAME,
        )
