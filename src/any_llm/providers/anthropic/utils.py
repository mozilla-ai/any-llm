import json
from typing import Any

from anthropic.pagination import SyncPage
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    MessageStopEvent,
)
from anthropic.types.model_info import ModelInfo as AnthropicModelInfo

from any_llm.exceptions import UnsupportedParameterError
from any_llm.logging import logger
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Choice,
    CompletionParams,
    CompletionUsage,
    Function,
    Reasoning,
)
from any_llm.types.model import Model

DEFAULT_MAX_TOKENS = 8192
REASONING_EFFORT_TO_THINKING_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 24576}


def _is_tool_call(message: dict[str, Any]) -> bool:
    """Check if the message is a tool call message."""
    return message["role"] == "assistant" and message.get("tool_calls") is not None


def _convert_messages_for_anthropic(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert messages to Anthropic format.

    - Extract messages with `role=system`.
    - Replace `role=tool` with `role=user`, according to examples in https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/.
    """
    system_message = None
    filtered_messages = []

    for n, message in enumerate(messages):
        if message["role"] == "system":
            if system_message is None:
                system_message = message["content"]
            else:
                system_message += "\n" + message["content"]
        else:
            # Handle messages inside agent loop.
            # See https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview#tool-use-examples
            if _is_tool_call(message):
                tool_call = message["tool_calls"][0]
                message = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                    ],
                }
            elif message["role"] == "tool":
                previous_message = messages[n - 1] if n > 0 else None
                tool_use_id = ""
                if previous_message and _is_tool_call(previous_message):
                    tool_use_id = previous_message["tool_calls"][0]["id"]
                message = {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": message["content"]}],
                }
            filtered_messages.append(message)

    return system_message, filtered_messages


def _create_openai_chunk_from_anthropic_chunk(chunk: Any, model_id: str) -> ChatCompletionChunk:
    """Convert Anthropic streaming chunk to OpenAI ChatCompletionChunk format."""
    chunk_dict = {
        "id": f"chatcmpl-{hash(str(chunk))}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model_id,
        "choices": [],
        "usage": None,
    }

    delta: dict[str, Any] = {}
    finish_reason = None

    if isinstance(chunk, ContentBlockStartEvent):
        if chunk.content_block.type == "text":
            delta = {"content": ""}
        elif chunk.content_block.type == "tool_use":
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": chunk.content_block.id,
                        "type": "function",
                        "function": {"name": chunk.content_block.name, "arguments": ""},
                    }
                ]
            }
        elif chunk.content_block.type == "thinking":
            delta = {"reasoning": {"content": ""}}

    elif isinstance(chunk, ContentBlockDeltaEvent):
        if chunk.delta.type == "text_delta":
            delta = {"content": chunk.delta.text}
        elif chunk.delta.type == "input_json_delta":
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": chunk.delta.partial_json},
                    }
                ]
            }
        elif chunk.delta.type == "thinking_delta":
            delta = {"reasoning": {"content": chunk.delta.thinking}}

    elif isinstance(chunk, ContentBlockStopEvent):
        if hasattr(chunk, "content_block") and chunk.content_block.type == "tool_use":
            finish_reason = "tool_calls"
        else:
            finish_reason = None

    elif isinstance(chunk, MessageStopEvent):
        finish_reason = "stop"
        if hasattr(chunk, "message") and chunk.message.usage:
            chunk_dict["usage"] = {
                "prompt_tokens": chunk.message.usage.input_tokens,
                "completion_tokens": chunk.message.usage.output_tokens,
                "total_tokens": chunk.message.usage.input_tokens + chunk.message.usage.output_tokens,
            }

    choice = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
        "logprobs": None,
    }

    chunk_dict["choices"] = [choice]

    return ChatCompletionChunk.model_validate(chunk_dict)


def _convert_response(response: Message) -> ChatCompletion:
    """Convert Anthropic Message to OpenAI ChatCompletion format."""
    finish_reason_raw = response.stop_reason or "end_turn"
    finish_reason_map = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
    finish_reason = finish_reason_map.get(finish_reason_raw, "stop")

    content_parts: list[str] = []
    tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
    reasoning_content: str | None = None
    for content_block in response.content:
        if content_block.type == "text":
            content_parts.append(content_block.text)
        elif content_block.type == "tool_use":
            tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=content_block.id,
                    type="function",
                    function=Function(
                        name=content_block.name,
                        arguments=json.dumps(content_block.input),
                    ),
                )
            )
        elif content_block.type == "thinking":
            if reasoning_content is None:
                reasoning_content = content_block.thinking
            else:
                reasoning_content += content_block.thinking
        else:
            msg = f"Unsupported content block type: {content_block.type}"
            raise ValueError(msg)

    message = ChatCompletionMessage(
        role="assistant",
        content="".join(content_parts),
        reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
        tool_calls=tool_calls or None,
    )

    usage = CompletionUsage(
        completion_tokens=response.usage.output_tokens,
        prompt_tokens=response.usage.input_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )

    from typing import Literal, cast

    choice = Choice(
        index=0,
        finish_reason=cast(
            "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']", finish_reason or "stop"
        ),
        message=message,
    )

    created_ts = int(response.created_at.timestamp()) if hasattr(response, "created_at") else 0

    return ChatCompletion(
        id=response.id,
        model=response.model,
        created=created_ts,
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


def _convert_tool_spec(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool specification to Anthropic format."""
    generic_tools = []

    for tool in openai_tools:
        if tool.get("type") != "function":
            continue

        function = tool["function"]
        generic_tool = {
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        }
        generic_tools.append(generic_tool)

    anthropic_tools = []
    for tool in generic_tools:
        anthropic_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": {
                "type": "object",
                "properties": tool["parameters"]["properties"],
                "required": tool["parameters"].get("required", []),
            },
        }
        anthropic_tools.append(anthropic_tool)

    return anthropic_tools


def _convert_tool_choice(params: CompletionParams) -> dict[str, Any]:
    parallel_tool_calls = params.parallel_tool_calls
    if parallel_tool_calls is None:
        parallel_tool_calls = True
    tool_choice = params.tool_choice or "any"
    if tool_choice == "required":
        tool_choice = "any"
    return {"type": tool_choice, "disable_parallel_tool_use": not parallel_tool_calls}


def _convert_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
    """Convert CompletionParams to kwargs for Anthropic API."""
    provider_name: str = kwargs.pop("provider_name")
    result_kwargs: dict[str, Any] = kwargs.copy()

    if params.response_format:
        msg = "response_format"
        raise UnsupportedParameterError(
            msg,
            provider_name,
            "Check the following links:\n- https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency\n- https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview#json-mode",
        )
    if params.max_tokens is None:
        logger.warning(f"max_tokens is required for Anthropic, setting to {DEFAULT_MAX_TOKENS}")
        params.max_tokens = DEFAULT_MAX_TOKENS

    if params.tools:
        params.tools = _convert_tool_spec(params.tools)

    if params.tool_choice or params.parallel_tool_calls:
        params.tool_choice = _convert_tool_choice(params)

    if params.reasoning_effort is None:
        result_kwargs["thinking"] = {"type": "disabled"}
    # in "auto" mode, we just don't pass `thinking`
    elif params.reasoning_effort != "auto":
        result_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": REASONING_EFFORT_TO_THINKING_BUDGETS[params.reasoning_effort],
        }

    result_kwargs.update(
        params.model_dump(
            exclude_none=True,
            exclude={"model_id", "messages", "reasoning_effort", "response_format", "parallel_tool_calls"},
        )
    )
    result_kwargs["model"] = params.model_id

    system_message, filtered_messages = _convert_messages_for_anthropic(params.messages)
    if system_message:
        result_kwargs["system"] = system_message
    result_kwargs["messages"] = filtered_messages

    return result_kwargs


def _convert_models_list(models_list: SyncPage[AnthropicModelInfo]) -> list[Model]:
    """Convert Anthropic models list to OpenAI format."""
    return [
        Model(id=model.id, object="model", created=int(model.created_at.timestamp()), owned_by="anthropic")
        for model in models_list
    ]
