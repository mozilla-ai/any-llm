import json
from typing import TYPE_CHECKING, Any, cast

from anthropic import transform_schema
from openresponses_types import ResponseResource
from openresponses_types.types import (
    AssistantMessageItemParam,
    DeveloperMessageItemParam,
    FunctionCallItemParam,
    FunctionCallOutputItemParam,
    ItemReferenceParam,
    Reasoning as ResponseReasoningConfig,
    SystemMessageItemParam,
    UserMessageItemParam,
)
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
    PromptTokensDetails,
    Reasoning,
)
from any_llm.types.model import Model
from any_llm.types.request import RequestInput, RequestParams, RequestReasoningItemParam, normalize_request_input
from any_llm.utils.request_output import (
    make_function_call_item,
    make_response_resource,
    make_text_message,
    make_usage,
    make_reasoning_item,
)
from any_llm.utils.request_state import decode_request_state, encode_request_state
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_custom_tool_call import (
        ChatCompletionMessageCustomToolCall,
    )
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
    )

    ChatCompletionMessageToolCallType = (
        OpenAIChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall
    )

DEFAULT_MAX_TOKENS = 8192
REASONING_EFFORT_TO_ANTHROPIC_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
}


def _is_tool_call(message: dict[str, Any]) -> bool:
    """Check if the message is a tool call message."""
    return message["role"] == "assistant" and message.get("tool_calls") is not None


def _convert_content_for_anthropic(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert content blocks from OpenAI format to Anthropic format.
    - Parse the "content" field block by block
    - Convert image_url blocks to Anthropic image format
    - Convert file blocks (PDFs) to Anthropic document format
    """
    converted_content = []
    for block in content:
        if block.get("type") == "image_url":
            converted_block: dict[str, Any] = {"type": "image"}
            url = block.get("image_url", {}).get("url", "")
            if url[:5] == "data:":
                mime_part = url[5:]
                semi_idx = mime_part.find(";")
                media_type = mime_part[:semi_idx] if semi_idx != -1 else mime_part
                converted_block["source"] = {
                    "type": "base64",
                    "media_type": media_type,
                    "data": url.split("base64,")[1],
                }
            else:
                converted_block["source"] = {"type": "url", "url": url}
            converted_content.append(converted_block)
        elif block.get("type") == "file":
            file_data = block.get("file", {}).get("file_data", "")
            converted_block = {"type": "document"}
            if file_data[:5] == "data:":
                mime_part = file_data[5:]
                semi_idx = mime_part.find(";")
                media_type = mime_part[:semi_idx] if semi_idx != -1 else mime_part
                converted_block["source"] = {
                    "type": "base64",
                    "media_type": media_type,
                    "data": file_data.split("base64,")[1],
                }
            else:
                converted_block["source"] = {"type": "url", "url": file_data}
            converted_content.append(converted_block)
        else:
            converted_content.append(block)
    return converted_content


def _convert_messages_for_anthropic(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert messages to Anthropic format.

    - Extract messages with `role=system`.
    - Replace `role=tool` with `role=user`, according to examples in https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/.
    - Handle multiple tool calls in a single assistant message.
    - Merge consecutive tool results into a single user message.
    """
    system_message = None
    filtered_messages: list[dict[str, Any]] = []

    for message in messages:
        if message["role"] == "system":
            if system_message is None:
                system_message = message["content"]
            else:
                system_message += "\n" + message["content"]
        else:
            # Handle messages inside agent loop.
            # See https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview#tool-use-examples
            if _is_tool_call(message):
                # Convert ALL tool calls from the assistant message
                tool_use_blocks = []
                for tool_call in message["tool_calls"]:
                    tool_use_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                    )
                message = {
                    "role": "assistant",
                    "content": tool_use_blocks,
                }
            elif message["role"] == "tool":
                # Use tool_call_id from the message itself
                tool_use_id = message.get("tool_call_id", "")
                tool_result = {"type": "tool_result", "tool_use_id": tool_use_id, "content": message["content"]}

                # Check if the previous message is already a user message with tool_results
                # If so, merge this tool_result into it
                if (
                    filtered_messages
                    and filtered_messages[-1]["role"] == "user"
                    and isinstance(filtered_messages[-1]["content"], list)
                    and filtered_messages[-1]["content"]
                    and filtered_messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    filtered_messages[-1]["content"].append(tool_result)
                    continue

                message = {
                    "role": "user",
                    "content": [tool_result],
                }

            if "content" in message and isinstance(message["content"], list):
                message["content"] = _convert_content_for_anthropic(message["content"])

            # Only keep Anthropic-compatible fields (strips OpenAI-specific fields like 'refusal')
            filtered_messages.append({"role": message["role"], "content": message.get("content", "")})

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
                        "index": chunk.index,
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
                        "index": chunk.index,
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
            anthropic_usage = chunk.message.usage
            cache_read = anthropic_usage.cache_read_input_tokens or 0
            cache_creation = anthropic_usage.cache_creation_input_tokens or 0
            total_prompt_tokens = anthropic_usage.input_tokens + cache_read + cache_creation
            chunk_dict["usage"] = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": anthropic_usage.output_tokens,
                "total_tokens": total_prompt_tokens + anthropic_usage.output_tokens,
                "prompt_tokens_details": PromptTokensDetails(cached_tokens=cache_read) if cache_read else None,
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
        tool_calls=cast("list[ChatCompletionMessageToolCallType] | None", tool_calls or None),
    )

    cache_read = response.usage.cache_read_input_tokens or 0
    cache_creation = response.usage.cache_creation_input_tokens or 0
    total_prompt_tokens = response.usage.input_tokens + cache_read + cache_creation

    usage = CompletionUsage(
        completion_tokens=response.usage.output_tokens,
        prompt_tokens=total_prompt_tokens,
        total_tokens=total_prompt_tokens + response.usage.output_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cache_read) if cache_read else None,
    )

    from typing import Literal

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
            "parameters": function.get("parameters") or {},
        }
        generic_tools.append(generic_tool)

    anthropic_tools = []
    for tool in generic_tools:
        params: dict[str, Any] = tool["parameters"] or {}
        anthropic_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": {
                "type": "object",
                "properties": tool["parameters"].get("properties") or {},
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
    elif isinstance(tool_choice, dict):
        if tool_choice_type := tool_choice.get("type"):
            if tool_choice_type in ("custom", "function"):
                return {"type": "tool", "name": tool_choice[tool_choice_type]["name"]}
        msg = f"Unsupported tool_choice format: {tool_choice}"
        raise ValueError(msg)
    return {"type": tool_choice, "disable_parallel_tool_use": not parallel_tool_calls}


def _convert_response_format(response_format: dict[str, Any] | type, provider_name: str) -> dict[str, Any]:
    """Convert any-llm response_format to Anthropic's output_config."""
    if is_structured_output_type(response_format):
        schema = get_json_schema(response_format)
    elif isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
        elif response_format.get("type") == "json_object":
            msg = "response_format with type 'json_object'"
            raise UnsupportedParameterError(
                msg,
                provider_name,
                "Use a Pydantic model or json_schema format instead.",
            )
        else:
            msg = f"Unsupported response_format type: {response_format.get('type')}"
            raise ValueError(msg)
    else:
        msg = f"Unsupported response_format: {response_format}"
        raise ValueError(msg)

    return {"format": {"type": "json_schema", "schema": transform_schema(schema)}}


def _convert_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
    """Convert CompletionParams to kwargs for Anthropic API."""
    provider_name: str = kwargs.pop("provider_name")
    result_kwargs: dict[str, Any] = kwargs.copy()

    if params.response_format:
        if params.stream:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, provider_name)
        result_kwargs["output_config"] = _convert_response_format(params.response_format, provider_name)
    if params.max_tokens is None:
        logger.warning(f"max_tokens is required for Anthropic, setting to {DEFAULT_MAX_TOKENS}")
        params.max_tokens = DEFAULT_MAX_TOKENS

    if params.tools:
        params.tools = _convert_tool_spec(params.tools)

    if params.tool_choice or params.parallel_tool_calls:
        params.tool_choice = _convert_tool_choice(params)

    if params.reasoning_effort is None or params.reasoning_effort == "none":
        result_kwargs["thinking"] = {"type": "disabled"}
    elif params.reasoning_effort != "auto":
        result_kwargs["thinking"] = {"type": "adaptive"}
        effort = REASONING_EFFORT_TO_ANTHROPIC_EFFORT[params.reasoning_effort]
        output_config = result_kwargs.get("output_config", {})
        output_config["effort"] = effort
        result_kwargs["output_config"] = output_config

    result_kwargs.update(
        params.model_dump(
            exclude_none=True,
            exclude={
                "model_id",
                "messages",
                "reasoning_effort",
                "response_format",
                "parallel_tool_calls",
                "stream_options",
            },
        )
    )
    result_kwargs["model"] = params.model_id

    system_message, filtered_messages = _convert_messages_for_anthropic(params.messages)
    if system_message:
        result_kwargs["system"] = system_message
    result_kwargs["messages"] = filtered_messages

    return result_kwargs


def _request_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type in ("input_text", "output_text", "text", "reasoning_text", "summary_text"):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif part_type == "refusal":
                    refusal = part.get("refusal")
                    if isinstance(refusal, str):
                        parts.append(refusal)
        return "".join(parts)
    return ""


def _convert_request_tools(openai_tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not openai_tools:
        return None
    return _convert_tool_spec(openai_tools)


def _convert_request_tool_choice(tool_choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "required":
            tool_choice = "any"
        return {"type": tool_choice, "disable_parallel_tool_use": False}
    if tool_choice_type := tool_choice.get("type"):
        if tool_choice_type in ("custom", "function"):
            return {"type": "tool", "name": tool_choice[tool_choice_type]["name"]}
        if tool_choice_type in ("tool", "auto", "any", "none"):
            return cast("dict[str, Any]", tool_choice)
    msg = f"Unsupported tool_choice format: {tool_choice}"
    raise ValueError(msg)


def _convert_request_input(
    input_data: RequestInput, instructions: str | None
) -> tuple[str | None, list[dict[str, Any]]]:
    if isinstance(input_data, str):
        return instructions, [{"role": "user", "content": input_data}]

    system_parts: list[str] = [instructions] if instructions else []
    messages: list[dict[str, Any]] = []
    pending_assistant_blocks: list[dict[str, Any]] = []

    def flush_assistant_blocks() -> None:
        nonlocal pending_assistant_blocks
        if pending_assistant_blocks:
            messages.append({"role": "assistant", "content": pending_assistant_blocks})
            pending_assistant_blocks = []

    for item in normalize_request_input(input_data):
        if isinstance(item, ItemReferenceParam):
            msg = "item_reference is not supported for anthropic arequest"
            raise UnsupportedParameterError(msg, "anthropic")
        if isinstance(item, (SystemMessageItemParam, DeveloperMessageItemParam)):
            system_text = _request_message_text(item.content)
            if system_text:
                system_parts.append(system_text)
            continue
        if isinstance(item, UserMessageItemParam):
            flush_assistant_blocks()
            messages.append({"role": "user", "content": _request_message_text(item.content)})
            continue
        if isinstance(item, AssistantMessageItemParam):
            assistant_text = _request_message_text(item.content)
            if assistant_text:
                pending_assistant_blocks.append({"type": "text", "text": assistant_text})
            continue
        if isinstance(item, RequestReasoningItemParam):
            state = decode_request_state(item.encrypted_content, "anthropic")
            thinking_text = _request_message_text(item.content)
            signature = state.get("signature") if isinstance(state, dict) else None
            if signature is not None and isinstance(signature, str):
                pending_assistant_blocks.append(
                    {"type": "thinking", "thinking": thinking_text, "signature": signature}
                )
            elif thinking_text:
                pending_assistant_blocks.append({"type": "text", "text": thinking_text})
            continue
        if isinstance(item, FunctionCallItemParam):
            pending_assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": item.call_id,
                    "name": item.name,
                    "input": json.loads(item.arguments),
                }
            )
            continue
        if isinstance(item, FunctionCallOutputItemParam):
            flush_assistant_blocks()
            tool_output = item.output if isinstance(item.output, str) else _request_message_text(item.output)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.call_id,
                            "content": tool_output,
                        }
                    ],
                }
            )
            continue

    flush_assistant_blocks()
    system_message = "\n".join(part for part in system_parts if part) or None
    return system_message, messages


def _convert_request_response(
    response: Message,
    *,
    params: RequestParams,
) -> ResponseResource:
    output_items: list[object] = []
    reasoning_tokens = 0
    if hasattr(response.usage, "output_tokens") and isinstance(response.usage.output_tokens, int):
        reasoning_tokens = 0

    for content_block in response.content:
        if content_block.type == "thinking":
            encrypted_content = encode_request_state("anthropic", {"signature": content_block.signature})
            output_items.append(make_reasoning_item(content_block.thinking, encrypted_content=encrypted_content))
        elif content_block.type == "tool_use":
            output_items.append(
                make_function_call_item(
                    call_id=content_block.id,
                    name=content_block.name,
                    arguments=json.dumps(content_block.input),
                )
            )
        elif content_block.type == "text":
            output_items.append(make_text_message(content_block.text))
        else:
            msg = f"Unsupported content block type: {content_block.type}"
            raise ValueError(msg)

    cache_read = response.usage.cache_read_input_tokens or 0
    cache_creation = response.usage.cache_creation_input_tokens or 0
    input_tokens = response.usage.input_tokens + cache_read + cache_creation
    usage = make_usage(
        input_tokens=input_tokens,
        output_tokens=response.usage.output_tokens,
        cached_tokens=cache_read,
        reasoning_tokens=reasoning_tokens,
    )
    reasoning_cfg = None
    if params.reasoning is not None:
        reasoning_cfg = ResponseReasoningConfig(
            effort=cast("Any", params.reasoning.get("effort")),
            summary=cast("Any", params.reasoning.get("summary")),
        )

    return make_response_resource(
        model=response.model,
        output=cast("list[Any]", output_items),
        tools=params.tools,
        tool_choice=params.tool_choice,
        temperature=params.temperature,
        top_p=params.top_p,
        max_output_tokens=params.max_output_tokens,
        reasoning=reasoning_cfg,
        usage=usage,
        instructions=params.instructions,
        metadata=params.metadata,
    )


def _convert_models_list(models_list: list[AnthropicModelInfo]) -> list[Model]:
    """Convert Anthropic models list to OpenAI format."""
    return [
        Model(id=model.id, object="model", created=int(model.created_at.timestamp()), owned_by="anthropic")
        for model in models_list
    ]
