import json
from time import time
from typing import Any, Literal, cast

from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionParams,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Function,
    Reasoning,
    Usage,
)

INFERENCE_PARAMETERS = ["maxTokens", "temperature", "topP", "stopSequences"]

REASONING_EFFORT_TO_THINKING_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 24576}


def _convert_params(params: CompletionParams, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert CompletionParams to kwargs for AWS API."""
    result_kwargs: dict[str, Any] = kwargs.copy()

    if params.response_format:
        msg = "response_format"
        raise UnsupportedParameterError(
            msg,
            "bedrock",
            "Check the following links:\n- https://docs.aws.amazon.com/nova/latest/userguide/prompting-structured-output.html",
        )

    if params.tools:
        result_kwargs["toolConfig"] = _convert_tool_spec(params.tools, params.tool_choice)

    reasoning_enabled = (
        params.reasoning_effort is not None and params.reasoning_effort != "auto" and params.reasoning_effort != "none"
    )

    inference_config: dict[str, Any] = {}
    if params.max_tokens:
        inference_config["maxTokens"] = params.max_tokens
    if params.temperature:
        inference_config["temperature"] = params.temperature
    if params.top_p:
        inference_config["topP"] = params.top_p
    if params.stop:
        inference_config["stopSequences"] = params.stop

    if inference_config:
        result_kwargs["inferenceConfig"] = inference_config

    if reasoning_enabled:
        additional_fields: dict[str, Any] = result_kwargs.get("additionalModelRequestFields", {})
        reasoning_config: dict[str, Any] = {"type": "enabled"}
        if params.reasoning_effort in REASONING_EFFORT_TO_THINKING_BUDGETS:
            reasoning_config["budget_tokens"] = REASONING_EFFORT_TO_THINKING_BUDGETS[params.reasoning_effort]
        additional_fields["reasoning_config"] = reasoning_config
        result_kwargs["additionalModelRequestFields"] = additional_fields

    system_message, formatted_messages = _convert_messages(params.messages)
    result_kwargs["messages"] = formatted_messages
    if system_message:
        result_kwargs["system"] = system_message

    result_kwargs["modelId"] = params.model_id

    return result_kwargs


def _convert_tool_spec(tools: list[dict[str, Any]], tool_choice: str | dict[str, Any] | None) -> dict[str, Any]:
    """Convert tool specifications to Bedrock format."""
    tool_config: dict[str, Any] = {
        "tools": [
            {
                "toolSpec": {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", " "),
                    "inputSchema": {"json": tool["function"]["parameters"]},
                }
            }
            for tool in tools
        ]
    }
    if tool_choice:
        if tool_choice == "required":
            tool_config["toolChoice"] = {"any": {}}
    return tool_config


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert messages to AWS Bedrock format.

    Bedrock requires that consecutive tool results are merged into a single user message.
    This is necessary because Bedrock expects all toolResult blocks that correspond to
    tool calls from a single assistant message to be grouped together.
    """
    system_message = []
    if messages and messages[0]["role"] == "system":
        system_message = [{"text": messages[0]["content"]}]
        messages = messages[1:]

    formatted_messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        """Flush accumulated tool results into a single user message."""
        if pending_tool_results:
            formatted_messages.append({"role": "user", "content": pending_tool_results.copy()})
            pending_tool_results.clear()

    for message in messages:
        if message["role"] == "system":
            continue

        if message["role"] == "tool":
            tool_result_content = _convert_tool_result_content(message)
            if tool_result_content:
                pending_tool_results.append(tool_result_content)
        elif message["role"] == "assistant":
            flush_tool_results()
            bedrock_message = _convert_assistant(message)
            if bedrock_message:
                formatted_messages.append(bedrock_message)
        else:  # user messages
            flush_tool_results()
            formatted_messages.append(
                {
                    "role": message["role"],
                    "content": [{"text": message["content"]}],
                }
            )

    # Flush any remaining tool results at the end
    flush_tool_results()

    return system_message, formatted_messages


def _convert_tool_result_content(message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert OpenAI tool result format to AWS Bedrock toolResult content block.

    Returns just the toolResult content block, not the full message.
    The caller is responsible for grouping these into a user message.
    """
    if message["role"] != "tool" or "content" not in message:
        return None

    tool_call_id = message.get("tool_call_id")
    if not tool_call_id:
        msg = "Tool result message must include tool_call_id"
        raise RuntimeError(msg)

    try:
        content_json = json.loads(message["content"])
        content = [{"json": content_json}]
    except json.JSONDecodeError:
        content = [{"text": message["content"]}]

    return {"toolResult": {"toolUseId": tool_call_id, "content": content}}


def _convert_assistant(message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert OpenAI assistant format to AWS Bedrock format."""
    if message["role"] != "assistant":
        return None

    content = []

    if message.get("content"):
        content.append({"text": message["content"]})

    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            if tool_call["type"] == "function":
                try:
                    input_json = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    input_json = tool_call["function"]["arguments"]

                content.append(
                    {
                        "toolUse": {
                            "toolUseId": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": input_json,
                        }
                    }
                )

    return {"role": "assistant", "content": content} if content else None


def _convert_response(response: dict[str, Any]) -> ChatCompletion:
    """Convert AWS Bedrock response to OpenAI format directly."""
    choices_out: list[Choice] = []
    usage: CompletionUsage | None = None

    reasoning_content: str | None = None
    content_parts: list[str] = []
    tool_calls_list: list[dict[str, Any]] = []

    for content_block in response["output"]["message"]["content"]:
        if "text" in content_block:
            content_parts.append(content_block["text"])
        elif "reasoningContent" in content_block:
            reasoning_text = content_block["reasoningContent"].get("reasoningText", {}).get("text", "")
            if reasoning_content is None:
                reasoning_content = reasoning_text
            else:
                reasoning_content += reasoning_text
        elif "toolUse" in content_block:
            tool = content_block["toolUse"]
            tool_calls_list.append(
                {
                    "id": tool["toolUseId"],
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "arguments": json.dumps(tool["input"]),
                    },
                }
            )

    if response.get("stopReason") == "tool_use" and tool_calls_list:
        message = ChatCompletionMessage(
            role="assistant",
            content=None,
            reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
            tool_calls=[
                ChatCompletionMessageFunctionToolCall(
                    id=tc["id"],
                    type="function",
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tool_calls_list
            ],
        )
        choices_out.append(Choice(index=0, finish_reason="tool_calls", message=message))

        if "usage" in response:
            usage_data = response["usage"]
            usage = CompletionUsage(
                completion_tokens=usage_data.get("outputTokens", 0),
                prompt_tokens=usage_data.get("inputTokens", 0),
                total_tokens=usage_data.get("totalTokens", 0),
            )

        return ChatCompletion(
            id=response.get("id", ""),
            model=response.get("model", ""),
            created=response.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )

    content = "".join(content_parts)
    stop_reason = response.get("stopReason")
    finish_reason: Literal["stop", "length"] = "length" if stop_reason == "max_tokens" else "stop"

    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
        tool_calls=None,
    )

    choices_out.append(
        Choice(
            index=0,
            finish_reason=cast(
                "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']", finish_reason
            ),
            message=message,
        )
    )

    if "usage" in response:
        usage_data = response["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("outputTokens", 0),
            prompt_tokens=usage_data.get("inputTokens", 0),
            total_tokens=usage_data.get("totalTokens", 0),
        )

    return ChatCompletion(
        id=response.get("id", ""),
        model=response.get("model", ""),
        created=response.get("created", 0),
        object="chat.completion",
        choices=choices_out,
        usage=usage,
    )


def _create_openai_chunk_from_aws_chunk(
    chunk: dict[str, Any], model: str, tool_index_map: dict[int, int] | None = None
) -> ChatCompletionChunk | None:
    """Create an OpenAI ChatCompletionChunk from an AWS Bedrock chunk.

    Args:
        chunk: The AWS Bedrock streaming chunk
        model: The model identifier
        tool_index_map: Optional mapping from contentBlockIndex to tool call index (0-based).
                       This should be passed in and maintained by the caller for multi-turn streaming.
    """
    if tool_index_map is None:
        tool_index_map = {}

    content: str | None = None
    reasoning_content: str | None = None
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None
    tool_call: ChoiceDeltaToolCall | None = None

    if "contentBlockStart" in chunk:
        block_start = chunk["contentBlockStart"]
        block = block_start.get("start", {})
        block_index = block_start.get("contentBlockIndex", 0)
        if "reasoningContent" in block:
            reasoning_content = ""
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            tool_idx = len(tool_index_map)
            tool_index_map[block_index] = tool_idx
            tool_call = ChoiceDeltaToolCall(
                index=tool_idx,
                id=tool_use.get("toolUseId", ""),
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tool_use.get("name", ""),
                    arguments="",
                ),
            )
        else:
            content = ""
    elif "contentBlockDelta" in chunk:
        block_delta = chunk["contentBlockDelta"]
        delta = block_delta.get("delta", {})
        block_index = block_delta.get("contentBlockIndex", 0)
        if "text" in delta:
            content = delta["text"]
        elif "reasoningContent" in delta:
            reasoning_content = delta["reasoningContent"].get("text", "")
        elif "toolUse" in delta:
            tool_use = delta["toolUse"]
            tool_idx = tool_index_map.get(block_index, 0)
            tool_call = ChoiceDeltaToolCall(
                index=tool_idx,
                id="",
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name="",
                    arguments=tool_use.get("input", ""),
                ),
            )
    elif "messageStop" in chunk:
        stop_reason = chunk["messageStop"]["stopReason"]
        if stop_reason == "max_tokens":
            finish_reason = "length"
        elif stop_reason == "tool_use":
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"
    elif "messageStart" in chunk:
        content = ""
    else:
        return None

    delta_dict: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        delta_dict["content"] = content
    if reasoning_content is not None:
        delta_dict["reasoning"] = {"content": reasoning_content}

    delta = ChoiceDelta(**delta_dict)
    if tool_call is not None:
        delta.tool_calls = [tool_call]

    choice = ChunkChoice(delta=delta, finish_reason=finish_reason, index=0)
    return ChatCompletionChunk(
        id=f"chatcmpl-{time()}",  # AWS doesn't provide an ID in the chunk
        choices=[choice],
        model=model,
        created=int(time()),
        object="chat.completion.chunk",
    )


def _create_openai_embedding_response_from_aws(
    embedding_data: list[dict[str, Any]], model: str, total_tokens: int
) -> CreateEmbeddingResponse:
    """Convert AWS Bedrock embedding response to OpenAI CreateEmbeddingResponse format."""
    openai_embeddings = []
    for data in embedding_data:
        openai_embedding = Embedding(embedding=data["embedding"], index=data["index"], object="embedding")
        openai_embeddings.append(openai_embedding)

    usage = Usage(
        prompt_tokens=total_tokens,
        total_tokens=total_tokens,
    )

    return CreateEmbeddingResponse(
        data=openai_embeddings,
        model=model,
        object="list",
        usage=usage,
    )
