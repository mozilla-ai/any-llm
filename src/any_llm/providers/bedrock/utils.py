import json
from time import time
from typing import Any, Literal, cast

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionParams,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Function,
    Usage,
)

INFERENCE_PARAMETERS = ["maxTokens", "temperature", "topP", "stopSequences"]


def _convert_params(params: CompletionParams, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert CompletionParams to kwargs for AWS API."""
    result_kwargs: dict[str, Any] = kwargs.copy()

    if params.tools:
        result_kwargs["toolConfig"] = _convert_tool_spec(params.tools, params.tool_choice)

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
    """Convert messages to AWS Bedrock format."""
    system_message = []
    if messages and messages[0]["role"] == "system":
        system_message = [{"text": messages[0]["content"]}]
        messages = messages[1:]

    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            continue

        if message["role"] == "tool":
            bedrock_message = _convert_tool_result(message)
            if bedrock_message:
                formatted_messages.append(bedrock_message)
        elif message["role"] == "assistant":
            bedrock_message = _convert_assistant(message)
            if bedrock_message:
                formatted_messages.append(bedrock_message)
        else:  # user messages
            formatted_messages.append(
                {
                    "role": message["role"],
                    "content": [{"text": message["content"]}],
                }
            )

    return system_message, formatted_messages


def _convert_tool_result(message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert OpenAI tool result format to AWS Bedrock format."""
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

    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_call_id, "content": content}}],
    }


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

    if response.get("stopReason") == "tool_use":
        tool_calls_list: list[dict[str, Any]] = []
        for content in response["output"]["message"]["content"]:
            if "toolUse" in content:
                tool = content["toolUse"]
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

        if tool_calls_list:
            message = ChatCompletionMessage(
                role="assistant",
                content=None,
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

    content = response["output"]["message"]["content"][0]["text"]
    stop_reason = response.get("stopReason")
    finish_reason: Literal["stop", "length"] = "length" if stop_reason == "max_tokens" else "stop"

    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=None)

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


def _create_openai_chunk_from_aws_chunk(chunk: dict[str, Any], model: str) -> ChatCompletionChunk | None:
    """Create an OpenAI ChatCompletionChunk from an AWS Bedrock chunk."""
    content: str | None = None
    finish_reason: Literal["stop", "length"] | None = None
    if "contentBlockDelta" in chunk:
        content = chunk["contentBlockDelta"]["delta"]["text"]
    elif "messageStop" in chunk:
        if chunk["messageStop"]["stopReason"] == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "stop"
    elif "messageStart" in chunk:
        content = ""
    else:
        return None
    delta = ChoiceDelta(content=content, role="assistant")
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
