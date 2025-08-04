import json
from time import time
from typing import Any, Literal

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from openai.types import CreateEmbeddingResponse
from openai.types.embedding import Embedding
from openai.types.create_embedding_response import Usage


INFERENCE_PARAMETERS = ["maxTokens", "temperature", "topP", "stopSequences"]


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for AWS Bedrock."""
    kwargs = kwargs.copy()

    # Convert tools and remove from kwargs
    tool_config = _convert_tool_spec(kwargs)
    kwargs.pop("tools", None)  # Remove tools from kwargs if present

    # Prepare inference config
    inference_config = {key: kwargs[key] for key in INFERENCE_PARAMETERS if key in kwargs}

    additional_fields = {key: value for key, value in kwargs.items() if key not in INFERENCE_PARAMETERS}

    request_config = {
        "inferenceConfig": inference_config,
        "additionalModelRequestFields": additional_fields,
    }

    if tool_config is not None:
        request_config["toolConfig"] = tool_config

    return request_config


def _convert_tool_spec(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Convert tool specifications to Bedrock format."""
    if "tools" not in kwargs:
        return None

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", " "),
                    "inputSchema": {"json": tool["function"]["parameters"]},
                }
            }
            for tool in kwargs["tools"]
        ]
    }
    return tool_config


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert messages to AWS Bedrock format."""
    # Handle system message
    system_message = []
    if messages and messages[0]["role"] == "system":
        system_message = [{"text": messages[0]["content"]}]
        messages = messages[1:]

    formatted_messages = []
    for message in messages:
        # Skip any additional system messages
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
        raise RuntimeError("Tool result message must include tool_call_id")

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
    """Convert AWS Bedrock response to OpenAI ChatCompletion format."""
    # Check if the model is requesting tool use
    if response.get("stopReason") == "tool_use":
        tool_calls = []
        for content in response["output"]["message"]["content"]:
            if "toolUse" in content:
                tool = content["toolUse"]
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tool["toolUseId"],
                        type="function",
                        function=Function(
                            name=tool["name"],
                            arguments=json.dumps(tool["input"]),
                        ),
                    )
                )

        if tool_calls:
            message = ChatCompletionMessage(
                content=None,
                role="assistant",
                tool_calls=tool_calls,
            )

            choice = Choice(
                finish_reason="tool_calls",
                index=0,
                message=message,
            )

            usage = None
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
                object="chat.completion",
                created=response.get("created", 0),
                choices=[choice],
                usage=usage,
            )

    content = response["output"]["message"]["content"][0]["text"]

    stop_reason = response.get("stopReason")
    finish_reason: Literal["stop", "length"]
    if stop_reason == "max_tokens":
        finish_reason = "length"
    else:
        finish_reason = "stop"

    message = ChatCompletionMessage(
        content=content,
        role="assistant",
        tool_calls=None,
    )

    choice = Choice(
        finish_reason=finish_reason,
        index=0,
        message=message,
    )

    usage = None
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
        object="chat.completion",
        created=response.get("created", 0),
        choices=[choice],
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
