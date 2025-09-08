import json
from time import time
from typing import Any, Literal, cast

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
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


def _convert_params(params: CompletionParams, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert CompletionParams to kwargs for SageMaker API."""
    result_kwargs: dict[str, Any] = kwargs.copy()

    messages = []
    system_message = None

    for message in params.messages:
        if message["role"] == "system":
            system_message = message["content"]
        else:
            messages.append(message)

    result_kwargs["messages"] = messages
    if system_message:
        result_kwargs["system"] = system_message

    if params.max_tokens:
        result_kwargs["max_tokens"] = params.max_tokens
    if params.temperature:
        result_kwargs["temperature"] = params.temperature
    if params.top_p:
        result_kwargs["top_p"] = params.top_p
    if params.stop:
        result_kwargs["stop"] = params.stop

    if params.tools:
        result_kwargs["tools"] = params.tools
        if params.tool_choice:
            result_kwargs["tool_choice"] = params.tool_choice

    return result_kwargs


def _convert_response(response: dict[str, Any], model: str) -> ChatCompletion:
    """Convert SageMaker response to OpenAI format."""
    choices_out: list[Choice] = []
    usage: CompletionUsage | None = None

    if "choices" in response:
        for choice_data in response["choices"]:
            message_data = choice_data.get("message", {})
            tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] | None = None
            if message_data.get("tool_calls"):
                tool_calls = [
                    ChatCompletionMessageFunctionToolCall(
                        id=tc.get("id", f"call_{int(time())}"),
                        type="function",
                        function=Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in message_data["tool_calls"]
                ]

            message = ChatCompletionMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
                tool_calls=tool_calls,
            )

            finish_reason = choice_data.get("finish_reason", "stop")
            choices_out.append(
                Choice(
                    index=choice_data.get("index", 0),
                    finish_reason=cast(
                        "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']", finish_reason
                    ),
                    message=message,
                )
            )
    else:
        content = None
        if "generated_text" in response:
            content = response["generated_text"]
        elif "outputs" in response:
            if isinstance(response["outputs"], list) and response["outputs"]:
                content = response["outputs"][0]
            else:
                content = response["outputs"]
        elif "content" in response:
            content = response["content"]
        else:
            content = str(response)

        message = ChatCompletionMessage(role="assistant", content=content, tool_calls=None)
        choices_out.append(
            Choice(
                index=0,
                finish_reason="stop",
                message=message,
            )
        )

    if "usage" in response:
        usage_data = response["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

    return ChatCompletion(
        id=response.get("id", f"chatcmpl-{int(time())}"),
        model=model,
        created=response.get("created", int(time())),
        object="chat.completion",
        choices=choices_out,
        usage=usage,
    )


def _create_openai_chunk_from_sagemaker_chunk(event: dict[str, Any], model: str) -> ChatCompletionChunk | None:
    """Create an OpenAI ChatCompletionChunk from a SageMaker streaming event."""
    if "PayloadPart" not in event:
        return None

    try:
        payload = json.loads(event["PayloadPart"]["Bytes"].decode("utf-8"))
    except (json.JSONDecodeError, KeyError):
        return None

    content: str | None = None
    finish_reason: Literal["stop", "length"] | None = None

    if "token" in payload:
        content = payload["token"]["text"]
    elif "outputs" in payload:
        if isinstance(payload["outputs"], list) and payload["outputs"]:
            content = payload["outputs"][0].get("text", "")
        else:
            content = payload["outputs"].get("text", "")
    elif "generated_text" in payload:
        content = payload["generated_text"]
    elif payload.get("choices"):
        delta = payload["choices"][0].get("delta", {})
        content = delta.get("content")
        finish_reason_raw = payload["choices"][0].get("finish_reason")
        if finish_reason_raw:
            finish_reason = "length" if finish_reason_raw == "length" else "stop"

    if payload.get("is_finished"):
        finish_reason = "stop"

    delta = ChoiceDelta(content=content, role="assistant")
    choice = ChunkChoice(delta=delta, finish_reason=finish_reason, index=0)

    return ChatCompletionChunk(
        id=f"chatcmpl-{int(time())}",
        choices=[choice],
        model=model,
        created=int(time()),
        object="chat.completion.chunk",
    )


def _create_openai_embedding_response_from_sagemaker(
    embedding_data: list[dict[str, Any]], model: str, total_tokens: int
) -> CreateEmbeddingResponse:
    """Convert SageMaker embedding response to OpenAI CreateEmbeddingResponse format."""
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
