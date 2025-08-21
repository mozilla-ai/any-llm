import json
from typing import Any, Literal, cast

from azure.ai.inference.models import (
    ChatCompletions,
    EmbeddingsResult,
    JsonSchemaFormat,
    StreamingChatCompletionsUpdate,
)
from pydantic import BaseModel

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Function,
    Usage,
)


def _convert_response_format(
    response_format: type[BaseModel] | dict[str, Any],
) -> JsonSchemaFormat | str | Any:
    """Convert Pydantic model to Azure JsonSchemaFormat."""
    if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
        return response_format

    schema = response_format.model_json_schema()
    # Azure requires additionalProperties to be false for structured output
    schema["additionalProperties"] = False

    # Ensure all nested objects also have additionalProperties: false
    def add_additional_properties_false(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "properties" in obj:
                obj["additionalProperties"] = False
            for value in obj.values():
                add_additional_properties_false(value)
        elif isinstance(obj, list):
            for item in obj:
                add_additional_properties_false(item)

    add_additional_properties_false(schema)

    return JsonSchemaFormat(
        name=response_format.__name__,
        description=schema.get("description", f"Schema for {response_format.__name__}"),
        schema=schema,
        strict=True,
    )


def _convert_response(response_data: ChatCompletions) -> ChatCompletion:
    """Convert Azure response to OpenAI ChatCompletion format directly."""
    choice_data = response_data.choices[0]
    message_data = choice_data.message

    # Convert tool calls
    tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] | None = None
    if message_data.tool_calls:
        tool_calls_list: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
        for tc in message_data.tool_calls:
            func = tc.function
            tool_calls_list.append(
                ChatCompletionMessageFunctionToolCall(
                    id=tc.id,
                    type="function",
                    function=Function(name=func.name if func else "", arguments=func.arguments if func else ""),
                )
            )
        tool_calls = tool_calls_list

    # Usage
    usage = None
    if response_data.usage:
        usage = CompletionUsage(
            prompt_tokens=response_data.usage.prompt_tokens,
            completion_tokens=response_data.usage.completion_tokens,
            total_tokens=response_data.usage.total_tokens,
        )

    message = ChatCompletionMessage(
        role=cast("Literal['assistant']", "assistant"),
        content=message_data.content,
        tool_calls=tool_calls,
    )

    choice = Choice(
        index=choice_data.index,
        finish_reason=cast(
            "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']", choice_data.finish_reason
        ),
        message=message,
    )

    return ChatCompletion(
        id=response_data.id,
        model=response_data.model,
        created=int(response_data.created.timestamp()),
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


def _create_openai_chunk_from_azure_chunk(azure_chunk: StreamingChatCompletionsUpdate) -> ChatCompletionChunk:
    """Convert an Azure AI Inference streaming chunk to OpenAI ChatCompletionChunk format."""
    openai_choices = []
    choices = azure_chunk.choices

    for choice in choices:
        delta = choice.delta
        if not delta:
            continue

        delta_content = delta.content
        role_value = delta.role
        delta_role = None

        if role_value and role_value in ["developer", "system", "user", "assistant", "tool"]:
            delta_role = cast("Literal['developer', 'system', 'user', 'assistant', 'tool']", role_value)

        delta_tool_calls: list[ChoiceDeltaToolCall] | None = None
        if delta.tool_calls:
            delta_tool_calls = []
            for tool_call in delta.tool_calls:
                function_data = tool_call.function
                function = None
                if function_data:
                    function = ChoiceDeltaToolCallFunction(
                        name=function_data.name,
                        arguments=function_data.arguments,
                    )

                delta_tool_calls.append(
                    ChoiceDeltaToolCall(
                        index=len(delta_tool_calls),  # Use current length as index
                        id=tool_call.id,
                        type="function",
                        function=function,
                    )
                )

        openai_choice = ChunkChoice(
            index=choice.index,
            delta=ChoiceDelta(
                content=delta_content,
                role=delta_role,
                tool_calls=delta_tool_calls,
            ),
            finish_reason=cast(
                "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']", choice.finish_reason
            )
            if choice.finish_reason
            else None,
        )
        openai_choices.append(openai_choice)

    usage = None
    usage_data = azure_chunk.usage
    if usage_data:
        usage = CompletionUsage(
            prompt_tokens=usage_data.prompt_tokens,
            completion_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
        )

    return ChatCompletionChunk(
        id=azure_chunk.id,
        choices=openai_choices,
        created=int(azure_chunk.created.timestamp()),
        model=azure_chunk.model,
        object="chat.completion.chunk",
        usage=usage,
    )


def _create_openai_embedding_response_from_azure(
    azure_response: EmbeddingsResult,
) -> CreateEmbeddingResponse:
    """Convert an Azure AI Inference embedding response to OpenAI CreateEmbeddingResponse format."""
    data = azure_response.data
    model_name = azure_response.model
    usage_data = azure_response.usage

    openai_embeddings: list[Embedding] = []
    if isinstance(data, list):
        for embedding_data in data:
            embedding_vector = embedding_data.embedding
            index = embedding_data.index

            if isinstance(embedding_vector, str):
                try:
                    embedding_list = json.loads(embedding_vector)
                    if isinstance(embedding_list, list):
                        embedding_vector = embedding_list
                    else:
                        embedding_vector = []
                except (json.JSONDecodeError, TypeError):
                    embedding_vector = []
            elif not isinstance(embedding_vector, list):
                embedding_vector = []

            openai_embedding = Embedding(embedding=embedding_vector, index=index, object="embedding")
            openai_embeddings.append(openai_embedding)

    usage = Usage(
        prompt_tokens=usage_data.prompt_tokens,
        total_tokens=usage_data.total_tokens,
    )

    return CreateEmbeddingResponse(
        data=openai_embeddings,
        model=model_name,
        object="list",
        usage=usage,
    )
