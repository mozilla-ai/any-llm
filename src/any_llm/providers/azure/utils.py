from typing import Any, Union, Optional, List, cast, Literal
import json

from azure.ai.inference.models import (
    JsonSchemaFormat,
    ChatCompletions,
    EmbeddingsResult,
    StreamingChatCompletionsUpdate,
)
from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.embedding import Embedding
from openai.types.create_embedding_response import Usage


def _convert_response_format(
    response_format: Union[type[BaseModel], str, JsonSchemaFormat, Any],
) -> Union[JsonSchemaFormat, str, Any]:
    """Convert Pydantic model to Azure JsonSchemaFormat."""
    if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
        return response_format

    # Convert Pydantic model to Azure JsonSchemaFormat
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
    """Convert Azure response to OpenAI ChatCompletion format."""
    choices_data = response_data.choices
    response_id = response_data.id
    model_name = response_data.model
    created_time = response_data.created
    usage_data = response_data.usage

    choice_data = choices_data[0]

    message_data = choice_data.message
    finish_reason = choice_data.finish_reason
    index = choice_data.index

    # Handle tool calls
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    if message_data.tool_calls:
        tool_calls = []
        for tool_call in message_data.tool_calls:
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=Function(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
            )

    message = ChatCompletionMessage(
        content=message_data.content,
        role=cast(Literal["assistant"], message_data.role),
        tool_calls=tool_calls,
    )

    choice = Choice(
        finish_reason=cast(Literal["stop", "length", "tool_calls", "content_filter", "function_call"], finish_reason),
        index=index,
        message=message,
    )

    usage: Optional[CompletionUsage] = None
    if usage_data:
        usage = CompletionUsage(
            completion_tokens=usage_data.completion_tokens,
            prompt_tokens=usage_data.prompt_tokens,
            total_tokens=usage_data.total_tokens,
        )

    return ChatCompletion(
        id=response_id,
        model=model_name,
        object="chat.completion",
        created=int(created_time.timestamp()),
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
            delta_role = cast(Literal["developer", "system", "user", "assistant", "tool"], role_value)

        delta_tool_calls: Optional[List[ChoiceDeltaToolCall]] = None
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
                Literal["stop", "length", "tool_calls", "content_filter", "function_call"], choice.finish_reason
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

    openai_embeddings: List[Embedding] = []
    if isinstance(data, list):
        for i, embedding_data in enumerate(data):
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
