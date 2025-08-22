import json
import uuid
from typing import Any, Literal, cast

from pydantic import BaseModel

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from any_llm.types.model import Model


def _convert_response(response: dict[str, Any]) -> ChatCompletion:
    """Convert Watsonx response to OpenAI ChatCompletion format."""
    choice_data = response["choices"][0]
    message_data = choice_data["message"]

    message = ChatCompletionMessage(
        content=message_data.get("content"),
        role=message_data.get("role", "assistant"),
        tool_calls=message_data.get("tool_calls"),
    )

    choice = Choice(
        finish_reason=choice_data.get("finish_reason", "stop"),
        index=choice_data.get("index", 0),
        message=message,
    )

    usage = None
    if "usage" in response:
        usage_data = response["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

    return ChatCompletion(
        id=response.get("id", ""),
        model=response.get("model", ""),
        object="chat.completion",
        created=response.get("created", 0),
        choices=[choice],
        usage=usage,
    )


def _convert_streaming_chunk(chunk: dict[str, Any]) -> ChatCompletionChunk:
    """Convert Watsonx streaming chunk to OpenAI ChatCompletionChunk format."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created = chunk.get("created", 0)
    model = chunk.get("model", "")

    choices = []
    chunk_choices = chunk.get("choices", [])

    for i, chunk_choice in enumerate(chunk_choices):
        delta_data = chunk_choice.get("delta", {})
        content = delta_data.get("content")
        role = delta_data.get("role")

        openai_role = None
        if role:
            openai_role = cast("Literal['developer', 'system', 'user', 'assistant', 'tool']", role)

        delta = ChoiceDelta(content=content, role=openai_role)

        choice = ChunkChoice(
            index=i,
            delta=delta,
            finish_reason=cast(
                "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None",
                chunk_choice.get("finish_reason"),
            ),
        )
        choices.append(choice)

    usage = None
    chunk_usage = chunk.get("usage")
    if chunk_usage:
        prompt_tokens = chunk_usage.get("prompt_tokens", 0)
        completion_tokens = chunk_usage.get("completion_tokens", 0)
        total_tokens = chunk_usage.get("total_tokens", 0)

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    return ChatCompletionChunk(
        id=chunk_id,
        choices=choices,
        created=created,
        model=model,
        object="chat.completion.chunk",
        usage=usage,
    )


def _convert_pydantic_to_watsonx_json(
    pydantic_model: type[BaseModel], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert a Pydantic model to an inline JSON instruction for Watsonx.

    This mirrors the generic JSON-mode prompt approach used for providers
    without native structured output support.
    """
    schema = pydantic_model.model_json_schema()

    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        json_instruction = f"""
Please respond with a JSON object that can be loaded into a pydantic model that matches the following schema:

{json.dumps(schema, indent=2)}

Return the JSON object only, no other text, do not wrap it in ```json or ```.

{original_content}
"""
        modified_messages[-1]["content"] = json_instruction
    else:
        msg = "Last message is not a user message"
        raise ValueError(msg)

    return modified_messages


def _convert_models_list(models: dict[str, Any]) -> list[Model]:
    models_list = models.get("resources", [])
    created = 0  # watsonx doesn't provide a created timestamp
    return [Model(id=model["model_id"], object="model", created=created, owned_by="watsonx") for model in models_list]
