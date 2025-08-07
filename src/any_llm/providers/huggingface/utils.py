import json
from typing import Any
import uuid

from pydantic import BaseModel
from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    CompletionUsage,
    ChunkChoice,
)
from typing import Literal, cast

from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
    ChatCompletionStreamOutput as HuggingFaceChatCompletionStreamOutput,
)


def _convert_pydantic_to_huggingface_json(
    pydantic_model: type[BaseModel], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert Pydantic model to HuggingFace-compatible JSON instructions.

    Following a similar pattern to the DeepSeek provider but adapted for HuggingFace.

    Returns:
        modified_messages
    """
    schema = pydantic_model.model_json_schema()

    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        json_instruction = f"""Answer the following question and format your response as a JSON object matching this schema:

Schema: {json.dumps(schema, indent=2)}

DO NOT return the schema itself. Instead, answer the question and put your answer in the correct JSON format.

For example, if the question asks for a name and you want to answer "Paris", return: {{"name": "Paris"}}

Question: {original_content}

Answer (as JSON):"""
        modified_messages[-1]["content"] = json_instruction
    else:
        msg = "Last message is not a user message"
        raise ValueError(msg)

    return modified_messages


def _create_openai_chunk_from_huggingface_chunk(chunk: HuggingFaceChatCompletionStreamOutput) -> ChatCompletionChunk:
    """Convert a HuggingFace streaming chunk to OpenAI ChatCompletionChunk format."""

    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created = chunk.created
    model = chunk.model

    choices = []
    hf_choices = chunk.choices

    for i, hf_choice in enumerate(hf_choices):
        hf_delta = hf_choice.delta
        content = hf_delta.content
        role = hf_delta.role

        openai_role = None
        if role:
            openai_role = cast(Literal["developer", "system", "user", "assistant", "tool"], role)

        delta = ChoiceDelta(content=content, role=openai_role)

        choice = ChunkChoice(
            index=i,
            delta=delta,
            finish_reason=cast(
                Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None,
                hf_choice.finish_reason,
            ),
        )
        choices.append(choice)

    usage = None
    hf_usage = chunk.usage
    if hf_usage:
        prompt_tokens = hf_usage.prompt_tokens
        completion_tokens = hf_usage.completion_tokens
        total_tokens = hf_usage.total_tokens

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
