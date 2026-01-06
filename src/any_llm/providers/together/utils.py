import uuid
from datetime import datetime
from typing import Any, Literal, cast

from together.types.chat import ChatCompletionChunk as TogetherChatCompletionChunk

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionUsage,
    Reasoning,
)
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags


def _create_openai_chunk_from_together_chunk(together_chunk: TogetherChatCompletionChunk) -> ChatCompletionChunk:
    """Convert a Together streaming chunk to OpenAI ChatCompletionChunk format."""

    openai_choices: list[ChunkChoice] = []
    for choice in together_chunk.choices or []:
        delta_content = choice.delta
        content = None
        role = None
        reasoning = None

        if delta_content:
            content = delta_content.content
            if delta_content.role:
                role = cast("Literal['assistant', 'user', 'system']", delta_content.role)
            if hasattr(delta_content, "reasoning") and delta_content.reasoning:
                reasoning = Reasoning(content=delta_content.reasoning)

        delta = ChoiceDelta(content=content, role=role, reasoning=reasoning)

        if delta_content and hasattr(delta_content, "tool_calls") and delta_content.tool_calls:
            openai_tool_calls = []
            for idx, tool_call in enumerate(delta_content.tool_calls):
                if isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    tc_id = tool_call.get("id") or str(uuid.uuid4())
                    raw_index = tool_call.get("index")
                    tc_index = raw_index if raw_index is not None else idx
                    name = func.get("name", "")
                    arguments = func.get("arguments", "")
                else:
                    tc_id = getattr(tool_call, "id", None) or str(uuid.uuid4())
                    raw_index = getattr(tool_call, "index", None)
                    tc_index = raw_index if raw_index is not None else idx
                    func = getattr(tool_call, "function", None)
                    name = getattr(func, "name", "") if func else ""
                    arguments = getattr(func, "arguments", "") if func else ""

                openai_tool_call = ChoiceDeltaToolCall(
                    index=tc_index,
                    id=tc_id,
                    type="function",
                    function=ChoiceDeltaToolCallFunction(
                        name=name,
                        arguments=arguments,
                    ),
                )
                openai_tool_calls.append(openai_tool_call)
            delta.tool_calls = openai_tool_calls

        openai_choice = ChunkChoice(
            index=choice.index or len(openai_choices),
            delta=delta,
            finish_reason=cast(
                "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None",
                choice.finish_reason,
            ),
        )
        openai_choices.append(openai_choice)

    usage = None
    if together_chunk.usage:
        usage = CompletionUsage(
            prompt_tokens=together_chunk.usage.prompt_tokens or 0,
            completion_tokens=together_chunk.usage.completion_tokens or 0,
            total_tokens=together_chunk.usage.total_tokens or 0,
        )

    return ChatCompletionChunk(
        id=together_chunk.id or f"chatcmpl-{uuid.uuid4()}",
        choices=openai_choices,
        created=together_chunk.created or int(datetime.now().timestamp()),
        model=together_chunk.model or "unknown",
        object="chat.completion.chunk",
        usage=usage,
    )


def _convert_together_response_to_chat_completion(response_data: dict[str, Any], model_id: str) -> ChatCompletion:
    """Convert Together API response to OpenAI ChatCompletion format."""
    choices_out: list[Choice] = []
    for i, ch in enumerate(response_data.get("choices", [])):
        msg = ch.get("message", {})

        normalize_reasoning_from_provider_fields_and_xml_tags(msg)

        message = ChatCompletionMessage(
            role=cast("Literal['assistant']", msg.get("role")),
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            reasoning=msg.get("reasoning"),
        )
        choices_out.append(
            Choice(
                index=i,
                finish_reason=cast(
                    "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
                    ch.get("finish_reason"),
                ),
                message=message,
            )
        )

    usage = None
    if response_data.get("usage"):
        u = response_data["usage"]
        usage = CompletionUsage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )

    return ChatCompletion(
        id=response_data.get("id", ""),
        model=model_id,
        created=response_data.get("created", 0),
        object="chat.completion",
        choices=choices_out,
        usage=usage,
    )
