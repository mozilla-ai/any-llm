import uuid
from collections.abc import Sequence
from typing import Any, Literal

from xai_sdk.chat import Chunk as XaiChunk
from xai_sdk.chat import Response as XaiResponse
from xai_sdk.chat import tool as xai_make_tool
from xai_sdk.proto import chat_pb2 as xai_chat_pb2
from xai_sdk.proto import models_pb2 as xai_models_pb2

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
    Function,
    Reasoning,
)
from any_llm.types.model import Model


def _map_xai_role_to_openai(
    role: int,
) -> Literal["developer", "system", "user", "assistant", "tool"] | None:
    if role == xai_chat_pb2.ROLE_USER:
        return "user"
    if role == xai_chat_pb2.ROLE_ASSISTANT:
        return "assistant"
    if role == xai_chat_pb2.ROLE_SYSTEM:
        return "system"
    if role == xai_chat_pb2.ROLE_TOOL or role == xai_chat_pb2.ROLE_FUNCTION:
        return "tool"
    return None


def _convert_xai_chunk_to_anyllm_chunk(chunk: XaiChunk) -> ChatCompletionChunk:
    # Collect deltas from the first (and only) choice index 0
    choices: list[ChunkChoice] = []
    for i, choice in enumerate(chunk.choices):
        reasoning = Reasoning(content=choice.reasoning_content) if choice.reasoning_content else None

        delta_tool_calls: list[ChoiceDeltaToolCall] | None = None
        tool_calls = choice.tool_calls
        if tool_calls:
            delta_tool_calls_list: list[ChoiceDeltaToolCall] = []
            for idx, tc in enumerate(tool_calls):
                func = tc.function

                delta_tool_calls_list.append(
                    ChoiceDeltaToolCall(
                        index=idx,
                        id=str(uuid.uuid4()),
                        type="function",
                        function=ChoiceDeltaToolCallFunction(name=func.name, arguments=func.arguments),
                    )
                )
            delta_tool_calls = delta_tool_calls_list or None

        openai_role = _map_xai_role_to_openai(choice.role)
        delta = ChoiceDelta(content=choice.content, role=openai_role, reasoning=reasoning)
        delta.tool_calls = delta_tool_calls

        choices.append(
            ChunkChoice(
                index=i,
                delta=delta,
                finish_reason=None,
            )
        )

    return ChatCompletionChunk(
        id=chunk.proto.id,
        choices=choices,
        created=chunk.proto.created.seconds,
        model=chunk.proto.model,
        object="chat.completion.chunk",
        usage=CompletionUsage(
            prompt_tokens=chunk.proto.usage.prompt_text_tokens,
            completion_tokens=chunk.proto.usage.completion_tokens,
            total_tokens=chunk.proto.usage.total_tokens,
        ),
    )


def _convert_xai_completion_to_anyllm_response(response: XaiResponse) -> ChatCompletion:
    reasoning = Reasoning(content=response.reasoning_content) if response.reasoning_content else None

    tool_calls_resp = response.tool_calls
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    if tool_calls_resp:
        calls: list[ChatCompletionMessageToolCall] = []
        for tc in tool_calls_resp:
            func = tc.function

            calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=tc.id,
                    type="function",
                    function=Function(name=func.name, arguments=func.arguments),
                )
            )
        tool_calls = calls or None

    message = ChatCompletionMessage(
        role="assistant",
        content=response.content,
        tool_calls=tool_calls,
        reasoning=reasoning,
    )

    choice = Choice(index=0, finish_reason="tool_calls" if tool_calls else "stop", message=message)

    usage = CompletionUsage(
        prompt_tokens=response.proto.usage.prompt_text_tokens,
        completion_tokens=response.proto.usage.completion_tokens,
        total_tokens=response.proto.usage.total_tokens,
    )

    return ChatCompletion(
        id=response.id,
        model=response.proto.model,
        created=response.proto.created.seconds,
        object="chat.completion",
        choices=[choice],
        usage=usage,
    )


def _convert_openai_tools_to_xai_tools(tools: Sequence[dict[str, Any]]) -> list[xai_chat_pb2.Tool]:
    xai_tools: list[xai_chat_pb2.Tool] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn_spec = t.get("function") if t.get("type") == "function" else None
        if not isinstance(fn_spec, dict):
            msg = f"Invalid tool: {t}"
            raise ValueError(msg)

        name = str(fn_spec.get("name", ""))
        description = str(fn_spec.get("description", ""))
        parameters = fn_spec.get("parameters")
        if not isinstance(parameters, dict):
            msg = f"Invalid parameters: {parameters}"
            raise ValueError(msg)

        xai_tools.append(xai_make_tool(name=name, description=description, parameters=parameters))
    return xai_tools


def _convert_models_list(models_list: Sequence[xai_models_pb2.LanguageModel]) -> list[Model]:
    return [
        Model(id=model.name, object="model", created=model.created.seconds, owned_by="xai") for model in models_list
    ]
