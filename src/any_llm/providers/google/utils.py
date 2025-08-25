import json
from time import time
from typing import Any

from google.genai import types
from google.genai.pagers import Pager

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    CreateEmbeddingResponse,
    Embedding,
    Reasoning,
    Usage,
)
from any_llm.types.model import Model


def _convert_tool_spec(openai_tools: list[dict[str, Any]]) -> list[types.Tool]:
    """Convert OpenAI tool specification to Google GenAI format."""
    function_declarations = []

    for tool in openai_tools:
        if tool.get("type") != "function":
            continue

        function = tool["function"]
        # Preserve nested schema details such as items/additionalProperties for arrays/objects
        properties: dict[str, dict[str, Any]] = {}
        for param_name, param_info in function["parameters"]["properties"].items():
            prop: dict[str, Any] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            # Google requires explicit items for arrays
            if "items" in param_info:
                prop["items"] = param_info["items"]
            if prop.get("type") == "array" and "items" not in prop:
                prop["items"] = {"type": "string"}
            # Google tool schema does not accept additionalProperties; drop it
            properties[param_name] = prop

        parameters_dict = {
            "type": "object",
            "properties": properties,
            "required": function["parameters"].get("required", []),
        }

        function_declarations.append(
            types.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters=types.Schema(**parameters_dict),
            )
        )

    return [types.Tool(function_declarations=function_declarations)]


def _convert_tool_choice(tool_choice: str) -> types.ToolConfig:
    tool_choice_to_mode = {
        "required": types.FunctionCallingConfigMode.ANY,
        "auto": types.FunctionCallingConfigMode.AUTO,
    }

    return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=tool_choice_to_mode[tool_choice]))


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[types.Content], str | None]:
    """Convert messages to Google GenAI format."""
    formatted_messages = []
    system_instruction = None

    for message in messages:
        if message["role"] == "system":
            if system_instruction is None:
                system_instruction = message["content"]
            else:
                system_instruction += f"\n{message['content']}"
        elif message["role"] == "user":
            parts = [types.Part.from_text(text=message["content"])]
            formatted_messages.append(types.Content(role="user", parts=parts))
        elif message["role"] == "assistant":
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]  # Assuming single function call for now
                function_call = tool_call["function"]

                parts = [
                    types.Part.from_function_call(
                        name=function_call["name"], args=json.loads(function_call["arguments"])
                    )
                ]
            else:
                parts = [types.Part.from_text(text=message["content"])]

            formatted_messages.append(types.Content(role="model", parts=parts))
        elif message["role"] == "tool":
            try:
                content_json = json.loads(message["content"])
                part = types.Part.from_function_response(name=message.get("name", "unknown"), response=content_json)
                formatted_messages.append(types.Content(role="function", parts=[part]))
            except json.JSONDecodeError:
                part = types.Part.from_function_response(
                    name=message.get("name", "unknown"), response={"result": message["content"]}
                )
                formatted_messages.append(types.Content(role="function", parts=[part]))

    return formatted_messages, system_instruction


def _convert_response_to_response_dict(response: types.GenerateContentResponse) -> dict[str, Any]:
    response_dict = {
        "id": "google_genai_response",
        "model": "google/genai",
        "created": 0,
        "usage": {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0)
            if hasattr(response, "usage_metadata")
            else 0,
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
            if hasattr(response, "usage_metadata")
            else 0,
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
            if hasattr(response, "usage_metadata")
            else 0,
        },
    }

    choices: list[dict[str, Any]] = []
    if (
        response.candidates
        and len(response.candidates) > 0
        and response.candidates[0].content
        and response.candidates[0].content.parts
        and len(response.candidates[0].content.parts) > 0
    ):
        reasoning = None
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", None):
                reasoning = part.text
            elif function_call := getattr(part, "function_call", None):
                args_dict = {}
                if args := getattr(function_call, "args", None):
                    for key, value in args.items():
                        args_dict[key] = value

                choices.append(
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": reasoning,
                            "tool_calls": [
                                {
                                    "id": f"call_{hash(function_call.name)}",
                                    "function": {
                                        "name": function_call.name,
                                        "arguments": json.dumps(args_dict),
                                    },
                                    "type": "function",
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                        "index": 0,
                    }
                )
            elif getattr(part, "text", None):
                choices.append(
                    {
                        "message": {
                            "role": "assistant",
                            "content": part.text,
                            "reasoning": reasoning,
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                )

    response_dict["choices"] = choices

    return response_dict


def _create_openai_embedding_response_from_google(
    model: str, result: types.EmbedContentResponse
) -> CreateEmbeddingResponse:
    """Convert a Google embedding response to an OpenAI-compatible format."""

    data = [
        Embedding(
            embedding=embedding.values,
            index=i,
            object="embedding",
        )
        for i, embedding in enumerate(result.embeddings or [])
        if embedding.values
    ]

    # Google does not provide usage data in the embedding response
    usage = Usage(prompt_tokens=0, total_tokens=0)

    return CreateEmbeddingResponse(
        data=data,
        model=model,
        object="list",
        usage=usage,
    )


def _create_openai_chunk_from_google_chunk(
    response: types.GenerateContentResponse,
) -> ChatCompletionChunk:
    """Convert a Google GenerateContentResponse to an OpenAI ChatCompletionChunk."""

    assert response.candidates
    candidate = response.candidates[0]
    assert candidate.content
    assert candidate.content.parts

    content = ""
    reasoning_content = ""

    for part in candidate.content.parts:
        if part.thought:
            # This is a thinking/reasoning part
            reasoning_content += part.text or ""
        else:
            # Regular content part
            content += part.text or ""

    delta = ChoiceDelta(
        content=content or None,
        role="assistant",
        reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
    )

    choice = ChunkChoice(
        index=0,
        delta=delta,
        finish_reason="stop" if getattr(candidate.finish_reason, "value", None) == "STOP" else None,
    )

    return ChatCompletionChunk(
        id=f"chatcmpl-{time()}",  # Google doesn't provide an ID in the chunk
        choices=[choice],
        created=int(time()),
        model=str(response.model_version),
        object="chat.completion.chunk",
    )


def _convert_models_list(models_list: Pager[types.Model]) -> list[Model]:
    # Google doesn't provide a creation date for models
    return [Model(id=model.name or "Unknown", object="model", created=0, owned_by="google") for model in models_list]
