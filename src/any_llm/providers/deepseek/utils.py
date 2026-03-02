import json
from typing import Any

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, PromptTokensDetails
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type


def _convert_structured_type_to_deepseek_json(
    response_format: type, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert a structured type to DeepSeek JSON format.

    DeepSeek requires:
    1. response_format = {'type': 'json_object'}
    2. The word "json" in the prompt
    3. An example of the desired JSON structure

    Following the instructions in the DeepSeek documentation:
    https://api-docs.deepseek.com/guides/json_mode

    Returns:
        modified_messages
    """
    schema = get_json_schema(response_format)

    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        json_instruction = f"""
Please respond with a JSON object that matches the following schema:

{json.dumps(schema, indent=2)}

Return the JSON object only, no other text, do not wrap it in ```json or ```.

{original_content}
"""
        modified_messages[-1]["content"] = json_instruction
    else:
        msg = "Last message is not a user message"
        raise ValueError(msg)

    return modified_messages


def _preprocess_messages(params: CompletionParams) -> CompletionParams:
    """Preprocess messages"""
    if params.response_format:
        if is_structured_output_type(params.response_format):
            modified_messages = _convert_structured_type_to_deepseek_json(params.response_format, params.messages)
            params.response_format = {"type": "json_object"}
            params.messages = modified_messages

    return params


def _inject_cached_tokens(completion: ChatCompletion) -> ChatCompletion:
    """Populate ``prompt_tokens_details.cached_tokens`` from DeepSeek's ``prompt_cache_hit_tokens``.

    DeepSeek's ``prompt_tokens`` already includes cached tokens
    (``prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens``).

    Reference: https://api-docs.deepseek.com/api/create-chat-completion
    """
    if completion.usage is None:
        return completion
    cached = getattr(completion.usage, "prompt_cache_hit_tokens", None)
    if cached:
        completion.usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cached)
    return completion


def _inject_cached_tokens_chunk(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
    """Same as ``_inject_cached_tokens`` but for streaming chunks."""
    if chunk.usage is None:
        return chunk
    cached = getattr(chunk.usage, "prompt_cache_hit_tokens", None)
    if cached:
        chunk.usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cached)
    return chunk
