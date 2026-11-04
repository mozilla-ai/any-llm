import json
from collections.abc import AsyncIterator
from typing import Any

from typing_extensions import override

from any_llm.exceptions import InvalidRequestError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CompletionUsage,
    PromptTokensDetails,
)
from any_llm.utils.aio import aclose_quietly
from any_llm.utils.reasoning import strip_extra_content
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type


def _asks_for_json(response_format: Any) -> bool:
    """Whether ``response_format`` asks for JSON output that DeepSeek must serve as JSON mode.

    DeepSeek's ``response_format`` takes only ``text`` and ``json_object``, so a type and an
    OpenAI ``json_schema`` dict (which the Messages bridge produces) both become JSON mode.
    https://api-docs.deepseek.com/api/create-chat-completion
    """
    if is_structured_output_type(response_format):
        return True
    return isinstance(response_format, dict) and response_format.get("type") == "json_schema"


def _json_schema_of(response_format: Any) -> dict[str, Any] | None:
    """The JSON schema a JSON-mode ``response_format`` carries, or None when it names none.

    OpenAI's ``json_schema`` format requires the ``json_schema`` object but not its
    ``schema``, so a format without one asks for JSON with no shape.

    Raises:
        InvalidRequestError: when the ``json_schema`` object is missing or its ``schema``
            is not a JSON schema object, which no request could mean.

    """
    if is_structured_output_type(response_format):
        return get_json_schema(response_format)
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        msg = "response_format of type 'json_schema' must include a 'json_schema' object"
        raise InvalidRequestError(msg, provider_name="deepseek")
    if "schema" not in json_schema:
        return None
    schema = json_schema["schema"]
    if not isinstance(schema, dict):
        msg = "response_format 'json_schema.schema' must be a JSON schema object"
        raise InvalidRequestError(msg, provider_name="deepseek")
    return schema


def _convert_schema_to_deepseek_json(
    schema: dict[str, Any] | None, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert a JSON schema, or a request for unconstrained JSON, to DeepSeek JSON format.

    DeepSeek requires:
    1. response_format = {'type': 'json_object'}
    2. The word "json" in the prompt
    3. An example of the desired JSON structure, given here when there is a schema

    Following the instructions in the DeepSeek documentation:
    https://api-docs.deepseek.com/guides/json_mode

    Returns:
        modified_messages
    """
    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        if schema is None:
            shape = "Please respond with a JSON object."
        else:
            shape = f"Please respond with a JSON object that matches the following schema:\n\n{json.dumps(schema, indent=2)}"
        json_instruction = f"""
{shape}

Return the JSON object only, no other text, do not wrap it in ```json or ```.
"""
        if isinstance(original_content, list):
            # Content parts (an image among them) keep their shape; the instruction leads as text.
            content: str | list[Any] = [{"type": "text", "text": json_instruction}, *original_content]
        else:
            content = f"{json_instruction}\n{original_content}\n"
        modified_messages[-1] = {**modified_messages[-1], "content": content}
    else:
        msg = "Last message is not a user message"
        raise ValueError(msg)

    return modified_messages


def _reinject_reasoning_content(messages: list[dict[str, Any]], *, replay_reasoning: bool) -> list[dict[str, Any]]:
    """Restore ``reasoning_content`` when the current request carries tools.

    DeepSeek requires all previous assistant reasoning to be replayed when ``tools`` is present,
    including turns that did not call a tool. any_llm's shared message serialization strips the
    normalized ``reasoning`` field, so this restores it from the provider side-channel populated
    by ``_inject_reasoning_extra_content``.

    Reference: https://api-docs.deepseek.com/guides/thinking_mode#tool-calls

    ``extra_content`` is read here and then dropped with ``strip_extra_content``, so only
    ``reasoning_content`` reaches DeepSeek's API.
    """
    result = []
    for message, cleaned in zip(messages, strip_extra_content(messages), strict=True):
        if replay_reasoning and message.get("role") == "assistant":
            extra_content = message.get("extra_content")
            deepseek_extra = extra_content.get("deepseek") if isinstance(extra_content, dict) else None
            if isinstance(deepseek_extra, dict) and isinstance(deepseek_extra.get("reasoning_content"), str):
                result.append({**cleaned, "reasoning_content": deepseek_extra["reasoning_content"]})
                continue
        result.append(cleaned)
    return result


def _preprocess_messages(params: CompletionParams) -> CompletionParams:
    """Preprocess messages"""
    if params.response_format and _asks_for_json(params.response_format):
        params.messages = _convert_schema_to_deepseek_json(_json_schema_of(params.response_format), params.messages)
        params.response_format = {"type": "json_object"}

    params.messages = _reinject_reasoning_content(params.messages, replay_reasoning=params.tools is not None)

    return params


def _apply_cache_hit_tokens(usage: CompletionUsage) -> None:
    """Populate ``prompt_tokens_details.cached_tokens`` from DeepSeek's ``prompt_cache_hit_tokens``.

    DeepSeek's ``prompt_tokens`` already includes cached tokens
    (``prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens``). A reported zero is kept so
    it stays distinguishable from a response without cache accounting.

    Reference: https://api-docs.deepseek.com/api/create-chat-completion
    """
    cached = getattr(usage, "prompt_cache_hit_tokens", None)
    if not isinstance(cached, int):
        return
    details = usage.prompt_tokens_details or PromptTokensDetails()
    details.cached_tokens = cached
    usage.prompt_tokens_details = details


def _inject_cached_tokens(completion: ChatCompletion) -> ChatCompletion:
    if completion.usage is not None:
        _apply_cache_hit_tokens(completion.usage)
    return completion


def _inject_cached_tokens_chunk(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
    if chunk.usage is not None:
        _apply_cache_hit_tokens(chunk.usage)
    return chunk


def _inject_reasoning_extra_content(completion: ChatCompletion) -> ChatCompletion:
    """Stash reasoning text into ``extra_content`` so it survives any_llm's outgoing message dump.

    See ``_reinject_reasoning_content`` for why this round-trip is necessary.
    """
    for choice in completion.choices:
        reasoning = choice.message.reasoning
        if reasoning is not None:
            choice.message.extra_content = {"deepseek": {"reasoning_content": reasoning.content}}
    return completion


class ReasoningReplayStream(AsyncIterator[ChatCompletionChunk]):
    """Stash a streamed turn's reasoning into ``extra_content`` on the chunk that finishes the choice.

    DeepSeek sends ``reasoning_content`` in fragments, but it must be replayed whole (see
    ``_reinject_reasoning_content``). The fragments pass through unchanged; the complete text is
    added once to the choice's finishing delta, so ``ChatCompletionAccumulator`` yields the same
    ``message.extra_content`` as a non-streamed response.
    """

    def __init__(self, stream: AsyncIterator[ChatCompletionChunk]) -> None:
        self._stream = stream
        self._reasoning: dict[int, list[str]] = {}

    @override
    def __aiter__(self) -> "ReasoningReplayStream":
        return self

    @override
    async def __anext__(self) -> ChatCompletionChunk:
        chunk = await anext(self._stream)
        for choice in chunk.choices:
            if choice.delta.reasoning is not None and choice.delta.reasoning.content:
                self._reasoning.setdefault(choice.index, []).append(choice.delta.reasoning.content)
            if choice.finish_reason is not None and (fragments := self._reasoning.pop(choice.index, None)):
                choice.delta.extra_content = {
                    **(choice.delta.extra_content or {}),
                    "deepseek": {"reasoning_content": "".join(fragments)},
                }
        return chunk

    async def aclose(self) -> None:
        await aclose_quietly(self._stream)
