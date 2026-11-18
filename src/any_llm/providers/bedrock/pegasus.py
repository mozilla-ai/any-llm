"""Chat with the TwelveLabs Pegasus video models, which Bedrock serves only on InvokeModel.

Request and response bodies:
https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html and
https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus-1-5.html
"""

from __future__ import annotations

from time import time
from typing import Any, Literal

from any_llm.exceptions import InvalidRequestError, UnsupportedParameterError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionParams,
    CompletionUsage,
)
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type
from any_llm.utils.token_limit import resolve_output_token_limit

_PEGASUS = "twelvelabs.pegasus-"
# Pegasus 1.2 takes the schema alone under responseFormat; 1.5 also needs responseFormat.type.
_PEGASUS_1_2 = "twelvelabs.pegasus-1-2-"

_PROMPT_ROLES = frozenset({"system", "developer", "user"})
_FinishReason = Literal["stop", "length", "content_filter"]
_FINISH_REASONS: dict[str, _FinishReason] = {"stop": "stop", "length": "length"}
# InvokeModel adds this field to the body when a guardrail is applied to the request.
_GUARDRAIL_ACTION = "amazon-bedrock-guardrailAction"


def is_pegasus_model(model_id: str) -> bool:
    """Return whether ``model_id`` is a Pegasus model or one of its inference profiles."""
    return _PEGASUS in model_id


def _media_source(part: dict[str, Any], model_id: str, provider_name: str) -> dict[str, Any]:
    video = part.get("video_url")
    url = video.get("url") if isinstance(video, dict) else None
    if not isinstance(video, dict) or not isinstance(url, str):
        msg = "A video_url part needs video_url.url."
        raise InvalidRequestError(msg, provider_name=provider_name)
    if url.startswith("s3://"):
        location = {"uri": url}
        if bucket_owner := video.get("bucket_owner"):
            location["bucketOwner"] = bucket_owner
        elif _PEGASUS_1_2 not in model_id:
            msg = "Pegasus 1.5 needs the bucket owner's account ID beside an s3:// video, as video_url.bucket_owner."
            raise InvalidRequestError(msg, provider_name=provider_name)
        return {"s3Location": location}
    if url.startswith("data:") and ";base64," in url:
        return {"base64String": url.split(";base64,", 1)[1]}
    msg = f"Pegasus takes a video as a base64 data URL or an s3:// URI. Got: {url[:100]}"
    raise InvalidRequestError(msg, provider_name=provider_name)


def _prompt_and_video(params: CompletionParams, model_id: str, provider_name: str) -> tuple[str, dict[str, Any]]:
    """Join the text of the messages into one prompt and find the one video.

    Pegasus answers one prompt about one video, so system and developer text goes before the
    user text, and there is no place for an assistant or tool turn.
    """
    texts: list[str] = []
    videos: list[dict[str, Any]] = []
    for message in params.messages:
        role = message.get("role")
        if role not in _PROMPT_ROLES:
            parameter_name = f"{role} messages"
            raise UnsupportedParameterError(
                parameter_name, provider_name, "Pegasus answers one prompt about one video, with no other turns."
            )
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            msg = "Message content must be a string or a list of content parts."
            raise InvalidRequestError(msg, provider_name=provider_name)
        for part in content:
            if not isinstance(part, dict):
                msg = "Each content part must be an object with a type."
                raise InvalidRequestError(msg, provider_name=provider_name)
            if part.get("type") == "text":
                text = part.get("text", "")
                if not isinstance(text, str):
                    msg = "A text part's text must be a string."
                    raise InvalidRequestError(msg, provider_name=provider_name)
                texts.append(text)
            elif part.get("type") == "video_url":
                videos.append(_media_source(part, model_id, provider_name))
            else:
                parameter_name = f"{part.get('type')} content"
                raise UnsupportedParameterError(parameter_name, provider_name, "Pegasus takes text and one video.")

    if len(videos) != 1:
        msg = f"Pegasus takes one video as a video_url content part, got {len(videos)}."
        raise InvalidRequestError(msg, provider_name=provider_name)
    prompt = "\n\n".join(text for text in texts if text)
    if not prompt:
        msg = "Pegasus needs a text prompt beside the video."
        raise InvalidRequestError(msg, provider_name=provider_name)
    return prompt, videos[0]


def _json_schema(response_format: dict[str, Any] | type, provider_name: str) -> dict[str, Any] | None:
    if is_structured_output_type(response_format):
        return get_json_schema(response_format)
    if isinstance(response_format, dict):
        if response_format.get("type") == "text":
            return None
        json_schema = response_format.get("json_schema")
        schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
        if response_format.get("type") == "json_schema" and isinstance(schema, dict):
            return schema
    parameter_name = "response_format"
    raise UnsupportedParameterError(
        parameter_name, provider_name, "Pegasus takes a JSON schema, as a Pydantic model, a dataclass or json_schema."
    )


def _check_options(params: CompletionParams, provider_name: str) -> None:
    unsupported = {
        "tools": params.tools,
        "tool_choice": params.tool_choice,
        "top_p": params.top_p,
        "stop": params.stop,
        "reasoning_effort": None if params.reasoning_effort in ("auto", "none") else params.reasoning_effort,
        "seed": params.seed,
        "logprobs": params.logprobs,
        "top_logprobs": params.top_logprobs,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
        "logit_bias": params.logit_bias,
    }
    for parameter_name, value in unsupported.items():
        if value is not None:
            raise UnsupportedParameterError(parameter_name, provider_name, "Pegasus does not take this parameter.")


def _check_extra_fields(extra_fields: dict[str, Any], provider_name: str) -> None:
    """Refuse the Pegasus 1.5 fields that replace the prompt the chat messages make."""
    if extra_fields.get("analysis_mode", "general") != "general":
        parameter_name = "analysis_mode"
        raise UnsupportedParameterError(
            parameter_name,
            provider_name,
            "Chat reads the message of the general mode; time_based_metadata takes no prompt and returns segments.",
        )
    if "prompt_v2" in extra_fields:
        parameter_name = "prompt_v2"
        raise UnsupportedParameterError(
            parameter_name, provider_name, "The chat messages make inputPrompt, and Pegasus takes only one of the two."
        )


def pegasus_body(
    params: CompletionParams, model_id: str, provider_name: str, extra_fields: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the InvokeModel body for a Pegasus model from a chat completion request.

    ``model_id`` names the Pegasus model whose version shapes the body. It differs from
    ``params.model_id`` when the request names an inference profile ARN.

    The video is a ``video_url`` content part whose URL is a base64 data URL, or an ``s3://``
    URI with a ``bucket_owner`` account ID beside it, which Pegasus 1.5 requires.
    ``extra_fields`` is merged into the body, and may not set a field the chat request sets.
    """
    extra_fields = extra_fields or {}
    _check_options(params, provider_name)
    _check_extra_fields(extra_fields, provider_name)
    prompt, media_source = _prompt_and_video(params, model_id, provider_name)
    body: dict[str, Any] = {"inputPrompt": prompt, "mediaSource": media_source}
    if params.temperature is not None:
        body["temperature"] = params.temperature
    if max_tokens := resolve_output_token_limit(params):
        body["maxOutputTokens"] = max_tokens
    if params.response_format is not None:
        schema = _json_schema(params.response_format, provider_name)
        if schema is not None:
            response_format: dict[str, Any] = {"jsonSchema": schema}
            if _PEGASUS_1_2 not in model_id:
                response_format = {"type": "json_schema", **response_format}
            body["responseFormat"] = response_format
    if clashes := sorted(extra_fields.keys() & body.keys()):
        msg = (
            f"additionalModelRequestFields sets {', '.join(clashes)}, which the chat request already sets. "
            "Use the messages and chat parameters for these."
        )
        raise InvalidRequestError(msg, provider_name=provider_name)
    return {**body, **extra_fields}


def _finish_reason(payload: dict[str, Any], *, final: bool) -> _FinishReason | None:
    """The finish reason of a response, or of a stream event when it is the last one."""
    # Pegasus 1.5 names the field stopReason, and 1.2 names it finishReason.
    reason = payload.get("stopReason") or payload.get("finishReason")
    if not reason and not final:
        return None
    if payload.get(_GUARDRAIL_ACTION) == "INTERVENED":
        return "content_filter"
    return _FINISH_REASONS.get(reason or "stop", "stop")


def _usage(input_tokens: Any, output_tokens: Any) -> CompletionUsage | None:
    if input_tokens is None and output_tokens is None:
        return None
    prompt_tokens, completion_tokens = int(input_tokens or 0), int(output_tokens or 0)
    return CompletionUsage(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens
    )


def convert_pegasus_response(payload: dict[str, Any], model_id: str, request_id: str) -> ChatCompletion:
    """Read a Pegasus InvokeModel response as a chat completion."""
    return ChatCompletion(
        id=request_id,
        object="chat.completion",
        created=int(time()),
        model=model_id,
        choices=[
            Choice(
                index=0,
                finish_reason=_finish_reason(payload, final=True) or "stop",
                message=ChatCompletionMessage(role="assistant", content=payload.get("message", "")),
            )
        ],
        # Pegasus 1.2 reports no token counts in the body.
        usage=_usage(payload.get("input_tokens"), payload.get("output_tokens")),
    )


def convert_pegasus_stream_event(payload: dict[str, Any], model_id: str, request_id: str) -> list[ChatCompletionChunk]:
    """Read one Pegasus stream event as chat completion chunks.

    A text delta or the finish reason makes one chunk. The invocation metrics on the last event
    make a usage chunk of its own with no choices, as the Converse stream does. In those metrics
    Bedrock counts 0 input tokens for a request whose InvokeModel body reports more; Pegasus bills
    by the second of input video, not by input token.
    """
    chunks: list[ChatCompletionChunk] = []
    created = int(time())
    text = payload.get("message") or None
    finish_reason = _finish_reason(payload, final=False)
    if text is not None or finish_reason is not None:
        chunks.append(
            ChatCompletionChunk(
                id=request_id,
                object="chat.completion.chunk",
                created=created,
                model=model_id,
                choices=[
                    ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content=text), finish_reason=finish_reason)
                ],
            )
        )
    metrics = payload.get("amazon-bedrock-invocationMetrics")
    if isinstance(metrics, dict):
        chunks.append(
            ChatCompletionChunk(
                id=request_id,
                object="chat.completion.chunk",
                created=created,
                model=model_id,
                choices=[],
                usage=_usage(metrics.get("inputTokenCount"), metrics.get("outputTokenCount")),
            )
        )
    return chunks
