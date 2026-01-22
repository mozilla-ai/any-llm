# mypy: disable-error-code="union-attr"
"""Converter from OpenAI SDK Response to OpenResponses types.

The OpenAI SDK Response and OpenResponses ResponseResource have incompatible schemas:
- Different field names (prompt_tokens vs input_tokens)
- Required fields in OpenResponses that OpenAI returns as None
- Different nested structure schemas

This module provides conversion functions to bridge the gap.
"""

from __future__ import annotations

from typing import Any

from openai.types.responses import Response as OpenAIResponse  # noqa: TC002
from openresponses_types import ResponseResource


def _convert_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Convert OpenAI usage format to OpenResponses format."""
    if not usage:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    return {
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens") or 0,
        "total_tokens": usage.get("total_tokens") or 0,
        "input_tokens_details": usage.get("input_tokens_details") or {"cached_tokens": 0},
        "output_tokens_details": usage.get("output_tokens_details") or {"reasoning_tokens": 0},
    }


def _convert_content_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert content item, ensuring required fields are present."""
    result = dict(item)
    content_type = result.get("type", "")

    if content_type == "output_text":
        if result.get("annotations") is None:
            result["annotations"] = []
        if result.get("logprobs") is None:
            result["logprobs"] = []

    return result


def _convert_output_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert an output item, handling nested content."""
    result = dict(item)

    if "content" in result and isinstance(result["content"], list):
        result["content"] = [_convert_content_item(c) for c in result["content"]]

    return result


def _convert_reasoning(reasoning: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert reasoning config to OpenResponses format."""
    if not reasoning:
        return None
    result = dict(reasoning)
    result.setdefault("effort", "medium")
    result.setdefault("summary", None)
    return result


def _convert_text(text: dict[str, Any] | None) -> dict[str, Any]:
    """Convert text config, ensuring required structure is present."""
    if text:
        return text
    return {"format": {"type": "text"}}


def convert_response(openai_response: OpenAIResponse) -> ResponseResource:
    """Convert OpenAI SDK Response to OpenResponses ResponseResource type."""
    data = openai_response.model_dump(warnings=False)

    # Handle fields that OpenResponses requires as non-nullable but OpenAI returns as None
    top_p = data.get("top_p")
    temperature = data.get("temperature")
    presence_penalty = data.get("presence_penalty")
    frequency_penalty = data.get("frequency_penalty")
    top_logprobs = data.get("top_logprobs")
    truncation = data.get("truncation")

    converted: dict[str, Any] = {
        "id": data.get("id", ""),
        "object": data.get("object", "response"),
        "created_at": int(data.get("created_at", 0)),
        "completed_at": data.get("completed_at"),
        "status": data.get("status", "completed"),
        "incomplete_details": data.get("incomplete_details"),
        "model": data.get("model", ""),
        "previous_response_id": data.get("previous_response_id"),
        "instructions": data.get("instructions"),
        "error": data.get("error"),
        "tools": data.get("tools") or [],
        "tool_choice": data.get("tool_choice") or "auto",
        "truncation": truncation if truncation is not None else "auto",
        "parallel_tool_calls": data.get("parallel_tool_calls", True),
        "text": _convert_text(data.get("text")),
        "top_p": top_p if top_p is not None else 1.0,
        "presence_penalty": presence_penalty if presence_penalty is not None else 0.0,
        "frequency_penalty": frequency_penalty if frequency_penalty is not None else 0.0,
        "top_logprobs": top_logprobs if top_logprobs is not None else 0,
        "temperature": temperature if temperature is not None else 1.0,
        "reasoning": _convert_reasoning(data.get("reasoning")),
        "usage": _convert_usage(data.get("usage")),
        "max_output_tokens": data.get("max_output_tokens"),
        "max_tool_calls": data.get("max_tool_calls"),
        "store": data.get("store") if data.get("store") is not None else True,
        "background": data.get("background") if data.get("background") is not None else False,
        "service_tier": data.get("service_tier") or "default",
        "metadata": data.get("metadata") or {},
        "safety_identifier": data.get("safety_identifier"),
        "prompt_cache_key": data.get("prompt_cache_key"),
    }

    # Convert output items
    output = data.get("output") or []
    converted["output"] = [_convert_output_item(item) for item in output]

    return ResponseResource.model_validate(converted)


def convert_stream_event(event: Any) -> dict[str, Any]:
    """Convert OpenAI SDK stream event to dict for OpenResponses streaming."""
    if hasattr(event, "model_dump"):
        return event.model_dump()  # type: ignore[no-any-return]
    return event  # type: ignore[no-any-return]
