"""z.ai Provider Utilities."""

from typing import Any, Literal

# z.ai documents choices[].finish_reason as one of stop, tool_calls, length,
# sensitive, model_context_window_exceeded or network_error:
# https://docs.z.ai/api-reference/llm/chat-completion
# The last three are outside the OpenAI literal set, so a GLM response carrying one
# fails validation in ChatCompletion.model_validate. The equivalences below are
# inferred from the failure reported in #1200 rather than described by z.ai, which
# enumerates the values without defining them: model_context_window_exceeded arrives
# on a 200 mid generation with content already delivered, which is truncation, and
# sensitive is a safety interception, which is what content_filter reports.
# network_error is deliberately absent: a transport failure is not a completion
# reason and no legal value fits.
_FINISH_REASON_MAP: dict[str, Literal["length", "content_filter"]] = {
    "model_context_window_exceeded": "length",
    "sensitive": "content_filter",
}


def _normalize_finish_reason(response: Any) -> None:
    """Rewrite z.ai specific finish reasons on an SDK response object, in place.

    The OpenAI SDK builds responses leniently through ``construct_type``, so the raw
    string survives to our stricter ``model_validate``. Values outside the table are
    left alone so unexpected stop reasons keep failing instead of being coerced.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        return
    for choice in choices:
        finish_reason = getattr(choice, "finish_reason", None)
        if not isinstance(finish_reason, str):
            continue
        mapped = _FINISH_REASON_MAP.get(finish_reason)
        if mapped is not None:
            choice.finish_reason = mapped