from collections.abc import Mapping
from typing import Any

from any_llm.types.completion import CompletionTokensDetails, CompletionUsage, PromptTokensDetails


def completion_usage_from_dict(usage: Mapping[str, Any]) -> CompletionUsage:
    """Build CompletionUsage from an OpenAI-shaped usage dict, keeping cached and reasoning token details.

    Provider SDKs often declare only the three token counts, so the details arrive as
    undeclared extras and are lost unless they are read from the dumped dict. A top-level
    ``cached_tokens`` is accepted where ``prompt_tokens_details.cached_tokens`` is missing.
    """
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = usage.get("cached_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return CompletionUsage(
        prompt_tokens=usage.get("prompt_tokens") or 0,
        completion_tokens=usage.get("completion_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens) if cached_tokens is not None else None,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=reasoning_tokens)
        if reasoning_tokens is not None
        else None,
    )
