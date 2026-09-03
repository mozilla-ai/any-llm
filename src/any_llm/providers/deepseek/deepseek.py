import re
from collections.abc import AsyncIterator
from typing import Any

from typing_extensions import override

from any_llm.exceptions import InvalidRequestError
from any_llm.providers.deepseek.utils import (
    _inject_cached_tokens,
    _inject_cached_tokens_chunk,
    _inject_reasoning_extra_content,
    _preprocess_messages,
)
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, ReasoningEffort

# DeepSeek Chat accepts low, high, and max. Its compatibility mapping collapses medium and
# xhigh to high. OpenAI's minimal value is not part of the DeepSeek Chat contract.
# https://api-docs.deepseek.com/guides/thinking_mode/
_REASONING_EFFORT_MAP: dict[ReasoningEffort, str] = {
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "high",
    "max": "max",
}

# These normalized fields are absent from the current DeepSeek Chat schema, except for the two
# penalty fields, which the API marks deprecated and ineffective. They remain accepted by
# any_llm's shared interface for compatibility but are not sent to DeepSeek.
# https://api-docs.deepseek.com/api/create-chat-completion
_UNSUPPORTED_DEEPSEEK_FIELDS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "prompt_cache_key",
        "seed",
        "service_tier",
    }
)


class DeepseekProvider(BaseOpenAIProvider):
    API_BASE = "https://api.deepseek.com"
    ENV_API_KEY_NAME = "DEEPSEEK_API_KEY"
    ENV_API_BASE_NAME = "DEEPSEEK_API_BASE"
    PROVIDER_NAME = "deepseek"
    PROVIDER_DOCUMENTATION_URL = "https://platform.deepseek.com/"

    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False  # DeepSeek doesn't host an embedding model
    SUPPORTS_COMPLETION_REASONING = True

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """DeepSeek only accepts ``max_tokens``, not ``max_completion_tokens``.

        DeepSeek's V4 models default to enabled thinking with high effort, so ``None`` and the
        normalized ``auto`` sentinel leave both controls absent. An explicit ``none`` uses the
        provider's thinking toggle. Caller-supplied ``extra_body`` values take precedence.
        """
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if "max_completion_tokens" in converted_params:
            converted_params["max_tokens"] = converted_params.pop("max_completion_tokens")

        user_id = converted_params.pop("user", None)
        for field in _UNSUPPORTED_DEEPSEEK_FIELDS:
            converted_params.pop(field, None)

        converted_params.pop("reasoning_effort", None)
        reasoning_effort = params.reasoning_effort
        thinking: dict[str, str] | None = None
        if reasoning_effort == "none":
            thinking = {"type": "disabled"}
        elif reasoning_effort not in (None, "auto"):
            mapped_effort = _REASONING_EFFORT_MAP.get(reasoning_effort)
            if mapped_effort is None:
                msg = f"reasoning_effort {reasoning_effort!r} is not supported by DeepSeek Chat"
                raise InvalidRequestError(msg, provider_name=DeepseekProvider.PROVIDER_NAME)
            converted_params["reasoning_effort"] = mapped_effort
            thinking = {"type": "enabled"}

        if user_id is not None or thinking is not None:
            extra_body = converted_params.get("extra_body")
            if extra_body is None:
                extra_body = {}
                converted_params["extra_body"] = extra_body
            if user_id is not None and "user_id" not in extra_body:
                # DeepSeek's user_id contract is stricter than any-llm's shared user field.
                # https://api-docs.deepseek.com/quick_start/rate_limit/#setting-user_id
                if re.fullmatch(r"[a-zA-Z0-9_-]{1,512}", user_id) is None:
                    msg = (
                        "DeepSeek user_id must contain only ASCII letters, digits, underscores, or hyphens "
                        "and be at most 512 characters"
                    )
                    raise InvalidRequestError(msg, provider_name=DeepseekProvider.PROVIDER_NAME)
                extra_body["user_id"] = user_id
            if thinking is not None:
                extra_body.setdefault("thinking", thinking)
        return converted_params

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        result = BaseOpenAIProvider._convert_completion_response(response)
        result = _inject_cached_tokens(result)
        return _inject_reasoning_extra_content(result)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        result = BaseOpenAIProvider._convert_completion_chunk_response(response, **kwargs)
        return _inject_cached_tokens_chunk(result)

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        return await super()._acompletion(_preprocess_messages(params), **kwargs)
