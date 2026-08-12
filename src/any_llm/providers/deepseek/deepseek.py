from collections.abc import AsyncIterator
from typing import Any

from typing_extensions import override

from any_llm.providers.deepseek.utils import (
    _inject_cached_tokens,
    _inject_cached_tokens_chunk,
    _inject_reasoning_extra_content,
    _preprocess_messages,
)
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams

# The two legacy API model names being discontinued 2026-07-24 in favor of deepseek-v4-flash /
# deepseek-v4-pro. See https://api-docs.deepseek.com/updates/#date-2026-04-24. They hard-code
# their own thinking behavior (non-thinking / thinking respectively) and don't need, and may not
# accept, the `thinking` request toggle added below for the new model family.
_LEGACY_MODEL_IDS = frozenset({"deepseek-chat", "deepseek-reasoner"})


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

        Also maps ``reasoning_effort`` to DeepSeek's ``thinking`` toggle for the V4 model
        family. DeepSeek's V4 models default to thinking mode ENABLED when the toggle is
        omitted from the request (see https://api-docs.deepseek.com/guides/thinking_mode), so
        any_llm explicitly defaults it to disabled here -- matching the legacy ``deepseek-chat``
        behavior -- unless the caller opts in via ``reasoning_effort``. A caller-supplied
        ``extra_body`` override is respected and not clobbered.
        """
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if "max_completion_tokens" in converted_params:
            converted_params["max_tokens"] = converted_params.pop("max_completion_tokens")

        if params.model_id not in _LEGACY_MODEL_IDS:
            # ``"auto"`` means "no explicit reasoning requested" (BaseOpenAIProvider._acompletion
            # normalizes it to this provider's default before we run), so it is treated the same
            # as ``None``/``"none"`` here -- matching every other provider's converter and keeping
            # this self-contained even if called directly with ``"auto"``.
            thinking_disabled = params.reasoning_effort in (None, "none", "auto")
            extra_body = converted_params.setdefault("extra_body", {})
            extra_body.setdefault("thinking", {"type": "disabled" if thinking_disabled else "enabled"})
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
