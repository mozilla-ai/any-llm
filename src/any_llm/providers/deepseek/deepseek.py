from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from typing_extensions import override

from any_llm.exceptions import InvalidRequestError, UnsupportedParameterError
from any_llm.logging import logger
from any_llm.providers.deepseek.utils import (
    ReasoningReplayStream,
    _inject_cached_tokens,
    _inject_cached_tokens_chunk,
    _inject_reasoning_extra_content,
    _preprocess_messages,
)
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion
from any_llm.types.messages import MessageResponse
from any_llm.utils.messages_compat import (
    anthropic_client_kwargs,
    prepare_blocks_for_native_messages,
    stream_native_messages,
)
from any_llm.utils.structured_output import normalize_output_config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
    from anthropic.types.parsed_message import ParsedMessage

    from any_llm.types.completion import ChatCompletionChunk, CompletionParams, ReasoningEffort
    from any_llm.types.messages import MessagesParams, MessageStreamEvent

# Each entry maps an any-llm effort to DeepSeek's top-level effort and thinking toggle.
# DeepSeek Chat accepts low, high, and max, maps medium and xhigh to high, and does not
# accept OpenAI's minimal value.
# https://api-docs.deepseek.com/guides/thinking_mode/
_REASONING_CONTROLS: dict[ReasoningEffort | None, tuple[str | None, str | None]] = {
    None: (None, None),
    "auto": (None, None),
    "none": (None, "disabled"),
    "low": ("low", "enabled"),
    "medium": ("high", "enabled"),
    "high": ("high", "enabled"),
    "xhigh": ("high", "enabled"),
    "max": ("max", "enabled"),
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
        "seed",
        "service_tier",
    }
)


def _derive_anthropic_base(openai_base: str) -> str:
    """The Anthropic-compatible base URL beside an OpenAI-compatible one.

    DeepSeek serves the Anthropic API under ``/anthropic`` on the same host, and the
    Anthropic SDK appends ``/v1/messages`` itself, so a ``/v1`` on the OpenAI base is dropped.
    https://api-docs.deepseek.com/guides/anthropic_api
    """
    return openai_base.rstrip("/").removesuffix("/v1") + "/anthropic"


class DeepseekProvider(BaseOpenAIProvider):
    """DeepSeek provider.

    Chat Completions and Responses go through the inherited OpenAI-compatible machinery.
    Messages go to DeepSeek's Anthropic-compatible endpoint through the Anthropic SDK, so
    thinking blocks, tool use and server-side web search blocks keep their native shape.
    """

    API_BASE = "https://api.deepseek.com"
    ENV_API_KEY_NAME = "DEEPSEEK_API_KEY"
    ENV_API_BASE_NAME = "DEEPSEEK_API_BASE"
    PROVIDER_NAME = "deepseek"
    PROVIDER_DOCUMENTATION_URL = "https://platform.deepseek.com/"

    # deepseek-flash accepts images; deepseek-v4-pro does not, and answers an image with a 400.
    # https://api-docs.deepseek.com/guides/vision
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False  # DeepSeek doesn't host an embedding model
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_MESSAGES_NATIVE = True

    client: AsyncOpenAI
    _anthropic_client: AsyncAnthropic

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        resolved_base = api_base or self.API_BASE
        self.client = AsyncOpenAI(base_url=resolved_base, api_key=api_key, **kwargs)
        self._anthropic_client = AsyncAnthropic(
            api_key=api_key,
            base_url=_derive_anthropic_base(resolved_base),
            **anthropic_client_kwargs(kwargs),
        )

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """DeepSeek only accepts ``max_tokens``, not ``max_completion_tokens``.

        DeepSeek's V4 models default to enabled thinking with high effort, so ``None`` and the
        normalized ``auto`` sentinel leave both controls absent. An explicit ``none`` uses the
        provider's thinking toggle. Caller-supplied ``extra_body`` values take precedence.

        When thinking is enabled, DeepSeek rejects forced tool choices (``required`` or a
        named function) with HTTP 400. Raise ``InvalidRequestError`` for these choices after
        resolving caller ``extra_body.thinking`` overrides. ``auto`` and ``none`` are forwarded
        unchanged, as are all tool choices when thinking is disabled.
        """
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if "max_completion_tokens" in converted_params:
            converted_params["max_tokens"] = converted_params.pop("max_completion_tokens")

        user_id = converted_params.pop("user", None)
        for field in _UNSUPPORTED_DEEPSEEK_FIELDS:
            converted_params.pop(field, None)

        converted_params.pop("reasoning_effort", None)
        controls = _REASONING_CONTROLS.get(params.reasoning_effort)
        if controls is None:
            msg = f"reasoning_effort {params.reasoning_effort!r} is not supported by DeepSeek Chat"
            raise InvalidRequestError(msg, provider_name=DeepseekProvider.PROVIDER_NAME)
        reasoning_effort, thinking_type = controls
        if reasoning_effort is not None:
            converted_params["reasoning_effort"] = reasoning_effort
        thinking = {"type": thinking_type} if thinking_type is not None else None

        if user_id is not None or thinking is not None:
            extra_body = dict(converted_params.get("extra_body") or {})
            converted_params["extra_body"] = extra_body
            if user_id is not None and "user_id" not in extra_body:
                # DeepSeek's user_id contract is stricter than any-llm's shared user field.
                # https://api-docs.deepseek.com/quick_start/rate_limit/#setting-user_id
                if re.fullmatch(r"[a-zA-Z0-9_-]{1,512}", user_id) is None:
                    msg = (
                        "DeepSeek user_id must contain only ASCII letters, digits, underscores, or hyphens "
                        "and be between 1 and 512 characters"
                    )
                    raise InvalidRequestError(msg, provider_name=DeepseekProvider.PROVIDER_NAME)
                extra_body["user_id"] = user_id
            if thinking is not None:
                extra_body.setdefault("thinking", thinking)

        if "tool_choice" in converted_params:
            # Resolve the effective mode after the caller's own extra_body.thinking override
            # (via setdefault above) has had a chance to win over the reasoning_effort mapping.
            # DeepSeek V4 defaults thinking to enabled, so anything short of an explicit
            # "disabled" wire value counts as effectively enabled.
            final_extra_body = converted_params.get("extra_body")
            final_thinking = final_extra_body.get("thinking") if isinstance(final_extra_body, dict) else None
            final_thinking_type = final_thinking.get("type") if isinstance(final_thinking, dict) else None
            thinking_effectively_enabled = final_thinking_type != "disabled"

            tool_choice = converted_params["tool_choice"]
            forced_tool_choice = tool_choice == "required" or isinstance(tool_choice, dict)
            if thinking_effectively_enabled and forced_tool_choice:
                msg = (
                    f"tool_choice={tool_choice!r} is not supported while DeepSeek thinking is enabled; "
                    "DeepSeek returns HTTP 400 for required or named-function tool choices in thinking mode. "
                    "Disable thinking (extra_body={'thinking': {'type': 'disabled'}}) or use "
                    "tool_choice='auto' or 'none'."
                )
                raise InvalidRequestError(msg, provider_name=DeepseekProvider.PROVIDER_NAME)

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
        response = await super()._acompletion(_preprocess_messages(params), **kwargs)
        if isinstance(response, ChatCompletion):
            return response
        return ReasoningReplayStream(response)

    @override
    async def _amessages(
        self, params: MessagesParams, **kwargs: Any
    ) -> MessageResponse | ParsedMessage[Any] | ParsedBetaMessage[Any] | AsyncIterator[MessageStreamEvent]:
        """Send a Messages request to DeepSeek's Anthropic-compatible endpoint.

        ``betas`` travel as the ``anthropic-beta`` header, which DeepSeek ignores except to
        accept a Files API image reference. ``context_management`` is dropped: it configures
        Anthropic's server-side context editing, which DeepSeek does not document, and Claude
        Code sends it on every request. ``container`` is refused here rather than sent, because
        DeepSeek ignores it and runs no code execution. DeepSeek's ``output_config`` takes only
        ``effort``, so an effort-only ``output_format`` dict is sent natively and a schema goes
        through the Messages-to-Completions bridge.
        https://api-docs.deepseek.com/guides/anthropic_api
        """
        if params.container is not None:
            msg = "container"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)

        output_config: dict[str, Any] | None = None
        if isinstance(params.output_format, dict):
            output_config = normalize_output_config(params.output_format)
        if params.output_format is not None and (output_config is None or "format" in output_config):
            bridged = params.model_copy(update={"betas": None, "context_management": None})
            return await super()._amessages(bridged, **kwargs)

        if params.context_management is not None:
            logger.debug("DeepSeek does not support context_management; dropping it from the Messages request.")

        api_kwargs = params.model_dump(
            exclude_none=True, exclude={"stream", "betas", "context_management", "output_format"}
        )
        api_kwargs["messages"] = prepare_blocks_for_native_messages(api_kwargs["messages"])
        if output_config is not None:
            api_kwargs["output_config"] = output_config
        api_kwargs.update(kwargs)
        if params.betas:
            headers = dict(api_kwargs.get("extra_headers") or {})
            headers["anthropic-beta"] = ",".join(params.betas)
            api_kwargs["extra_headers"] = headers

        if params.stream:
            return self._stream_messages_async(**api_kwargs)

        message = await self._anthropic_client.messages.create(**api_kwargs)
        return MessageResponse.model_validate(message, from_attributes=True)

    def _stream_messages_async(self, **kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return stream_native_messages(self._anthropic_client, **kwargs)
