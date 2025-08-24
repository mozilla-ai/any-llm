from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openai.utils import _convert_chat_completion, _normalize_openai_dict_response
from any_llm.providers.openrouter.utils import build_reasoning_directive
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class OpenrouterProvider(BaseOpenAIProvider):

    API_BASE = "https://openrouter.ai/api/v1"
    ENV_API_KEY_NAME = "OPENROUTER_API_KEY"
    PROVIDER_NAME = "openrouter"
    PROVIDER_DOCUMENTATION_URL = "https://openrouter.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = False

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        messages = params.messages
        if messages and not isinstance(messages[0], dict):
            messages = [msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else msg for msg in messages]

        model = params.model_id or getattr(params, "model", None)
        if not model:
            msg = "OpenrouterProvider: `model` (or `model_id`) is required"
            raise ValueError(msg)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "stream": bool(params.stream),
        }
        body = {k: v for k, v in body.items() if v is not None}

        extra_body: dict[str, Any] = {}

        reasoning_directive = build_reasoning_directive(
            reasoning=(getattr(params, "reasoning", None) or kwargs.get("reasoning")),
            include_reasoning=kwargs.get("include_reasoning"),
            reasoning_effort=(getattr(params, "reasoning_effort", None) or kwargs.get("reasoning_effort")),
        )

        if reasoning_directive is not None:
            extra_body["reasoning"] = reasoning_directive

        client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.api_base or self.API_BASE)
        stream_mode = body.pop("stream", False)

        client_kwargs = body.copy()
        if extra_body:
            client_kwargs["extra_body"] = extra_body

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["include_reasoning", "reasoning", "reasoning_effort"]}
        client_kwargs.update(filtered_kwargs)

        if stream_mode:
            stream = await client.chat.completions.create(stream=True, **client_kwargs)

            async def _stream() -> AsyncIterator[ChatCompletionChunk]:
                async for chunk in stream:
                    raw = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    normalized = _normalize_openai_dict_response(raw)
                    yield ChatCompletionChunk.model_validate(normalized)

            return _stream()

        resp = await client.chat.completions.create(**client_kwargs)
        return _convert_chat_completion(resp)
