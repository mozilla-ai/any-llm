from typing import Any, Iterator

try:
    from fireworks import LLM
except ImportError as exc:
    msg = "fireworks-ai is not installed. Please install it with `pip install any-llm-sdk[fireworks]`"
    raise ImportError(msg) from exc

from pydantic import BaseModel
from any_llm.types.completion import ChatCompletionChunk, ChatCompletion
from any_llm.provider import Provider
from any_llm.types.completion import ChatCompletionMessage, Choice, CompletionUsage
from any_llm.providers.fireworks.utils import _create_openai_chunk_from_fireworks_chunk


class FireworksProvider(Provider):
    PROVIDER_NAME = "Fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False

    def _stream_completion(
        self,
        llm: LLM,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response_generator = llm.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        for chunk in response_generator:
            yield _create_openai_chunk_from_fireworks_chunk(chunk)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        llm = LLM(
            model=model,
            deployment_type="auto",
            api_key=self.config.api_key,
        )

        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": response_format.__name__, "schema": response_format.model_json_schema()},
                }
            else:
                kwargs["response_format"] = response_format

        if kwargs.get("stream", False):
            return self._stream_completion(llm, messages, **kwargs)

        response = llm.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        response_data = response.model_dump()
        choices_out: list[Choice] = []
        for i, ch in enumerate(response_data.get("choices", [])):
            msg = ch.get("message", {})
            message = ChatCompletionMessage(
                role="assistant",
                content=msg.get("content"),
                tool_calls=msg.get("tool_calls"),  # Already OpenAI compatible
            )
            choices_out.append(Choice(index=i, finish_reason=ch.get("finish_reason"), message=message))
        usage = None
        if response_data.get("usage"):
            u = response_data["usage"]
            usage = CompletionUsage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )
        return ChatCompletion(
            id=response_data.get("id", ""),
            model=model,
            created=response_data.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )
