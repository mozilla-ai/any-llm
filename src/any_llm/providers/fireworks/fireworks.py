from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any, cast

from openai import AsyncOpenAI, AsyncStream, OpenAI
from pydantic import BaseModel

from any_llm.provider import Provider
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionParams,
    CompletionUsage,
    Reasoning,
)
from any_llm.types.model import Model
from any_llm.types.responses import Response, ResponseStreamEvent

MISSING_PACKAGES_ERROR = None
try:
    from fireworks import LLM

    from .utils import _create_openai_chunk_from_fireworks_chunk
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class FireworksProvider(Provider):
    PROVIDER_NAME = "fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"
    BASE_URL = "https://api.fireworks.ai/inference/v1"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    async def _stream_completion_async(
        self, llm: "LLM", messages: list[dict[str, Any]], params: CompletionParams, **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""

        response_generator = await llm.chat.completions.acreate(
            messages=messages,  # type: ignore[arg-type]
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages"}),
            **kwargs,
        )

        async for chunk in cast("AsyncGenerator[ChatCompletionChunk, None]", response_generator):
            yield _create_openai_chunk_from_fireworks_chunk(chunk)

    async def acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        llm = LLM(
            model=params.model_id,
            deployment_type="serverless",
            api_key=self.config.api_key,
        )

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        if params.response_format is not None:
            response_format = params.response_format
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": response_format.__name__, "schema": response_format.model_json_schema()},
                }
            else:
                kwargs["response_format"] = response_format

        if params.stream:
            return self._stream_completion_async(llm, params.messages, params, **kwargs)

        response = await llm.chat.completions.acreate(
            messages=params.messages,  # type: ignore[arg-type]
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}),
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
            model=params.model_id,
            created=response_data.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )

    async def aresponses(
        self, model: str, input_data: Any, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Call Fireworks Responses API and normalize into ChatCompletion/Chunks."""
        client = AsyncOpenAI(
            base_url=self.BASE_URL,
            api_key=self.config.api_key,
        )
        response = await client.responses.create(
            model=model,
            input=input_data,
            **kwargs,
        )
        if not isinstance(response, Response | AsyncStream):
            err_msg = f"Responses API returned an unexpected type: {type(response)}"
            raise ValueError(err_msg)
        if isinstance(response, Response) and not isinstance(response, AsyncStream):
            # See https://fireworks.ai/blog/response-api for details about Fireworks Responses API support
            reasoning = response.output[-1].content[0].text.split("</think>")[-1]  # type: ignore[union-attr,index]
            if reasoning:
                reasoning = reasoning.strip()
                response.output[-1].content[0].text = response.output[-1].content[0].text.split("</think>")[0]  # type: ignore[union-attr,index]
            response.reasoning = Reasoning(content=reasoning) if reasoning else None  # type: ignore[assignment]

        return response

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self.config.api_key,
        )
        return client.models.list(**kwargs).data
