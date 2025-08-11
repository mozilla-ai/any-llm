import os
from typing import Any, Iterator

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    msg = "google-genai is not installed. Please install it with `pip install any-llm-sdk[google]`"
    raise ImportError(msg) from exc

from pydantic import BaseModel

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChatCompletion,
    CreateEmbeddingResponse,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.google.utils import (
    _convert_tool_choice,
    _convert_tool_spec,
    _convert_messages,
    _convert_response_to_response_dict,
    _create_openai_chunk_from_google_chunk,
    _create_openai_embedding_response_from_google,
)


class GoogleProvider(Provider):
    """Google Provider using the new response conversion utilities."""

    PROVIDER_NAME = "Google"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = True

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Google GenAI provider."""
        self.use_vertex_ai = os.getenv("GOOGLE_USE_VERTEX_AI", "false").lower() == "true"

        if self.use_vertex_ai:
            self.project_id = os.getenv("GOOGLE_PROJECT_ID")
            self.location = os.getenv("GOOGLE_REGION", "us-central1")

            if not self.project_id:
                raise MissingApiKeyError("Google Vertex AI", "GOOGLE_PROJECT_ID")

            self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        else:
            api_key = getattr(config, "api_key", None) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

            if not api_key:
                raise MissingApiKeyError("Google Gemini Developer API", "GEMINI_API_KEY/GOOGLE_API_KEY")

            self.client = genai.Client(api_key=api_key)

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        result = self.client.models.embed_content(
            model=model,
            contents=inputs,  # type: ignore[arg-type]
            **kwargs,
        )

        return _create_openai_embedding_response_from_google(model, result)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if kwargs.get("stream", False) and kwargs.get("response_format", None) is not None:
            raise UnsupportedParameterError("stream and response_format", self.PROVIDER_NAME)

        if kwargs.get("parallel_tool_calls", None) is not None:
            raise UnsupportedParameterError("parallel_tool_calls", self.PROVIDER_NAME)
        tools = None
        if "tools" in kwargs:
            tools = _convert_tool_spec(kwargs["tools"])
            kwargs["tools"] = tools

        if tool_choice := kwargs.pop("tool_choice", None):
            kwargs["tool_config"] = _convert_tool_choice(tool_choice)

        stream = kwargs.pop("stream", False)
        response_format = kwargs.pop("response_format", None)
        generation_config = types.GenerateContentConfig(
            **kwargs,
        )
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = response_format

        formatted_messages = _convert_messages(messages)

        content_text = ""
        if len(formatted_messages) == 1 and formatted_messages[0].role == "user":
            # Single user message
            parts = formatted_messages[0].parts
            if parts and hasattr(parts[0], "text"):
                content_text = parts[0].text or ""
        else:
            # Multiple messages - concatenate user messages for simplicity
            content_parts = []
            for msg in formatted_messages:
                if msg.role == "user" and msg.parts:
                    if hasattr(msg.parts[0], "text") and msg.parts[0].text:
                        content_parts.append(msg.parts[0].text)

            content_text = "\n".join(content_parts)

        if stream:
            response_stream = self.client.models.generate_content_stream(
                model=model, contents=content_text, config=generation_config
            )
            return map(_create_openai_chunk_from_google_chunk, response_stream)
        else:
            response: types.GenerateContentResponse = self.client.models.generate_content(
                model=model, contents=content_text, config=generation_config
            )

            response_dict = _convert_response_to_response_dict(response)

            # Directly construct ChatCompletion
            choices_out: list[Choice] = []
            for i, choice_item in enumerate(response_dict.get("choices", [])):
                message_dict: dict[str, Any] = choice_item.get("message", {})
                tool_calls: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] | None = None
                if message_dict.get("tool_calls"):
                    tool_calls_list: list[ChatCompletionMessageFunctionToolCall | ChatCompletionMessageToolCall] = []
                    for tc in message_dict["tool_calls"]:
                        tool_calls_list.append(
                            ChatCompletionMessageFunctionToolCall(
                                id=tc.get("id"),
                                type="function",
                                function=Function(
                                    name=tc["function"]["name"],
                                    arguments=tc["function"]["arguments"],
                                ),
                            )
                        )
                    tool_calls = tool_calls_list
                message = ChatCompletionMessage(
                    role="assistant",
                    content=message_dict.get("content"),
                    tool_calls=tool_calls,
                )
                from typing import Literal, cast

                choices_out.append(
                    Choice(
                        index=i,
                        finish_reason=cast(
                            Literal["stop", "length", "tool_calls", "content_filter", "function_call"],
                            choice_item.get("finish_reason", "stop"),
                        ),
                        message=message,
                    )
                )

            usage_dict = response_dict.get("usage", {})
            usage = CompletionUsage(
                prompt_tokens=usage_dict.get("prompt_tokens", 0),
                completion_tokens=usage_dict.get("completion_tokens", 0),
                total_tokens=usage_dict.get("total_tokens", 0),
            )

            return ChatCompletion(
                id=response_dict.get("id", ""),
                model=model,
                created=response_dict.get("created", 0),
                object="chat.completion",
                choices=choices_out,
                usage=usage,
            )
