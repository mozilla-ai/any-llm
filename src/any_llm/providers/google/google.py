import os
from collections.abc import Iterator
from typing import Any

try:
    from google import genai
    from google.genai import types

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from pydantic import BaseModel

from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.provider import ApiConfig, Provider
from any_llm.providers.google.utils import (
    _convert_messages,
    _convert_response_to_response_dict,
    _convert_tool_choice,
    _convert_tool_spec,
    _create_openai_chunk_from_google_chunk,
    _create_openai_embedding_response_from_google,
)
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Choice,
    CompletionParams,
    CompletionUsage,
    CreateEmbeddingResponse,
    Function,
    Reasoning,
)

# From https://ai.google.dev/gemini-api/docs/openai#thinking
REASONING_EFFORT_TO_THINKING_BUDGETS = {"minimal": 256, "low": 1024, "medium": 8192, "high": 24576}


class GoogleProvider(Provider):
    """Google Provider using the new response conversion utilities."""

    PROVIDER_NAME = "google"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"
    ENV_API_KEY_NAME = "GOOGLE_API_KEY/GEMINI_API_KEY"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Google GenAI provider."""

        if not self.PACKAGES_INSTALLED:
            msg = "google required packages are not installed"
            raise ImportError(msg)

        self.use_vertex_ai = os.getenv("GOOGLE_USE_VERTEX_AI", "false").lower() == "true"

        if self.use_vertex_ai:
            self.project_id = os.getenv("GOOGLE_PROJECT_ID")
            self.location = os.getenv("GOOGLE_REGION", "us-central1")

            if not self.project_id:
                msg = "Google Vertex AI"
                raise MissingApiKeyError(msg, "GOOGLE_PROJECT_ID")

            self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        else:
            api_key = getattr(config, "api_key", None) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

            if not api_key:
                msg = "Google Gemini Developer API"
                raise MissingApiKeyError(msg, "GEMINI_API_KEY/GOOGLE_API_KEY")

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

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if params.stream and params.response_format is not None:
            error_message = "stream and response_format"
            raise UnsupportedParameterError(error_message, self.PROVIDER_NAME)

        if params.parallel_tool_calls is not None:
            error_message = "parallel_tool_calls"
            raise UnsupportedParameterError(error_message, self.PROVIDER_NAME)
        tools = None
        if params.tools is not None:
            tools = _convert_tool_spec(params.tools)
            kwargs["tools"] = tools

        if isinstance(params.tool_choice, str):
            kwargs["tool_config"] = _convert_tool_choice(params.tool_choice)

        if params.reasoning_effort is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True, thinking_budget=REASONING_EFFORT_TO_THINKING_BUDGETS[params.reasoning_effort]
            )

        stream = bool(params.stream)
        response_format = params.response_format
        # Build generation config without duplicating keys (e.g., tools)
        base_kwargs = params.model_dump(
            exclude_none=True,
            exclude={"model_id", "messages", "reasoning_effort", "response_format", "stream", "tools", "tool_choice"},
        )
        base_kwargs.update(kwargs)
        generation_config = types.GenerateContentConfig(**base_kwargs)
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = response_format

        formatted_messages, system_instruction = _convert_messages(params.messages)
        if system_instruction:
            generation_config.system_instruction = system_instruction

        if stream:
            response_stream = self.client.models.generate_content_stream(
                model=params.model_id,
                contents=formatted_messages,  # type: ignore[arg-type]
                config=generation_config,
            )
            return map(_create_openai_chunk_from_google_chunk, response_stream)
        response: types.GenerateContentResponse = self.client.models.generate_content(
            model=params.model_id,
            contents=formatted_messages,  # type: ignore[arg-type]
            config=generation_config,
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

            reasoning_content = message_dict.get("reasoning")
            message = ChatCompletionMessage(
                role="assistant",
                content=message_dict.get("content"),
                tool_calls=tool_calls,
                reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
            )
            from typing import Literal, cast

            choices_out.append(
                Choice(
                    index=i,
                    finish_reason=cast(
                        "Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']",
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
            model=params.model_id,
            created=response_dict.get("created", 0),
            object="chat.completion",
            choices=choices_out,
            usage=usage,
        )
