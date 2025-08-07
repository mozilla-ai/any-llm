import os
import json
from typing import Any, Iterator

try:
    from google import genai
    from google.genai import types
except ImportError:
    msg = "google-genai is not installed. Please install it with `pip install any-llm-sdk[google]`"
    raise ImportError(msg)

from pydantic import BaseModel

from any_llm.types.completion import ChatCompletionChunk, ChatCompletion, CreateEmbeddingResponse
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.helpers import (
    create_completion_from_response,
)
from any_llm.providers.google.utils import (
    _convert_tool_spec,
    _convert_messages,
    _create_openai_chunk_from_google_chunk,
    _create_openai_embedding_response_from_google,
)


class GoogleProvider(Provider):
    """Google Provider using the new response conversion utilities."""

    PROVIDER_NAME = "Google"
    PROVIDER_DOCUMENTATION_URL = "https://cloud.google.com/vertex-ai/docs"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
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

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Google provider."""
        if kwargs.get("stream", False) and kwargs.get("response_format", None) is not None:
            raise UnsupportedParameterError("stream and response_format", cls.PROVIDER_NAME)

        if kwargs.get("parallel_tool_calls", None) is not None:
            raise UnsupportedParameterError("parallel_tool_calls", cls.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        tools = None
        if "tools" in kwargs:
            tools = _convert_tool_spec(kwargs["tools"])
            kwargs["tools"] = tools

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

            response_dict = {
                "id": "google_genai_response",
                "model": "google/genai",
                "created": 0,
                "usage": {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0)
                    if hasattr(response, "usage_metadata")
                    else 0,
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0)
                    if hasattr(response, "usage_metadata")
                    else 0,
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
                    if hasattr(response, "usage_metadata")
                    else 0,
                },
            }

            if (
                response.candidates
                and len(response.candidates) > 0
                and response.candidates[0].content
                and response.candidates[0].content.parts
                and len(response.candidates[0].content.parts) > 0
                and hasattr(response.candidates[0].content.parts[0], "function_call")
                and response.candidates[0].content.parts[0].function_call
            ):
                function_call = response.candidates[0].content.parts[0].function_call

                args_dict = {}
                if hasattr(function_call, "args") and function_call.args:
                    for key, value in function_call.args.items():
                        args_dict[key] = value

                response_dict["choices"] = [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{hash(function_call.name)}",
                                    "function": {
                                        "name": function_call.name,
                                        "arguments": json.dumps(args_dict),
                                    },
                                    "type": "function",
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                        "index": 0,
                    }
                ]
            else:
                content = ""
                if (
                    response.candidates
                    and len(response.candidates) > 0
                    and response.candidates[0].content
                    and response.candidates[0].content.parts
                    and len(response.candidates[0].content.parts) > 0
                    and hasattr(response.candidates[0].content.parts[0], "text")
                ):
                    content = response.candidates[0].content.parts[0].text or ""

                response_dict["choices"] = [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ]

            return create_completion_from_response(
                response_data=response_dict,
                model=model,
                provider_name=self.PROVIDER_NAME,
            )
