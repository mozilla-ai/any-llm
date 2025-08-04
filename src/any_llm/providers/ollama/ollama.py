import os
from datetime import datetime
from typing import Any
import json

try:
    from ollama import ChatResponse as OllamaChatResponse
    from ollama import Message as OllamaMessage
    from ollama import Client
except ImportError:
    msg = "ollama is not installed. Please install it with `pip install any-llm-sdk[ollama]`"
    raise ImportError(msg)

from pydantic import BaseModel
from openai.types.chat.chat_completion import ChatCompletion
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types import CreateEmbeddingResponse
from any_llm.provider import ApiConfig, Provider
from any_llm.providers.helpers import create_completion_from_response
from any_llm.exceptions import UnsupportedParameterError


from any_llm.providers.ollama.utils import _create_openai_embedding_response_from_ollama


class OllamaProvider(Provider):
    """
    Ollama Provider using the new response conversion utilities.

    It uses the ollama sdk.
    Read more here - https://github.com/ollama/ollama-python
    """

    PROVIDER_NAME = "Ollama"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/ollama/ollama"

    SUPPORTS_STREAMING = False
    SUPPORTS_EMBEDDING = True

    def __init__(self, config: ApiConfig) -> None:
        """We don't use the Provider init because by default we don't require an API key."""

        self.url = config.api_base or os.getenv("OLLAMA_API_URL")

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Ollama provider."""
        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Ollama."""

        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # response_format is a Pydantic model class
                format = response_format.model_json_schema()
            else:
                # response_format is already a dict/schema
                format = response_format
        else:
            format = None

        # Convert tool messages to user messages and remove tool_calls from assistant messages
        # (https://www.reddit.com/r/ollama/comments/1ked8x2/feeding_tool_output_back_to_llm/)
        cleaned_messages = []
        for message in messages:
            if message["role"] == "tool":
                cleaned_message = {
                    "role": "user",
                    "content": json.dumps(message["content"]),
                }
            elif message["role"] == "assistant" and "tool_calls" in message:
                content = message["content"] + "\n" + json.dumps(message["tool_calls"])
                cleaned_message = {
                    "role": "assistant",
                    "content": content,
                }
            else:
                cleaned_message = message.copy()

            cleaned_messages.append(cleaned_message)

        kwargs["num_ctx"] = kwargs.get("num_ctx", 32000)

        client = Client(host=self.url, timeout=kwargs.pop("timeout", None))
        response: OllamaChatResponse = client.chat(
            model=model,
            tools=kwargs.pop("tools", None),
            think=kwargs.pop("think", None),
            messages=cleaned_messages,
            format=format,
            options=kwargs,
        )

        # Convert Ollama's timestamp format to int
        created_str = response.created_at
        if created_str is None:
            raise ValueError("Expected Ollama to provide a created_at timestamp")
        # Convert Ollama's timestamp format to int
        created_str = response.created_at
        if created_str is None:
            raise ValueError("Expected Ollama to provide a created_at timestamp")
        # Handle both microseconds (6 digits) and nanoseconds (9 digits)
        if len(created_str.split(".")[1].rstrip("Z")) > 6:
            # Truncate nanoseconds to microseconds
            parts = created_str.split(".")
            microseconds = parts[1][:6]
            created_str = f"{parts[0]}.{microseconds}Z"
        created = int(datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

        # Normalize response structure for the utility
        prompt_tokens = response.prompt_eval_count or 0
        completion_tokens = response.eval_count or 0
        response_dict: dict[str, Any] = {
            "id": "chatcmpl-" + str(hash(created)),
            "model": response.model,
            "created": created,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        # Handle tool calls vs regular responses
        response_message: OllamaMessage = response.get("message")
        if not response_message or not isinstance(response_message, OllamaMessage):
            raise ValueError("Unexpected output from ollama")
        if response_message.tool_calls:
            # Convert tool calls to standard format
            tool_calls = []
            for tool_call in response_message.tool_calls:
                tool_calls.append(
                    {
                        "id": str(hash(str(tool_call))),  # Generate ID from hash
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": json.dumps(tool_call["function"]["arguments"]),
                        },
                        "type": "function",
                    }
                )

            response_dict["choices"] = [
                {
                    "message": {
                        "role": response_message.role,
                        "content": response_message.content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                    "index": 0,
                }
            ]
        else:
            # Regular text response
            response_dict["choices"] = [
                {
                    "message": {
                        "role": response_message.role,
                        "content": response_message.content,
                        "tool_calls": None,
                    },
                    "finish_reason": response.done_reason,
                    "index": 0,
                }
            ]

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response_dict,
            model=model,
            provider_name=self.PROVIDER_NAME,
        )

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Generate embeddings using Ollama."""
        client = Client(host=self.url, timeout=kwargs.pop("timeout", None))

        response = client.embed(
            model=model,
            input=inputs,
            **kwargs,
        )
        return _create_openai_embedding_response_from_ollama(response)
