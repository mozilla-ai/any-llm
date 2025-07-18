import os
from datetime import datetime
from typing import Any
import json

import httpx
from pydantic import BaseModel
from openai.types.chat.chat_completion import ChatCompletion
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from any_llm.logging import logger
from any_llm.provider import ApiConfig, Provider
from any_llm.providers.helpers import create_completion_from_response
from any_llm.exceptions import UnsupportedParameterError


class OllamaProvider(Provider):
    """
    Ollama Provider using the new response conversion utilities.

    It uses the /api/chat endpoint and makes HTTP calls to the Ollama API.
    Read more here - https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    If OLLAMA_API_URL is not set and not passed in config, then it will default to "http://localhost:11434"
    """

    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _DEFAULT_URL = "http://localhost:11434"
    PROVIDER_NAME = "Ollama"

    def __init__(self, config: ApiConfig) -> None:
        """We don't use the Provider init because by default we don't require an API key."""

        self.url = str(config.api_base or os.getenv("OLLAMA_API_URL", self._DEFAULT_URL))

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

        kwargs["stream"] = kwargs.get("stream", False)  # Ollama requires you to specifically set this to False.

        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # response_format is a Pydantic model class
                kwargs["format"] = response_format.model_json_schema()
            else:
                # response_format is already a dict/schema
                kwargs["format"] = response_format

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

        data = {
            "model": model,
            "messages": cleaned_messages,
            "options": {"num_ctx": 32000},  # Default is 4096 which is too small for most use cases.
            **kwargs,  # Pass any additional arguments to the API
        }

        timeout = kwargs.pop("timeout", None)

        try:
            response = httpx.post(
                self.url.rstrip("/") + self._CHAT_COMPLETION_ENDPOINT,
                json=data,
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"Error calling Ollama: {e.response.text}")
            raise e

        response_data = response.json()

        # Convert Ollama's timestamp format to int
        created_str = response_data.get("created_at", 0)
        try:
            created = int(datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())
        except (ValueError, TypeError):
            created = 0

        # Normalize response structure for the utility
        response_dict = {
            "id": "chatcmpl-" + str(hash(str(response_data))),
            "model": response_data["model"],
            "created": created,
            "usage": {
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "completion_tokens": response_data.get("eval_count", 0),
                "total_tokens": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0),
            },
        }

        # Handle tool calls vs regular responses
        message_data = response_data["message"]
        if "tool_calls" in message_data:
            # Convert tool calls to standard format
            tool_calls = []
            for tool_call in message_data["tool_calls"]:
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
                        "role": message_data["role"],
                        "content": message_data["content"],
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
                        "role": message_data["role"],
                        "content": message_data["content"],
                        "tool_calls": None,
                    },
                    "finish_reason": response_data.get("done_reason", "stop"),
                    "index": 0,
                }
            ]

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response_dict,
            model=model,
            provider_name=self.PROVIDER_NAME,
        )
