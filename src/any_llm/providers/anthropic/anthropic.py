import json
from typing import Any

from any_llm.logging import logger

try:
    from anthropic import Anthropic
    from anthropic.types import Message
    import instructor
except ImportError:
    msg = "anthropic or instructor is not installed. Please install it with `pip install any-llm-sdk[anthropic]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion

from any_llm.provider import ApiConfig, convert_instructor_response
from any_llm.providers.base_framework import (
    BaseProviderFramework,
    create_completion_from_response,
)

DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(BaseProviderFramework):
    """
    Anthropic Provider using enhanced BaseProviderFramework framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_NAME = "Anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"

    def _initialize_client(self, config: ApiConfig) -> None:
        """Initialize the Anthropic client."""
        self.client = Anthropic(api_key=config.api_key, base_url=config.api_base)

        # Create instructor client for structured output
        self.instructor_client = instructor.from_anthropic(self.client)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Anthropic with instructor support."""
        # Handle response_format for structured output
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            # Convert other kwargs
            converted_kwargs = self._convert_kwargs(kwargs)

            # Use instructor for structured output
            instructor_response = self.instructor_client.messages.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_format,
                **converted_kwargs,
            )

            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, "anthropic")

        # Fall back to standard completion flow
        return super().completion(model, messages, **kwargs)

    def _convert_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert kwargs to Anthropic format."""
        kwargs = kwargs.copy()

        if "max_tokens" not in kwargs:
            logger.warning(f"max_tokens is required for Anthropic, setting to {DEFAULT_MAX_TOKENS}")
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        # Convert tools if present
        if "tools" in kwargs:
            kwargs["tools"] = self._convert_tool_spec(kwargs["tools"])

        # Handle parallel_tool_calls
        if "parallel_tool_calls" in kwargs:
            parallel_tool_calls = kwargs.pop("parallel_tool_calls")
            if parallel_tool_calls is False:
                tool_choice = {"type": kwargs.get("tool_choice", "any"), "disable_parallel_tool_use": True}
                kwargs["tool_choice"] = tool_choice

        return kwargs

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format."""
        return messages

    def _make_api_call(self, model: str, messages: tuple[str, list[dict[str, Any]]], **kwargs: Any) -> Message:
        """Make the API call to Anthropic."""
        return self.client.messages.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

    def _convert_response(self, response: Message) -> ChatCompletion:
        """Convert Anthropic response to OpenAI format using base_framework utility."""
        finish_reason_mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }

        # Process content blocks into structured format
        tool_calls = []
        content = ""

        for content_block in response.content:
            if content_block.type == "text":
                content = content_block.text
            elif content_block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": content_block.id,
                        "function": {
                            "name": content_block.name,
                            "arguments": json.dumps(content_block.input),
                        },
                        "type": "function",
                    }
                )

        # Structure response data for the utility
        response_dict = {
            "id": response.id,
            "model": response.model,
            "created": int(response.created_at.timestamp()) if hasattr(response, "created_at") else 0,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": tool_calls if tool_calls else None,
                    },
                    "finish_reason": response.stop_reason or "end_turn",
                    "index": 0,
                }
            ],
            "usage": {
                "completion_tokens": response.usage.output_tokens,
                "prompt_tokens": response.usage.input_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }

        # Use base_framework utility for conversion
        return create_completion_from_response(
            response_data=response_dict,
            model=response.model,
            provider_name="anthropic",
            finish_reason_mapping=finish_reason_mapping,
        )

    def _convert_tool_spec(self, openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool specification to Anthropic format."""
        # Use the generic utility first
        generic_tools = []

        for tool in openai_tools:
            if tool.get("type") != "function":
                continue

            function = tool["function"]
            generic_tool = {
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
            generic_tools.append(generic_tool)

        # Convert to Anthropic-specific format
        anthropic_tools = []
        for tool in generic_tools:
            anthropic_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool["parameters"]["properties"],
                    "required": tool["parameters"].get("required", []),
                },
            }
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools
