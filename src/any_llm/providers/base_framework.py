import os
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional
import json

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError


class BaseProviderFramework(Provider, ABC):
    """
    Base framework for all providers that standardizes conversion patterns.

    This class provides a common structure for providers while allowing
    customization of specific conversion logic through abstract methods.

    All providers follow the same pattern:
    1. Convert kwargs to provider format
    2. Convert messages to provider format
    3. Make the API call
    4. Convert response back to OpenAI format

    This eliminates code duplication while maintaining flexibility.
    """

    # Provider-specific configuration (to be overridden by subclasses)
    PROVIDER_NAME: str
    ENV_API_KEY_NAME: str

    def __init__(self, config: ApiConfig) -> None:
        """Initialize provider with standardized configuration handling."""
        super().__init__(config)

        # Standardized API key handling
        if not config.api_key:
            config.api_key = os.getenv(self.ENV_API_KEY_NAME)

        if not config.api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)

        # Allow subclasses to perform custom initialization
        self._initialize_client(config)

    @abstractmethod
    def _initialize_client(self, config: ApiConfig) -> None:
        """Initialize the provider-specific client. Must be implemented by subclasses."""
        pass

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        Standard completion flow that all providers follow:
        1. Convert kwargs to provider format
        2. Convert messages to provider format
        3. Make the API call
        4. Convert response back to OpenAI format
        """
        # Step 1: Convert kwargs
        converted_kwargs = self._convert_kwargs(kwargs)

        # Step 2: Convert messages
        converted_messages = self._convert_messages(messages)

        # Step 3: Make API call (provider-specific)
        raw_response = self._make_api_call(model, converted_messages, **converted_kwargs)

        # Step 4: Convert response to OpenAI format
        return self._convert_response(raw_response)

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        return await asyncio.to_thread(self.completion, model, messages, **kwargs)

    @abstractmethod
    def _convert_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert standard kwargs to provider-specific format."""
        pass

    @abstractmethod
    def _convert_messages(self, messages: list[dict[str, Any]]) -> Any:
        """Convert standard messages to provider-specific format."""
        pass

    @abstractmethod
    def _make_api_call(self, model: str, messages: Any, **kwargs: Any) -> Any:
        """Make the actual API call to the provider."""
        pass

    @abstractmethod
    def _convert_response(self, raw_response: Any) -> ChatCompletion:
        """Convert provider response to OpenAI ChatCompletion format."""
        pass


# Common utility functions that can be shared across providers
def create_openai_tool_call(tool_call_id: str, name: str, arguments: str) -> ChatCompletionMessageToolCall:
    """Create a standardized OpenAI tool call object."""
    return ChatCompletionMessageToolCall(
        id=tool_call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def create_openai_message(
    role: str, content: Optional[str] = None, tool_calls: Optional[list[ChatCompletionMessageToolCall]] = None
) -> ChatCompletionMessage:
    """Create a standardized OpenAI message object."""
    return ChatCompletionMessage(
        role=role,  # type: ignore[arg-type]
        content=content,
        tool_calls=tool_calls,
    )


def create_openai_completion(
    id: str,
    model: str,
    choices: list[Any],
    usage: Optional[Any] = None,
    created: int = 0,
) -> ChatCompletion:
    """Create a standardized OpenAI ChatCompletion object."""
    return ChatCompletion(
        id=id,
        model=model,
        object="chat.completion",
        created=created,
        choices=choices,
        usage=usage,
    )


def create_tool_calls_from_list(tool_calls_data: list[dict[str, Any]]) -> list[ChatCompletionMessageToolCall]:
    """
    Convert a list of tool call dictionaries to ChatCompletionMessageToolCall objects.

    Handles common variations in tool call structure across providers.
    """
    tool_calls = []

    for tool_call in tool_calls_data:
        # Extract tool call ID (handle various formats)
        tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id") or f"call_{hash(str(tool_call))}"

        # Extract function info (handle nested structures)
        function_info = tool_call.get("function", {})
        if not function_info and "name" in tool_call:
            # Some providers put function info directly in the tool_call
            function_info = {
                "name": tool_call["name"],
                "arguments": tool_call.get("arguments", tool_call.get("input", {})),
            }

        name = function_info.get("name", "")
        arguments = function_info.get("arguments", {})

        # Ensure arguments is a JSON string
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = str(arguments)

        tool_calls.append(create_openai_tool_call(tool_call_id, name, arguments))

    return tool_calls


def create_choice_from_message_data(
    message_data: dict[str, Any],
    index: int = 0,
    finish_reason: str = "stop",
    finish_reason_mapping: Optional[dict[str, str]] = None,
) -> Choice:
    """
    Create a Choice object from message data, handling tool calls and content.

    Args:
        message_data: Dictionary containing message content and tool calls
        index: Choice index (default 0)
        finish_reason: Raw finish reason from provider
        finish_reason_mapping: Optional mapping to convert provider finish reasons to OpenAI format
    """
    # Apply finish reason mapping if provided
    if finish_reason_mapping and finish_reason in finish_reason_mapping:
        finish_reason = finish_reason_mapping[finish_reason]

    # Extract tool calls if present
    tool_calls = None
    tool_calls_data = message_data.get("tool_calls", [])
    if tool_calls_data:
        tool_calls = create_tool_calls_from_list(tool_calls_data)

    # Create the message
    message = create_openai_message(
        role=message_data.get("role", "assistant"),
        content=message_data.get("content"),
        tool_calls=tool_calls,
    )

    return Choice(
        finish_reason=finish_reason,  # type: ignore[arg-type]
        index=index,
        message=message,
    )


def create_usage_from_data(
    usage_data: dict[str, Any], token_field_mapping: Optional[dict[str, str]] = None
) -> CompletionUsage:
    """
    Create CompletionUsage from provider usage data.

    Args:
        usage_data: Dictionary containing usage information
        token_field_mapping: Optional mapping for field names (e.g., {"input_tokens": "prompt_tokens"})
    """
    # Default field mapping
    default_mapping = {
        "completion_tokens": "completion_tokens",
        "prompt_tokens": "prompt_tokens",
        "total_tokens": "total_tokens",
    }

    # Apply custom mapping if provided
    if token_field_mapping:
        for openai_field, provider_field in token_field_mapping.items():
            if provider_field in usage_data:
                default_mapping[openai_field] = provider_field

    return CompletionUsage(
        completion_tokens=usage_data.get(default_mapping["completion_tokens"], 0),
        prompt_tokens=usage_data.get(default_mapping["prompt_tokens"], 0),
        total_tokens=usage_data.get(default_mapping["total_tokens"], 0),
    )


def create_completion_from_response(
    response_data: dict[str, Any],
    model: str,
    provider_name: str = "provider",
    finish_reason_mapping: Optional[dict[str, str]] = None,
    token_field_mapping: Optional[dict[str, str]] = None,
    id_field: str = "id",
    created_field: str = "created",
    choices_field: str = "choices",
    usage_field: str = "usage",
) -> ChatCompletion:
    """
    Create a complete ChatCompletion from provider response data.

    This is the main utility that most providers can use to convert their responses.

    Args:
        response_data: The raw response from the provider
        model: Model name to use in the response
        provider_name: Name of the provider (for generating fallback IDs)
        finish_reason_mapping: Mapping for finish reasons
        token_field_mapping: Mapping for token count fields
        id_field: Field name for response ID
        created_field: Field name for creation timestamp
        choices_field: Field name for choices array
        usage_field: Field name for usage data
    """
    # Extract choices
    choices = []
    choices_data = response_data.get(choices_field, [])

    # Handle single choice responses (common pattern)
    if not choices_data and "message" in response_data:
        choices_data = [
            {"message": response_data["message"], "finish_reason": response_data.get("finish_reason", "stop")}
        ]

    for i, choice_data in enumerate(choices_data):
        choice = create_choice_from_message_data(
            choice_data.get("message", choice_data),
            index=i,
            finish_reason=choice_data.get("finish_reason", "stop"),
            finish_reason_mapping=finish_reason_mapping,
        )
        choices.append(choice)

    # Create usage if available
    usage = None
    if usage_field in response_data and response_data[usage_field]:
        usage = create_usage_from_data(response_data[usage_field], token_field_mapping)

    # Generate ID if not present
    response_id = response_data.get(id_field, f"{provider_name}_{hash(str(response_data))}")

    return create_openai_completion(
        id=response_id,
        model=model,
        choices=choices,
        usage=usage,
        created=response_data.get(created_field, 0),
    )
