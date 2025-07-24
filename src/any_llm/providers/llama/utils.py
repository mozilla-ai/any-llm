from typing import Any
from pydantic import BaseModel

def convert_to_llama_format(messages: list[dict[str, Any]], response_format: type[BaseModel] | None = None) -> list[dict[str, Any]]:
    """Utility function to convert messages to Llama-specific format."""
    # Example conversion logic
    converted_messages = []
    for message in messages:
        converted_message = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if response_format:
            # Example logic for utilizing response_format
            converted_message["format"] = response_format.schema()  # Assuming schema() provides necessary format
        converted_messages.append(converted_message)
    return converted_messages
