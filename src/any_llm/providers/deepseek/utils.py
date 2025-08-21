import json
from typing import Any

from pydantic import BaseModel

from any_llm.types.completion import CompletionParams


def _convert_pydantic_to_deepseek_json(
    pydantic_model: type[BaseModel], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert Pydantic model to DeepSeek JSON format.

    DeepSeek requires:
    1. response_format = {'type': 'json_object'}
    2. The word "json" in the prompt
    3. An example of the desired JSON structure

    Following the instructions in the DeepSeek documentation:
    https://api-docs.deepseek.com/guides/json_mode

    Returns:
        modified_messages
    """
    # Get the JSON schema from the Pydantic model
    schema = pydantic_model.model_json_schema()

    # Add JSON instructions to the last user message
    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        json_instruction = f"""
Please respond with a JSON object that can be loaded into a pydantic model that matches the following schema:

{json.dumps(schema, indent=2)}

Return the JSON object only, no other text, do not wrap it in ```json or ```.

{original_content}
"""
        modified_messages[-1]["content"] = json_instruction
    else:
        msg = "Last message is not a user message"
        raise ValueError(msg)

    return modified_messages


def _preprocess_messages(params: CompletionParams) -> CompletionParams:
    """Preprocess messages"""
    if params.response_format:
        if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
            modified_messages = _convert_pydantic_to_deepseek_json(params.response_format, params.messages)
            params.response_format = {"type": "json_object"}
            params.messages = modified_messages

    return params
