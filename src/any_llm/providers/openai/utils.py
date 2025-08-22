"""OpenAI Provider Utilities."""

from collections.abc import Sequence
from typing import Any

from openai.pagination import SyncPage
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.model import Model as OpenAIModel

from any_llm.logging import logger
from any_llm.types.completion import ChatCompletion
from any_llm.types.model import Model


def _normalize_reasoning_on_message(message_dict: dict[str, Any]) -> None:
    """Mutate a message dict to move provider-specific reasoning fields to our Reasoning type."""
    if isinstance(message_dict.get("reasoning"), dict) and "content" in message_dict["reasoning"]:
        return

    possible_fields = [
        "reasoning_content",
        "thinking",
        "chain_of_thought",
    ]
    value: Any | None = None
    for field_name in possible_fields:
        if field_name in message_dict and message_dict[field_name] is not None:
            value = message_dict[field_name]
            break

    if value is None and isinstance(message_dict.get("reasoning"), str):
        value = message_dict["reasoning"]

    if value is not None:
        message_dict["reasoning"] = {"content": str(value)}


def _normalize_openai_dict_response(response_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a dict where non-standard reasoning fields are normalized.

    - For non-streaming: response.choices[*].message
    - For streaming: chunk.choices[*].delta
    """
    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                _normalize_reasoning_on_message(message)

            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                _normalize_reasoning_on_message(delta)

    return response_dict


def _convert_chat_completion(response: OpenAIChatCompletion) -> ChatCompletion:
    if response.object != "chat.completion":
        # Force setting this here because it's a requirement Literal in the OpenAI API, but the Llama API has
        # a typo where they set it to "chat.completions". I filed a ticket with them to fix it. No harm in setting it here
        # Because this is the only accepted value anyways.
        logger.warning(
            "API returned an unexpected object type: %s. Setting to 'chat.completion'.",
            response.object,
        )
        response.object = "chat.completion"
    if not isinstance(response.created, int):
        # Sambanova returns a float instead of an int.
        logger.warning(
            "API returned an unexpected created type: %s. Setting to int.",
            type(response.created),
        )
        response.created = int(response.created)
    normalized = _normalize_openai_dict_response(response.model_dump())
    return ChatCompletion.model_validate(normalized)


def _convert_models_list(
    openai_models: SyncPage[OpenAIModel],
    provider_name: str,
) -> Sequence[Model]:
    """
    Convert OpenAI models list to OpenAI Model format.
    Each openai_model can be a dict or a pydantic BaseModel.
    """
    results = []
    for openai_model in openai_models:
        model = Model(
            id=openai_model.id,
            label=openai_model.id,
            created=openai_model.created,
            object=openai_model.object or "model",
            provider=provider_name,
            owned_by=openai_model.owned_by or provider_name,
            attributes=openai_model.model_dump(),
        )
        results.append(model)
    return results
