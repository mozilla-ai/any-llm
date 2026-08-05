"""OpenAI Provider Utilities."""

from typing import Any

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion as OpenAIParsedChatCompletion

from any_llm.constants import REASONING_FIELD_NAMES
from any_llm.exceptions import ProviderError
from any_llm.logging import logger
from any_llm.types.completion import ChatCompletion, ParsedChatCompletion
from any_llm.types.moderation import ModerationResponse, ModerationResult


def _normalize_reasoning_on_message(message_dict: dict[str, Any]) -> None:
    """Mutate a message dict to move provider-specific reasoning fields to our Reasoning type."""
    if isinstance(message_dict.get("reasoning"), dict) and "content" in message_dict["reasoning"]:
        return

    possible_fields = REASONING_FIELD_NAMES
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

    # Detect completely empty responses (all required fields None).
    # Some providers return 200 OK with an empty body on malformed input
    # instead of a proper error. Fail fast with a clear message.
    if response.id is None and response.choices is None and response.model is None:
        msg = (
            "Provider returned an empty response with no id, choices, or model. "
            "This usually means the provider failed to process the request."
        )
        raise ProviderError(msg)

    if not isinstance(response.created, int):
        # Sambanova returns a float instead of an int.
        logger.warning(
            "API returned an unexpected created type: %s. Setting to int.",
            type(response.created),
        )
        response.created = int(response.created)
    normalized = _normalize_openai_dict_response(response.model_dump())
    return ChatCompletion.model_validate(normalized)


def _convert_parsed_chat_completion(response: OpenAIParsedChatCompletion[Any]) -> ParsedChatCompletion[Any]:
    """Convert an OpenAI ParsedChatCompletion preserving the .parsed field on each choice."""
    base = _convert_chat_completion(response)
    parsed_completion: ParsedChatCompletion[Any] = ParsedChatCompletion.model_validate(base, from_attributes=True)
    for base_choice, parsed_choice in zip(response.choices, parsed_completion.choices, strict=True):
        parsed_choice.message.parsed = base_choice.message.parsed
    return parsed_completion


def _dump_if_model(obj: Any) -> Any:
    """Return ``model_dump()`` when the value is a pydantic model, else the value itself."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _convert_moderation_response_from_openai(raw: Any, *, include_raw: bool) -> ModerationResponse:
    """Convert an OpenAI ``ModerationCreateResponse`` to our ``ModerationResponse``.

    Drops categories/category_scores keys whose values are ``None`` (i.e. not
    returned for the requested input). Preserves the full provider response
    in ``ModerationResult.provider_raw`` only when ``include_raw`` is True.
    """
    results: list[ModerationResult] = []
    for item in raw.results:
        categories_raw = _dump_if_model(item.categories)
        scores_raw = _dump_if_model(item.category_scores)
        types_raw = getattr(item, "category_applied_input_types", None)

        categories = {key: value for key, value in categories_raw.items() if isinstance(value, bool)}
        # ``bool`` is a subclass of ``int`` in Python; explicitly exclude it
        # so boolean flags from ``categories`` do not leak into ``scores``.
        scores = {
            key: float(value)
            for key, value in scores_raw.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

        applied_types: dict[str, list[str]] | None
        if types_raw is None:
            applied_types = None
        else:
            types_dump = _dump_if_model(types_raw)
            if isinstance(types_dump, dict):
                applied_types = {key: list(value) for key, value in types_dump.items() if value is not None}
            else:
                applied_types = None

        results.append(
            ModerationResult(
                flagged=bool(item.flagged),
                categories=categories,
                category_scores=scores,
                category_applied_input_types=applied_types,
                provider_raw=_dump_if_model(item) if include_raw else None,
            )
        )

    return ModerationResponse(
        id=raw.id,
        model=raw.model,
        results=results,
    )
