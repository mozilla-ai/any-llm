"""Shared conversion utilities for providers that use XML reasoning tags.

These utilities extract reasoning content from XML tags in non-streaming responses.
For streaming responses, use XMLReasoningOpenAIProvider or wrap_chunks_with_xml_reasoning
from the xml_reasoning module.
"""

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

from any_llm.providers.openai.utils import _normalize_openai_dict_response
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.utils.reasoning import normalize_reasoning_from_provider_fields_and_xml_tags


def convert_chat_completion_with_xml_reasoning(response: OpenAIChatCompletion) -> ChatCompletion:
    """Convert OpenAI ChatCompletion, extracting reasoning from XML tags.

    This function handles non-streaming responses by:
    1. Normalizing the response dict
    2. Extracting reasoning from provider fields AND XML tags in content
    3. Validating as a ChatCompletion

    Args:
        response: OpenAI ChatCompletion response

    Returns:
        ChatCompletion with reasoning extracted from XML tags
    """
    response_dict = _normalize_openai_dict_response(response.model_dump())

    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                normalize_reasoning_from_provider_fields_and_xml_tags(message)

            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                normalize_reasoning_from_provider_fields_and_xml_tags(delta)

    return ChatCompletion.model_validate(response_dict)


def convert_chat_completion_chunk_with_xml_reasoning(
    response: OpenAIChatCompletionChunk,
) -> ChatCompletionChunk:
    """Convert OpenAI ChatCompletionChunk, extracting reasoning from XML tags.

    This function handles individual streaming chunks by:
    1. Normalizing the response dict
    2. Extracting reasoning from provider fields AND XML tags in delta content
    3. Setting the correct object type
    4. Validating as a ChatCompletionChunk

    Note: For proper streaming XML tag handling across chunk boundaries,
    use XMLReasoningOpenAIProvider or wrap_chunks_with_xml_reasoning instead.

    Args:
        response: OpenAI ChatCompletionChunk response

    Returns:
        ChatCompletionChunk with reasoning extracted from XML tags
    """
    response_dict = _normalize_openai_dict_response(response.model_dump())

    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                normalize_reasoning_from_provider_fields_and_xml_tags(delta)

    response_dict["object"] = "chat.completion.chunk"
    return ChatCompletionChunk.model_validate(response_dict)
