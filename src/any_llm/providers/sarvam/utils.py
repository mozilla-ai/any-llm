from typing import Any

from sarvamai import CreateChatCompletionV2Response, ModelListResponse
from sarvamai.types.chat_completion_chunk import ChatCompletionChunk as SarvamChatCompletionChunk

from any_llm.providers.openai.utils import _normalize_openai_dict_response
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.types.model import Model
from any_llm.utils.usage import completion_usage_from_dict


def _to_openai_dict(response: CreateChatCompletionV2Response | SarvamChatCompletionChunk) -> dict[str, Any]:
    """Dump a Sarvam v2 object, which already has the OpenAI shape, with reasoning normalized.

    Sarvam returns the reasoning trace as ``reasoning_content``, which becomes any_llm's ``reasoning``.
    """
    data = _normalize_openai_dict_response(response.model_dump())
    if isinstance(data.get("usage"), dict):
        data["usage"] = completion_usage_from_dict(data["usage"])
    return data


def _convert_response(response: CreateChatCompletionV2Response) -> ChatCompletion:
    return ChatCompletion.model_validate(_to_openai_dict(response))


def _convert_chunk(chunk: SarvamChatCompletionChunk) -> ChatCompletionChunk:
    # The final usage chunk of a stream with include_usage has an empty choices list, kept as is.
    return ChatCompletionChunk.model_validate(_to_openai_dict(chunk))


def _convert_models_list(response: ModelListResponse) -> list[Model]:
    # Extra fields such as `access` ("standard" or "provisioned") stay on the Model.
    return [Model.model_validate({**model.model_dump(), "object": "model"}) for model in response.data]
