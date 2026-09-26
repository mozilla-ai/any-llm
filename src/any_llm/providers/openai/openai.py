from collections.abc import AsyncIterator
from typing import Any, ClassVar

from openresponses_types import ResponseResource
from typing_extensions import override

from any_llm.types.files import FileOperation
from any_llm.types.messages import MessageResponse, MessageStreamEvent, MessagesParams, ParsedBetaMessage, ParsedMessage
from any_llm.types.responses import Response

from .base import BaseOpenAIProvider
from .files import OpenAIFileMethods
from .messages_responses import (
    convert_responses_stream,
    messages_needs_responses,
    messages_params_to_responses_params,
    response_to_message_response,
)


class OpenaiProvider(OpenAIFileMethods, BaseOpenAIProvider):
    API_BASE = "https://api.openai.com/v1"

    ENV_API_KEY_NAME = "OPENAI_API_KEY"
    ENV_API_BASE_NAME = "OPENAI_BASE_URL"
    PROVIDER_NAME = "openai"
    PROVIDER_DOCUMENTATION_URL = "https://platform.openai.com/docs/api-reference"
    PROMPT_CACHE_KEY_SUPPORT = "supported"
    SUPPORTS_RESPONSES = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True
    SUPPORTS_IMAGE_GENERATION = True
    SUPPORTS_AUDIO_TRANSCRIPTION = True
    SUPPORTS_AUDIO_SPEECH = True
    SUPPORTED_FILE_OPERATIONS: ClassVar[frozenset[FileOperation]] = frozenset(
        {"upload", "list", "retrieve", "download", "delete"}
    )

    @override
    async def _amessages(
        self, params: MessagesParams, **kwargs: Any
    ) -> MessageResponse | ParsedMessage[Any] | ParsedBetaMessage[Any] | AsyncIterator[MessageStreamEvent]:
        """Route tools + thinking through Responses; otherwise keep Completions bridge.

        Chat Completions rejects function tools combined with a non-none reasoning_effort on
        newer OpenAI reasoning models. The Responses API accepts that combination, so when the
        caller asked for Messages with both, serve via ``_aresponses`` instead of guessing by
        model name (#1432).
        """
        if not messages_needs_responses(params):
            return await super()._amessages(params, **kwargs)

        if params.container is not None:
            msg = "container requires a provider with a native Anthropic Messages API"
            raise NotImplementedError(msg)
        if params.context_management is not None or params.betas:
            msg = "context_management and betas require a provider with a native Anthropic Messages API"
            raise NotImplementedError(msg)

        responses_params = messages_params_to_responses_params(params)
        result = await self._aresponses(responses_params, **kwargs)

        if isinstance(result, (Response, ResponseResource)):
            return response_to_message_response(result)

        return convert_responses_stream(result)
