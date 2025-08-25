import os
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI
from pydantic import BaseModel

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.utils.instructor import _convert_instructor_response

MISSING_PACKAGES_ERROR = None
try:
    import instructor

except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class SambanovaProvider(BaseOpenAIProvider):
    API_BASE = "https://api.sambanova.ai/v1/"
    ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"
    PROVIDER_NAME = "sambanova"
    PROVIDER_DOCUMENTATION_URL = "https://sambanova.ai/"

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Make the API call to SambaNova service with instructor for structured output."""
        client = AsyncOpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        if params.response_format:
            instructor_client = instructor.from_openai(client)
            if not isinstance(params.response_format, type) or not issubclass(params.response_format, BaseModel):
                msg = "response_format must be a pydantic model"
                raise ValueError(msg)
            response = await instructor_client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                response_model=params.response_format,
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format"}),
                **kwargs,
            )
            return _convert_instructor_response(response, params.model_id, self.PROVIDER_NAME)
        return self._convert_completion_response_async(
            await client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages"}),
                **kwargs,
            )
        )
