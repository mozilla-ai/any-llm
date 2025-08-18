import os
from collections.abc import Iterator
from typing import Any, cast

try:
    import instructor

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from openai import OpenAI
from pydantic import BaseModel

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.utils.instructor import _convert_instructor_response


class SambanovaProvider(BaseOpenAIProvider):
    API_BASE = "https://api.sambanova.ai/v1/"
    ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"
    PROVIDER_NAME = "sambanova"
    PROVIDER_DOCUMENTATION_URL = "https://sambanova.ai/"

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def completion(self, params: CompletionParams, **kwargs: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Make the API call to SambaNova service with instructor for structured output."""
        client = OpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )

        if params.response_format:
            instructor_client = instructor.from_openai(client)
            if not isinstance(params.response_format, type) or not issubclass(params.response_format, BaseModel):
                msg = "response_format must be a pydantic model"
                raise ValueError(msg)
            response = instructor_client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                response_model=params.response_format,
                **params.model_dump(
                    exclude_none=True, exclude={"model_id", "messages", "reasoning_effort", "response_format"}
                ),
                **kwargs,
            )
            return _convert_instructor_response(response, params.model_id, self.PROVIDER_NAME)
        return self._convert_completion_response(
            client.chat.completions.create(
                model=params.model_id,
                messages=cast("Any", params.messages),
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages"}),
                **kwargs,
            )
        )
