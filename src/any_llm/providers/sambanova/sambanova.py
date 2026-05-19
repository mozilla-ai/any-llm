from typing import Any

from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from typing_extensions import override

from any_llm.providers.openai.xml_reasoning import XMLReasoningOpenAIProvider
from any_llm.providers.openai.xml_reasoning_utils import (
    convert_chat_completion_chunk_with_xml_reasoning,
    convert_chat_completion_with_xml_reasoning,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type


class SambanovaProvider(XMLReasoningOpenAIProvider):
    API_BASE = "https://api.sambanova.ai/v1/"
    ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"
    ENV_API_BASE_NAME = "SAMBANOVA_API_BASE"
    PROVIDER_NAME = "sambanova"
    PROVIDER_DOCUMENTATION_URL = "https://sambanova.ai/"

    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_REASONING = True

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """SambaNova does not support the ``encoding_format`` parameter.

        The OpenAI SDK injects ``encoding_format`` into every
        ``embeddings.create()`` call automatically.  We bypass that by
        calling the client-level ``post()`` directly so only the
        parameters SambaNova accepts are included in the request body.
        """
        if not self.SUPPORTS_EMBEDDING:
            msg = "This provider does not support embeddings."
            raise NotImplementedError(msg)

        body: dict[str, Any] = {"input": inputs, "model": model}
        if "dimensions" in kwargs:
            body["dimensions"] = kwargs["dimensions"]

        return self._convert_embedding_response(
            await self.client.post(
                "/embeddings",
                body=body,
                cast_to=CreateEmbeddingResponse,
            )
        )

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        if isinstance(response, OpenAIChatCompletion):
            return convert_chat_completion_with_xml_reasoning(response)
        if isinstance(response, ChatCompletion):
            return response
        return ChatCompletion.model_validate(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        if isinstance(response, OpenAIChatCompletionChunk):
            return convert_chat_completion_chunk_with_xml_reasoning(response)
        if isinstance(response, ChatCompletionChunk):
            return response
        return ChatCompletionChunk.model_validate(response)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenAI API."""
        if is_structured_output_type(params.response_format):
            params.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": get_json_schema(params.response_format),
                },
            }
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params
