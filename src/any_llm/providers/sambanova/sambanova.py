from collections.abc import AsyncIterator
from typing import Any

from openai._streaming import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.sambanova.utils import _convert_chat_completion, _convert_chat_completion_chunk
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, Reasoning
from any_llm.utils.reasoning import find_reasoning_tag, is_partial_reasoning_tag


class SambanovaProvider(BaseOpenAIProvider):
    API_BASE = "https://api.sambanova.ai/v1/"
    ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"
    PROVIDER_NAME = "sambanova"
    PROVIDER_DOCUMENTATION_URL = "https://sambanova.ai/"

    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_REASONING = True

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        if isinstance(response, OpenAIChatCompletion):
            return _convert_chat_completion(response)
        if isinstance(response, ChatCompletion):
            return response
        return ChatCompletion.model_validate(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        if isinstance(response, OpenAIChatCompletionChunk):
            return _convert_chat_completion_chunk(response)
        if isinstance(response, ChatCompletionChunk):
            return response
        return ChatCompletionChunk.model_validate(response)

    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert an OpenAI completion response with streaming reasoning support."""
        if isinstance(response, OpenAIChatCompletion):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            buffer = ""
            current_tag = None
            reasoning_buffer = ""

            async for chunk in response:
                original_chunk = self._convert_completion_chunk_response(chunk)

                if not (len(original_chunk.choices) > 0 and original_chunk.choices[0].delta.content):
                    yield original_chunk
                    continue

                buffer += original_chunk.choices[0].delta.content
                content_parts = []
                reasoning_parts = []

                while buffer:
                    if current_tag is None:
                        tag_info = find_reasoning_tag(buffer, opening=True)
                        if tag_info:
                            tag_start, tag_name = tag_info
                            if tag_start > 0:
                                content_parts.append(buffer[:tag_start])
                            tag_full = f"<{tag_name}>"
                            buffer = buffer[tag_start + len(tag_full) :]
                            current_tag = tag_name
                        elif is_partial_reasoning_tag(buffer, opening=True):
                            break
                        else:
                            content_parts.append(buffer)
                            buffer = ""
                    else:
                        tag_close = f"</{current_tag}>"
                        tag_end = buffer.find(tag_close)
                        if tag_end != -1:
                            reasoning_parts.append(reasoning_buffer + buffer[:tag_end])
                            reasoning_buffer = ""
                            buffer = buffer[tag_end + len(tag_close) :]
                            current_tag = None
                        elif is_partial_reasoning_tag(buffer, opening=False):
                            reasoning_buffer += buffer
                            buffer = ""
                            break
                        else:
                            reasoning_buffer += buffer
                            buffer = ""

                if content_parts or reasoning_parts:
                    modified_chunk = original_chunk.model_copy(deep=True)
                    modified_chunk.choices[0].delta.content = "".join(content_parts) if content_parts else None
                    if reasoning_parts:
                        modified_chunk.choices[0].delta.reasoning = Reasoning(content="".join(reasoning_parts))
                    yield modified_chunk
                elif not buffer:
                    modified_chunk = original_chunk.model_copy(deep=True)
                    modified_chunk.choices[0].delta.content = None
                    yield modified_chunk

        return chunk_iterator()

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenAI API."""
        if isinstance(params.response_format, type) and issubclass(params.response_format, BaseModel):
            params.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": params.response_format.model_json_schema(),
                },
            }
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params
