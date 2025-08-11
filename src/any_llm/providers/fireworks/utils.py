import uuid
from typing import Any
from datetime import datetime

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)


def _create_openai_chunk_from_fireworks_chunk(fireworks_chunk: Any) -> ChatCompletionChunk:
    """Convert a Fireworks streaming chunk to OpenAI ChatCompletionChunk format."""

    content = None
    if hasattr(fireworks_chunk, "choices") and fireworks_chunk.choices:
        choice = fireworks_chunk.choices[0]
        if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
            content = choice.delta.content
        elif hasattr(choice, "text"):
            content = choice.text

    delta = ChoiceDelta(content=content)

    finish_reason = None
    if hasattr(fireworks_chunk, "choices") and fireworks_chunk.choices:
        choice = fireworks_chunk.choices[0]
        if hasattr(choice, "finish_reason"):
            finish_reason = choice.finish_reason

    choice_obj = ChunkChoice(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )

    usage = None
    if hasattr(fireworks_chunk, "usage") and fireworks_chunk.usage:
        usage_data = fireworks_chunk.usage
        usage = CompletionUsage(
            prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
            completion_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0),
        )

    created = int(datetime.now().timestamp())

    return ChatCompletionChunk(
        id=f"chatcmpl-{uuid.uuid4()}",
        choices=[choice_obj],
        created=created,
        model=getattr(fireworks_chunk, "model", "unknown"),
        object="chat.completion.chunk",
        usage=usage,
    )

class ResponseWrapper:
    """Wrapper class to add output_text property to Fireworks Response objects."""
    def __init__(self, response):
        self._response = response
        self._output_text = None
    
    def __getattr__(self, name):
        return getattr(self._response, name)
    
    @property
    def output_text(self):
        if self._output_text is None and hasattr(self._response, 'output') and self._response.output:
            try:
                raw_output = self._response.output[-1].content[0].text
                self._output_text = raw_output.split("</think>")[-1].strip()
            except (IndexError, AttributeError):
                self._output_text = ""
        return self._output_text or ""


def _create_response_with_output_text(fireworks_response: Any) -> Any:
    """Wrap a Fireworks response to add output_text property."""
    return ResponseWrapper(fireworks_response)