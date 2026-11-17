import base64
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, ParsedChatCompletion
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.BEDROCK])]

# Pegasus 1.2 is Legacy, and Bedrock refuses Legacy models to accounts that have not called them
# in the last 30 days. Pegasus 1.5 takes no fewer than 512 output tokens.
PEGASUS = "us.twelvelabs.pegasus-1-5-v1:0"
MAX_TOKENS = 512

# The asset is 480x360, since Pegasus refuses a video under 360x360 pixels.
VIDEO = base64.b64encode((Path(__file__).parent.parent / "assets" / "test-pattern.mp4").read_bytes()).decode()


class VideoSummary(BaseModel):
    summary: str


def _messages() -> list[dict[str, Any] | ChatCompletionMessage]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video in one sentence."},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{VIDEO}"}},
            ],
        }
    ]


def _client(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> AnyLLM:
    try:
        return AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Bedrock credentials missing")


async def test_pegasus_describes_a_video(
    provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]
) -> None:
    result = await _client(provider, provider_client_config).acompletion(
        model=PEGASUS, messages=_messages(), max_tokens=MAX_TOKENS
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content
    assert result.choices[0].finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.completion_tokens > 0


async def test_pegasus_streams_a_description(
    provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]
) -> None:
    stream = await _client(provider, provider_client_config).acompletion(
        model=PEGASUS, messages=_messages(), max_tokens=MAX_TOKENS, stream=True
    )

    chunks = [chunk async for chunk in stream]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices][-1] == "stop"
    assert chunks[-1].usage is not None


async def test_pegasus_answers_in_a_schema(
    provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]
) -> None:
    result = await _client(provider, provider_client_config).acompletion(
        model=PEGASUS, messages=_messages(), max_tokens=MAX_TOKENS, response_format=VideoSummary
    )

    assert isinstance(result, ParsedChatCompletion)
    assert isinstance(result.choices[0].message.parsed, VideoSummary)
    assert result.choices[0].message.parsed.summary
