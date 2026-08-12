import os

import pytest

from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

MODEL_ID = "gpt-5-nano"


def _create_custom_provider() -> AnyLLM:
    """Reach OpenAI's real endpoint through the custom path, not the openai provider.

    This exercises the full create_openai_compatible stack (auth, base-URL binding,
    identity reporting) against a live endpoint using a key CI already holds. The
    verification bar matches the community-provider policy: completion, streaming,
    and list_models.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping custom-path integration test")
    return AnyLLM.create_openai_compatible(
        name="custom-openai",
        api_base="https://api.openai.com/v1",
        api_key=api_key,
        timeout=10,
    )


@pytest.mark.asyncio
async def test_custom_path_completion() -> None:
    llm = _create_custom_provider()
    result = await llm.acompletion(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content
    assert llm.PROVIDER_NAME == "custom-openai"
    assert llm.get_provider_metadata().name == "custom-openai"


@pytest.mark.asyncio
async def test_custom_path_streaming() -> None:
    llm = _create_custom_provider()
    chunks = []
    async for chunk in await llm.acompletion(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    ):
        assert isinstance(chunk, ChatCompletionChunk)
        chunks.append(chunk)
    content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert content


@pytest.mark.asyncio
async def test_custom_path_list_models() -> None:
    llm = _create_custom_provider()
    models = await llm.alist_models()
    assert len(models) > 0
    assert all(model.id for model in models)
