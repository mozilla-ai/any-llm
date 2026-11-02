"""OVHcloud AI Endpoints, reached through its registry entry: chat, embeddings and the model listing.

OVHcloud has no ``LLMProvider`` member, so the parametrized suites never reach it.
"""

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion, CreateEmbeddingResponse
from tests.constants import EXPECTED_PROVIDERS

pytestmark = pytest.mark.asyncio

CHAT_MODEL = "Meta-Llama-3_3-70B-Instruct"
EMBEDDING_MODELS = ["bge-multilingual-gemma2", "bge-m3"]


@pytest_asyncio.fixture
async def ovhcloud() -> AsyncIterator[AnyLLM]:
    if not os.getenv("OVHCLOUD_API_KEY"):
        if "ovhcloud" in EXPECTED_PROVIDERS:
            pytest.fail("OVHCLOUD_API_KEY not set")
        pytest.skip("OVHCLOUD_API_KEY not set, skipping")
    llm = AnyLLM.create("ovhcloud")
    try:
        yield llm
    finally:
        await llm.client.close()  # type: ignore[attr-defined]


async def test_ovhcloud_completion(ovhcloud: AnyLLM) -> None:
    result = await ovhcloud.acompletion(
        model=CHAT_MODEL, messages=[{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=10
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content


@pytest.mark.parametrize("model", EMBEDDING_MODELS)
async def test_ovhcloud_embedding(ovhcloud: AnyLLM, model: str) -> None:
    result = await ovhcloud.aembedding(model=model, inputs=["Hello world", "Bonjour le monde"])

    assert isinstance(result, CreateEmbeddingResponse)
    assert len(result.data) == 2
    for entry in result.data:
        assert entry.embedding
        assert all(isinstance(v, float) for v in entry.embedding)
    assert result.usage.prompt_tokens > 0


async def test_ovhcloud_list_models_marks_models_that_write_no_text(ovhcloud: AnyLLM) -> None:
    """The listing's one sign of a model's kind is ``max_completion_tokens``: 0 for a model that writes no text."""
    limits = {
        model.id: (model.model_extra or {}).get("max_completion_tokens") for model in await ovhcloud.alist_models()
    }

    assert limits[CHAT_MODEL]
    for model in EMBEDDING_MODELS:
        assert limits[model] == 0
