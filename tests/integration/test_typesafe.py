from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio

from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.typesafe import TypesafeProvider
from any_llm.types.decide import Choice, Noul, Score
from any_llm.utils.aio import run_async_in_sync
from tests.constants import EXPECTED_PROVIDERS

_MODEL = "jev-latest"
_STATE = "I was charged twice. Please fix this ASAP."


def _create_provider() -> TypesafeProvider:
    try:
        # The suite's flaky reruns are the only retry layer.
        return TypesafeProvider(max_retries=0)
    except MissingApiKeyError:
        if LLMProvider.TYPESAFE in EXPECTED_PROVIDERS:
            raise
        pytest.skip("TYPESAFE_API_KEY is not configured")


@pytest_asyncio.fixture
async def async_llm() -> AsyncIterator[TypesafeProvider]:
    llm = _create_provider()
    yield llm
    await llm.client.aclose()


@pytest.fixture
def llm() -> Iterator[TypesafeProvider]:
    llm = _create_provider()
    yield llm
    # decide runs on any-llm's runner loop, which owns the client's connections.
    run_async_in_sync(llm.client.aclose())


@pytest.mark.asyncio
async def test_adecide_answers_every_question_type(async_llm: TypesafeProvider) -> None:
    result = await async_llm.adecide(
        _MODEL,
        _STATE,
        {
            "billing": Noul(instructions="Is this ticket about billing?"),
            "tone": Choice(
                instructions="What is the customer's tone?",
                criteria={"calm": None, "frustrated": None, "angry": None},
            ),
            "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
        },
    )

    assert 0.0 <= result.nouls["billing"].noul <= 1.0
    assert result.choices["tone"].choice in {"calm", "frustrated", "angry"}
    assert 0.0 <= result.scores["urgency"].score <= 2.0
    assert result.usage is not None
    assert result.usage.input_tokens


def test_decide_sync(llm: TypesafeProvider) -> None:
    result = llm.decide(_MODEL, _STATE, {"billing": Noul(instructions="Is this ticket about billing?")})

    assert result.nouls["billing"].noul > 0.5
