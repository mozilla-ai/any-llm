from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import TypeAdapter, ValidationError

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai import OpenaiProvider
from any_llm.types.decide import (
    Choice,
    ChoiceAnswer,
    DecideResponse,
    DecideUsage,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Score,
    ScoreAnswer,
)


def _response() -> DecideResponse:
    return DecideResponse.model_validate(
        {
            "model": "jev-1",
            "answers": {
                "billing": {"type": "noul", "noul": 0.9},
                "tone": {"type": "choice", "choice": "angry", "confidence": 0.8, "probabilities": {"angry": 0.8}},
                "urgency": {
                    "type": "score",
                    "score": 1.5,
                    "confidence": 0.7,
                    "legend": {"0": "low", "1": "high"},
                    "probabilities": {"0": 0.3, "1": 0.7},
                },
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )


def test_questions_carry_their_type_tag() -> None:
    assert Noul().type == "noul"
    assert Choice(criteria={"calm": None}).type == "choice"
    assert Score(criteria=["low"]).type == "score"


def test_question_dicts_validate_by_type() -> None:
    adapter: TypeAdapter[Question] = TypeAdapter(Question)

    question = adapter.validate_python({"type": "noul", "criteria": {"true": "spam"}})

    assert question == Noul(criteria=NoulCriteria(true="spam"))


def test_question_dict_with_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Question).validate_python({"type": "rank"})


def test_question_with_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Noul.model_validate({"instructions": "Spam?", "criterion": "typo"})


def test_score_without_criteria_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Score(criteria=[])


def test_score_criteria_keep_their_order() -> None:
    assert Score(criteria=("low", "high")).criteria == ("low", "high")


@pytest.mark.parametrize("criteria", [{"low", "high"}, frozenset({"low"})])
def test_score_criteria_reject_unordered_collections(criteria: set[str] | frozenset[str]) -> None:
    with pytest.raises(ValidationError, match="Sequence"):
        Score(criteria=criteria)  # type: ignore[arg-type]


def test_choice_keeps_undescribed_criteria() -> None:
    assert Choice(criteria={"calm": None, "angry": "Hostile"}).criteria == {"calm": None, "angry": "Hostile"}


def test_question_assignment_is_validated() -> None:
    question = Noul()

    question.criteria = {"true": "Advertising"}  # type: ignore[assignment]
    assert question.criteria == NoulCriteria(true="Advertising")

    with pytest.raises(ValidationError):
        question.criteria = {"yes": "Advertising"}  # type: ignore[assignment]


def test_answers_are_frozen() -> None:
    answer = NoulAnswer(noul=0.9)

    assert {answer} == {NoulAnswer(noul=0.9)}
    with pytest.raises(ValidationError, match="frozen"):
        answer.noul = 0.1


def test_response_answers_validate_by_type() -> None:
    response = _response()

    assert response.answers["billing"] == NoulAnswer(noul=0.9)
    assert isinstance(response.answers["tone"], ChoiceAnswer)
    assert isinstance(response.answers["urgency"], ScoreAnswer)
    assert response.usage == DecideUsage(input_tokens=10, output_tokens=2)


def test_score_answer_keys_are_integer_levels() -> None:
    answer = _response().scores["urgency"]

    assert answer.legend == {0: "low", 1: "high"}
    assert answer.probabilities == {0: 0.3, 1: 0.7}


def test_response_groups_answers_by_type() -> None:
    response = _response()

    assert list(response.nouls) == ["billing"]
    assert list(response.choices) == ["tone"]
    assert list(response.scores) == ["urgency"]


def test_response_optional_fields_default_to_none() -> None:
    response = DecideResponse(answers={})

    assert response.id is None
    assert response.model is None
    assert response.usage is None


@pytest.fixture
def provider() -> Iterator[OpenaiProvider]:
    """A provider that claims decide support, so `_adecide` can be mocked."""
    provider = OpenaiProvider(api_key="test-key")
    with patch.object(provider, "SUPPORTS_DECIDE", True):
        yield provider


def test_provider_metadata_decide_defaults_to_false() -> None:
    assert OpenaiProvider.get_provider_metadata().decide is False


@pytest.mark.parametrize("unified_exceptions", [False, True])
@pytest.mark.asyncio
async def test_unsupported_provider_raises(unified_exceptions: bool) -> None:
    provider = OpenaiProvider(api_key="test-key", unified_exceptions=unified_exceptions)

    with pytest.raises(NotImplementedError, match="doesn't support decide"):
        await provider.adecide("model", "state", {"billing": Noul()})


@pytest.mark.asyncio
async def test_unsupported_provider_is_reported_before_the_timeout_is_checked() -> None:
    provider = OpenaiProvider(api_key="test-key")

    with (
        patch.object(provider, "TIMEOUT_SUPPORT", "unsupported"),
        pytest.raises(NotImplementedError, match="doesn't support decide"),
    ):
        await provider.adecide("model", "state", {"billing": Noul()}, timeout=5)


@pytest.mark.asyncio
async def test_unsupported_provider_raises_from_the_hook_too() -> None:
    with pytest.raises(NotImplementedError, match="doesn't support decide"):
        await OpenaiProvider(api_key="test-key")._adecide("model", "state", {"billing": Noul()})


@pytest.mark.asyncio
async def test_supported_provider_must_implement_adecide(provider: OpenaiProvider) -> None:
    with pytest.raises(NotImplementedError, match="must implement _adecide"):
        await provider.adecide("model", "state", {"billing": Noul()})


@pytest.mark.asyncio
async def test_adecide_validates_question_dicts(provider: OpenaiProvider) -> None:
    with patch.object(provider, "_adecide", new=AsyncMock(return_value=_response())) as adecide:
        await provider.adecide("model", "state", {"billing": {"type": "noul"}, "risk": Score(criteria=["low"])})

    adecide.assert_awaited_once_with("model", "state", {"billing": Noul(), "risk": Score(criteria=["low"])})


@pytest.mark.asyncio
async def test_adecide_rejects_an_invalid_question_dict(provider: OpenaiProvider) -> None:
    with (
        patch.object(provider, "_adecide", new=AsyncMock()) as adecide,
        pytest.raises(ValidationError),
    ):
        await provider.adecide("model", "state", {"risk": {"type": "score", "criteria": []}})

    adecide.assert_not_awaited()


@pytest.mark.asyncio
async def test_adecide_reports_every_invalid_question_by_name(provider: OpenaiProvider) -> None:
    with patch.object(provider, "_adecide", new=AsyncMock()), pytest.raises(ValidationError) as exc_info:
        await provider.adecide(
            "model", "state", {"billing": Noul(), "risk": {"type": "score", "criteria": []}, "tone": {"type": "choice"}}
        )

    assert [error["loc"] for error in exc_info.value.errors()] == [
        ("risk", "score", "criteria"),
        ("tone", "choice", "criteria"),
    ]


@pytest.mark.asyncio
async def test_adecide_rejects_questions_that_are_not_a_mapping(provider: OpenaiProvider) -> None:
    with patch.object(provider, "_adecide", new=AsyncMock()) as adecide, pytest.raises(ValidationError):
        await provider.adecide("model", "state", None)  # type: ignore[arg-type]

    adecide.assert_not_awaited()


@pytest.mark.asyncio
async def test_adecide_accepts_question_objects_by_their_attributes(provider: OpenaiProvider) -> None:
    class ForeignScore:
        type = "score"
        instructions = "How risky?"
        criteria = ("low", "high")

    with patch.object(provider, "_adecide", new=AsyncMock(return_value=_response())) as adecide:
        await provider.adecide("model", "state", {"risk": ForeignScore()})  # type: ignore[dict-item]

    adecide.assert_awaited_once_with(
        "model", "state", {"risk": Score(instructions="How risky?", criteria=("low", "high"))}
    )


@pytest.mark.asyncio
async def test_adecide_fills_in_the_requested_model(provider: OpenaiProvider) -> None:
    with patch.object(provider, "_adecide", new=AsyncMock(return_value=DecideResponse(answers={}))):
        response = await provider.adecide("requested", "state", {"billing": Noul()})

    assert response.model == "requested"


@pytest.mark.asyncio
async def test_adecide_keeps_the_provider_reported_model(provider: OpenaiProvider) -> None:
    with patch.object(provider, "_adecide", new=AsyncMock(return_value=_response())):
        response = await provider.adecide("requested", "state", {"billing": Noul()})

    assert response.model == "jev-1"


@pytest.mark.asyncio
async def test_adecide_forwards_timeout(provider: OpenaiProvider) -> None:
    with (
        patch.object(provider, "TIMEOUT_SUPPORT", "native"),
        patch.object(provider, "_adecide", new=AsyncMock(return_value=_response())) as adecide,
    ):
        await provider.adecide("model", "state", {"billing": Noul()}, timeout=5)

    assert adecide.await_args is not None
    assert adecide.await_args.kwargs == {"timeout": 5}


@pytest.mark.asyncio
async def test_adecide_rejects_timeout_when_unsupported(provider: OpenaiProvider) -> None:
    with (
        patch.object(provider, "TIMEOUT_SUPPORT", "unsupported"),
        patch.object(provider, "_adecide", new=AsyncMock()) as adecide,
        pytest.raises(UnsupportedParameterError, match="timeout"),
    ):
        await provider.adecide("model", "state", {"billing": Noul()}, timeout=5)

    adecide.assert_not_awaited()


def test_decide_runs_adecide_synchronously(provider: OpenaiProvider) -> None:
    with patch.object(provider, "_adecide", new=AsyncMock(return_value=_response())) as adecide:
        response = provider.decide("model", "state", {"billing": Noul()}, extra="value")

    assert response == _response()
    adecide.assert_awaited_once_with("model", "state", {"billing": Noul()}, extra="value")
