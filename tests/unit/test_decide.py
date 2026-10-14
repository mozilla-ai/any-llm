import pytest
from pydantic import TypeAdapter, ValidationError

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
