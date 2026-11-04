from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

JSONContent: TypeAlias = str | Mapping[str, Any] | Sequence[Any]
"""Text, a JSON object, or an array. Nested values are sent as the provider encodes them."""


class _Question(BaseModel):
    # A misspelled field would otherwise be dropped and change the question without an error, and a
    # value assigned after construction would otherwise reach the provider unchecked.
    model_config = ConfigDict(extra="forbid", validate_assignment=True, revalidate_instances="always")


class NoulCriteria(_Question):
    """Optional descriptions of the yes and no outcomes of a `Noul` question."""

    true: JSONContent | None = None
    """Description of the yes outcome; `None` leaves it undescribed."""
    false: JSONContent | None = None
    """Description of the no outcome; `None` leaves it undescribed."""


class Noul(_Question):
    """A yes/no question, answered with the probability of yes."""

    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    """The question to ask; optional."""
    criteria: NoulCriteria | None = None
    """Optional descriptions of the yes and no outcomes."""


class Choice(_Question):
    """A question that picks one of the named criteria."""

    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    """The question to ask; optional."""
    criteria: Mapping[str, JSONContent | None] = Field(
        description="Choice names mapped to a description of when each applies, or None to use the name alone"
    )


class Score(_Question):
    """A question that rates the state on an ordered scale."""

    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    """The question to ask; optional."""
    # A Sequence rather than a list: a set would be accepted in hash order and scramble the levels.
    criteria: Sequence[JSONContent] = Field(
        min_length=1, description="Ordered score levels; each level's position is its score, starting at zero"
    )


Question: TypeAlias = Annotated[Noul | Choice | Score, Field(discriminator="type")]
"""A question, identified by its `type`."""

QuestionsInput: TypeAlias = Mapping[str, Question | Mapping[str, Any]]
"""Question names mapped to questions, or to their dict forms."""


class _Answer(BaseModel):
    # Frozen, so answers stay hashable and a mutation cannot leave one that no longer fits its type.
    # Fields a provider adds are kept, so a caller that passes the answer on loses nothing.
    model_config = ConfigDict(frozen=True, extra="allow")


class NoulAnswer(_Answer):
    """The answer to a `Noul` question."""

    type: Literal["noul"] = "noul"
    noul: float = Field(description="Probability of yes, from 0 to 1")


class ChoiceAnswer(_Answer):
    """The answer to a `Choice` question."""

    type: Literal["choice"] = "choice"
    choice: str = Field(description="The most likely choice name")
    confidence: float = Field(description="Confidence in the choice, from 0 to 1")
    probabilities: dict[str, float] = Field(description="Probability of each choice name")


class ScoreAnswer(_Answer):
    """The answer to a `Score` question."""

    type: Literal["score"] = "score"
    score: float = Field(description="Probability-weighted average of the score levels")
    confidence: float = Field(description="Confidence in the score, from 0 to 1")
    legend: dict[int, JSONContent] = Field(description="Score levels mapped to the criteria that define them")
    probabilities: dict[int, float] = Field(description="Probability of each score level")


class UnknownAnswer(_Answer):
    """An answer of a type any-llm does not model, such as OpenRouter's `box`, with all its fields."""

    type: str


_KNOWN_ANSWER_TYPES = frozenset({"noul", "choice", "score"})


def _answer_tag(value: Any) -> str | None:
    answer_type = value.get("type") if isinstance(value, Mapping) else getattr(value, "type", None)
    if not isinstance(answer_type, str):
        # No tag fails validation, so an answer must still name its type.
        return None
    return answer_type if answer_type in _KNOWN_ANSWER_TYPES else "unknown"


Answer: TypeAlias = Annotated[
    Annotated[NoulAnswer, Tag("noul")]
    | Annotated[ChoiceAnswer, Tag("choice")]
    | Annotated[ScoreAnswer, Tag("score")]
    | Annotated[UnknownAnswer, Tag("unknown")],
    Discriminator(_answer_tag),
]
"""An answer, identified by its `type`. A type any-llm does not model is kept as an `UnknownAnswer`."""

AnswerT = TypeVar("AnswerT", NoulAnswer, ChoiceAnswer, ScoreAnswer)


class DecideUsage(BaseModel):
    """Token usage of a decide request. Fields a provider adds are kept."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    """Input tokens used, or `None` when the provider did not report it."""
    output_tokens: int | None = None
    """Output tokens used, or `None` when the provider did not report it."""
    cost: float | None = None
    """What the provider charged for the request in USD, or `None` when it did not report it."""


class DecideResponse(BaseModel):
    """Answers keyed by the question names of the request, provider-agnostic.

    Fields a provider adds, such as OpenRouter's `provider`, are kept and appear in `model_dump()`.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = Field(default=None, description="Provider-assigned request ID, when the provider reports one")
    model: str | None = Field(
        default=None,
        description="Model that served the request: the provider-reported one when available, else the requested model",
    )
    answers: dict[str, Answer] = Field(description="One answer per question, keyed by question name")
    usage: DecideUsage | None = None
    """Token usage, when the provider reports it."""

    def _answers_of(self, answer_type: type[AnswerT]) -> dict[str, AnswerT]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, answer_type)}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Yes/no answers keyed by question name."""
        return self._answers_of(NoulAnswer)

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Choice answers keyed by question name."""
        return self._answers_of(ChoiceAnswer)

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Score answers keyed by question name."""
        return self._answers_of(ScoreAnswer)
