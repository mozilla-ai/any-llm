from typing import Any

from any_llm.types.decide import Noul, Question


def question_to_dict(question: Question) -> dict[str, Any]:
    """Return the System One dict form of a question, with unset optional fields left off.

    Nested values stay as they are, so the provider's own encoder decides how to send them.
    """
    data = question.model_dump()
    if isinstance(question, Noul) and question.criteria is not None:
        # An undescribed outcome is left off rather than sent as null.
        data["criteria"] = {side: value for side, value in data["criteria"].items() if value is not None}
    return {field: value for field, value in data.items() if value is not None}
