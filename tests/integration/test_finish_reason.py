"""Integration test for issue #1196: Gemini truncation must surface as LengthFinishReasonError.

A small ``max_tokens`` budget forces Gemini to stop mid-generation with
``finishReason=MAX_TOKENS``. With a pydantic ``response_format`` this must raise
``LengthFinishReasonError`` so callers can retry with a larger budget, instead of a
misleading malformed-JSON ``ValidationError`` from parsing the truncated output.
"""

import pytest
from pydantic import BaseModel

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import LengthFinishReasonError, MissingApiKeyError


class Assessment(BaseModel):
    explanation: str
    result: bool


@pytest.mark.asyncio
async def test_gemini_truncated_structured_output_raises_length_error(
    provider_model_map: dict[LLMProvider, str],
) -> None:
    try:
        llm = AnyLLM.create(LLMProvider.GEMINI)
    except MissingApiKeyError:
        pytest.skip("Gemini API key not provided, skipping")

    with pytest.raises(LengthFinishReasonError):
        await llm.acompletion(
            model=provider_model_map[LLMProvider.GEMINI],
            messages=[{"role": "user", "content": "Assess: water is wet. Explain in great detail."}],
            response_format=Assessment,
            max_tokens=20,
        )
