"""`max_completion_tokens` is the OpenAI name for the output limit; providers with their own field must honor it."""

import logging
from typing import Any

import pytest

from any_llm.types.completion import CompletionParams

pytest.importorskip("anthropic")
pytest.importorskip("google.genai")
pytest.importorskip("boto3")
pytest.importorskip("mistralai")

from any_llm.providers.anthropic.utils import _convert_params as anthropic_convert
from any_llm.providers.bedrock.bedrock import BedrockProvider
from any_llm.providers.gemini.gemini import GeminiProvider
from any_llm.providers.mistral.mistral import MistralProvider


def _params(**limits: int | None) -> CompletionParams:
    return CompletionParams(model_id="model", messages=[{"role": "user", "content": "Hello"}], **limits)


def _anthropic(params: CompletionParams) -> Any:
    return anthropic_convert(params, provider_name="anthropic")["max_tokens"]


def _gemini(params: CompletionParams) -> Any:
    return GeminiProvider._convert_completion_params(params, provider_name="gemini")["config"].max_output_tokens


def _bedrock(params: CompletionParams) -> Any:
    return BedrockProvider._convert_completion_params(params)["inferenceConfig"]["maxTokens"]


def _mistral(params: CompletionParams) -> Any:
    converted = MistralProvider._convert_completion_params(params)
    assert "max_completion_tokens" not in converted
    return converted["max_tokens"]


PROVIDERS = [
    pytest.param(_anthropic, id="anthropic"),
    pytest.param(_gemini, id="gemini"),
    pytest.param(_bedrock, id="bedrock"),
    pytest.param(_mistral, id="mistral"),
]


@pytest.mark.parametrize("convert", PROVIDERS)
def test_max_completion_tokens_sets_the_output_limit(convert: Any) -> None:
    assert convert(_params(max_completion_tokens=500)) == 500


@pytest.mark.parametrize("convert", PROVIDERS)
def test_max_tokens_still_sets_the_output_limit(convert: Any) -> None:
    assert convert(_params(max_tokens=300)) == 300


@pytest.mark.parametrize("convert", PROVIDERS)
def test_max_completion_tokens_wins_over_max_tokens(convert: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert convert(_params(max_tokens=300, max_completion_tokens=500)) == 500
    assert "Ignoring max_tokens (300) in favor of max_completion_tokens (500)" in caplog.text
