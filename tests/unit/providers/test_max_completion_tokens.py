"""`max_completion_tokens` is the OpenAI name for the output limit; providers with their own field must honor it."""

import logging
from typing import Any

import pytest

from any_llm.types.completion import CompletionParams
from any_llm.utils.token_limit import resolve_output_token_limit


def _params(max_tokens: int | None = None, max_completion_tokens: int | None = None) -> CompletionParams:
    return CompletionParams(
        model_id="model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
    )


def _anthropic(params: CompletionParams) -> Any:
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.utils import _convert_params

    converted = _convert_params(params, provider_name="anthropic")
    assert "max_completion_tokens" not in converted
    return converted["max_tokens"]


def _gemini(params: CompletionParams) -> Any:
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.gemini import GeminiProvider

    return GeminiProvider._convert_completion_params(params, provider_name="gemini")["config"].max_output_tokens


def _bedrock(params: CompletionParams) -> Any:
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    return BedrockProvider._convert_completion_params(params).get("inferenceConfig", {}).get("maxTokens")


def _mistral(params: CompletionParams) -> Any:
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    converted = MistralProvider._convert_completion_params(params)
    assert "max_completion_tokens" not in converted
    return converted.get("max_tokens")


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


@pytest.mark.parametrize("convert", PROVIDERS)
def test_equal_limits_do_not_warn(convert: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert convert(_params(max_tokens=400, max_completion_tokens=400)) == 400
    assert "Ignoring max_tokens" not in caplog.text


@pytest.mark.parametrize(
    "convert",
    [pytest.param(_gemini, id="gemini"), pytest.param(_bedrock, id="bedrock"), pytest.param(_mistral, id="mistral")],
)
def test_no_limit_sends_no_limit(convert: Any) -> None:
    assert convert(_params()) is None


def test_resolve_output_token_limit_without_limits_returns_none() -> None:
    assert resolve_output_token_limit(_params()) is None
