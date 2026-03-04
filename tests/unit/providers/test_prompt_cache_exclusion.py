"""Verify that prompt_cache_key and prompt_cache_retention are excluded by non-OpenAI providers."""

import pytest

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams


def _make_params() -> CompletionParams:
    return CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "hi"}],
        prompt_cache_key="my-key",
        prompt_cache_retention="1h",
    )


def _assert_excluded(result: dict) -> None:  # type: ignore[type-arg]
    assert "prompt_cache_key" not in result, "prompt_cache_key should be excluded"
    assert "prompt_cache_retention" not in result, "prompt_cache_retention should be excluded"


def test_base_openai_provider_excludes_prompt_cache() -> None:
    result = BaseOpenAIProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_anthropic_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.anthropic.utils import _convert_params

    result = _convert_params(_make_params(), provider_name="anthropic")
    _assert_excluded(result)


def test_mistral_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.mistral.mistral import MistralProvider

    result = MistralProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_groq_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.groq.groq import GroqProvider

    result = GroqProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_cohere_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.cohere.cohere import CohereProvider

    result = CohereProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_together_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.together.together import TogetherProvider

    result = TogetherProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_cerebras_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.cerebras.cerebras import CerebrasProvider

    result = CerebrasProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_ollama_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.ollama.ollama import OllamaProvider

    result = OllamaProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_watsonx_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.watsonx.watsonx import WatsonxProvider

    result = WatsonxProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_xai_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    result = XaiProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_sambanova_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.sambanova.sambanova import SambanovaProvider

    result = SambanovaProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_portkey_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.portkey.portkey import PortkeyProvider

    result = PortkeyProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_azure_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.azure.azure import AzureProvider

    result = AzureProvider._convert_completion_params(_make_params())
    _assert_excluded(result)


def test_huggingface_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.huggingface.utils import _convert_params

    result = _convert_params(_make_params())
    _assert_excluded(result)


def test_minimax_provider_excludes_prompt_cache() -> None:
    from any_llm.providers.minimax.minimax import MinimaxProvider

    result = MinimaxProvider._convert_completion_params(_make_params())
    _assert_excluded(result)
