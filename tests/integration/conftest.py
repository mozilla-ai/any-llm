import os

import pytest

from any_llm.constants import LLMProvider
from tests.constants import EXPECTED_PROVIDERS

# otari is a hosted gateway (like openrouter) and gateway is its legacy alias. Neither has a
# server started in CI by default, so their integration tests can only run against a real
# endpoint. otari authenticates with a platform token (OTARI_AI_TOKEN) and defaults to the
# hosted api.otari.ai; a self-hosted deployment uses OTARI_API_BASE / GATEWAY_API_BASE. When
# none of those are configured we skip rather than fail (otari 0.1.0's SDK is urllib3-based, so
# an unreachable host raises errors the per-test skip clauses do not catch). Listing the
# provider in EXPECTED_PROVIDERS forces a run so a misconfigured CI job fails loudly.
_GATEWAY_PROVIDER_ENV_VARS: dict[LLMProvider, tuple[str, ...]] = {
    LLMProvider.OTARI: ("OTARI_AI_TOKEN", "OTARI_API_BASE"),
    LLMProvider.GATEWAY: ("GATEWAY_API_BASE", "OTARI_API_BASE", "OTARI_AI_TOKEN"),
}


@pytest.fixture(autouse=True)
def _skip_unconfigured_gateway_providers(request: pytest.FixtureRequest) -> None:
    """Skip otari/gateway integration tests when no real endpoint is configured."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    provider = callspec.params.get("provider")
    if not isinstance(provider, LLMProvider) or provider in EXPECTED_PROVIDERS:
        return
    env_vars = _GATEWAY_PROVIDER_ENV_VARS.get(provider)
    if env_vars and not any(os.getenv(var) for var in env_vars):
        pytest.skip(f"{provider.value} endpoint not configured (set {' or '.join(env_vars)}), skipping")


# The hosted otari test account only routes Anthropic completions, and a few response shapes
# hit known bugs. These tests are skipped for otari so the e2e stays green on the capabilities
# that work; each entry points at the tracking issue and should be removed once it is fixed.
# Mirrors the existing per-provider capability skips (LMSTUDIO, LLAMAFILE, HUGGINGFACE).
_OTARI_SKIPPED_TESTS: dict[str, str] = {
    # Account / gateway capability gaps
    "test_embedding_providers_async": "no embedding model on the test account (otari-ai#1036)",
    "test_list_models": "gateway /v1/models returns 404 (otari-ai#758)",
    "test_responses_async": "no Responses-API upstream on the test account (otari-ai#907)",
    "test_responses_format_basemodel": "no Responses-API upstream on the test account (otari-ai#907)",
    "test_responses_format_dataclass": "no Responses-API upstream on the test account (otari-ai#907)",
    "test_completion_reasoning": "reasoning content not surfaced by the gateway",
    "test_completion_reasoning_streaming": "reasoning content not surfaced by the gateway",
    "test_create_and_retrieve_batch": "batch needs provider_name + gateway batch support",
    "test_list_batches": "batch needs provider_name + gateway batch support",
    "test_retrieve_batch_results_not_complete": "batch needs provider_name + gateway batch support",
    "test_retrieve_batch_results_with_api_function": "batch needs provider_name + gateway batch support",
    "test_completion_with_image": "gateway returns 502 on multimodal image content",
    "test_completion_with_pdf": "gateway returns 502 on multimodal pdf content",
}


@pytest.fixture(autouse=True)
def _skip_otari_unsupported_capabilities(request: pytest.FixtureRequest) -> None:
    """Skip otari tests for capabilities the hosted test account/provider can't serve yet."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is None or callspec.params.get("provider") is not LLMProvider.OTARI:
        return
    reason = _OTARI_SKIPPED_TESTS.get(request.node.function.__name__)
    if reason:
        pytest.skip(f"otari: {reason}")
