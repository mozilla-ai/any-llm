import os

import pytest

from any_llm.constants import LLMProvider
from tests.constants import EXPECTED_PROVIDERS

# The otari/gateway gateway has no server started in CI (unlike ollama, llamacpp, lmstudio and
# llamafile, which the local integration job boots up). Its integration tests can therefore
# only run against a real endpoint configured via these env vars, for example the dedicated
# otari e2e job. When the endpoint is not configured we skip rather than fail: otari 0.1.0's
# SDK is urllib3-based, so an unreachable placeholder host raises errors the per-test skip
# clauses do not catch, and some requests fail during client-side validation before any
# network call. Listing the provider in EXPECTED_PROVIDERS forces a run so a misconfigured
# e2e job fails loudly instead of skipping silently.
_GATEWAY_PROVIDER_ENV_VARS: dict[LLMProvider, tuple[str, ...]] = {
    LLMProvider.OTARI: ("OTARI_API_BASE",),
    LLMProvider.GATEWAY: ("GATEWAY_API_BASE", "OTARI_API_BASE"),
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
