import re
from pathlib import Path

import pytest

from any_llm.any_llm import AnyLLM
from any_llm.constants import (
    VERIFIED_LOCAL_PROVIDERS,
    VERIFIED_PROVIDERS,
    ProviderTier,
    get_provider_tier,
)
from tests.constants import LOCAL_PROVIDERS

WORKFLOW_PATH = Path(__file__).parent.parent.parent / ".github" / "workflows" / "tests-integration.yaml"


def _workflow_expected_providers() -> tuple[set[str], set[str]]:
    """Parse the two EXPECTED_PROVIDERS lists out of the integration workflow.

    Returns (non_local, local) in the order the workflow declares them.
    """
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    matches = re.findall(r'expected_providers=([a-z0-9,]+)"', text)
    assert len(matches) == 2, f"expected exactly 2 expected_providers assignments, found {len(matches)}"
    return ({name for name in matches[0].split(",") if name}, {name for name in matches[1].split(",") if name})


def test_verified_set_matches_workflow_non_local_keys() -> None:
    """The advertised verified tier must match the keys CI actually holds.

    If these drift, docs/providers.md promises support that CI does not exercise.
    """
    non_local, _ = _workflow_expected_providers()
    assert VERIFIED_PROVIDERS - VERIFIED_LOCAL_PROVIDERS == non_local


def test_verified_local_set_matches_workflow_local_providers() -> None:
    _, local = _workflow_expected_providers()
    assert VERIFIED_LOCAL_PROVIDERS == local


def test_verified_local_providers_are_a_subset_of_verified() -> None:
    assert VERIFIED_LOCAL_PROVIDERS <= VERIFIED_PROVIDERS


def test_every_verified_name_is_a_real_provider() -> None:
    """Catches a typo in VERIFIED_PROVIDERS, which would silently mark a provider community."""
    unknown = VERIFIED_PROVIDERS - set(AnyLLM.get_supported_providers())
    assert not unknown, f"VERIFIED_PROVIDERS names that are not providers: {unknown}"


@pytest.mark.parametrize("name", ["openai", "anthropic", "moonshot", "ollama"])
def test_known_verified_providers_report_verified(name: str) -> None:
    assert get_provider_tier(name) is ProviderTier.VERIFIED


@pytest.mark.parametrize("name", ["atlascloud", "telnyx", "qiniu", "not-a-provider-at-all"])
def test_unlisted_providers_report_community(name: str) -> None:
    assert get_provider_tier(name) is ProviderTier.COMMUNITY


def test_get_provider_tier_normalizes_name() -> None:
    assert get_provider_tier("  OpenAI ") is ProviderTier.VERIFIED


def test_metadata_carries_the_tier() -> None:
    metadata = {entry.name: entry for entry in AnyLLM.get_all_provider_metadata()}
    assert metadata["openai"].tier is ProviderTier.VERIFIED
    # moonshot is a config-only registry row that we hold a key for, so the tier
    # is independent of code shape.
    assert metadata["moonshot"].tier is ProviderTier.VERIFIED
    assert metadata["atlascloud"].tier is ProviderTier.COMMUNITY


@pytest.mark.parametrize("name", ["somegateway", "openai", "moonshot", " OpenAI "])
def test_custom_endpoints_never_report_verified(name: str) -> None:
    """A custom endpoint is an arbitrary URL, so it is never advertised as verified.

    The tier is derived from the provider name, and create_openai_compatible lets the
    caller pick any name, so naming an endpoint after a provider we hold a key for
    must not inherit its tier. An earlier version of this test only used a name that
    happened not to be verified, so it passed while the property was broken.
    """
    provider = AnyLLM.create_openai_compatible(name=name, api_base="https://untrusted.test/v1", api_key="k")
    assert provider.get_provider_metadata().tier is ProviderTier.COMMUNITY
    # The name itself is still reported, so the endpoint is not misattributed either way.
    assert provider.get_provider_metadata().name == name


def test_verified_local_providers_are_covered_by_the_test_suite_local_list() -> None:
    """Guard the two separate notions of "local provider" against drifting apart.

    tests/constants.py LOCAL_PROVIDERS drives test parametrization and is the wider
    set: it also contains cascadia, which is in CI_EXCLUDED_PROVIDERS and therefore
    has no CI run to be verified by. So every verified-local name must appear there,
    but not the reverse.
    """
    assert VERIFIED_LOCAL_PROVIDERS <= {provider.value for provider in LOCAL_PROVIDERS}
