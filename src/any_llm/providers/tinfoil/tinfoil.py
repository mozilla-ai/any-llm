"""Tinfoil: OpenAI-compatible inference served from attested secure enclaves."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from typing_extensions import override

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider

MISSING_PACKAGES_ERROR = None
try:
    from tinfoil import SecureClient
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

SECURE_CLIENT_OPTIONS = ("enclave", "repo", "measurement", "transport", "attestation_bundle_url", "user_cache_secret")
"""Client options that configure Tinfoil's enclave verification rather than the OpenAI client."""


class TinfoilProvider(BaseOpenAIProvider):
    """Provider for Tinfoil, whose OpenAI-compatible API runs inside secure hardware enclaves.

    Requests go through the `tinfoil` SDK's attested HTTP client, which verifies the enclave's
    attestation against the code published in `repo` (or a pinned `measurement`) and then encrypts
    request bodies to the enclave's attested key (`transport="ehbp"`, the default) or pins its TLS
    certificate (`transport="tls"`). A plain OpenAI client pointed at Tinfoil gets none of this,
    which is why this is a provider rather than a registry row.

    Creating the provider makes blocking network calls: without an `enclave`, it asks Tinfoil's
    router list for one, and it verifies that enclave's attestation before the first request.

    `api_base`, when given, is where requests are sent verbatim, as with the SDK's `base_url`; in
    `ehbp` mode it may be a proxy, while in `tls` mode it must be the verified enclave's origin.
    """

    API_BASE = None
    ENV_API_KEY_NAME = "TINFOIL_API_KEY"
    ENV_API_BASE_NAME = "TINFOIL_API_BASE"
    PROVIDER_NAME = "tinfoil"
    PROVIDER_DOCUMENTATION_URL = "https://docs.tinfoil.sh/sdk/python-sdk"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_N = False
    SUPPORTS_RESPONSES = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_MODERATION = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        if "http_client" in kwargs:
            # The attested HTTP client is what verifies the enclave; replacing it would send
            # requests unverified.
            param = "http_client"
            msg = "Tinfoil requests must go through the tinfoil SDK's attested HTTP client."
            raise UnsupportedParameterError(param, self.PROVIDER_NAME, msg)
        secure_options = {name: kwargs.pop(name) for name in SECURE_CLIENT_OPTIONS if name in kwargs}
        secure_client = SecureClient(base_url=api_base, **secure_options)
        # The tinfoil SDK builds an httpx client. Newer openai releases annotate `http_client` as an
        # httpx2 client but still accept httpx ones, which is what the SDK's own AsyncTinfoilAI relies on.
        http_client: Any = secure_client.make_secure_async_http_client()
        self.client = AsyncOpenAI(
            base_url=api_base or f"https://{secure_client.enclave}/v1/",
            api_key=api_key,
            http_client=http_client,
            **kwargs,
        )
