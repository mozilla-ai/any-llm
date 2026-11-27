"""TrustedRouter: OpenAI-compatible LLM router running in GCP Confidential Space."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from openresponses_types import ResponseResource
from typing_extensions import override

from any_llm.exceptions import AnyLLMError, ProviderError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.trustedrouter.attestation import (
    DEFAULT_TRUST_RELEASE_URL,
    GCP_JWKS_URI,
    AttestationPolicy,
    AttestationVerificationError,
    GatewayAttestation,
    verify_gateway_attestation,
)
from any_llm.providers.trustedrouter.utils import _convert_models_list, merge_min_privacy, normalize_min_privacy
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.model import Model
from any_llm.types.responses import ParsedResponse, Response, ResponsesParams, ResponseStreamEvent


class TrustedrouterProvider(BaseOpenAIProvider):
    """Provider for TrustedRouter, an OpenAI-compatible router in Google Confidential Space.

    TrustedRouter serves chat completions, streaming, Responses API, embeddings, and a live
    model catalog from an OpenAI-compatible gateway running on Intel TDX hardware enclaves.

    Extra client options accepted by ``AnyLLM.create("trustedrouter", ...)``:
    - ``min_privacy``: optional default privacy tier (``"any"``, ``"no_store"``, ``"zdr"``,
      ``"confidential"``, ``"e2e"``, ``"e2ee"``) merged into ``extra_body["provider"]["min_privacy"]``.
      May also be passed per request on ``completion``/``responses``/``embedding``.
    - ``verify_attestation``: when ``True``, fetches ``GET <origin>/attestation?nonce=<hex>``
      and verifies the Google Confidential Space RS256 JWT against Google's JWKS, the release
      manifest at ``trust_release_url``, the fresh challenge nonce, and (on direct HTTPS) the
      gateway's TLS leaf certificate. Off by default to avoid blocking network calls during
      client construction; call ``verify_gateway_attestation()`` at any time to verify on demand.
    - ``trust_release_url``: override the release manifest URL
      (default: ``https://trust.trustedrouter.com/gcp-release.json``).
    - ``attestation_jwks_url``: override Google's Confidential Space JWKS URL.
    - ``attestation_timeout``: timeout in seconds for attestation HTTP fetches (default: ``10.0``).
    - ``attestation_policy`` / ``attestation_jwks``: optional pre-built policy or JWKS dict.
    """

    API_BASE = "https://api.trustedrouter.com/v1"
    ENV_API_KEY_NAME = "TRUSTEDROUTER_API_KEY"
    ENV_API_BASE_NAME = "TRUSTEDROUTER_API_BASE"
    PROVIDER_NAME = "trustedrouter"
    PROVIDER_DOCUMENTATION_URL = "https://trustedrouter.com/docs"

    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_N = False
    SUPPORTS_RESPONSES = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_MODERATION = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False
    SUPPORTS_IMAGE_GENERATION = False
    SUPPORTS_RERANK = False

    _default_min_privacy: str | None = None
    _effective_api_base: str = API_BASE
    _trust_release_url: str = DEFAULT_TRUST_RELEASE_URL
    _attestation_jwks_url: str = GCP_JWKS_URI
    _attestation_timeout: float = 10.0
    attestation: GatewayAttestation | None = None

    @staticmethod
    @override
    def _classify_exception(exception: Exception) -> type[AnyLLMError] | None:
        if isinstance(exception, AttestationVerificationError):
            return ProviderError
        return None

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        verify_attestation = bool(kwargs.pop("verify_attestation", False))
        trust_release_url: str = kwargs.pop("trust_release_url", DEFAULT_TRUST_RELEASE_URL)
        attestation_jwks_url: str = kwargs.pop("attestation_jwks_url", GCP_JWKS_URI)
        attestation_timeout: float = kwargs.pop("attestation_timeout", 10.0)
        attestation_policy: AttestationPolicy | None = kwargs.pop("attestation_policy", None)
        attestation_jwks: Mapping[str, Any] | None = kwargs.pop("attestation_jwks", None)
        min_privacy: str | None = kwargs.pop("min_privacy", None)

        self._default_min_privacy = normalize_min_privacy(min_privacy) if min_privacy is not None else None
        self._effective_api_base = api_base or self.API_BASE
        self._trust_release_url = trust_release_url
        self._attestation_jwks_url = attestation_jwks_url
        self._attestation_timeout = attestation_timeout
        self.attestation = None

        if verify_attestation:
            self.attestation = verify_gateway_attestation(
                api_base=self._effective_api_base,
                trust_release_url=trust_release_url,
                jwks_url=attestation_jwks_url,
                timeout=attestation_timeout,
                policy=attestation_policy,
                jwks=attestation_jwks,
            )

        super()._init_client(api_key=api_key, api_base=self._effective_api_base, **kwargs)

    def verify_gateway_attestation(
        self,
        *,
        trust_release_url: str | None = None,
        jwks_url: str | None = None,
        timeout: float | None = None,
        policy: AttestationPolicy | None = None,
        jwks: Mapping[str, Any] | None = None,
    ) -> GatewayAttestation:
        """Fetch and verify the gateway's Confidential Space attestation on demand."""
        result = verify_gateway_attestation(
            api_base=self._effective_api_base,
            trust_release_url=trust_release_url or self._trust_release_url,
            jwks_url=jwks_url or self._attestation_jwks_url,
            timeout=self._attestation_timeout if timeout is None else timeout,
            policy=policy,
            jwks=jwks,
        )
        self.attestation = result
        return result

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert TrustedRouter list models response to filtered, valid Model objects."""
        return _convert_models_list(response)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for TrustedRouter API, including ``min_privacy``."""
        min_privacy = kwargs.pop("min_privacy", None)
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        merge_min_privacy(converted_params, min_privacy=min_privacy, default_min_privacy=None)
        return converted_params

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        min_privacy = kwargs.pop("min_privacy", None)
        merge_min_privacy(kwargs, min_privacy=min_privacy, default_min_privacy=self._default_min_privacy)
        return await super()._acompletion(params, **kwargs)

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> ResponseResource | Response | ParsedResponse[Any] | AsyncIterator[ResponseStreamEvent]:
        min_privacy = kwargs.pop("min_privacy", None)
        merge_min_privacy(kwargs, min_privacy=min_privacy, default_min_privacy=self._default_min_privacy)
        return await super()._aresponses(params, **kwargs)

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        min_privacy = kwargs.pop("min_privacy", None)
        merge_min_privacy(kwargs, min_privacy=min_privacy, default_min_privacy=self._default_min_privacy)
        return await super()._aembedding(model, inputs, **kwargs)
