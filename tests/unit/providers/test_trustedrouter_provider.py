import base64
import hashlib
import json
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from any_llm import AnyLLM
from any_llm.constants import LLMProvider, ProviderTier, get_provider_tier
from any_llm.exceptions import MissingApiKeyError, ProviderError, UnsupportedParameterError
from any_llm.providers.trustedrouter import (
    AttestationPolicy,
    AttestationVerificationError,
    GatewayAttestation,
    TrustedrouterProvider,
)
from any_llm.providers.trustedrouter.attestation import (
    DEFAULT_ATTESTATION_AUDIENCE,
    DEFAULT_TRUST_RELEASE_URL,
    GCP_ISSUER,
    GCP_JWKS_URI,
    policy_from_trust_release,
    verify_attestation_jwt,
    verify_gateway_attestation,
)
from any_llm.providers.trustedrouter.utils import _convert_models_list, normalize_min_privacy
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model

IMAGE_DIGEST = "sha256:fe0c91d70b77f129cb4c08cd1a760eefa0a1ea6e3c678e81b3b94c68070a0add"
IMAGE_REFERENCE = "us-central1-docker.pkg.dev/quill-cloud-proxy/quill/enclave-multi:gcp-release-7cf7e348"
SAMPLE_RELEASE: dict[str, Any] = {
    "accepted_image_digests": [IMAGE_DIGEST],
    "accepted_image_references": [IMAGE_REFERENCE],
    "attestation_audience": DEFAULT_ATTESTATION_AUDIENCE,
    "attestation_issuer": GCP_ISSUER,
    "image_digest": IMAGE_DIGEST,
    "image_reference": IMAGE_REFERENCE,
    "platform": "gcp-confidential-space",
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _int_to_b64url(value: int) -> str:
    length = (value.bit_length() + 7) // 8
    return _b64url(value.to_bytes(length, "big"))


def _generate_rsa_jwk(kid: str = "test-kid-1") -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = private_key.public_key().public_numbers()
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "alg": "RS256",
                "kid": kid,
                "n": _int_to_b64url(numbers.n),
                "e": _int_to_b64url(numbers.e),
            }
        ]
    }
    return private_key, jwks


def _sign_jwt(
    private_key: rsa.RSAPrivateKey,
    claims: dict[str, Any],
    *,
    header: dict[str, Any] | None = None,
) -> bytes:
    jwt_header = header or {"alg": "RS256", "kid": "test-kid-1", "typ": "JWT"}
    h_b64 = _b64url(json.dumps(jwt_header, separators=(",", ":")).encode("utf-8"))
    p_b64 = _b64url(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{h_b64}.{p_b64}.{_b64url(signature)}".encode("ascii")


def _valid_claims(
    *,
    nonce: str,
    cert_sha256: str | None = None,
    now: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    current = int(time.time()) if now is None else now
    eat_nonce = [nonce]
    if cert_sha256 is not None:
        eat_nonce.insert(0, cert_sha256)
    claims: dict[str, Any] = {
        "iss": GCP_ISSUER,
        "aud": DEFAULT_ATTESTATION_AUDIENCE,
        "iat": current - 10,
        "nbf": current - 10,
        "exp": current + 3600,
        "eat_nonce": eat_nonce,
        "secboot": True,
        "hwmodel": "GCP_INTEL_TDX",
        "swname": "CONFIDENTIAL_SPACE",
        "dbgstat": "disabled-since-boot",
        "submods": {
            "container": {
                "image_digest": IMAGE_DIGEST,
                "image_reference": IMAGE_REFERENCE,
            }
        },
    }
    claims.update(overrides)
    return claims


def _mock_provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> TrustedrouterProvider:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TrustedrouterProvider(api_key="sk-tr-test", http_client=http_client, **kwargs)


def _make_trustedrouter_model(**overrides: object) -> Model:
    defaults: dict[str, object] = {
        "id": "trustedrouter/auto",
        "name": "TrustedRouter Auto",
        "created": 0,
        "object": None,
        "owned_by": None,
        "trustedrouter": {
            "provider": "trustedrouter",
            "aliases": ["nyte/auto"],
            "configuration_hidden": False,
            "internal_only": False,
            "synthetic_monitor": False,
            "supports_chat": True,
            "supports_responses": True,
            "supports_embeddings": False,
        },
    }
    defaults.update(overrides)
    return Model.model_construct(**defaults)  # type: ignore[arg-type]


# ---- metadata, registration, config ----


def test_provider_metadata_and_registration() -> None:
    provider = TrustedrouterProvider(api_key="sk-tr-test")
    assert provider.PROVIDER_NAME == "trustedrouter"
    assert provider.API_BASE == "https://api.trustedrouter.com/v1"
    assert provider.ENV_API_KEY_NAME == "TRUSTEDROUTER_API_KEY"
    assert provider.ENV_API_BASE_NAME == "TRUSTEDROUTER_API_BASE"
    assert provider.PROVIDER_DOCUMENTATION_URL == "https://trustedrouter.com/docs"
    assert provider.attestation is None
    assert AnyLLM.get_provider_class("trustedrouter") is TrustedrouterProvider
    assert "trustedrouter" in AnyLLM.get_supported_providers()
    assert LLMProvider("trustedrouter") is LLMProvider.TRUSTEDROUTER
    assert get_provider_tier("trustedrouter") is ProviderTier.COMMUNITY
    assert AnyLLM.split_model_provider("trustedrouter:trustedrouter/auto") == (
        LLMProvider.TRUSTEDROUTER,
        "trustedrouter/auto",
    )


def test_capability_flags() -> None:
    assert TrustedrouterProvider.SUPPORTS_COMPLETION
    assert TrustedrouterProvider.SUPPORTS_COMPLETION_STREAMING
    assert TrustedrouterProvider.SUPPORTS_COMPLETION_REASONING
    assert TrustedrouterProvider.SUPPORTS_COMPLETION_IMAGE
    assert TrustedrouterProvider.SUPPORTS_RESPONSES
    assert TrustedrouterProvider.SUPPORTS_EMBEDDING
    assert TrustedrouterProvider.SUPPORTS_LIST_MODELS
    assert not TrustedrouterProvider.SUPPORTS_COMPLETION_PDF
    assert not TrustedrouterProvider.SUPPORTS_COMPLETION_N
    assert not TrustedrouterProvider.SUPPORTS_MODERATION
    assert not TrustedrouterProvider.SUPPORTS_BATCH
    assert not TrustedrouterProvider.SUPPORTS_IMAGE_GENERATION
    assert not TrustedrouterProvider.SUPPORTS_RERANK


def test_missing_api_key_raises_before_attestation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRUSTEDROUTER_API_KEY", raising=False)
    with (
        patch("any_llm.providers.trustedrouter.trustedrouter.verify_gateway_attestation") as mock_verify,
        pytest.raises(MissingApiKeyError),
    ):
        TrustedrouterProvider(verify_attestation=True)
    mock_verify.assert_not_called()


def test_api_key_and_base_from_env_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTEDROUTER_API_KEY", "env-key")
    monkeypatch.setenv("TRUSTEDROUTER_API_BASE", "https://eu.trustedrouter.example/v1")

    from_env = TrustedrouterProvider()
    assert from_env.client.api_key == "env-key"
    assert str(from_env.client.base_url).rstrip("/") == "https://eu.trustedrouter.example/v1"

    explicit = TrustedrouterProvider(api_key="explicit-key", api_base="https://explicit.example/v1", max_retries=0)
    assert explicit.client.api_key == "explicit-key"
    assert str(explicit.client.base_url).rstrip("/") == "https://explicit.example/v1"
    assert explicit.client.max_retries == 0


def test_api_base_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRUSTEDROUTER_API_BASE", raising=False)
    provider = TrustedrouterProvider(api_key="sk-tr-test")
    assert str(provider.client.base_url).rstrip("/") == "https://api.trustedrouter.com/v1"


# ---- model listing & filtering ----


def test_convert_models_list_filters_hidden_internal_and_synthetic_entries() -> None:
    raw = SimpleNamespace(
        data=[
            _make_trustedrouter_model(id="trustedrouter/auto"),
            _make_trustedrouter_model(
                id="openai/text-embedding-3-small",
                created=None,
                trustedrouter={
                    "provider": "openai",
                    "configuration_hidden": False,
                    "internal_only": False,
                    "synthetic_monitor": False,
                    "supports_embeddings": True,
                },
            ),
            _make_trustedrouter_model(
                id="trustedrouter/athena-1.0",
                trustedrouter={"provider": "trustedrouter", "configuration_hidden": True},
            ),
            _make_trustedrouter_model(
                id="trustedrouter/internal-eval",
                trustedrouter={"provider": "trustedrouter", "internal_only": True},
            ),
            _make_trustedrouter_model(
                id="trustedrouter/canary-probe",
                trustedrouter={"provider": "trustedrouter", "synthetic_monitor": True},
            ),
            {"id": "plain/model", "created": 123},
            SimpleNamespace(id="ns/model", created=456, object=None, owned_by=None),
        ]
    )

    models = _convert_models_list(raw)
    assert [m.id for m in models] == [
        "trustedrouter/auto",
        "openai/text-embedding-3-small",
        "plain/model",
        "ns/model",
    ]
    assert models[0].object == "model"
    assert models[0].owned_by == "trustedrouter"
    assert models[0].model_extra is not None
    assert models[0].model_extra["trustedrouter"]["aliases"] == ["nyte/auto"]

    assert models[1].owned_by == "openai"
    assert models[1].created == 0

    assert models[2].owned_by == "trustedrouter"
    assert models[3].owned_by == "trustedrouter"

    # Round-trip JSON serialization must succeed for every normalized model.
    for model in models:
        restored = Model.model_validate_json(model.model_dump_json())
        assert restored.id == model.id
        assert restored.object == "model"


@pytest.mark.asyncio
async def test_alist_models_through_http_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "trustedrouter/auto",
                        "name": "TrustedRouter Auto",
                        "created": 0,
                        "trustedrouter": {
                            "provider": "trustedrouter",
                            "aliases": ["nyte/auto"],
                            "configuration_hidden": False,
                            "internal_only": False,
                            "synthetic_monitor": False,
                        },
                    },
                    {
                        "id": "trustedrouter/hidden-route",
                        "created": 0,
                        "trustedrouter": {"provider": "trustedrouter", "configuration_hidden": True},
                    },
                ]
            },
        )

    provider = _mock_provider(handler)
    models = await provider.alist_models()

    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.trustedrouter.com/v1/models"
    assert [m.id for m in models] == ["trustedrouter/auto"]
    assert models[0].object == "model"
    assert models[0].owned_by == "trustedrouter"


# ---- completions, streaming, responses, embeddings, min_privacy ----


@pytest.mark.asyncio
async def test_completion_streaming_responses_and_embeddings_with_min_privacy() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            body = json.loads(request.content)
            if body.get("stream"):
                sse = (
                    'data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1,'
                    '"model":"trustedrouter/auto","choices":[{"index":0,"delta":{"role":"assistant",'
                    '"reasoning_content":"step 1","content":"hi"},"finish_reason":null}]}\n\n'
                    'data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1,'
                    '"model":"trustedrouter/auto","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "trustedrouter/auto",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "hello",
                                "reasoning_content": "thought",
                            },
                        }
                    ],
                },
            )
        if path.endswith("/responses"):
            return httpx.Response(
                200,
                json={
                    "id": "resp-1",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "trustedrouter/auto",
                    "output": [
                        {
                            "id": "msg-1",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "Paris", "annotations": []}],
                        }
                    ],
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
            )
        if path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "openai/text-embedding-3-small",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )
        return httpx.Response(404)

    provider = _mock_provider(handler, min_privacy="ZDR")

    # 1. Non-streaming completion inherits default min_privacy="zdr" and normalizes reasoning.
    completion_resp: Any = await provider.acompletion(
        model="trustedrouter/auto",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ],
    )
    assert completion_resp.choices[0].message.content == "hello"
    assert completion_resp.choices[0].message.reasoning.content == "thought"
    assert json.loads(requests[0].content)["provider"] == {"min_privacy": "zdr"}
    assert requests[0].headers["authorization"] == "Bearer sk-tr-test"

    # 2. Streaming completion with per-request min_privacy="e2e" overriding instance default.
    caller_extra: dict[str, Any] = {"provider": {"allow_fallbacks": False}, "user_tag": "legal"}
    stream: Any = await provider.acompletion(
        model="trustedrouter/auto",
        messages=[{"role": "user", "content": "Stream"}],
        stream=True,
        min_privacy="e2e",
        extra_body=caller_extra,
    )
    chunks = [chunk async for chunk in stream]
    assert chunks[0].choices[0].delta.content == "hi"
    assert chunks[0].choices[0].delta.reasoning.content == "step 1"
    stream_body = json.loads(requests[1].content)
    assert stream_body["provider"] == {"allow_fallbacks": False, "min_privacy": "e2e"}
    assert stream_body["user_tag"] == "legal"
    # Caller dict must not be mutated.
    assert caller_extra == {"provider": {"allow_fallbacks": False}, "user_tag": "legal"}

    # 3. Responses API respects explicit extra_body["provider"]["min_privacy"] over instance default.
    resp: Any = await provider.aresponses(
        "trustedrouter/auto",
        "Capital of France?",
        extra_body={"provider": {"min_privacy": "confidential"}},
    )
    assert resp.id == "resp-1"
    assert json.loads(requests[2].content)["provider"] == {"min_privacy": "confidential"}

    # 4. Embeddings API inherits default min_privacy="zdr".
    emb = await provider.aembedding("openai/text-embedding-3-small", "hello")
    assert emb.data[0].embedding == [0.1, 0.2]
    assert json.loads(requests[3].content)["provider"] == {"min_privacy": "zdr"}


def test_convert_completion_params_handles_min_privacy_and_validation() -> None:
    params = CompletionParams(model_id="trustedrouter/auto", messages=[{"role": "user", "content": "hi"}])

    plain = TrustedrouterProvider._convert_completion_params(params)
    assert "extra_body" not in plain

    with_privacy = TrustedrouterProvider._convert_completion_params(params, min_privacy="confidential")
    assert with_privacy["extra_body"] == {"provider": {"min_privacy": "confidential"}}

    assert normalize_min_privacy(" E2EE ") == "e2ee"
    with pytest.raises(ValueError, match="unsupported min_privacy"):
        normalize_min_privacy("invalid-tier")
    with pytest.raises(ValueError, match="min_privacy must be one of"):
        normalize_min_privacy(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="extra_body must be a mapping"):
        TrustedrouterProvider._convert_completion_params(params, min_privacy="zdr", extra_body="bad")
    with pytest.raises(ValueError, match="extra_body\\['provider'\\] must be a mapping"):
        TrustedrouterProvider._convert_completion_params(params, min_privacy="zdr", extra_body={"provider": "bad"})


@pytest.mark.asyncio
async def test_unsupported_n_raises() -> None:
    provider = TrustedrouterProvider(api_key="sk-tr-test")
    with pytest.raises(UnsupportedParameterError, match="n"):
        await provider.acompletion(model="trustedrouter/auto", messages=[{"role": "user", "content": "hi"}], n=2)


# ---- Confidential Space attestation verification ----


def test_policy_from_trust_release_and_failures() -> None:
    policy = policy_from_trust_release(SAMPLE_RELEASE)
    assert policy.gcp_audience == DEFAULT_ATTESTATION_AUDIENCE
    assert policy.expected_image_digests == (IMAGE_DIGEST,)
    assert policy.expected_image_references == (IMAGE_REFERENCE,)
    assert policy.pins_image_identity

    fallback = policy_from_trust_release({"image_digest": IMAGE_DIGEST, "image_reference": IMAGE_REFERENCE})
    assert fallback.expected_image_digests == (IMAGE_DIGEST,)
    assert fallback.expected_image_references == (IMAGE_REFERENCE,)

    with pytest.raises(AttestationVerificationError, match="JSON object"):
        policy_from_trust_release("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(AttestationVerificationError, match="pins no image identity"):
        policy_from_trust_release({"attestation_audience": "quill-cloud"})
    with pytest.raises(AttestationVerificationError, match="string or null"):
        policy_from_trust_release({"image_digest": 123})
    with pytest.raises(AttestationVerificationError, match="array of non-empty strings"):
        policy_from_trust_release({"accepted_image_digests": [""]})


def test_verify_attestation_jwt_valid_document_and_tls_cert_binding() -> None:
    private_key, jwks = _generate_rsa_jwk()
    policy = policy_from_trust_release(SAMPLE_RELEASE)
    nonce = "a" * 64
    cert_der = b"fake-der-leaf-certificate"
    cert_sha = hashlib.sha256(cert_der).hexdigest()

    token = _sign_jwt(private_key, _valid_claims(nonce=nonce, cert_sha256=cert_sha))

    # Document-only mode (no TLS cert supplied).
    doc_result = verify_attestation_jwt(token, policy=policy, nonce_hex=nonce, jwks=jwks)
    assert isinstance(doc_result, GatewayAttestation)
    assert doc_result.binding == "none"
    assert doc_result.cert_sha256 is None
    assert doc_result.hwmodel == "GCP_INTEL_TDX"
    assert doc_result.image_digest == IMAGE_DIGEST
    assert doc_result.image_reference == IMAGE_REFERENCE
    assert doc_result.nonce == nonce

    # Live TLS leaf cert binding mode.
    bound_result = verify_attestation_jwt(
        token,
        policy=policy_from_trust_release(SAMPLE_RELEASE, cert_sha256=cert_sha),
        nonce_hex=nonce,
        jwks=jwks,
        tls_cert_der=cert_der,
    )
    assert bound_result.binding == "tls-cert"
    assert bound_result.cert_sha256 == cert_sha

    # Policy expected_cert_sha256 verified via labelled claim without tls_cert_der.
    labelled_token = _sign_jwt(
        private_key,
        _valid_claims(nonce=nonce, tls_cert_sha256=cert_sha, aud=[DEFAULT_ATTESTATION_AUDIENCE]),
    )
    pinned_result = verify_attestation_jwt(
        labelled_token,
        policy=policy_from_trust_release(SAMPLE_RELEASE, cert_sha256=cert_sha),
        nonce_hex=nonce,
        jwks=jwks,
    )
    assert pinned_result.binding == "none"
    assert pinned_result.cert_sha256 == cert_sha


def test_verify_attestation_jwt_rejects_invalid_tokens_and_claims() -> None:
    private_key, jwks = _generate_rsa_jwk()
    other_key, _ = _generate_rsa_jwk(kid="test-kid-1")
    policy = policy_from_trust_release(SAMPLE_RELEASE)
    nonce = "b" * 64
    now = 1_800_000_000

    # Policy without image pins fails closed.
    with pytest.raises(AttestationVerificationError, match="pin at least one"):
        verify_attestation_jwt(b"a.b.c", policy=AttestationPolicy(), nonce_hex=nonce, jwks=jwks)

    # Empty nonce fails closed.
    with pytest.raises(AttestationVerificationError, match="non-empty challenge nonce"):
        verify_attestation_jwt(b"a.b.c", policy=policy, nonce_hex="", jwks=jwks)

    # Malformed JWTs & header/JWK errors.
    with pytest.raises(AttestationVerificationError, match="3 JWT segments"):
        verify_attestation_jwt(b"not-a-jwt", policy=policy, nonce_hex=nonce, jwks=jwks)
    with pytest.raises(AttestationVerificationError, match="invalid JWT encoding"):
        verify_attestation_jwt(b"!!!.!!!.!!!", policy=policy, nonce_hex=nonce, jwks=jwks)
    with pytest.raises(AttestationVerificationError, match="JSON objects"):
        verify_attestation_jwt(
            f"{_b64url(b'[]')}.{_b64url(b'{}')}.{_b64url(b'x')}".encode(),
            policy=policy,
            nonce_hex=nonce,
            jwks=jwks,
        )

    bad_alg = _sign_jwt(private_key, _valid_claims(nonce=nonce, now=now), header={"alg": "HS256", "kid": "test-kid-1"})
    with pytest.raises(AttestationVerificationError, match="unsupported JWT alg"):
        verify_attestation_jwt(bad_alg, policy=policy, nonce_hex=nonce, jwks=jwks, now=now)

    no_kid = _sign_jwt(private_key, _valid_claims(nonce=nonce, now=now), header={"alg": "RS256", "kid": ""})
    with pytest.raises(AttestationVerificationError, match="JWT kid"):
        verify_attestation_jwt(no_kid, policy=policy, nonce_hex=nonce, jwks=jwks, now=now)

    good_token = _sign_jwt(private_key, _valid_claims(nonce=nonce, now=now))
    with pytest.raises(AttestationVerificationError, match="JWKS must be a JSON object"):
        verify_attestation_jwt(good_token, policy=policy, nonce_hex=nonce, jwks="bad", now=now)  # type: ignore[arg-type]
    with pytest.raises(AttestationVerificationError, match="JWKS keys"):
        verify_attestation_jwt(good_token, policy=policy, nonce_hex=nonce, jwks={"keys": "bad"}, now=now)
    with pytest.raises(AttestationVerificationError, match="no JWK with kid"):
        verify_attestation_jwt(good_token, policy=policy, nonce_hex=nonce, jwks={"keys": []}, now=now)
    with pytest.raises(AttestationVerificationError, match="expected RSA key"):
        verify_attestation_jwt(
            good_token,
            policy=policy,
            nonce_hex=nonce,
            jwks={"keys": [{"kid": "test-kid-1", "kty": "EC"}]},
            now=now,
        )
    with pytest.raises(AttestationVerificationError, match="JWK n and e"):
        verify_attestation_jwt(
            good_token,
            policy=policy,
            nonce_hex=nonce,
            jwks={"keys": [{"kid": "test-kid-1", "kty": "RSA"}]},
            now=now,
        )
    with pytest.raises(AttestationVerificationError, match="malformed RSA JWK"):
        verify_attestation_jwt(
            good_token,
            policy=policy,
            nonce_hex=nonce,
            jwks={"keys": [{"kid": "test-kid-1", "kty": "RSA", "n": "AA", "e": "AA"}]},
            now=now,
        )

    wrong_sig = _sign_jwt(other_key, _valid_claims(nonce=nonce, now=now))
    with pytest.raises(AttestationVerificationError, match="signature verification failed"):
        verify_attestation_jwt(wrong_sig, policy=policy, nonce_hex=nonce, jwks=jwks, now=now)

    # Claim mismatches.
    bad_claim_cases: list[tuple[dict[str, Any], str]] = [
        ({"exp": "not-an-int"}, "valid integer exp"),
        ({"exp": now - 1}, "JWT expired"),
        ({"nbf": "bad"}, "nbf must be an integer"),
        ({"nbf": now + 120}, "not valid until"),
        ({"iss": "https://evil.example"}, "unexpected JWT issuer"),
        ({"aud": 123}, "JWT aud must be"),
        ({"aud": "wrong-aud"}, "not in JWT aud"),
        ({"dbgstat": "enabled"}, "disabled-since-boot"),
        ({"swname": "OTHER"}, "CONFIDENTIAL_SPACE"),
        ({"secboot": False}, "Secure Boot"),
        ({"hwmodel": "UNKNOWN_HW"}, "unsupported confidential hardware"),
        ({"submods": {}}, "submods.container"),
        ({"submods": {"container": {"image_digest": 1, "image_reference": IMAGE_REFERENCE}}}, "must be strings"),
        (
            {"submods": {"container": {"image_digest": "sha256:bad", "image_reference": IMAGE_REFERENCE}}},
            "image_digest mismatch",
        ),
        (
            {"submods": {"container": {"image_digest": IMAGE_DIGEST, "image_reference": "other:tag"}}},
            "image_reference mismatch",
        ),
        ({"eat_nonce": ["different-nonce"]}, "challenge nonce"),
    ]
    for overrides, pattern in bad_claim_cases:
        token = _sign_jwt(private_key, _valid_claims(nonce=nonce, now=now, **overrides))
        with pytest.raises(AttestationVerificationError, match=pattern):
            verify_attestation_jwt(token, policy=policy, nonce_hex=nonce, jwks=jwks, now=now)

    # TLS cert mismatches.
    with pytest.raises(AttestationVerificationError, match="TLS leaf certificate SHA-256"):
        verify_attestation_jwt(good_token, policy=policy, nonce_hex=nonce, jwks=jwks, tls_cert_der=b"other", now=now)

    cert_der = b"actual-cert"
    cert_sha = hashlib.sha256(cert_der).hexdigest()
    bound_token = _sign_jwt(private_key, _valid_claims(nonce=nonce, cert_sha256=cert_sha, now=now))
    with pytest.raises(AttestationVerificationError, match="does not match policy pin"):
        verify_attestation_jwt(
            bound_token,
            policy=policy_from_trust_release(SAMPLE_RELEASE, cert_sha256="0" * 64),
            nonce_hex=nonce,
            jwks=jwks,
            tls_cert_der=cert_der,
            now=now,
        )
    with pytest.raises(AttestationVerificationError, match="expected_cert_sha256"):
        verify_attestation_jwt(
            good_token,
            policy=policy_from_trust_release(SAMPLE_RELEASE, cert_sha256="0" * 64),
            nonce_hex=nonce,
            jwks=jwks,
            now=now,
        )


def test_verify_gateway_attestation_over_mock_transport_with_tls_stream() -> None:
    private_key, jwks = _generate_rsa_jwk()
    cert_der = b"trustedrouter-enclave-leaf-cert"
    cert_sha = hashlib.sha256(cert_der).hexdigest()
    seen_urls: list[str] = []

    class _FakeSSLObject:
        def getpeercert(self, binary_form: bool = False) -> bytes | None:
            return cert_der if binary_form else None

    class _FakeStream:
        def get_extra_info(self, key: str) -> Any:
            return _FakeSSLObject() if key == "ssl_object" else None

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen_urls.append(url)
        if url == DEFAULT_TRUST_RELEASE_URL:
            return httpx.Response(200, json=SAMPLE_RELEASE)
        if url == GCP_JWKS_URI:
            return httpx.Response(200, json=jwks)
        if request.url.path == "/attestation":
            nonce = request.url.params["nonce"]
            jwt_bytes = _sign_jwt(private_key, _valid_claims(nonce=nonce, cert_sha256=cert_sha))
            return httpx.Response(200, content=jwt_bytes, extensions={"network_stream": _FakeStream()})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = verify_gateway_attestation(api_base="https://api.trustedrouter.com/v1", http_client=client)

    assert result.binding == "tls-cert"
    assert result.cert_sha256 == cert_sha
    assert result.image_digest == IMAGE_DIGEST
    assert seen_urls[0] == DEFAULT_TRUST_RELEASE_URL
    assert seen_urls[1] == GCP_JWKS_URI
    assert seen_urls[2].startswith("https://api.trustedrouter.com/attestation?nonce=")


def test_verify_gateway_attestation_error_paths() -> None:
    with pytest.raises(AttestationVerificationError, match="invalid TrustedRouter api_base"):
        verify_gateway_attestation(api_base="not-a-url")

    def empty_att_handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == DEFAULT_TRUST_RELEASE_URL:
            return httpx.Response(200, json=SAMPLE_RELEASE)
        if str(request.url) == GCP_JWKS_URI:
            return httpx.Response(200, json={"keys": []})
        return httpx.Response(200, content=b"   ")

    with (
        httpx.Client(transport=httpx.MockTransport(empty_att_handler)) as client,
        pytest.raises(AttestationVerificationError, match="empty response"),
    ):
        verify_gateway_attestation(api_base="https://api.trustedrouter.com/v1", http_client=client)

    def bad_json_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    with (
        httpx.Client(transport=httpx.MockTransport(bad_json_handler)) as client,
        pytest.raises(AttestationVerificationError, match="malformed JSON"),
    ):
        verify_gateway_attestation(api_base="https://api.trustedrouter.com/v1", http_client=client)

    def list_json_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    with (
        httpx.Client(transport=httpx.MockTransport(list_json_handler)) as client,
        pytest.raises(AttestationVerificationError, match="JSON object"),
    ):
        verify_gateway_attestation(api_base="https://api.trustedrouter.com/v1", http_client=client)


def test_provider_init_and_on_demand_attestation_verification() -> None:
    fake_attestation = GatewayAttestation(
        image_digest=IMAGE_DIGEST,
        image_reference=IMAGE_REFERENCE,
        nonce="c" * 64,
        expires_at=2_000_000_000,
        issuer=GCP_ISSUER,
        audience=DEFAULT_ATTESTATION_AUDIENCE,
        hwmodel="GCP_INTEL_TDX",
        cert_sha256="d" * 64,
        binding="tls-cert",
        raw_claims={},
    )
    policy = policy_from_trust_release(SAMPLE_RELEASE)

    with patch(
        "any_llm.providers.trustedrouter.trustedrouter.verify_gateway_attestation",
        return_value=fake_attestation,
    ) as mock_verify:
        provider = TrustedrouterProvider(
            api_key="sk-tr-test",
            verify_attestation=True,
            trust_release_url="https://trust.example/release.json",
            attestation_jwks_url="https://jwks.example/keys",
            attestation_timeout=5.0,
            attestation_policy=policy,
            attestation_jwks={"keys": []},
        )
        assert provider.attestation is fake_attestation
        mock_verify.assert_called_once_with(
            api_base="https://api.trustedrouter.com/v1",
            trust_release_url="https://trust.example/release.json",
            jwks_url="https://jwks.example/keys",
            timeout=5.0,
            policy=policy,
            jwks={"keys": []},
        )

        # On-demand verification updates provider.attestation.
        provider.attestation = None
        verified = provider.verify_gateway_attestation()
        assert verified is fake_attestation
        assert provider.attestation is fake_attestation


def test_attestation_failure_classification_with_and_without_unified_exceptions() -> None:
    with (
        patch(
            "any_llm.providers.trustedrouter.trustedrouter.verify_gateway_attestation",
            side_effect=AttestationVerificationError("invalid JWT signature"),
        ),
        pytest.raises(AttestationVerificationError, match="invalid JWT signature"),
    ):
        AnyLLM.create("trustedrouter", api_key="sk-tr-test", verify_attestation=True, unified_exceptions=False)

    with (
        patch(
            "any_llm.providers.trustedrouter.trustedrouter.verify_gateway_attestation",
            side_effect=AttestationVerificationError("invalid JWT signature"),
        ),
        pytest.raises(ProviderError, match="invalid JWT signature") as exc_info,
    ):
        AnyLLM.create("trustedrouter", api_key="sk-tr-test", verify_attestation=True, unified_exceptions=True)
    assert isinstance(exc_info.value.original_exception, AttestationVerificationError)
    assert TrustedrouterProvider._classify_exception(RuntimeError("other")) is None


def test_missing_cryptography_raises_helpful_import_error() -> None:
    private_key, jwks = _generate_rsa_jwk()
    policy = policy_from_trust_release(SAMPLE_RELEASE)
    nonce = "e" * 64
    token = _sign_jwt(private_key, _valid_claims(nonce=nonce))

    blocked = {name: None for name in list(sys.modules) if name == "cryptography" or name.startswith("cryptography.")}
    blocked["cryptography"] = None
    with (
        patch.dict(sys.modules, blocked),
        pytest.raises(ImportError, match="any-llm-sdk\\[trustedrouter\\]"),
    ):
        verify_attestation_jwt(token, policy=policy, nonce_hex=nonce, jwks=jwks)
