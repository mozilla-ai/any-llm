"""Google Cloud Confidential Space attestation verification for TrustedRouter."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

DEFAULT_TRUST_RELEASE_URL = "https://trust.trustedrouter.com/gcp-release.json"
GCP_ISSUER = "https://confidentialcomputing.googleapis.com"
GCP_JWKS_URI = (
    "https://www.googleapis.com/service_accounts/v1/metadata/jwk/signer@confidentialspace-sign.iam.gserviceaccount.com"
)
DEFAULT_ATTESTATION_AUDIENCE = "quill-cloud"
SUPPORTED_HWMODELS = frozenset({"GCP_AMD_SEV", "GCP_AMD_SEV_ES", "GCP_INTEL_TDX"})
_CLOCK_SKEW_SECONDS = 60

AttestationBinding = Literal["none", "tls-cert"]


class AttestationVerificationError(ValueError):
    """Raised when TrustedRouter gateway attestation verification fails."""


@dataclass(frozen=True, slots=True)
class AttestationPolicy:
    """Pinned identity claims required on a TrustedRouter Confidential Space JWT."""

    gcp_audience: str = DEFAULT_ATTESTATION_AUDIENCE
    expected_image_digests: tuple[str, ...] = ()
    expected_image_references: tuple[str, ...] = ()
    expected_cert_sha256: str | None = None

    @property
    def pins_image_identity(self) -> bool:
        """Return whether the policy pins at least one image digest or reference."""
        return bool(self.expected_image_digests or self.expected_image_references)


@dataclass(frozen=True, slots=True)
class GatewayAttestation:
    """Verified Confidential Space attestation result for a TrustedRouter gateway."""

    image_digest: str
    image_reference: str
    nonce: str
    expires_at: int
    issuer: str
    audience: str
    hwmodel: str
    cert_sha256: str | None
    binding: AttestationBinding
    raw_claims: Mapping[str, Any]


def _read_optional_string(release: Mapping[str, Any], field: str) -> str | None:
    value = release.get(field)
    if value is not None and not isinstance(value, str):
        msg = f"trust release {field} must be a string or null"
        raise AttestationVerificationError(msg)
    return value or None


def _read_string_pins(release: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = release.get(field, [])
    if not isinstance(value, list) or any(not isinstance(pin, str) or not pin for pin in value):
        msg = f"trust release {field} must be an array of non-empty strings"
        raise AttestationVerificationError(msg)
    return tuple(value)


def policy_from_trust_release(
    release: Mapping[str, Any],
    *,
    audience: str | None = None,
    cert_sha256: str | None = None,
) -> AttestationPolicy:
    """Build an ``AttestationPolicy`` from a TrustedRouter ``gcp-release.json`` payload."""
    if not isinstance(release, Mapping):
        msg = "trust release must be a JSON object"
        raise AttestationVerificationError(msg)

    accepted_digests = _read_string_pins(release, "accepted_image_digests")
    image_digest = _read_optional_string(release, "image_digest")
    if not accepted_digests and image_digest:
        accepted_digests = (image_digest,)

    accepted_references = _read_string_pins(release, "accepted_image_references")
    image_reference = _read_optional_string(release, "image_reference")
    if not accepted_references and image_reference:
        accepted_references = (image_reference,)

    resolved_audience = (
        audience or _read_optional_string(release, "attestation_audience") or DEFAULT_ATTESTATION_AUDIENCE
    )
    policy = AttestationPolicy(
        gcp_audience=resolved_audience,
        expected_image_digests=accepted_digests,
        expected_image_references=accepted_references,
        expected_cert_sha256=cert_sha256,
    )
    if not policy.pins_image_identity:
        msg = (
            "trust release pins no image identity (none of accepted_image_digests, image_digest, "
            "accepted_image_references, image_reference)"
        )
        raise AttestationVerificationError(msg)
    return policy


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def _jwt_split(token: bytes | str) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    text = token.decode("ascii", errors="replace").strip() if isinstance(token, bytes) else token.strip()
    parts = text.split(".")
    if len(parts) != 3:
        msg = f"expected 3 JWT segments, got {len(parts)}"
        raise AttestationVerificationError(msg)
    header_b64, payload_b64, sig_b64 = parts
    try:
        header: object = json.loads(_b64url_decode(header_b64))
        payload: object = json.loads(_b64url_decode(payload_b64))
        signature = _b64url_decode(sig_b64)
    except (ValueError, UnicodeError) as exc:
        msg = f"invalid JWT encoding: {exc}"
        raise AttestationVerificationError(msg) from exc
    if not isinstance(header, dict) or not isinstance(payload, dict):
        msg = "JWT header and claims must be JSON objects"
        raise AttestationVerificationError(msg)
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    return header, payload, signing_input, signature


def _verify_rs256(
    jwks: Mapping[str, Any],
    header: Mapping[str, Any],
    signing_input: bytes,
    signature: bytes,
) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError as exc:
        msg = (
            "trustedrouter attestation verification requires the `cryptography` package. "
            "Please install it with `pip install any-llm-sdk[trustedrouter]`."
        )
        raise ImportError(msg) from exc

    if header.get("alg") != "RS256":
        msg = f"unsupported JWT alg {header.get('alg')!r}; expected RS256"
        raise AttestationVerificationError(msg)
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        msg = "JWT kid must be a non-empty string"
        raise AttestationVerificationError(msg)
    if not isinstance(jwks, Mapping):
        msg = "JWKS must be a JSON object"
        raise AttestationVerificationError(msg)
    keys = jwks.get("keys")
    if not isinstance(keys, list) or any(not isinstance(key, Mapping) for key in keys):
        msg = "JWKS keys must be a list of JWK objects"
        raise AttestationVerificationError(msg)

    matching = next((key for key in keys if key.get("kid") == kid), None)
    if matching is None:
        msg = f"no JWK with kid={kid!r} in JWKS"
        raise AttestationVerificationError(msg)
    if matching.get("kty") != "RSA":
        msg = "expected RSA key in JWKS"
        raise AttestationVerificationError(msg)

    modulus = matching.get("n")
    exponent = matching.get("e")
    if not isinstance(modulus, str) or not isinstance(exponent, str):
        msg = "JWK n and e must be strings"
        raise AttestationVerificationError(msg)
    try:
        n_int = int.from_bytes(_b64url_decode(modulus), "big")
        e_int = int.from_bytes(_b64url_decode(exponent), "big")
        public_key = rsa.RSAPublicNumbers(e=e_int, n=n_int).public_key()
    except (ValueError, UnicodeError) as exc:
        msg = f"malformed RSA JWK: {exc}"
        raise AttestationVerificationError(msg) from exc

    try:
        public_key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature as exc:
        msg = "JWT RS256 signature verification failed"
        raise AttestationVerificationError(msg) from exc


def _safe_eq(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode(), right.encode())


def _normalize_string_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    msg = f"JWT {field} must be a string or list of strings"
    raise AttestationVerificationError(msg)


def _committed_cert_hashes(claims: Mapping[str, Any], eat_nonces: list[str]) -> list[str]:
    values: list[str] = []
    for field in ("tls_cert_sha256", "workload_tls_cert_sha256"):
        value = claims.get(field)
        if isinstance(value, str):
            values.append(value.lower())
    values.extend(nonce.lower() for nonce in eat_nonces)
    return values


def verify_attestation_jwt(
    token: bytes | str,
    *,
    policy: AttestationPolicy,
    nonce_hex: str,
    jwks: Mapping[str, Any],
    tls_cert_der: bytes | None = None,
    now: int | None = None,
) -> GatewayAttestation:
    """Verify a TrustedRouter Confidential Space attestation JWT and its claims."""
    if not policy.pins_image_identity:
        msg = "attestation policy must pin at least one image digest or reference"
        raise AttestationVerificationError(msg)
    if not isinstance(nonce_hex, str) or not nonce_hex:
        msg = "attestation verification requires a non-empty challenge nonce"
        raise AttestationVerificationError(msg)

    header, claims, signing_input, signature = _jwt_split(token)
    _verify_rs256(jwks, header, signing_input, signature)

    current_time = int(time.time()) if now is None else now
    exp = claims.get("exp")
    if not isinstance(exp, int) or isinstance(exp, bool):
        msg = "JWT is missing a valid integer exp claim"
        raise AttestationVerificationError(msg)
    if exp <= current_time:
        msg = f"JWT expired at {exp} (now={current_time})"
        raise AttestationVerificationError(msg)

    nbf = claims.get("nbf")
    if nbf is not None:
        if not isinstance(nbf, int) or isinstance(nbf, bool):
            msg = "JWT nbf must be an integer"
            raise AttestationVerificationError(msg)
        if nbf > current_time + _CLOCK_SKEW_SECONDS:
            msg = f"JWT is not valid until {nbf} (now={current_time})"
            raise AttestationVerificationError(msg)

    issuer = claims.get("iss")
    if issuer != GCP_ISSUER:
        msg = f"unexpected JWT issuer {issuer!r}; expected {GCP_ISSUER!r}"
        raise AttestationVerificationError(msg)

    aud_list = _normalize_string_list(claims.get("aud"), "aud")
    if policy.gcp_audience not in aud_list:
        msg = f"audience {policy.gcp_audience!r} not in JWT aud {aud_list!r}"
        raise AttestationVerificationError(msg)

    if claims.get("dbgstat") != "disabled-since-boot":
        msg = "attested Confidential Space workload must report dbgstat='disabled-since-boot'"
        raise AttestationVerificationError(msg)
    if claims.get("swname") != "CONFIDENTIAL_SPACE":
        msg = "attested workload is not running CONFIDENTIAL_SPACE"
        raise AttestationVerificationError(msg)
    if claims.get("secboot") is not True:
        msg = "attested workload does not report Secure Boot enabled"
        raise AttestationVerificationError(msg)
    hwmodel = claims.get("hwmodel")
    if not isinstance(hwmodel, str) or hwmodel not in SUPPORTED_HWMODELS:
        msg = f"unsupported confidential hardware model {hwmodel!r}"
        raise AttestationVerificationError(msg)

    submods = claims.get("submods")
    container = submods.get("container") if isinstance(submods, Mapping) else None
    if not isinstance(container, Mapping):
        msg = "JWT is missing submods.container claims"
        raise AttestationVerificationError(msg)
    image_digest = container.get("image_digest", "")
    image_reference = container.get("image_reference", "")
    if not isinstance(image_digest, str) or not isinstance(image_reference, str):
        msg = "JWT container image_digest and image_reference must be strings"
        raise AttestationVerificationError(msg)

    if policy.expected_image_digests and not any(
        _safe_eq(image_digest, expected) for expected in policy.expected_image_digests
    ):
        msg = f"image_digest mismatch: workload={image_digest!r}, policy={policy.expected_image_digests!r}"
        raise AttestationVerificationError(msg)
    if policy.expected_image_references and not any(
        _safe_eq(image_reference, expected) for expected in policy.expected_image_references
    ):
        msg = f"image_reference mismatch: workload={image_reference!r}, policy={policy.expected_image_references!r}"
        raise AttestationVerificationError(msg)

    eat_nonces = _normalize_string_list(claims.get("eat_nonce"), "eat_nonce")
    nonce_lower = nonce_hex.lower()
    if not any(_safe_eq(entry.lower(), nonce_lower) for entry in eat_nonces):
        msg = f"challenge nonce {nonce_hex!r} not present in JWT eat_nonce"
        raise AttestationVerificationError(msg)

    committed_hashes = _committed_cert_hashes(claims, eat_nonces)
    binding: AttestationBinding = "none"
    cert_sha256: str | None = None
    if tls_cert_der is not None:
        actual_sha = hashlib.sha256(tls_cert_der).hexdigest()
        if not any(_safe_eq(candidate, actual_sha) for candidate in committed_hashes):
            msg = f"TLS leaf certificate SHA-256 {actual_sha!r} is not committed in the attestation JWT"
            raise AttestationVerificationError(msg)
        if policy.expected_cert_sha256 and not _safe_eq(actual_sha, policy.expected_cert_sha256.lower()):
            msg = "TLS leaf certificate SHA-256 does not match policy pin"
            raise AttestationVerificationError(msg)
        cert_sha256 = actual_sha
        binding = "tls-cert"
    elif policy.expected_cert_sha256:
        expected_cert = policy.expected_cert_sha256.lower()
        if not any(_safe_eq(candidate, expected_cert) for candidate in committed_hashes):
            msg = "JWT does not commit to policy expected_cert_sha256"
            raise AttestationVerificationError(msg)
        cert_sha256 = expected_cert

    return GatewayAttestation(
        image_digest=image_digest,
        image_reference=image_reference,
        nonce=nonce_hex,
        expires_at=exp,
        issuer=GCP_ISSUER,
        audience=policy.gcp_audience,
        hwmodel=hwmodel,
        cert_sha256=cert_sha256,
        binding=binding,
        raw_claims=dict(claims),
    )


def _attestation_url(api_base: str, nonce_hex: str) -> str:
    parsed = urlsplit(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        msg = f"invalid TrustedRouter api_base URL for attestation: {api_base!r}"
        raise AttestationVerificationError(msg)
    return f"{parsed.scheme}://{parsed.netloc}/attestation?nonce={nonce_hex}"


def _extract_peer_cert_der(response: httpx.Response) -> bytes | None:
    stream = response.extensions.get("network_stream")
    get_extra_info = getattr(stream, "get_extra_info", None)
    if not callable(get_extra_info):
        return None
    ssl_obj = get_extra_info("ssl_object")
    getpeercert = getattr(ssl_obj, "getpeercert", None)
    if not callable(getpeercert):
        return None
    peer_cert = getpeercert(True)
    return peer_cert if isinstance(peer_cert, bytes) else None


def _fetch_json_object(client: httpx.Client, url: str, description: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    try:
        payload: object = response.json()
    except ValueError as exc:
        msg = f"{description} returned malformed JSON"
        raise AttestationVerificationError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{description} must return a JSON object"
        raise AttestationVerificationError(msg)
    return payload


def verify_gateway_attestation(
    *,
    api_base: str,
    trust_release_url: str = DEFAULT_TRUST_RELEASE_URL,
    jwks_url: str = GCP_JWKS_URI,
    timeout: float = 10.0,
    policy: AttestationPolicy | None = None,
    jwks: Mapping[str, Any] | None = None,
    http_client: httpx.Client | None = None,
) -> GatewayAttestation:
    """Fetch and verify TrustedRouter's live Google Confidential Space attestation."""
    nonce_hex = secrets.token_hex(32)
    attestation_endpoint = _attestation_url(api_base, nonce_hex)

    def _run(client: httpx.Client) -> GatewayAttestation:
        resolved_policy = (
            policy
            if policy is not None
            else policy_from_trust_release(_fetch_json_object(client, trust_release_url, "trust release"))
        )
        resolved_jwks = (
            jwks if jwks is not None else _fetch_json_object(client, jwks_url, "Google Confidential Space JWKS")
        )
        response = client.get(attestation_endpoint, headers={"Accept": "application/jwt, */*"})
        response.raise_for_status()
        if not response.content.strip():
            msg = "gateway /attestation returned an empty response"
            raise AttestationVerificationError(msg)
        tls_cert_der = _extract_peer_cert_der(response)
        return verify_attestation_jwt(
            response.content,
            policy=resolved_policy,
            nonce_hex=nonce_hex,
            jwks=resolved_jwks,
            tls_cert_der=tls_cert_der,
        )

    if http_client is not None:
        return _run(http_client)
    with httpx.Client(timeout=timeout) as client:
        return _run(client)
