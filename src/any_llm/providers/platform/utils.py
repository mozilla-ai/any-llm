from __future__ import annotations

import base64
import hashlib
import os
import re
import uuid
from typing import TYPE_CHECKING, Any

import nacl.bindings
import nacl.public
import requests

from any_llm.logging import logger

if TYPE_CHECKING:
    import httpx

    from any_llm.any_llm import AnyLLM
    from any_llm.types.completion import ChatCompletion


API_V1_STR = "/api/v1"
ANY_LLM_PLATFORM_URL = os.getenv("ANY_LLM_PLATFORM_URL", "https://platform-api.any-llm.ai")
ANY_LLM_PLATFORM_API_URL = f"{ANY_LLM_PLATFORM_URL}{API_V1_STR}"


def _parse_any_llm_key(any_api_key: str) -> tuple[str, str, str]:
    """Parse `ANY_LLM_KEY` format and extract components.

    The `ANY_LLM_KEY` adheres to the following format:
    ANY.<version>.<kid>.<fingerprint>-<base64_32byte_private_key>
    """
    match = re.match(r"^ANY\.v\d+\.([^.]+)\.([^-]+)-(.+)$", any_api_key)

    if not match:
        msg = "Invalid ANY_API_KEY format. Expected: ANY.v1.<kid>.<fingerprint>-<base64_key>"
        raise ValueError(msg)

    kid, fingerprint, base64_private_key = match.groups()
    return kid, fingerprint, base64_private_key


def _load_private_key(private_key_base64: str) -> nacl.public.PrivateKey:
    """Load X25519 private key from base64 string."""
    private_key_bytes = base64.b64decode(private_key_base64)
    if len(private_key_bytes) != 32:
        msg = f"X25519 private key must be 32 bytes, got {len(private_key_bytes)}"
        raise ValueError(msg)
    return nacl.public.PrivateKey(private_key_bytes)


def _extract_public_key(private_key: nacl.public.PrivateKey) -> str:
    """Extract public key as base64 from X25519 private key."""
    public_key_bytes = bytes(private_key.public_key)
    return base64.b64encode(public_key_bytes).decode("utf-8")


def _create_challenge(public_key: str, any_api_url: str) -> Any:
    """Create an authentication challenge."""
    logger.info("Creating authentication challenge...")

    response = requests.post(
        f"{any_api_url}/auth/",
        json={
            "encryption_key": public_key,
            "key_type": "RSA",  # Backend auto-detects X25519, this is just for tracking
        },
    )

    if response.status_code != 200:
        logger.error(f"Error creating challenge: {response.status_code}")
        logger.error(response.json())
        raise RuntimeError(response.text)

    data = response.json()
    logger.info("Challenge created")

    return data


def _decrypt_data(encrypted_data_base64: str, private_key: nacl.public.PrivateKey) -> str:
    """Decrypt data using X25519 sealed box with XChaCha20-Poly1305."""
    encrypted_data = base64.b64decode(encrypted_data_base64)

    # Extract ephemeral public key (first 32 bytes) and ciphertext
    if len(encrypted_data) < 32:
        msg = "Invalid sealed box format: too short"
        raise ValueError(msg)

    ephemeral_public_key = encrypted_data[:32]
    ciphertext = encrypted_data[32:]

    # Get recipient's public key from private key
    recipient_public_key = bytes(private_key.public_key)

    # Compute shared secret using X25519 ECDH
    shared_secret = nacl.bindings.crypto_scalarmult(bytes(private_key), ephemeral_public_key)

    # Derive nonce from hash(ephemeral_pubkey || recipient_pubkey)
    # This matches the sealed box construction in the frontend/backend
    combined = ephemeral_public_key + recipient_public_key
    nonce_hash = hashlib.sha512(combined).digest()[:24]  # Take first 24 bytes of SHA-512

    # Decrypt with XChaCha20-Poly1305 AEAD
    decrypted_data = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
        ciphertext, None, nonce_hash, shared_secret
    )

    return decrypted_data.decode("utf-8")


def _solve_challenge(encrypted_challenge: str, private_key: nacl.public.PrivateKey) -> uuid.UUID:
    """Decrypt the challenge to get the UUID."""
    logger.info("Decrypting challenge...")

    decrypted_uuid_str = _decrypt_data(encrypted_challenge, private_key)
    solved_challenge = uuid.UUID(decrypted_uuid_str)

    logger.info(f"Challenge solved: {solved_challenge}")

    return solved_challenge


def _fetch_provider_key(
    provider: str,
    public_key: str,
    solved_challenge: uuid.UUID,
    any_api_url: str,
) -> Any:
    """Fetch the provider key using the solved challenge."""
    logger.info(f"Fetching provider key for {provider}...")

    response = requests.get(
        f"{any_api_url}/provider-keys/{provider}",
        headers={"encryption-key": public_key, "AnyLLM-Challenge-Response": str(solved_challenge)},
    )

    if response.status_code != 200:
        logger.error(f"Error fetching provider key: {response.status_code}")
        logger.error(response.json())
        raise RuntimeError(response.text)

    data = response.json()
    logger.info("Provider key fetched")

    return data


def _decrypt_provider_key(encrypted_key: str, private_key: nacl.public.PrivateKey) -> str:
    """Decrypt the provider API key."""
    logger.info("Decrypting provider API key...")

    decrypted_key = _decrypt_data(encrypted_key, private_key)
    logger.info("Decrypted successfully!")

    return decrypted_key


def get_provider_key(any_llm_key: str, provider: type[AnyLLM]) -> str:
    """Get the provider key from Any LLM Platform.

    The client first has to prove the ownership of the private key that decrypts the encrypted provider key,
    without sharing it. To this end, the user asks Any LLM platform to create a new challenge, based on the public key,
    and then the client attempts to solve it.

    If the client solves the challenge correctly, Any LLM platform handles the encrypted provider key,
    which can be decrypted using the private key (`ANY_LLM_KEY`).

    Args:
        any_llm_key: The Any LLM platform key, tied to a specific project.
        provider: The name of the LLM provider.

    Returns:
        The provider key.
    """
    # Parse the `ANY_LLM_KEY`, load the private key and extract the public key from it.
    _, _, private_key_base64 = _parse_any_llm_key(any_llm_key)
    private_key = _load_private_key(private_key_base64)
    public_key = _extract_public_key(private_key)

    # Create and solve a new challenge to prove ownership of the private key without sharing it.
    challenge_data = _create_challenge(public_key, ANY_LLM_PLATFORM_API_URL)
    solved_challenge = _solve_challenge(challenge_data["encrypted_challenge"], private_key)

    # Fetch and decrypt the provider key
    provider_key_data = _fetch_provider_key(
        provider=provider.PROVIDER_NAME,
        public_key=public_key,
        solved_challenge=solved_challenge,
        any_api_url=ANY_LLM_PLATFORM_API_URL,
    )
    return _decrypt_provider_key(provider_key_data["encrypted_key"], private_key)


async def post_completion_usage_event(
    client: httpx.AsyncClient, any_llm_key: str, provider: str, completion: ChatCompletion
) -> None:
    """Posts completion usage events.

    The client has to create and solve two challenges, one to prove ownership of the private key
    that decrypts the encrypted provider key, without sharing it, and one to prove ownership of the project
    it wants to post data to.

    Args:
        client: An httpx client to perform post request.
        any_llm_key: The Any LLM platform key, tied to a specific project.
        provider: The name of the LLM provider.
        completion: The LLM response.
    """
    # Parse the `ANY_LLM_KEY`, load the private key and extract the public key from it.
    _, _, private_key_base64 = _parse_any_llm_key(any_llm_key)
    private_key = _load_private_key(private_key_base64)
    public_key = _extract_public_key(private_key)

    # Create and solve a new challenge to prove ownership of the private key without sharing it.
    challenge_data = _create_challenge(public_key, ANY_LLM_PLATFORM_API_URL)
    solved_challenge = _solve_challenge(challenge_data["encrypted_challenge"], private_key)

    # Fetch the provider key info
    provider_key_data = _fetch_provider_key(
        provider=provider,
        public_key=public_key,
        solved_challenge=solved_challenge,
        any_api_url=ANY_LLM_PLATFORM_API_URL,
    )
    provider_key_id = provider_key_data.get("id")

    # Create and solve a new challenge to prove ownership of the project
    challenge_data = _create_challenge(public_key, ANY_LLM_PLATFORM_API_URL)
    solved_challenge = _solve_challenge(challenge_data["encrypted_challenge"], private_key)

    # Send usage event data
    input_tokens = completion.usage.prompt_tokens  # type: ignore[union-attr]
    output_tokens = completion.usage.completion_tokens  # type: ignore[union-attr]
    model = completion.model

    event_id = str(uuid.uuid4())

    payload = {
        "provider_key_id": provider_key_id,
        "provider": provider,
        "model": model,
        "data": {
            "input_tokens": str(input_tokens),
            "output_tokens": str(output_tokens),
        },
        "id": event_id,
    }

    response = await client.post(
        f"{ANY_LLM_PLATFORM_API_URL}/usage-events/",
        json=payload,
        headers={"encryption-key": public_key, "AnyLLM-Challenge-Response": str(solved_challenge)},
    )
    response.raise_for_status()
