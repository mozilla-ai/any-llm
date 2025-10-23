from __future__ import annotations

import base64
import hashlib
import re
import uuid
from typing import TYPE_CHECKING

import nacl.bindings
import nacl.public
import requests

from any_llm.logging import logger

if TYPE_CHECKING:
    from any_llm.any_llm import AnyLLM


def parse_any_api_key(any_api_key: str) -> tuple:
    """Parse ANY_API_KEY format and extract components.

    Format: ANY.v1.<kid>.<fingerprint>-<base64_32byte_private_key>

    Returns:
        tuple: (kid, fingerprint, base64_private_key)
    """
    match = re.match(r"^ANY\.v\d+\.([^.]+)\.([^-]+)-(.+)$", any_api_key)

    if not match:
        msg = "Invalid ANY_API_KEY format. Expected: ANY.v1.<kid>.<fingerprint>-<base64_key>"
        raise ValueError(msg)

    kid, fingerprint, base64_private_key = match.groups()
    return kid, fingerprint, base64_private_key


def load_private_key(private_key_base64: str):
    """Load X25519 private key from base64 string."""
    private_key_bytes = base64.b64decode(private_key_base64)
    if len(private_key_bytes) != 32:
        msg = f"X25519 private key must be 32 bytes, got {len(private_key_bytes)}"
        raise ValueError(msg)
    return nacl.public.PrivateKey(private_key_bytes)


def extract_public_key(private_key) -> str:
    """Extract public key as base64 from X25519 private key."""
    public_key_bytes = bytes(private_key.public_key)
    return base64.b64encode(public_key_bytes).decode("utf-8")


def create_challenge(public_key: str, any_api_url: str) -> dict:
    """Step 1: Create an authentication challenge."""
    logger.info("Creating authentication challenge...")

    response = requests.post(
        f"{any_api_url}/auth/",
        json={
            "encryption_key": public_key,
            "key_type": "RSA"  # Backend auto-detects X25519, this is just for tracking
        }
    )

    if response.status_code != 200:
        logger.error(f"Error creating challenge: {response.status_code}")
        logger.error(response.json())
        raise RuntimeError(response.text)

    data = response.json()
    logger.info("Challenge created")

    return data


def decrypt_data(encrypted_data_base64: str, private_key) -> str:
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
    shared_secret = nacl.bindings.crypto_scalarmult(
        bytes(private_key), ephemeral_public_key
    )

    # Derive nonce from hash(ephemeral_pubkey || recipient_pubkey)
    # This matches the sealed box construction in the frontend/backend
    combined = ephemeral_public_key + recipient_public_key
    nonce_hash = hashlib.sha512(combined).digest()[:24]  # Take first 24 bytes of SHA-512

    # Decrypt with XChaCha20-Poly1305 AEAD
    decrypted_data = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
        ciphertext, None, nonce_hash, shared_secret
    )

    return decrypted_data.decode("utf-8")


def solve_challenge(encrypted_challenge: str, private_key) -> uuid.UUID:
    """Step 2: Decrypt the challenge to get the UUID."""
    logger.info("Decrypting challenge...")

    decrypted_uuid_str = decrypt_data(encrypted_challenge, private_key)
    solved_challenge = uuid.UUID(decrypted_uuid_str)

    logger.info(f"Challenge solved: {solved_challenge}")

    return solved_challenge


def fetch_provider_key(
    project_id: str,
    provider: str,
    public_key: str,
    solved_challenge: uuid.UUID,
    any_api_url: str,
) -> dict:
    """Fetch the provider key using the solved challenge."""
    logger.info(f"Fetching provider key for {provider}...")

    response = requests.get(
        f"{any_api_url}/provider-keys/{project_id}/{provider}",
        headers={
            "encryption-key": public_key,
            "X-Solved-Challenge": str(solved_challenge)
        }
    )

    if response.status_code != 200:
        logger.error(f"Error fetching provider key: {response.status_code}")
        logger.error(response.json())
        raise RuntimeError(response.text)

    data = response.json()
    logger.info("Provider key fetched")

    return data


def decrypt_provider_key(encrypted_key: str, private_key) -> str:
    """Decrypt the provider API key."""
    logger.info("Decrypting provider API key...")

    decrypted_key = decrypt_data(encrypted_key, private_key)
    logger.info("Decrypted successfully!")

    return decrypted_key


def get_provider_key(any_api_key: str, project_id: str, provider: type[AnyLLM], any_api_url: str) -> str:
    """Get the provider key."""
    _,_, private_key_base64 = parse_any_api_key(any_api_key)
    private_key = load_private_key(private_key_base64)
    public_key = extract_public_key(private_key)
    challenge_data = create_challenge(public_key, any_api_url)
    solved_challenge = solve_challenge(challenge_data["encrypted_challenge"], private_key)
    provider_key_data = fetch_provider_key(
            project_id=project_id,
            provider=provider.PROVIDER_NAME,
            public_key=public_key,
            solved_challenge=solved_challenge,
            any_api_url=any_api_url
        )
    return decrypt_provider_key(
            provider_key_data["encrypted_key"],
            private_key
        )
