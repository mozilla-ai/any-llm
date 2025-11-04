import hashlib
import re
import secrets


def generate_api_key() -> str:
    """Generate a new API key with prefix.

    Returns:
        A new API key with format 'gw-' followed by 48 random characters

    Raises:
        RuntimeError: If generated key doesn't match expected format (should never happen)

    """
    api_key = f"gw-{secrets.token_urlsafe(48)}"

    try:
        validate_api_key_format(api_key)
    except ValueError as e:
        msg = f"Generated API key failed validation: {e}"
        raise RuntimeError(msg) from e

    return api_key


def validate_api_key_format(api_key: str) -> None:
    """Validate API key format.

    Args:
        api_key: The API key to validate

    Raises:
        ValueError: If the API key format is invalid

    """
    if not isinstance(api_key, str):
        msg = f"API key must be a string, got {type(api_key).__name__}"
        raise ValueError(msg)

    if not api_key.startswith("gw-"):
        msg = "API key must start with 'gw-' prefix"
        raise ValueError(msg)

    if len(api_key) < 50:
        msg = f"API key is too short. Expected at least 50 characters, got {len(api_key)}"
        raise ValueError(msg)

    if not re.match(r"^gw-[A-Za-z0-9_-]+$", api_key):
        msg = "API key contains invalid characters. Must match pattern: gw-[A-Za-z0-9_-]+"
        raise ValueError(msg)


def hash_key(api_key: str) -> str:
    """Hash an API key using SHA-256.

    Args:
        api_key: The API key to hash

    Returns:
        Hexadecimal string of the SHA-256 hash

    Raises:
        ValueError: If the API key format is invalid

    """
    validate_api_key_format(api_key)
    return hashlib.sha256(api_key.encode()).hexdigest()
