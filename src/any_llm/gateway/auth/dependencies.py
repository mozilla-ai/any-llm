import secrets
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from any_llm.gateway.auth.models import hash_key
from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.db import APIKey, get_db

_config: GatewayConfig | None = None


def set_config(config: GatewayConfig) -> None:
    """Set the global config instance."""
    global _config  # noqa: PLW0603
    _config = config


def get_config() -> GatewayConfig:
    """Get the global config instance."""
    if _config is None:
        msg = "Config not initialized"
        raise RuntimeError(msg)
    return _config


def _extract_bearer_token(request: Request, config: GatewayConfig) -> str:
    """Extract and validate Bearer token from request header.

    Checks X-AnyLLM-Key first, then falls back to standard Authorization header
    for OpenAI client compatibility.
    """
    auth_header = request.headers.get(API_KEY_HEADER) or request.headers.get("Authorization")

    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_HEADER} or Authorization header",
        )

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid header format. Expected 'Bearer <token>'",
        )

    return auth_header[7:]


def _verify_and_update_api_key(db: Session, token: str) -> APIKey:
    """Verify API key token and update last_used_at."""
    try:
        key_hash = hash_key(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API key format: {e}",
        ) from e

    api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is inactive",
        )

    if api_key.expires_at and api_key.expires_at < datetime.now(UTC).replace(tzinfo=None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    api_key.last_used_at = datetime.now(UTC).replace(tzinfo=None)
    db.commit()

    return api_key


def _is_valid_master_key(token: str, config: GatewayConfig) -> bool:
    """Check if token matches the master key."""
    return config.master_key is not None and secrets.compare_digest(token, config.master_key)


async def verify_api_key(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> APIKey:
    """Verify API key from X-AnyLLM-Key header.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        APIKey object if valid

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = _extract_bearer_token(request, config)
    return _verify_and_update_api_key(db, token)


async def verify_master_key(
    request: Request,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Verify master key from X-AnyLLM-Key header.

    Args:
        request: FastAPI request object
        config: Gateway configuration

    Raises:
        HTTPException: If master key is not configured or invalid

    """
    if not config.master_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Master key not configured. Set GATEWAY_MASTER_KEY environment variable.",
        )

    token = _extract_bearer_token(request, config)

    if not _is_valid_master_key(token, config):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid master key",
        )


async def verify_api_key_or_master_key(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> tuple[APIKey | None, bool]:
    """Verify either API key or master key from X-AnyLLM-Key header.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        Tuple of (APIKey object or None, is_master_key boolean)

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = _extract_bearer_token(request, config)

    if _is_valid_master_key(token, config):
        return None, True

    api_key = _verify_and_update_api_key(db, token)
    return api_key, False
