import secrets
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.auth.models import hash_key
from any_llm.gateway.core.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.core.database import get_db
from any_llm.gateway.metrics import record_auth_failure
from any_llm.gateway.models.entities import APIKey

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


def reset_config() -> None:
    """Reset config state. Intended for testing only."""
    global _config  # noqa: PLW0603
    _config = None


def _extract_bearer_token(request: Request, config: GatewayConfig) -> str:
    """Extract and validate Bearer token from request header.

    Checks X-AnyLLM-Key first, then falls back to standard Authorization header
    for OpenAI client compatibility, then falls back to x-api-key header
    for Anthropic client compatibility.
    """
    auth_header = request.headers.get(API_KEY_HEADER) or request.headers.get("Authorization")

    if auth_header:
        if not auth_header.startswith("Bearer "):
            record_auth_failure("invalid_format")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid header format. Expected 'Bearer <token>'",
            )
        return auth_header[7:]

    # Fallback: x-api-key header (Anthropic client compatibility, no Bearer prefix)
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key

    record_auth_failure("missing_credentials")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"Missing {API_KEY_HEADER} or Authorization header",
    )


async def _verify_and_update_api_key(db: AsyncSession, token: str) -> APIKey:
    """Verify API key token and update last_used_at."""
    try:
        key_hash = hash_key(token)
    except ValueError as e:
        record_auth_failure("invalid_format")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API key format: {e}",
        ) from e

    api_key = (await db.execute(select(APIKey).where(APIKey.key_hash == key_hash))).scalar_one_or_none()

    if not api_key:
        record_auth_failure("invalid_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not api_key.is_active:
        record_auth_failure("inactive_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is inactive",
        )

    if api_key.expires_at and api_key.expires_at < datetime.now(UTC):
        record_auth_failure("expired_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    api_key.last_used_at = datetime.now(UTC)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()

    return api_key


def _is_valid_master_key(token: str, config: GatewayConfig) -> bool:
    """Check if token matches the master key."""
    return config.master_key is not None and secrets.compare_digest(token, config.master_key)


async def verify_api_key(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
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
    return await _verify_and_update_api_key(db, token)


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
    db: Annotated[AsyncSession, Depends(get_db)],
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

    api_key = await _verify_and_update_api_key(db, token)
    return api_key, False


__all__ = [
    "get_config",
    "get_db",
    "reset_config",
    "set_config",
    "verify_api_key",
    "verify_api_key_or_master_key",
    "verify_master_key",
]
