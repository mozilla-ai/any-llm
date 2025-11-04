import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import generate_api_key, hash_key, verify_master_key
from any_llm.gateway.db import APIKey, User, get_db

router = APIRouter(prefix="/v1/keys", tags=["keys"])


class CreateKeyRequest(BaseModel):
    """Request model for creating a new API key."""

    key_name: str | None = Field(default=None, description="Optional name for the key")
    user_id: str | None = Field(default=None, description="Optional user ID to associate with this key")
    expires_at: datetime | None = Field(default=None, description="Optional expiration timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class CreateKeyResponse(BaseModel):
    """Response model for creating a new API key."""

    id: str
    key: str
    key_name: str | None
    user_id: str | None
    created_at: str
    expires_at: str | None
    is_active: bool
    metadata: dict[str, Any]


class KeyInfo(BaseModel):
    """Response model for key information."""

    id: str
    key_name: str | None
    user_id: str | None
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    is_active: bool
    metadata: dict[str, Any]


class UpdateKeyRequest(BaseModel):
    """Request model for updating a key."""

    key_name: str | None = None
    is_active: bool | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_key(
    request: CreateKeyRequest,
    db: Annotated[Session, Depends(get_db)],
) -> CreateKeyResponse:
    """Create a new API key.

    Requires master key authentication.

    If user_id is provided, the key will be associated with that user (creates user if it doesn't exist).
    If user_id is not provided, a new user will be created automatically and the key will be associated with it.
    """
    api_key = generate_api_key()
    key_hash = hash_key(api_key)
    key_id = uuid.uuid4()

    if request.user_id:
        user = db.query(User).filter(User.user_id == request.user_id).first()
        if not user:
            user = User(
                user_id=request.user_id,
                alias=f"User {request.user_id}",
            )
            db.add(user)
        user_id = request.user_id
    else:
        user_id = f"apikey-{key_id}"
        user = User(
            user_id=user_id,
            alias=f"Virtual user for API key: {request.key_name or 'unnamed'}",
        )
        db.add(user)

    db_key = APIKey(
        id=str(key_id),
        key_hash=key_hash,
        key_name=request.key_name,
        user_id=user_id,
        expires_at=request.expires_at,
        metadata_=request.metadata,
    )

    db.add(db_key)
    db.commit()
    db.refresh(db_key)

    return CreateKeyResponse(
        id=str(db_key.id),
        key=api_key,
        key_name=str(db_key.key_name) if db_key.key_name else None,
        user_id=str(db_key.user_id) if db_key.user_id else None,
        created_at=db_key.created_at.isoformat(),
        expires_at=db_key.expires_at.isoformat() if db_key.expires_at else None,
        is_active=bool(db_key.is_active),
        metadata=dict(db_key.metadata_) if db_key.metadata_ else {},
    )


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_keys(
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 100,
) -> list[KeyInfo]:
    """List all API keys.

    Requires master key authentication.
    """
    keys = db.query(APIKey).offset(skip).limit(limit).all()

    return [
        KeyInfo(
            id=str(key.id),
            key_name=str(key.key_name) if key.key_name else None,
            user_id=str(key.user_id) if key.user_id else None,
            created_at=key.created_at.isoformat(),
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            is_active=bool(key.is_active),
            metadata=dict(key.metadata_) if key.metadata_ else {},
        )
        for key in keys
    ]


@router.get("/{key_id}", dependencies=[Depends(verify_master_key)])
async def get_key(
    key_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> KeyInfo:
    """Get details of a specific API key.

    Requires master key authentication.
    """
    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    return KeyInfo(
        id=str(key.id),
        key_name=str(key.key_name) if key.key_name else None,
        user_id=str(key.user_id) if key.user_id else None,
        created_at=key.created_at.isoformat(),
        last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
        expires_at=key.expires_at.isoformat() if key.expires_at else None,
        is_active=bool(key.is_active),
        metadata=dict(key.metadata_) if key.metadata_ else {},
    )


@router.patch("/{key_id}", dependencies=[Depends(verify_master_key)])
async def update_key(
    key_id: str,
    request: UpdateKeyRequest,
    db: Annotated[Session, Depends(get_db)],
) -> KeyInfo:
    """Update an API key.

    Requires master key authentication.
    """
    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    if request.key_name is not None:
        key.key_name = request.key_name
    if request.is_active is not None:
        key.is_active = request.is_active
    if request.expires_at is not None:
        key.expires_at = request.expires_at
    if request.metadata is not None:
        key.metadata_ = request.metadata

    db.commit()
    db.refresh(key)

    return KeyInfo(
        id=str(key.id),
        key_name=str(key.key_name) if key.key_name else None,
        user_id=str(key.user_id) if key.user_id else None,
        created_at=key.created_at.isoformat(),
        last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
        expires_at=key.expires_at.isoformat() if key.expires_at else None,
        is_active=bool(key.is_active),
        metadata=dict(key.metadata_) if key.metadata_ else {},
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_key(
    key_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete (revoke) an API key.

    Requires master key authentication.
    """
    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    db.delete(key)
    db.commit()
