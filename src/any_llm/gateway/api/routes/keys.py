import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.api.deps import get_db, verify_master_key
from any_llm.gateway.auth.models import generate_api_key, hash_key
from any_llm.gateway.models.entities import APIKey, User

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

    @classmethod
    def from_model(cls, key: APIKey) -> "KeyInfo":
        return cls(
            id=str(key.id),
            key_name=str(key.key_name) if key.key_name else None,
            user_id=str(key.user_id) if key.user_id else None,
            created_at=key.created_at.isoformat(),
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            is_active=bool(key.is_active),
            metadata=dict(key.metadata_) if key.metadata_ else {},
        )


class UpdateKeyRequest(BaseModel):
    """Request model for updating a key."""

    key_name: str | None = None
    is_active: bool | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_key(
    request: CreateKeyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
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
        user = (await db.execute(select(User).where(User.user_id == request.user_id))).scalar_one_or_none()
        if not user:
            user = User(
                user_id=request.user_id,
                alias=f"User {request.user_id}",
            )
            db.add(user)
        elif user.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{request.user_id}' has been deleted. Recreate via POST /v1/users first.",
            )
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
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(db_key)

    key_info = KeyInfo.from_model(db_key)
    return CreateKeyResponse(
        **key_info.model_dump(exclude={"last_used_at"}),
        key=api_key,
    )


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_keys(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[KeyInfo]:
    """List all API keys.

    Requires master key authentication.
    """
    keys = (await db.execute(select(APIKey).offset(skip).limit(limit))).scalars().all()

    return [KeyInfo.from_model(key) for key in keys]


@router.get("/{key_id}", dependencies=[Depends(verify_master_key)])
async def get_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeyInfo:
    """Get details of a specific API key.

    Requires master key authentication.
    """
    key = (await db.execute(select(APIKey).where(APIKey.id == key_id))).scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    return KeyInfo.from_model(key)


@router.patch("/{key_id}", dependencies=[Depends(verify_master_key)])
async def update_key(
    key_id: str,
    request: UpdateKeyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeyInfo:
    """Update an API key.

    Requires master key authentication.
    """
    key = (await db.execute(select(APIKey).where(APIKey.id == key_id))).scalar_one_or_none()

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

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(key)

    return KeyInfo.from_model(key)


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete (revoke) an API key.

    Requires master key authentication.
    """
    key = (await db.execute(select(APIKey).where(APIKey.id == key_id))).scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    await db.delete(key)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
