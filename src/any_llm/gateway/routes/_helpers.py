from __future__ import annotations

from fastapi import HTTPException

from any_llm.gateway.db import APIKey


def resolve_user_id(
    user_id_from_request: str | None,
    api_key: APIKey | None,
    is_master_key: bool,
    *,
    master_key_error: HTTPException,
    no_api_key_error: HTTPException,
    no_user_error: HTTPException,
) -> str:
    """Resolve the effective user_id from request context.

    The resolution order is:
    1. If master key is used, the request *must* supply a user_id.
    2. If the request supplies a user_id, use it.
    3. Fall back to the user_id associated with the API key.

    Args:
        user_id_from_request: User identifier extracted from the request body
        api_key: Authenticated API key object (None when using master key)
        is_master_key: Whether the request was authenticated with a master key
        master_key_error: Raised when master key is used but no user_id is provided
        no_api_key_error: Raised when no API key is available
        no_user_error: Raised when the API key has no associated user

    Returns:
        Resolved user_id string

    """
    if is_master_key:
        if not user_id_from_request:
            raise master_key_error
        return user_id_from_request

    if user_id_from_request:
        return user_id_from_request

    if api_key is None:
        raise no_api_key_error
    if not api_key.user_id:
        raise no_user_error
    return str(api_key.user_id)
