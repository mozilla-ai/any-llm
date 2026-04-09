import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.auth import generate_api_key, hash_key
from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.log_config import logger
from any_llm.gateway.models.entities import APIKey, User


async def bootstrap_first_api_key(config: GatewayConfig, db: AsyncSession) -> None:
    """Create a first API key for new installations.

    If bootstrap is enabled and no API keys exist, this creates one API key
    and one associated virtual user, then prints the plain key once.
    """
    if not config.bootstrap_api_key:
        return

    existing_key = (await db.execute(select(APIKey.id).limit(1))).scalar_one_or_none()
    if existing_key:
        return

    api_key = generate_api_key()
    key_id = str(uuid.uuid4())
    user_id = f"apikey-{key_id}"

    user = User(
        user_id=user_id,
        alias="Virtual user for API key: bootstrap",
    )
    db.add(user)

    db_key = APIKey(
        id=key_id,
        key_hash=hash_key(api_key),
        key_name="bootstrap",
        user_id=user_id,
        metadata_={"bootstrap": True},
    )
    db.add(db_key)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise

    logger.warning(
        "No API keys found. Created bootstrap key for first run. Save this key now: %s",
        api_key,
    )
