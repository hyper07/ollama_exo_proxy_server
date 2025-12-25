# app/crud/apikey_crud.py
import secrets
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional

from app.database.models import APIKey, User
from app.core.security import get_api_key_hash


async def get_api_key_by_prefix(db: AsyncIOMotorDatabase, prefix: str) -> APIKey | None:
    return await APIKey.find_one(APIKey.key_prefix == prefix)


async def get_api_key_by_id(db: AsyncIOMotorDatabase, key_id: str) -> APIKey | None:
    return await APIKey.get(key_id)


async def get_api_keys_for_user(db: AsyncIOMotorDatabase, user_id: str) -> list[APIKey]:
    # Get the User object first, then query by comparing Link to User object directly
    user = await User.get(user_id)
    if not user:
        return []
    return await APIKey.find(APIKey.user == user).sort(-APIKey.created_at).to_list()


async def create_api_key(
    db: AsyncIOMotorDatabase, 
    user_id: str, 
    key_name: str,
    rate_limit_requests: Optional[int] = None,
    rate_limit_window_minutes: Optional[int] = None
) -> (str, APIKey):
    """
    Generates a new API key, stores its hash, and returns the plain key and the DB object.
    The plain key is only available at creation time.
    """
    # Get the User object for the Link reference
    user = await User.get(user_id)
    if not user:
        raise ValueError(f"User with id {user_id} not found")
    
    # --- CRITICAL FIX: Use token_hex to guarantee no underscores in random parts ---
    # This makes the '_' a reliable delimiter.
    prefix_random_part = secrets.token_hex(8)
    prefix = f"op_{prefix_random_part}"
    
    secret = secrets.token_hex(24)
    plain_key = f"{prefix}_{secret}"
    # --- END FIX ---

    hashed_key = get_api_key_hash(secret)

    db_api_key = APIKey(
        key_name=key_name,
        hashed_key=hashed_key,
        key_prefix=prefix,
        user=user,  # Use User object, not just user_id string
        rate_limit_requests=rate_limit_requests,
        rate_limit_window_minutes=rate_limit_window_minutes
    )
    await db_api_key.insert()
    return plain_key, db_api_key


async def revoke_api_key(db: AsyncIOMotorDatabase, key_id: str) -> APIKey | None:
    key = await get_api_key_by_id(db, key_id)
    if key:
        key.is_revoked = True
        key.is_active = False
        await key.save()
    return key


async def toggle_api_key_active(db: AsyncIOMotorDatabase, key_id: str) -> APIKey | None:
    """Toggles the is_active status of an API key."""
    key = await get_api_key_by_id(db, key_id)
    if not key or key.is_revoked:
        return None

    key.is_active = not key.is_active
    await key.save()
    return key


async def get_api_key_by_name_and_user_id(db: AsyncIOMotorDatabase, *, key_name: str, user_id: str) -> APIKey | None:
    """Gets an API key by its name for a specific user."""
    # Get the User object first, then query by comparing Link to User object directly
    user = await User.get(user_id)
    if not user:
        return None
    return await APIKey.find_one(APIKey.user == user, APIKey.key_name == key_name)