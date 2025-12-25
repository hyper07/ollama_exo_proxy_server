# app/crud/apikey_crud.py
import secrets
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional

from app.database.models import APIKey, User
from app.core.security import get_api_key_hash

logger = logging.getLogger(__name__)


async def get_api_key_by_prefix(db: AsyncIOMotorDatabase, prefix: str) -> APIKey | None:
    return await APIKey.find_one(APIKey.key_prefix == prefix)


async def get_api_key_by_id(db: AsyncIOMotorDatabase, key_id: str) -> APIKey | None:
    return await APIKey.get(key_id)


async def get_api_keys_for_user(db: AsyncIOMotorDatabase, user_id: str) -> list[APIKey]:
    from bson import ObjectId
    
    # Get the User object first, then query by comparing Link to User object directly
    logger.info(f"Fetching API keys for user_id: {user_id}")
    user = await User.get(user_id)
    if not user:
        logger.warning(f"User with id {user_id} not found")
        return []
    
    logger.info(f"Found user: {user.username} (ObjectId: {user.id}), querying for API keys...")
    
    # Check what's actually in the database using raw query
    raw_collection = db.get_collection("APIKey")
    raw_count = await raw_collection.count_documents({"user.$id": ObjectId(user_id)})
    logger.info(f"Raw MongoDB query found {raw_count} keys with user.$id == {user_id}")
    
    # Sample one document to see structure
    sample_doc = await raw_collection.find_one({"user.$id": ObjectId(user_id)})
    if sample_doc:
        logger.info(f"Sample document user field structure: {sample_doc.get('user', 'N/A')}")
    
    # Try multiple query approaches to find what works
    # Approach 1: Using Link reference
    logger.info(f"Trying query: APIKey.user.ref.id == {user.id}")
    keys1 = await APIKey.find(APIKey.user.ref.id == user.id).sort(-APIKey.created_at).to_list()
    logger.info(f"Approach 1 (user.ref.id): Found {len(keys1)} keys")
    
    # Approach 2: Direct Link comparison
    logger.info(f"Trying query: APIKey.user == user")
    keys2 = await APIKey.find(APIKey.user == user).sort(-APIKey.created_at).to_list()
    logger.info(f"Approach 2 (user == user): Found {len(keys2)} keys")
    
    # Use whichever approach returned results
    keys = keys1 if keys1 else keys2
    logger.info(f"Using result with {len(keys)} API keys for user {user.username}")
    
    # If still no results, try raw query and convert
    if not keys and raw_count > 0:
        logger.warning("Beanie queries failed but raw query found keys. Using raw query...")
        raw_docs = await raw_collection.find({"user.$id": ObjectId(user_id)}).sort("created_at", -1).to_list(length=None)
        keys = [await APIKey.get(doc["_id"]) for doc in raw_docs]
        keys = [k for k in keys if k is not None]
        logger.info(f"Retrieved {len(keys)} keys via raw query fallback")
    
    # Log details about each key
    for key in keys:
        logger.info(f"  - Key ID: {key.id}, Name: {key.key_name}, Prefix: {key.key_prefix}, Active: {key.is_active}")
    
    return keys


async def create_api_key(
    db: AsyncIOMotorDatabase, 
    user_id: str, 
    key_name: str,
    rate_limit_requests: Optional[int] = None,
    rate_limit_window_minutes: Optional[int] = None
) -> tuple[str, APIKey]:
    """
    Generates a new API key, stores its hash, and returns the plain key and the DB object.
    The plain key is only available at creation time.
    """
    logger.info(f"Creating API key '{key_name}' for user_id: {user_id}")
    
    # Get the User object for the Link reference
    user = await User.get(user_id)
    if not user:
        logger.error(f"User with id {user_id} not found")
        raise ValueError(f"User with id {user_id} not found")
    
    logger.info(f"Found user: {user.username} (ID: {user.id})")
    
    # --- CRITICAL FIX: Use token_hex to guarantee no underscores in random parts ---
    # This makes the '_' a reliable delimiter.
    prefix_random_part = secrets.token_hex(8)
    prefix = f"op_{prefix_random_part}"
    
    secret = secrets.token_hex(24)
    plain_key = f"{prefix}_{secret}"
    # --- END FIX ---

    hashed_key = get_api_key_hash(secret)
    
    logger.info(f"Generated key with prefix: {prefix}")

    db_api_key = APIKey(
        key_name=key_name,
        hashed_key=hashed_key,
        key_prefix=prefix,
        user=user,  # Use User object, not just user_id string
        rate_limit_requests=rate_limit_requests,
        rate_limit_window_minutes=rate_limit_window_minutes
    )
    
    logger.info(f"Inserting API key into database...")
    try:
        await db_api_key.insert()
        logger.info(f"API key inserted successfully with ID: {db_api_key.id}")
    except Exception as e:
        logger.error(f"Failed to insert API key: {e}", exc_info=True)
        raise
    
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
    return await APIKey.find_one(APIKey.user.ref.id == user.id, APIKey.key_name == key_name)