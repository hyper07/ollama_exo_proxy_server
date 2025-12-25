from app.database.models import User, APIKey, UsageLog, UserRole
from app.schema.user import UserCreate
from app.core.security import get_password_hash
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase


async def get_user_by_username(db: AsyncIOMotorDatabase, username: str) -> User | None:
    return await User.find_one(User.username == username)


async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> User | None:
    try:
        # Try to get by ObjectId first
        if len(user_id) == 24 and all(c in '0123456789abcdefABCDEF' for c in user_id):
            return await User.get(user_id)
        else:
            # If not a valid ObjectId, search by string ID field
            return await User.find_one(User.id == user_id)
    except Exception:
        return None


async def get_users(db: AsyncIOMotorDatabase, skip: int = 0, limit: int = 100, sort_by: str = "username", sort_order: str = "asc") -> list:
    """
    Retrieves a list of users.
    """
    # Build sort criteria for MongoDB
    sort_criteria = []
    if sort_by == "username":
        sort_criteria.append(("username", 1 if sort_order == "asc" else -1))
    elif sort_by == "created_at":
        sort_criteria.append(("created_at", 1 if sort_order == "asc" else -1))
    else:
        # Default sort by username ascending
        sort_criteria.append(("username", 1))

    # Use MongoDB aggregation to get users with API key and usage statistics
    pipeline = [
        # Sort first
        {"$sort": dict(sort_criteria)},
        # Skip and limit
        {"$skip": skip},
        {"$limit": limit},
        # Left join with API keys to count them
        {
            "$lookup": {
                "from": "apikey",
                "localField": "_id",
                "foreignField": "user.$id",  # Reference to user ObjectId
                "as": "api_keys"
            }
        },
        # Add computed fields
        {
            "$addFields": {
                "key_count": {"$size": "$api_keys"},
                "request_count": {"$sum": "$api_keys.usage_count"},  # Sum usage from all keys
                "last_used": {
                    "$max": "$api_keys.last_used"  # Most recent usage across all keys
                }
            }
        },
        # Remove the api_keys array to avoid sending too much data
        {
            "$project": {
                "api_keys": 0
            }
        }
    ]

    users = await User.aggregate(pipeline).to_list(length=None)

    # Convert ObjectIds to strings for template compatibility
    # Since aggregate returns dicts, we need to handle them differently
    for user in users:
        if '_id' in user:
            user['id'] = str(user['_id'])

    return users


async def cleanup_duplicate_users(db: AsyncIOMotorDatabase) -> int:
    """
    Remove duplicate users, keeping only the oldest one for each username.
    Returns the number of duplicate users removed.
    """
    # Find usernames that appear more than once
    pipeline = [
        {"$group": {"_id": "$username", "count": {"$sum": 1}, "users": {"$push": {"_id": "$_id", "created_at": "$created_at"}}}},
        {"$match": {"count": {"$gt": 1}}}
    ]

    duplicates = await db.User.aggregate(pipeline).to_list(length=None)
    removed_count = 0

    for dup in duplicates:
        # Sort by creation date, keep the oldest
        users_sorted = sorted(dup["users"], key=lambda x: x["created_at"])
        # Remove all except the first (oldest)
        users_to_remove = users_sorted[1:]

        for user in users_to_remove:
            await db.User.delete_one({"_id": user["_id"]})
            removed_count += 1

    return removed_count


async def ensure_user_uniqueness(db: AsyncIOMotorDatabase) -> dict:
    """
    Ensure all users have unique usernames and IDs.
    Returns a summary of actions taken.
    """
    result = {
        "duplicates_removed": 0,
        "indexes_created": False
    }

    # Clean up duplicates
    result["duplicates_removed"] = await cleanup_duplicate_users(db)

    # Ensure unique index exists
    try:
        await db.User.create_index([("username", 1)], unique=True)
        result["indexes_created"] = True
    except Exception:
        # Index might already exist
        pass

    return result


async def create_user(db: AsyncIOMotorDatabase, user: UserCreate, role: UserRole = UserRole.USER) -> User:
    # Check if user already exists
    existing_user = await get_user_by_username(db, username=user.username)
    if existing_user:
        raise ValueError(f"User with username '{user.username}' already exists")

    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        hashed_password=hashed_password,
        is_admin=(role == UserRole.ADMIN),  # Keep backward compatibility
        role=role,
    )
    await db_user.insert()
    return db_user
    await db.refresh(db_user)
    return db_user


async def update_user(db: AsyncIOMotorDatabase, user_id: str, username: str, password: Optional[str] = None, role: Optional[UserRole] = None) -> User | None:
    """Updates a user's username, password, and optionally their role."""
    user = await get_user_by_id(db, user_id=user_id)
    if not user:
        return None

    user.username = username
    if password:
        user.hashed_password = get_password_hash(password)
    if role is not None:
        user.role = role
        user.is_admin = (role == UserRole.ADMIN)  # Keep backward compatibility

    await user.save()
    return user


async def delete_user(db: AsyncIOMotorDatabase, user_id: str) -> User | None:
    user = await get_user_by_id(db, user_id=user_id)
    if user:
        await user.delete()
    return user