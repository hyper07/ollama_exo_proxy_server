#!/usr/bin/env python3
"""
Cleanup script to remove duplicate users from the database.
This script will keep only the oldest user for each username.

Usage:
    python cleanup_duplicate_users.py
"""

import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

async def cleanup_duplicates():
    """Remove duplicate users, keeping only the oldest one for each username."""
    print("Connecting to MongoDB...")
    client = AsyncIOMotorClient(settings.DATABASE_URL)
    db = client.get_database("exo_proxy")
    
    print("Checking for duplicate users...")
    
    # Find usernames that appear more than once
    pipeline = [
        {
            "$group": {
                "_id": "$username",
                "count": {"$sum": 1},
                "users": {
                    "$push": {
                        "_id": "$_id",
                        "created_at": "$created_at",
                        "role": "$role"
                    }
                }
            }
        },
        {"$match": {"count": {"$gt": 1}}}
    ]
    
    duplicates = await db.User.aggregate(pipeline).to_list(length=None)
    
    if not duplicates:
        print("✅ No duplicate users found!")
        return
    
    print(f"Found {len(duplicates)} username(s) with duplicates:")
    
    total_removed = 0
    
    for dup in duplicates:
        username = dup["_id"]
        count = dup["count"]
        users = dup["users"]
        
        print(f"\n  Username: '{username}' has {count} duplicate(s)")
        
        # Sort by creation date, keep the oldest
        users_sorted = sorted(users, key=lambda x: x.get("created_at", ""))
        oldest = users_sorted[0]
        users_to_remove = users_sorted[1:]
        
        print(f"    Keeping: User ID {oldest['_id']} (created: {oldest.get('created_at', 'unknown')})")
        
        for user in users_to_remove:
            print(f"    Removing: User ID {user['_id']} (created: {user.get('created_at', 'unknown')})")
            result = await db.User.delete_one({"_id": user["_id"]})
            if result.deleted_count > 0:
                total_removed += 1
    
    print(f"\n✅ Successfully removed {total_removed} duplicate user(s)!")
    print("The database now has unique usernames.")
    
    client.close()

if __name__ == "__main__":
    try:
        asyncio.run(cleanup_duplicates())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)



