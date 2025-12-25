#!/usr/bin/env python3
"""
Migration script to add role field to existing users and convert is_admin to role-based system.
This script should be run once after updating the codebase to the new role system.
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from app.database.models import User, UserRole
from app.database.session import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_user_roles():
    """
    Migrate existing users to use the new role-based system.
    - Convert is_admin boolean to appropriate role enum values
    - Add role field to all users that don't have it
    """
    logger.info("Starting user role migration...")

    try:
        # Get database connection
        db = await get_db()

        # Find all users
        users = await User.find().to_list(length=None)
        logger.info(f"Found {len(users)} users to migrate")

        migrated_count = 0

        for user in users:
            # Check if user already has role field
            if hasattr(user, 'role') and user.role is not None:
                logger.debug(f"User {user.username} already has role: {user.role}")
                continue

            # Determine role based on is_admin field
            if user.is_admin:
                user.role = UserRole.ADMIN
                logger.info(f"Setting role ADMIN for user {user.username}")
            else:
                user.role = UserRole.USER
                logger.info(f"Setting role USER for user {user.username}")

            # Save the updated user
            await user.save()
            migrated_count += 1

        logger.info(f"Successfully migrated {migrated_count} users")
        logger.info("User role migration completed successfully")

    except Exception as e:
        logger.error(f"Error during user role migration: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(migrate_user_roles())
