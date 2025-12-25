import json
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.models import AppSettings
from app.schema.settings import AppSettingsModel

async def get_app_settings(db: AsyncIOMotorDatabase) -> AppSettings | None:
    """Retrieves the application settings from the database."""
    return await AppSettings.find_one(AppSettings.id == "main")

async def update_app_settings(db: AsyncIOMotorDatabase, settings_data: AppSettingsModel) -> AppSettings:
    """Updates the application settings in the database."""
    db_settings = await get_app_settings(db)
    if not db_settings:
        db_settings = AppSettings(id="main")

    # Update fields from the Pydantic model
    db_settings.settings_data = json.loads(settings_data.model_dump_json())

    await db_settings.save()
    return db_settings

async def create_initial_settings(db: AsyncIOMotorDatabase) -> AppSettings:
    """Creates the very first default settings if none exist."""
    existing_settings = await get_app_settings(db)
    if existing_settings:
        return existing_settings

    # No longer contains a default server
    default_settings = AppSettingsModel()

    new_db_settings = AppSettings(
        id="main",
        settings_data=json.loads(default_settings.model_dump_json())
    )
    await new_db_settings.insert()
    return new_db_settings