from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Optional

from app.database.models import ModelMetadata

async def get_metadata_by_model_name(db: AsyncIOMotorDatabase, model_name: str) -> Optional[ModelMetadata]:
    return await ModelMetadata.find_one(ModelMetadata.model_name == model_name)

async def get_or_create_metadata(db: AsyncIOMotorDatabase, model_name: str) -> ModelMetadata:
    """Gets metadata for a model, creating a default entry if it doesn't exist."""
    metadata = await get_metadata_by_model_name(db, model_name)
    if not metadata:
        # Basic heuristic for multi-modal models
        supports_images_default = "llava" in model_name or "bakllava" in model_name

        metadata = ModelMetadata(
            model_name=model_name,
            supports_images=supports_images_default,
            description="Auto-discovered model."
        )
        await metadata.insert()
    return metadata

async def get_all_metadata(db: AsyncIOMotorDatabase) -> List[ModelMetadata]:
    """Gets all model metadata records, sorted by priority then name."""
    return await ModelMetadata.find().sort([("priority", 1), ("model_name", 1)]).to_list()

async def update_metadata(db: AsyncIOMotorDatabase, model_name: str, **kwargs) -> Optional[ModelMetadata]:
    """Updates metadata for a specific model."""
    metadata = await get_metadata_by_model_name(db, model_name)
    if not metadata:
        return None

    for key, value in kwargs.items():
        if hasattr(metadata, key):
            setattr(metadata, key, value)

    await metadata.save()
    return metadata
