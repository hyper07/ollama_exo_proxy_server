from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings

client = AsyncIOMotorClient(settings.DATABASE_URL)
database = client.get_database()


async def get_db() -> AsyncIOMotorDatabase:
    yield database