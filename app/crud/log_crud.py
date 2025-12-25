# app/crud/log_crud.py
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.database.models import UsageLog, APIKey, User, ExoServer
import datetime
from typing import List, Dict, Any, Optional
from beanie import PydanticObjectId

async def create_usage_log(
    db: AsyncIOMotorDatabase, *, api_key_id, endpoint: str, status_code: int, server_id=None, model: str | None = None
) -> UsageLog:
    """Create a usage log entry.
    
    Args:
        db: Database connection
        api_key_id: The ID of the APIKey (string or PydanticObjectId, will be fetched and used as Link reference)
        endpoint: The API endpoint that was called
        status_code: HTTP status code of the response
        server_id: The ID of the ExoServer (string or PydanticObjectId, will be fetched and used as Link reference)
        model: Optional model name used in the request
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Convert to string if needed for Beanie.get()
        api_key_id_str = str(api_key_id) if api_key_id else None
        server_id_str = str(server_id) if server_id else None
        
        logger.info(f"Creating usage log: endpoint={endpoint}, api_key_id={api_key_id_str}, server_id={server_id_str}, model={model}")
        
        # Fetch the actual APIKey object for proper Link reference
        api_key = await APIKey.get(api_key_id_str)
        if not api_key:
            logger.error(f"APIKey with id {api_key_id_str} not found when creating usage log")
            raise ValueError(f"APIKey with id {api_key_id_str} not found")
        
        logger.debug(f"Found API key: {api_key.key_prefix}")
        
        # Fetch the actual ExoServer object if server_id is provided
        server = None
        if server_id_str:
            server = await ExoServer.get(server_id_str)
            # If server not found, log a warning but don't fail the request
            if not server:
                logger.warning(f"ExoServer with id {server_id_str} not found when creating usage log")
            else:
                logger.debug(f"Found server: {server.name}")
        
        db_log = UsageLog(
            api_key=api_key,
            endpoint=endpoint,
            status_code=status_code,
            server=server,
            model=model
        )
        await db_log.insert()
        logger.info(f"Successfully created usage log with id {db_log.id}")
        return db_log
        
    except Exception as e:
        logger.error(f"Failed to create usage log: {type(e).__name__}: {str(e)}", exc_info=True)
        # Re-raise to let caller handle it
        raise

async def get_usage_statistics(db: AsyncIOMotorDatabase, sort_by: str = "request_count", sort_order: str = "desc", user_id: str = None):
    """
    Returns aggregated usage statistics for API keys, with sorting.
    Counts actual usage logs per API key.
    If user_id is provided, only returns stats for that user's API keys.
    """
    # MongoDB aggregation pipeline starting from usagelog collection
    pipeline = [
        # Lookup API key information
        {
            "$lookup": {
                "from": "apikey",
                "localField": "api_key",
                "foreignField": "_id",
                "as": "api_key_info"
            }
        },
        # Unwind the API key info array
        {
            "$unwind": "$api_key_info"
        },
        # Lookup user information through API key
        {
            "$lookup": {
                "from": "user",
                "localField": "api_key_info.user",
                "foreignField": "_id",
                "as": "user_info"
            }
        },
        # Unwind the user info array
        {
            "$unwind": "$user_info"
        },
        # Filter by user if specified
        *([
            {
                "$match": {
                    "user_info._id": user_id
                }
            }
        ] if user_id else []),
        # Group by API key details
        {
            "$group": {
                "_id": {
                    "username": "$user_info.username",
                    "key_name": "$api_key_info.key_name",
                    "key_prefix": "$api_key_info.key_prefix",
                    "is_revoked": "$api_key_info.is_revoked"
                },
                "request_count": {"$sum": 1}
            }
        },
        # Project to match expected format
        {
            "$project": {
                "username": "$_id.username",
                "key_name": "$_id.key_name",
                "key_prefix": "$_id.key_prefix",
                "is_revoked": "$_id.is_revoked",
                "request_count": 1,
                "_id": 0
            }
        }
    ]

    # Add sorting
    sort_direction = 1 if sort_order.lower() == "asc" else -1
    sort_field = "request_count"  # Default sort field
    if sort_by in ["username", "key_name", "key_prefix"]:
        sort_field = sort_by

    pipeline.append({"$sort": {sort_field: sort_direction}})

    # Execute aggregation on usagelog collection
    results = await UsageLog.aggregate(pipeline).to_list(length=None)

    return results

# --- NEW STATISTICS FUNCTIONS ---

async def get_daily_usage_stats(db: AsyncIOMotorDatabase, days: int = 30):
    """Returns total requests per day for the last N days."""
    start_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)

    # MongoDB aggregation pipeline
    pipeline = [
        # Filter by date range
        {
            "$match": {
                "request_timestamp": {"$gte": start_date}
            }
        },
        # Group by date (YYYY-MM-DD format)
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$request_timestamp"
                    }
                },
                "request_count": {"$sum": 1}
            }
        },
        # Sort by date
        {
            "$sort": {"_id": 1}
        },
        # Project to match expected format
        {
            "$project": {
                "date": {
                    "$dateFromString": {
                        "dateString": "$_id",
                        "format": "%Y-%m-%d"
                    }
                },
                "request_count": 1,
                "_id": 0
            }
        }
    ]

    # Execute aggregation
    result = await UsageLog.aggregate(pipeline).to_list(length=None)

    # Convert to objects with date and request_count attributes
    class DailyStat:
        def __init__(self, date, request_count):
            self.date = date
            self.request_count = request_count

    return [DailyStat(row["date"], row["request_count"]) for row in result]

async def get_hourly_usage_stats(db: AsyncIOMotorDatabase):
    """Returns total requests aggregated by the hour of the day (UTC)."""
    # MongoDB aggregation pipeline
    pipeline = [
        # Group by hour of day
        {
            "$group": {
                "_id": {
                    "$hour": "$request_timestamp"
                },
                "request_count": {"$sum": 1}
            }
        }
    ]

    # Execute aggregation
    result = await UsageLog.aggregate(pipeline).to_list(length=None)

    # Convert to dictionary for easy lookup
    stats_dict = {str(row["_id"]).zfill(2): row["request_count"] for row in result}

    # Ensure all 24 hours are present (00 to 23)
    return [{"hour": f"{h:02d}:00", "request_count": stats_dict.get(f"{h:02d}", 0)} for h in range(24)]

async def get_server_load_stats(db: AsyncIOMotorDatabase):
    """Returns total requests per backend server."""
    # MongoDB aggregation pipeline
    pipeline = [
        # Lookup server information
        {
            "$lookup": {
                "from": "exoserver",
                "localField": "server",
                "foreignField": "_id",
                "as": "server_info"
            }
        },
        # Unwind the server info array
        {
            "$unwind": {
                "path": "$server_info",
                "preserveNullAndEmptyArrays": True
            }
        },
        # Group by server name
        {
            "$group": {
                "_id": "$server_info.name",
                "request_count": {"$sum": 1}
            }
        },
        # Sort by request count descending
        {
            "$sort": {"request_count": -1}
        },
        # Project to match expected format
        {
            "$project": {
                "server_name": "$_id",
                "request_count": 1,
                "_id": 0
            }
        }
    ]

    # Execute aggregation
    result = await UsageLog.aggregate(pipeline).to_list(length=None)

    # Convert to objects with server_name and request_count attributes
    class ServerStat:
        def __init__(self, server_name, request_count):
            self.server_name = server_name or "Unknown Server"
            self.request_count = request_count

    return [ServerStat(row.get("server_name"), row["request_count"]) for row in result]

async def get_model_usage_stats(db: AsyncIOMotorDatabase):
    """Returns total requests per model."""
    # MongoDB aggregation pipeline
    pipeline = [
        # Filter out null models
        {
            "$match": {
                "model": {"$ne": None}
            }
        },
        # Group by model name
        {
            "$group": {
                "_id": "$model",
                "request_count": {"$sum": 1}
            }
        },
        # Sort by request count descending
        {
            "$sort": {"request_count": -1}
        },
        # Project to match expected format
        {
            "$project": {
                "model_name": "$_id",
                "request_count": 1,
                "_id": 0
            }
        }
    ]

    # Execute aggregation
    result = await UsageLog.aggregate(pipeline).to_list(length=None)

    # Convert to objects with model_name and request_count attributes
    class ModelStat:
        def __init__(self, model_name, request_count):
            self.model_name = model_name
            self.request_count = request_count

    return [ModelStat(row["model_name"], row["request_count"]) for row in result]

# --- NEW USER-SPECIFIC STATISTICS FUNCTIONS ---

async def get_daily_usage_stats_for_user(db: AsyncIOMotorDatabase, user_id: str, days: int = 30):
    """Returns total requests per day for the last N days for a specific user."""
    start_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)

    # MongoDB aggregation pipeline
    pipeline = [
        # Lookup API key information and filter by user
        {
            "$lookup": {
                "from": "apikey",
                "localField": "api_key",
                "foreignField": "_id",
                "as": "api_key_info"
            }
        },
        # Unwind the API key info array
        {
            "$unwind": "$api_key_info"
        },
        # Filter by user ID (user is a DBRef, so we need to match $id)
        {
            "$match": {
                "api_key_info.user.$id": PydanticObjectId(user_id),
                "request_timestamp": {"$gte": start_date}
            }
        },
        # Group by date (YYYY-MM-DD format)
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$request_timestamp"
                    }
                },
                "request_count": {"$sum": 1}
            }
        },
        # Sort by date
        {
            "$sort": {"_id": 1}
        },
        # Project to match expected format
        {
            "$project": {
                "date": {
                    "$dateFromString": {
                        "dateString": "$_id",
                        "format": "%Y-%m-%d"
                    }
                },
                "request_count": 1,
                "_id": 0
            }
        }
    ]

    # Execute aggregation
    result = await UsageLog.aggregate(pipeline).to_list(length=None)

    # Convert to objects with date and request_count attributes
    class DailyStat:
        def __init__(self, date, request_count):
            self.date = date
            self.request_count = request_count

    return [DailyStat(row["date"], row["request_count"]) for row in result]

async def get_hourly_usage_stats_for_user(db: AsyncIOMotorDatabase, user_id: str):
    """Returns total requests aggregated by the hour for a specific user."""
    # MongoDB aggregation pipeline
    pipeline = [
        # Lookup API key information and filter by user
        {
            "$lookup": {
                "from": "apikey",
                "localField": "api_key",
                "foreignField": "_id",
                "as": "api_key_info"
            }
        },
        # Unwind the API key info array
        {
            "$unwind": "$api_key_info"
        },
        # Filter by user ID
        {
            "$match": {
                "api_key_info.user.$id": PydanticObjectId(user_id)
            }
        },
        # Group by hour of day
        {
            "$group": {
                "_id": {
                    "$hour": "$request_timestamp"
                },
                "request_count": {"$sum": 1}
            }
        }
    ]

    # Execute aggregation
    result = await UsageLog.aggregate(pipeline).to_list(length=None)

    # Convert to dictionary for easy lookup
    stats_dict = {str(row["_id"]).zfill(2): row["request_count"] for row in result}

    # Ensure all 24 hours are present (00 to 23)
    return [{"hour": f"{h:02d}:00", "request_count": stats_dict.get(f"{h:02d}", 0)} for h in range(24)]

async def get_server_load_stats_for_user(db: AsyncIOMotorDatabase, user_id: str):
    """Returns total requests per backend server for a specific user."""
    # MongoDB aggregation pipeline
    pipeline = [
        # Lookup API key information and filter by user
        {
            "$lookup": {
                "from": "apikey",
                "localField": "api_key",
                "foreignField": "_id",
                "as": "api_key_info"
            }
        },
        # Unwind the API key info array
        {
            "$unwind": "$api_key_info"
        },
        # Filter by user ID
        {
            "$match": {
                "api_key_info.user.$id": PydanticObjectId(user_id)
            }
        },
        # Lookup server information
        {
            "$lookup": {
                "from": "exoserver",
                "localField": "server",
                "foreignField": "_id",
                "as": "server_info"
            }
        },
        # Unwind the server info array
        {
            "$unwind": {
                "path": "$server_info",
                "preserveNullAndEmptyArrays": True
            }
        },
        # Group by server name
        {
            "$group": {
                "_id": "$server_info.name",
                "request_count": {"$sum": 1}
            }
        },
        # Sort by request count descending
        {
            "$sort": {"request_count": -1}
        },
        # Project to match expected format
        {
            "$project": {
                "server_name": "$_id",
                "request_count": 1,
                "_id": 0
            }
        }
    ]

    # Execute aggregation
    result = await UsageLog.aggregate(pipeline).to_list(length=None)

    # Convert to objects with server_name and request_count attributes
    class ServerStat:
        def __init__(self, server_name, request_count):
            self.server_name = server_name or "Unknown Server"
            self.request_count = request_count

    return [ServerStat(row.get("server_name"), row["request_count"]) for row in result]

async def get_model_usage_stats_for_user(db: AsyncIOMotorDatabase, user_id: str):
    """Returns total requests per model for a specific user."""
    # MongoDB aggregation pipeline
    pipeline = [
        # Lookup API key information and filter by user
        {
            "$lookup": {
                "from": "apikey",
                "localField": "api_key",
                "foreignField": "_id",
                "as": "api_key_info"
            }
        },
        # Unwind the API key info array
        {
            "$unwind": "$api_key_info"
        },
        # Filter by user ID and non-null models
        {
            "$match": {
                "api_key_info.user.$id": PydanticObjectId(user_id),
                "model": {"$ne": None}
            }
        },
        # Group by model name
        {
            "$group": {
                "_id": "$model",
                "request_count": {"$sum": 1}
            }
        },
        # Sort by request count descending
        {
            "$sort": {"request_count": -1}
        },
        # Project to match expected format
        {
            "$project": {
                "model_name": "$_id",
                "request_count": 1,
                "_id": 0
            }
        }
    ]

    # Execute aggregation
    result = await UsageLog.aggregate(pipeline).to_list(length=None)

    # Convert to objects with model_name and request_count attributes
    class ModelStat:
        def __init__(self, model_name, request_count):
            self.model_name = model_name
            self.request_count = request_count

    return [ModelStat(row["model_name"], row["request_count"]) for row in result]