import os
import json
import asyncio
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.logging_config import LOGGING_CONFIG
from app.core.config import settings


# Gunicorn config variables
loglevel = os.environ.get("LOG_LEVEL", "info")
workers = int(os.environ.get("GUNICORN_WORKERS", "4"))
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8080")
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "uvicorn.workers.UvicornWorker")
accesslog = "-"  # Direct access logs to stdout
errorlog = "-"   # Direct error logs to stdout

# Use our custom JSON logging config
logconfig_dict = LOGGING_CONFIG

# --- Load SSL settings from MongoDB for Gunicorn ---
keyfile = None
certfile = None

async def load_ssl_settings():
    global keyfile, certfile
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.DATABASE_URL.replace("mongodb://", "").split("/")[0])
        db = client.get_database("exo_proxy")

        # Get SSL settings from app_settings collection
        settings_collection = db.app_settings
        app_settings = await settings_collection.find_one({"_id": "main"})

        if app_settings and "settings_data" in app_settings:
            db_settings = app_settings["settings_data"]
            keyfile_path = db_settings.get("ssl_keyfile")
            certfile_path = db_settings.get("ssl_certfile")

            if keyfile_path and certfile_path:
                if Path(keyfile_path).is_file() and Path(certfile_path).is_file():
                    keyfile = keyfile_path
                    certfile = certfile_path
                    print(f"[INFO] Gunicorn starting with HTTPS. Cert: {certfile}")
                else:
                    if not Path(keyfile_path).is_file():
                        print(f"[WARNING] SSL key file not found at '{keyfile_path}'. HTTPS disabled.")
                    if not Path(certfile_path).is_file():
                        print(f"[WARNING] SSL cert file not found at '{certfile_path}'. HTTPS disabled.")
    except Exception as e:
        print(f"[INFO] Could not load SSL settings from DB (this is normal on first run). Reason: {e}")

# Run the async SSL loading function
asyncio.run(load_ssl_settings())