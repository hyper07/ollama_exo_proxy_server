#!/bin/bash
# Start the proxy server locally (outside Docker)
export DATABASE_URL="mongodb://localhost:27017/exo_proxy"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload
