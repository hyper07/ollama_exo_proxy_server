#!/bin/bash
set -euo pipefail

# ====================================================================
#
#   Exo Proxy Fortress - Quick Start
#
# ====================================================================

echo "🚀 Exo Proxy Fortress"
echo "===================="
echo

# Check if .env exists
if [[ ! -f ".env" ]]; then
    echo "📝 First time setup required..."
    ./setup.sh
fi

echo "🐳 Starting services with Docker..."
docker-compose up -d

echo
echo "✅ Exo Proxy Fortress is starting!"
echo
echo "🌐 Access the web interface at: http://localhost:$(grep PROXY_PORT .env | cut -d'=' -f2 | tr -d '"')"
echo "👤 Default admin login: admin / (password from setup)"
echo
echo "📊 View logs: docker-compose logs -f app"
echo "🛑 Stop services: docker-compose down"
echo "🔄 Restart: docker-compose restart"