#!/bin/bash
set -euo pipefail

# ====================================================================
#
#   Exo Proxy Fortress - Setup Wizard
#
# ====================================================================

echo "🚀 Exo Proxy Fortress Setup"
echo "============================"
echo

# Check if .env already exists
if [[ -f ".env" ]]; then
    echo "⚠️  Configuration file (.env) already exists!"
    read -p "Do you want to recreate it? (y/n): " RECREATE
    if [[ ! "$RECREATE" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    rm .env
fi

echo "📝 Running setup wizard..."
python3 setup_wizard.py

if [[ -f ".env" ]]; then
    echo
    echo "✅ Setup complete! Your configuration has been saved to .env"
    echo
    echo "🚀 To start the server with Docker, run:"
    echo "   docker-compose up -d"
    echo
    echo "🌐 Then visit: http://localhost:$(grep PROXY_PORT .env | cut -d'=' -f2 | tr -d '"')"
    echo
    echo "📊 To view logs: docker-compose logs -f app"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Setup failed!"
    exit 1
fi