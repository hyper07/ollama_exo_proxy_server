#!/bin/bash

# exo_internet_on.sh - Restore Exo internet access
# This script removes the restrictions applied by exo_internet_off.sh

set -e

echo "🌐 Restoring Exo internet access..."

# Method 1: Remove iptables rules
if command -v iptables &> /dev/null; then
    echo "📡 Removing iptables rules..."

    # Remove the specific rules we added (note: this removes ALL matching rules)
    sudo iptables -D OUTPUT -m owner --exe-owner $(which exo) -p tcp --dport 80 -j REJECT 2>/dev/null || true
    sudo iptables -D OUTPUT -m owner --exe-owner $(which exo) -p tcp --dport 443 -j REJECT 2>/dev/null || true
    sudo iptables -D OUTPUT -m owner --exe-owner $(which exo) -p tcp --dport 52415 -j REJECT 2>/dev/null || true
    sudo iptables -D OUTPUT -m owner --exe-owner $(which exo) -p udp --dport 53 -j REJECT 2>/dev/null || true
    sudo iptables -D OUTPUT -m owner --exe-owner $(which exo) -p tcp --dport 53 -j REJECT 2>/dev/null || true

    echo "✅ iptables rules removed"
else
    echo "⚠️  iptables not available"
fi

# Method 2: Remove wrapper script
EXO_WRAPPER="$HOME/.local/bin/exo_offline"
if [[ -f "$EXO_WRAPPER" ]]; then
    rm "$EXO_WRAPPER"
    echo "✅ Removed offline wrapper script"
fi

# Method 3: Remove environment file
EXO_ENV_FILE="$HOME/.exo_offline_env"
if [[ -f "$EXO_ENV_FILE" ]]; then
    rm "$EXO_ENV_FILE"
    echo "✅ Removed offline environment file"
fi

# Method 4: Restore hosts file
if [[ -f /etc/hosts.exo_backup ]]; then
    echo "🔒 Restoring hosts file..."
    sudo cp /etc/hosts.exo_backup /etc/hosts 2>/dev/null || true
    sudo rm /etc/hosts.exo_backup 2>/dev/null || true
    echo "✅ Hosts file restored"
else
    # Manually remove the lines if backup doesn't exist
    if [[ -w /etc/hosts ]] || sudo -n true 2>/dev/null; then
        sudo sed -i '/# Exo internet blocking/,$d' /etc/hosts 2>/dev/null || true
        echo "✅ Removed Exo blocking entries from hosts file"
    fi
fi

# Method 5: Reset environment variables to defaults
echo "🔧 Resetting environment variables..."
unset EXO_HOST
unset EXO_ORIGINS
unset EXO_DEBUG

echo ""
echo "🎯 Exo internet access has been restored!"
echo "💡 You can now use Exo normally:"
echo "   exo run llama2"
echo "   exo pull mistral"
echo ""
echo "🔄 You may want to restart any running Exo processes:"
echo "   pkill exo"
echo "   exo serve"
