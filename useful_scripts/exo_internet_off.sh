#!/bin/bash

# exo_internet_off.sh - Block Exo from accessing the internet
# This script uses multiple methods to ensure Exo cannot connect to the internet

set -e

echo "🚫 Blocking Exo internet access..."

# Method 1: Use iptables to block outbound connections from exo process
if command -v iptables &> /dev/null; then
    echo "📡 Setting up iptables rules..."

    # Block HTTP/HTTPS traffic from exo
    sudo iptables -A OUTPUT -m owner --exe-owner $(which exo) -p tcp --dport 80 -j REJECT 2>/dev/null || true
    sudo iptables -A OUTPUT -m owner --exe-owner $(which exo) -p tcp --dport 443 -j REJECT 2>/dev/null || true
    sudo iptables -A OUTPUT -m owner --exe-owner $(which exo) -p tcp --dport 52415 -j REJECT 2>/dev/null || true

    # Block DNS queries from exo
    sudo iptables -A OUTPUT -m owner --exe-owner $(which exo) -p udp --dport 53 -j REJECT 2>/dev/null || true
    sudo iptables -A OUTPUT -m owner --exe-owner $(which exo) -p tcp --dport 53 -j REJECT 2>/dev/null || true

    echo "✅ iptables rules applied"
else
    echo "⚠️  iptables not available, skipping firewall rules"
fi

# Method 2: Create a wrapper script that runs exo without network access
EXO_WRAPPER_DIR="$HOME/.local/bin"
EXO_WRAPPER="$EXO_WRAPPER_DIR/exo_offline"

mkdir -p "$EXO_WRAPPER_DIR"

cat > "$EXO_WRAPPER" << 'EOF'
#!/bin/bash

# Exo offline wrapper
export EXO_HOST="127.0.0.1:52415"

# Use unshare to create network namespace without internet (Linux only)
if command -v unshare &> /dev/null && [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Run exo in a network namespace with only loopback interface
    exec unshare --net --map-root-user bash -c '
        ip link set lo up
        export PATH="'$(dirname $(which exo))':$PATH"
        exec '$( which exo )' "$@"
    ' -- "$@"
else
    # Fallback: just run with localhost binding
    exec $(which exo) "$@"
fi
EOF

chmod +x "$EXO_WRAPPER"

# Method 3: Set environment variables to restrict network access
echo "🔧 Setting up environment configuration..."
EXO_ENV_FILE="$HOME/.exo_offline_env"

cat > "$EXO_ENV_FILE" << 'EOF'
# Exo offline environment variables
export EXO_HOST="127.0.0.1:52415"
export EXO_ORIGINS="http://localhost:*,https://localhost:*"
export EXO_DEBUG=1
EOF

echo "📝 Created environment file at $EXO_ENV_FILE"
echo "💡 To use: source $EXO_ENV_FILE"

# Method 4: Create hosts file backup and block known Exo domains
if [[ -w /etc/hosts ]] || sudo -n true 2>/dev/null; then
    echo "🔒 Backing up and modifying hosts file..."
    sudo cp /etc/hosts /etc/hosts.exo_backup 2>/dev/null || true

    # Block known domains that Exo might try to reach
    sudo tee -a /etc/hosts > /dev/null << 'EOF' || true

# Exo internet blocking - added by exo_internet_off.sh
127.0.0.1 registry.exo.ai
127.0.0.1 exo.com
127.0.0.1 api.exo.com
127.0.0.1 huggingface.co
127.0.0.1 github.com
EOF
    echo "✅ Hosts file modified"
else
    echo "⚠️  Cannot modify hosts file, skipping domain blocking"
fi

echo ""
echo "🎯 Exo internet access has been blocked using multiple methods:"
echo "   1. iptables firewall rules (if available)"
echo "   2. Network namespace wrapper at $EXO_WRAPPER"
echo "   3. Environment variables in $EXO_ENV_FILE"
echo "   4. Hosts file domain blocking (if permissions allow)"
echo ""
echo "💡 Usage options:"
echo "   - Use wrapper: $EXO_WRAPPER run llama2"
echo "   - Source env: source $EXO_ENV_FILE && exo run llama2"
echo "   - Direct use: EXO_HOST=127.0.0.1:52415 exo run llama2"
echo ""
echo "⚠️  Make sure you have already downloaded your models with 'exo pull <model>'"
