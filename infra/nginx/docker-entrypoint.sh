#!/bin/sh
set -e

# Certificate paths
CERT_DIR="/etc/nginx/certs"
CERT_FILE="${CERT_DIR}/cert.pem"
KEY_FILE="${CERT_DIR}/key.pem"

# Create certificate directory if it doesn't exist
mkdir -p "${CERT_DIR}"

# Check if certificates already exist
if [ ! -f "${CERT_FILE}" ] || [ ! -f "${KEY_FILE}" ]; then
    echo "🔐 SSL certificates not found. Generating self-signed certificates for localhost..."
    
    # Generate self-signed certificate for localhost
    openssl req -x509 -newkey rsa:4096 \
        -keyout "${KEY_FILE}" \
        -out "${CERT_FILE}" \
        -sha256 -days 365 -nodes \
        -subj "/CN=localhost"
    
    echo "✅ SSL certificates generated successfully:"
    echo "   - Certificate: ${CERT_FILE}"
    echo "   - Private Key: ${KEY_FILE}"
else
    echo "✅ SSL certificates already exist. Skipping generation."
fi

# Execute the CMD (nginx)
exec "$@"

