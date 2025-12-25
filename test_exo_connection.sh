#!/bin/bash

# Test EXO Connection Script
echo "Testing connection to EXO instance..."
echo ""

EXO_URL="http://192.168.1.188:52415"
EXO_API_KEY="YOUR_API_KEY_HERE"  # Replace with your actual API key

echo "1. Testing basic connectivity (ping)..."
ping -c 3 192.168.1.188

echo ""
echo "2. Testing port connectivity..."
nc -zv 192.168.1.188 52415 2>&1 || echo "Port 52415 is not accessible"

echo ""
echo "3. Testing HTTP connection to EXO API..."
curl -v "$EXO_URL/node_id" \
  -H "Authorization: Bearer $EXO_API_KEY" \
  2>&1

echo ""
echo "4. Testing without API key (to see if it's an auth vs connection issue)..."
curl -v "$EXO_URL/node_id" 2>&1

echo ""
echo "Done!"

