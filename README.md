# Exo Proxy Fortress: Secure AI Cluster Management 🛡️

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![Built with](https://img.shields.io/badge/Built%20with-FastAPI-brightgreen)
![Release](https://img.shields.io/badge/release-v1.0.0-blue)
[![GitHub stars](https://img.shields.io/github/stars/ParisNeo/exo_proxy_server.svg?style=social&label=Star)](https://github.com/ParisNeo/exo_proxy_server/stargazers/)

Secure your distributed AI infrastructure. **Exo Proxy Fortress** is the ultimate security and management layer for your Exo AI clusters, designed to be set up in **60 seconds** by anyone, on any operating system.

Whether you're running a home AI cluster or managing enterprise-scale distributed inference, this tool transforms your vulnerable Exo endpoints into a managed, secure, and **deeply customizable** AI command center.

---

## Why You Need Exo Proxy Fortress

Exo enables running massive AI models across multiple devices with automatic discovery, RDMA over Thunderbolt, and tensor parallelism. However, exposing Exo clusters directly to the internet introduces significant security risks. Exo Proxy Fortress provides enterprise-grade security while preserving all the power of distributed AI inference.

### Key Benefits

*   ✨ **Centralized Cluster Management:** Monitor and manage all your Exo nodes from a single web interface. Track model distribution, device health, and cluster performance.

*   🛡️ **Rock-Solid Security:**
    *   **API Key Authentication:** Eliminate anonymous access to your AI clusters.
    *   **One-Click HTTPS/SSL:** Encrypt all traffic with easy certificate management.
    *   **IP Filtering:** Create granular allow/deny lists for cluster access control.
    *   **Rate Limiting & Brute-Force Protection:** Prevent abuse and secure your admin interface (powered by Redis).

*   🚀 **High-Performance Distributed Engine:**
    *   **Intelligent Load Balancing:** Distribute inference requests across your Exo cluster for optimal performance.
    *   **Smart Model Routing:** Automatically route requests to nodes with the required models available.
    *   **Automatic Retries:** Handle node failures gracefully with exponential backoff.

*   🧪 **Model Playgrounds & Benchmarking:**
    *   **Interactive Chat Playground:** Test your distributed models with streaming responses, multi-modal inputs, and conversation management.

*   📊 **Mission Control Dashboard:**
    *   Real-time monitoring of cluster health, device utilization, and model performance.
    *   Live load balancer status and queue monitoring.

*   📈 **Comprehensive Analytics Suite:**
    *   Interactive charts for request patterns, model usage, and cluster performance.
    *   Per-user analytics with exportable data.

*   🎨 **Radical Theming Engine:**

*   ✨ **Centralized Model Management:** Pull, update, and delete models on any of your connected Exo servers directly from the proxy's web UI. No more terminal commands or switching between machines.

*   🛡️ **Rock-Solid Security:**
    *   **Endpoint Blocking:** Prevent API key holders from accessing sensitive endpoints like `pull`, `delete`, and `create` to protect your servers from abuse.
    *   **API Key Authentication:** Eliminate anonymous access entirely.
    *   **One-Click HTTPS/SSL:** Encrypt all traffic with easy certificate uploads or path-based configuration.
    *   **IP Filtering:** Create granular allow/deny lists to control exactly which machines can connect.
    *   **Rate Limiting & Brute-Force Protection:** Prevent abuse and secure your admin login (powered by Redis).

*   🚀 **High-Performance Engine:**
    *   **Intelligent Load Balancing:** Distribute requests across multiple Exo servers for maximum speed and high availability.
    *   **Smart Model Routing:** Automatically sends requests only to servers that have the specific model available, preventing failed requests and saving compute resources.
    *   **Automatic Retries:** The proxy resiliently handles temporary server hiccups with an exponential backoff strategy, making your AI services more reliable.

*   🧪 **Model Playgrounds & Benchmarking:**
    *   **Interactive Chat Playground:** Go beyond simple API calls. Chat with any model in a familiar interface that supports streaming, multi-modal inputs (paste images directly!), and full conversation history management (import/export).

*   📊 **Mission Control Dashboard:**
    *   Real-time monitoring of your proxy's health (CPU, Memory, Disk), see all active models across all cluster nodes, monitor the **live health of your load balancer**, and watch API rate-limit queues fill and reset in real-time.

*   📈 **Comprehensive Analytics Suite:**
    *   Don't just guess your usage—know it. Dive into beautiful, interactive charts for daily and hourly requests, model popularity, and cluster load.
    *   With a single click, drill down into **per-user analytics** to understand individual usage patterns. All data is exportable to CSV or PNG.

*   🎨 **Radical Theming Engine:**
    *   Why should your tools be boring? Choose from over a dozen stunning UI themes to match your personal aesthetic. Whether you prefer a sleek **Material Design**, a futuristic **Cyberpunk** neon glow, a retro **CRT Terminal**, or a stark **Brutalist** look, you can make the interface truly yours.

*   👤 **Granular User & API Key Management:**
    *   Effortlessly create and manage users. The sortable user table gives you at-a-glance stats on key counts, total requests, and last activity.
    *   From there, manage individual API keys with per-key rate limits, and temporarily disable or re-enable keys on the fly.

*   🌐 **Multi-Node Cluster Management:**
    *   Centrally manage all your Exo cluster nodes. The proxy load-balances requests and provides a unified, federated view of all available models from all your distributed devices combined.

*   ✨ **Effortless 1-Click Setup:**
    *   No Docker, no `pip install`, no command-line wizardry required. Just download and run a single script.

---

## 🛡️ Harden Your Defenses: Endpoint Blocking

Giving every user an API key shouldn't mean giving them the keys to the kingdom. By default, **Exo Proxy Fortress blocks access to dangerous and resource-intensive API endpoints** for all API key holders.

-   **Prevent Resource Abuse:** Stop users from accessing sensitive management endpoints that could affect cluster performance.
-   **Protect Your Models:** Prevent API users from modifying model configurations on your backend nodes.
-   **Full Admin Control:** As an administrator, you can still perform all management actions securely through the web UI's **Cluster Management** page.
-   **Customizable:** You have full control to change which endpoints are blocked via the **Settings -> Endpoint Security** menu.

---

## 🔒 Encrypt Everything with One-Click HTTPS/SSL

Securing your AI traffic is now dead simple. In the **Settings -> HTTPS/SSL** menu, you have two easy options:

1.  **Upload & Go (Easiest):** Simply upload your `key.pem` and `cert.pem` files directly through the UI. The server handles the rest.
2.  **Path-Based:** If your certificates are already on the server (e.g., managed by Certbot), just provide the full file paths.

For local testing, you can generate a self-signed certificate with OpenSSL:

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj "/CN=localhost"
```

A server restart is required to apply changes, ensuring your connection is fully encrypted and secure from eavesdropping.

---

## 🚀 Quick Start

**One command (HTTP only):**

```bash
docker-compose up -d
```

This starts the app, MongoDB, and Redis (HTTP on port 8080).

**Access the web interface:** `http://localhost:8080`  
**Default admin:** `admin` / `changeme` (change this!)

### Optional: Full HTTPS with Certbot + Nginx

1) Point your DNS `A`/`AAAA` to this host (e.g., `example.com`).

2) Issue a cert (webroot):
```bash
docker-compose run --rm certbot certonly \
    --webroot -w /var/www/certbot \
    -d example.com \
    --email you@example.com --agree-tos --no-eff-email
```

3) Edit `infra/nginx/conf.d/exo.conf` and replace `example.com` with your domain (both server blocks).

4) Start/restart with Nginx TLS termination:
```bash
docker-compose up -d nginx app redis mongodb
```

5) If you want the app itself to also serve HTTPS (optional), set in settings or `.env`:
- `SSL_CERTFILE=/etc/letsencrypt/live/example.com/fullchain.pem`
- `SSL_KEYFILE=/etc/letsencrypt/live/example.com/privkey.pem`
and restart the app container.

Renewal (cron-friendly):
```bash
docker-compose run --rm certbot renew --webroot -w /var/www/certbot
docker-compose restart nginx
```

### Manual Setup (Optional)

If you want to customize settings before starting:

```bash
# 1. Run setup wizard
docker-compose run --rm app python setup_wizard.py

# 2. Start services
docker-compose up -d
```

### Native Python Installation (Advanced)

For development or custom deployments, use the legacy installer:

```bash
./run.sh
```
Then choose option 2 for native Python installation.

---

## 🔑 Using the API with API Keys

Once you've created an API key through the admin interface, you can use it to authenticate your requests to the proxy. All API requests must include your API key in the `Authorization` header using the `Bearer` token format.

### Getting Your API Key

1. Log in to the admin interface at `http://localhost:8080/admin`
2. Navigate to **Users** → Select a user → **Create API Key**
3. Copy the generated API key (you'll only see it once!)

### API Request Examples

#### Using cURL

**Chat Completion:**
```bash
curl -X POST http://localhost:8080/api/chat/completions \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "stream": false
  }'
```

#### Using Python

```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "http://localhost:8080"

# Chat completion
response = requests.post(
    f"{BASE_URL}/api/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "llama-3.2-1b",
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ],
        "stream": False
    }
)
print(response.json())
```

#### Using JavaScript/Node.js

```javascript
const API_KEY = "your_api_key_here";
const BASE_URL = "http://localhost:8080";

// Chat completion
fetch(`${BASE_URL}/api/chat/completions`, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${API_KEY}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    model: "llama-3.2-1b",
    messages: [
      { role: "user", content: "Explain quantum computing" }
    ],
    stream: false
  })
})
  .then(res => res.json())
  .then(data => console.log(data))
  .catch(err => console.error("Error:", err));
```

### Streaming Responses

For streaming responses, set `"stream": true` in your request:

```bash
curl -X POST http://localhost:8080/api/chat/completions \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

### Important Notes

- **API Key Format:** Your API key should be included exactly as generated (format: `op_prefix_secret`)
- **HTTPS:** If you've configured HTTPS, replace `http://` with `https://` in all examples
- **Rate Limiting:** Each API key has configurable rate limits set by the administrator
- **Endpoint Restrictions:** Some sensitive management endpoints may be blocked for API key users by default

### Troubleshooting API Requests

**"Method Not Allowed" Error:**
- Ensure you're using the correct HTTP method (POST for `/api/chat/completions`, etc.)
- Verify your API key is valid and active in the admin interface
- Check that the endpoint isn't blocked in **Settings → Endpoint Security**
- Try restarting the proxy server if you recently made configuration changes
- Verify the route is accessible: `curl -X GET https://proxy.local/api/models -H "Authorization: Bearer your_api_key"`

**"401 Unauthorized" Error:**
- Check that your API key is correctly formatted in the Authorization header: `Bearer your_api_key_here`
- Ensure there's a space between "Bearer" and your API key
- Verify the API key hasn't been revoked or disabled
- Make sure you're using the full API key (both prefix and secret parts)

**"403 Forbidden" Error:**
- The endpoint you're trying to access may be blocked for API key users
- Check **Settings → Endpoint Security** to see which endpoints are restricted
- Some management endpoints are blocked by default for security

**Connection Issues:**
- Verify the proxy server is running and accessible
- Check that you're using the correct base URL (including port if not using standard 80/443)
- For HTTPS, ensure your SSL certificates are properly configured

---

## Visual Showcase

### Step 1: Secure Admin Login

Log in with the secure credentials you created during setup.

![Secure Admin Login Page](assets/login.png)

### Step 2: The Command Center Dashboard

Your new mission control. Instantly see system health, active models, server status, and live rate-limit queues, all updating automatically.

![Dashboard](assets/DashBoard.gif)

### Step 3: Manage Your Servers & Models

No more SSH or terminal juggling. Add all your Exo instances, then pull, update, and delete models on any server with a few clicks.

![Server Management](assets/server_management.png)

### Step 4: Choose Your Look: The Theming Engine

Navigate to the Settings page and instantly transform the entire UI. Pick a style that matches your mood or your desktop setup.

![Theming](assets/theming.gif)

### Step 5: Manage Users & Drill Down into Analytics

The User Management page gives you a sortable, high-level overview. From here, click "View Usage" to dive into a dedicated analytics page for any specific user.

![User edit](assets/user_edit.gif)

### Step 6: Test & Benchmark in the Playgrounds

Use the built-in playground to evaluate your models. The **Chat Playground** provides a familiar UI to test conversational models with streaming and image support.

### Step 7: Master Your Analytics

The main "Usage Stats" page and the per-user pages give you a beautiful, exportable overview of exactly how your models are being used.

![API Usage Statistics](assets/stats.png)

### Step 8: Get Help When You Need It

The built-in Help page is now a rich document with a sticky table of contents that tracks your scroll position, making it effortless to find the information you need.

![Help and Credits Page](assets/help.png)

---

## 🐳 Docker Deployment (Recommended)

The easiest way to run Exo Proxy Fortress is with Docker Compose, which provides the full stack (app, MongoDB, Redis) in isolated containers.

### Quick Start
```bash
./start.sh
```

### Manual Control
```bash
# Setup (first time only)
./setup.sh

# Start services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Docker Commands
- **Start:** `docker-compose up -d`
- **Stop:** `docker-compose down`
- **Restart:** `docker-compose restart`
- **Logs:** `docker-compose logs -f app`
- **Rebuild:** `docker-compose up --build`

The setup automatically handles:
- MongoDB database initialization
- Redis caching setup
- Volume persistence for data
- Automatic service dependencies

---

## 🛠️ Development Mode (Hot Reload)

For developers who want to modify the codebase with instant feedback:

### Prerequisites
```bash
# Install development dependencies
pip install -r requirements.txt
pip install beanie motor redis

# Or with Poetry (if available)
poetry install
```

### Quick Development Start
```bash
# Use the development script with optimized hot reload
python dev.py
```

### Manual Development Start
```bash
# Activate virtual environment
source venv/bin/activate

# Start with hot reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 \
  --reload \
  --reload-delay 0.1 \
  --reload-include "*.py" \
  --reload-include "*.html" \
  --reload-include "*.css" \
  --reload-include "*.js" \
  --reload-include "*.jinja2" \
  --log-level info
```

### Hot Reload Features
- **Instant Python code reloading** - Changes to `.py` files reload automatically
- **Template hot reload** - HTML template changes reflect immediately (no caching)
- **Static file watching** - CSS/JS changes trigger reload
- **Fast reload delay** - 0.1 second response time
- **Comprehensive file watching** - Watches entire project directory

### Development Tips
- Server runs on `http://localhost:8000`
- Template changes are picked up immediately (no server restart needed)
- Use browser developer tools to clear cache if needed
- Check terminal for reload confirmation messages

---

## Resetting Your Installation (Troubleshooting)

> **WARNING: IRREVERSIBLE ACTION**
>
> The reset scripts are for troubleshooting or starting over completely. They will **PERMANENTLY DELETE** your database, configuration, and Python environment.

If you encounter critical errors or wish to perform a completely fresh installation, use the provided reset scripts.

**On Windows:**
Double-click the `reset.bat` file.

**On macOS or Linux:**
```bash
chmod +x reset.sh
./reset.sh
```

---

## Credits and Acknowledgements

The Exo Proxy was developed with passion by the open-source community. A special thank you to:

*   **[Exo](https://github.com/exo-explore/exo)** - The distributed AI inference framework that powers this proxy.
*   **ParisNeo** for creating and maintaining this project.
*   All contributors who have helped find and fix bugs.
*   The teams behind **FastAPI**, **MongoDB**, **Jinja2**, **Chart.js**, and **Tailwind CSS**.

Visit the project on [GitHub](https://github.com/ParisNeo/exo_proxy_server) to contribute, report issues, or star the repository!

---

## License

This project is licensed under the Apache License 2.0. Feel free to use, modify, and distribute.
