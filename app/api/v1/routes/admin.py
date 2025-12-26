# app/api/v1/routes/admin.py
import logging
from typing import Union, Optional, List, Dict, Any
import redis.asyncio as redis
import psutil
import shutil
import httpx
import asyncio
import secrets
from pathlib import Path
import os
from pydantic import AnyHttpUrl

from fastapi import APIRouter, Depends, Request, Form, HTTPException, status, Query, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings
from app.core.security import verify_password
from app.database.session import get_db
from app.database.models import User, UserRole, APIKey
from app.crud import user_crud, apikey_crud, log_crud, server_crud, settings_crud, model_metadata_crud
from app.schema.user import UserCreate
from app.schema.server import ServerCreate, ServerUpdate
from app.schema.settings import AppSettingsModel
from app.api.v1.dependencies import get_csrf_token, validate_csrf_token, login_rate_limiter


logger = logging.getLogger(__name__)
router = APIRouter()
# Configure Jinja2 templates with optimized settings for development
templates = Jinja2Templates(directory="app/templates")

# --- Constants for Logo Upload ---
MAX_LOGO_SIZE_MB = 2
MAX_LOGO_SIZE_BYTES = MAX_LOGO_SIZE_MB * 1024 * 1024
ALLOWED_LOGO_TYPES = ["image/png", "image/jpeg", "image/gif", "image/svg+xml", "image/webp"]
UPLOADS_DIR = Path("app/static/uploads")
SSL_DIR = Path(".ssl")


# --- Sync helper for system info (to be run in threadpool) ---
def get_system_info():
    """Returns a dictionary with system usage information."""
    psutil.cpu_percent(interval=None)
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    memory = psutil.virtual_memory()
    try:
        disk = shutil.disk_usage('/')
    except FileNotFoundError:
        # Fallback for Windows
        disk = shutil.disk_usage('C:\\')
        
    return {
        "cpu": {"percent": cpu_percent},
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "percent": round((disk.used / disk.total) * 100, 2),
        },
    }

# --- Helper for Redis Rate Limit Scan ---
async def get_active_rate_limits(
    redis_client: redis.Redis, 
    db: AsyncIOMotorDatabase, 
    settings: AppSettingsModel
) -> List[Dict[str, Any]]:
    if not redis_client:
        return []
        
    limits = []
    # Use SCAN to avoid blocking the server.
    async for key in redis_client.scan_iter("rate_limit:*"):
        try:
            pipe = redis_client.pipeline()
            pipe.get(key)
            pipe.ttl(key)
            results = await pipe.execute()
            count, ttl = results
            
            prefix = key.split(":", 1)[1]

            # Fetch API key details from DB to get the specific rate limit
            api_key = await apikey_crud.get_api_key_by_prefix(db, prefix=prefix)
            
            key_limit = settings.rate_limit_requests
            key_window = settings.rate_limit_window_minutes

            if api_key:
                if api_key.rate_limit_requests is not None:
                    key_limit = api_key.rate_limit_requests
                if api_key.rate_limit_window_minutes is not None:
                    key_window = api_key.rate_limit_window_minutes

            if count is not None and ttl is not None:
                limits.append({
                    "prefix": prefix,
                    "count": int(count),
                    "ttl_seconds": int(ttl),
                    "limit": key_limit,
                    "window_minutes": key_window
                })
        except Exception as e:
            logger.warning(f"Could not parse rate limit key {key}: {e}")
            
    # Sort by the percentage of the limit used
    def sort_key(item):
        if item['limit'] > 0:
            return item['count'] / item['limit']
        return 0
        
    return sorted(limits, key=sort_key, reverse=True)[:10]

# --- Helper to add common context to all templates ---
def get_template_context(request: Request) -> dict:
    return {
        "request": request,
        "is_redis_connected": request.app.state.redis is not None,
        "bootstrap_settings": settings
    }

def flash(request: Request, message: str, category: str = "info"):
    """
    FIX: Re-assign list to session to avoid mutation issues with modern SessionMiddleware.
    """
    messages = request.session.get("_messages", [])
    messages.append({"message": message, "category": category})
    request.session["_messages"] = messages

def get_flashed_messages(request: Request): return request.session.pop("_messages", [])
templates.env.globals["get_flashed_messages"] = get_flashed_messages
async def get_current_user_from_cookie(request: Request, db: AsyncIOMotorDatabase = Depends(get_db)) -> User | None:
    user_id = request.session.get("user_id")
    if user_id:
        user = await user_crud.get_user_by_id(db, user_id=user_id)
        return user
    return None
async def require_authenticated_user(request: Request, current_user: Union[User, None] = Depends(get_current_user_from_cookie)) -> User:
    if not current_user: raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, detail="Not authorized", headers={"Location": str(request.url_for("admin_login"))})
    request.state.user = current_user
    return current_user

async def require_admin_user(request: Request, current_user: Union[User, None] = Depends(get_current_user_from_cookie)) -> User:
    if not current_user or current_user.role != UserRole.ADMIN: raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, detail="Not authorized", headers={"Location": str(request.url_for("admin_login"))})
    request.state.user = current_user
    return current_user
    
@router.get("/login", response_class=HTMLResponse, name="admin_login")
async def admin_login_form(request: Request):
    context = get_template_context(request)
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/login.html", context)

@router.post("/login", name="admin_login_post", dependencies=[Depends(login_rate_limiter), Depends(validate_csrf_token)])
async def admin_login_post(request: Request, db: AsyncIOMotorDatabase = Depends(get_db), username: str = Form(...), password: str = Form(...)):
    user = await user_crud.get_user_by_username(db, username=username)
    
    is_valid = user and user.is_admin and verify_password(password, user.hashed_password)
    redis_client: redis.Redis = request.app.state.redis
    client_ip = request.client.host

    if not is_valid and redis_client:
        key = f"login_fail:{client_ip}"
        try:
            current_fails = await redis_client.incr(key)
            if current_fails == 1:
                await redis_client.expire(key, 300) 
        except Exception as e:
            logger.error(f"Redis failed during login attempt tracking: {e}")

    if not is_valid:
        flash(request, "Invalid username or password", "error")
        return RedirectResponse(url=request.url_for("admin_login"), status_code=status.HTTP_303_SEE_OTHER)

    if redis_client:
        await redis_client.delete(f"login_fail:{client_ip}")

    request.session["user_id"] = str(user.id)
    flash(request, "Successfully logged in.", "success")
    return RedirectResponse(url=request.url_for("admin_dashboard"), status_code=status.HTTP_303_SEE_OTHER)
    
@router.get("/logout", name="admin_logout")
async def admin_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url=request.url_for("admin_login"), status_code=status.HTTP_303_SEE_OTHER)
    
@router.get("/dashboard", response_class=HTMLResponse, name="admin_dashboard")
async def admin_dashboard(request: Request, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user)):
    context = get_template_context(request)
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/dashboard.html", context)

# --- API ENDPOINT FOR DYNAMIC DASHBOARD DATA ---
@router.get("/system-info", response_class=JSONResponse, name="admin_system_info")
async def get_system_and_exo_info(
    request: Request, 
    db: AsyncIOMotorDatabase = Depends(get_db), 
    admin_user: User = Depends(require_admin_user)
):
    http_client: httpx.AsyncClient = request.app.state.http_client
    redis_client: redis.Redis = request.app.state.redis
    app_settings: AppSettingsModel = request.app.state.settings

    # Run blocking psutil calls in a threadpool to avoid blocking the event loop
    system_info_task = run_in_threadpool(get_system_info)
    
    # Fetch active models, server health, and server load concurrently
    running_models_task = server_crud.get_active_models_all_servers(db, http_client)
    server_health_task = server_crud.check_all_servers_health(db, http_client)
    server_load_task = log_crud.get_server_load_stats(db)
    
    # Fetch rate limit info from Redis if available
    rate_limit_task = get_active_rate_limits(redis_client, db, app_settings)
    
    # Await all tasks
    (
        system_info, 
        running_models, 
        server_health, 
        server_load, 
        rate_limits
    ) = await asyncio.gather(
        system_info_task,
        running_models_task,
        server_health_task,
        server_load_task,
        rate_limit_task
    )
    
    # Combine server health and load data into a single structure
    server_load_map = {row.server_name: row.request_count for row in server_load}
    for server in server_health:
        server["request_count"] = server_load_map.get(server["name"], 0)
        
    return {
        "system_info": system_info, 
        "running_models": running_models,
        "load_balancer_status": server_health,
        "queue_status": rate_limits
    }
    
@router.get("/stats", response_class=HTMLResponse, name="admin_stats")
async def admin_stats(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: User = Depends(require_authenticated_user),
    sort_by: str = Query("request_count"),
    sort_order: str = Query("desc"),
):
    context = get_template_context(request)

    # Check if user is admin to determine which stats to show
    is_admin = current_user.role == UserRole.ADMIN
    user_id = str(current_user.id) if not is_admin else None

    if is_admin:
        # Admin sees all stats
        key_usage_stats = await log_crud.get_usage_statistics(db, sort_by=sort_by, sort_order=sort_order)
        daily_stats = await log_crud.get_daily_usage_stats(db, days=30)
        hourly_stats = await log_crud.get_hourly_usage_stats(db)
        server_stats = await log_crud.get_server_load_stats(db)
        model_stats = await log_crud.get_model_usage_stats(db)
    else:
        # Regular users see only their own stats
        key_usage_stats = await log_crud.get_usage_statistics(db, sort_by=sort_by, sort_order=sort_order, user_id=user_id)
        daily_stats = await log_crud.get_daily_usage_stats_for_user(db, user_id, days=30)
        hourly_stats = await log_crud.get_hourly_usage_stats_for_user(db, user_id)
        server_stats = await log_crud.get_server_load_stats_for_user(db, user_id)
        model_stats = await log_crud.get_model_usage_stats_for_user(db, user_id)
    context.update({
        "key_usage_stats": key_usage_stats,
        "daily_labels": [row.date.strftime('%Y-%m-%d') for row in daily_stats],
        "daily_data": [row.request_count for row in daily_stats],
        "hourly_labels": [row['hour'] for row in hourly_stats],
        "hourly_data": [row['request_count'] for row in hourly_stats],
        "server_labels": [row.server_name for row in server_stats],
        "server_data": [row.request_count for row in server_stats],
        "model_labels": [row.model_name for row in model_stats],
        "model_data": [row.request_count for row in model_stats],
        "sort_by": sort_by,
        "sort_order": sort_order,
        "is_admin_view": is_admin,
        "current_username": current_user.username,
    })
    return templates.TemplateResponse("admin/statistics.html", context)


@router.get("/help", response_class=HTMLResponse, name="admin_help")
async def admin_help_page(request: Request, admin_user: User = Depends(require_admin_user)):
    return templates.TemplateResponse("admin/help.html", get_template_context(request))

@router.get("/api-tester", response_class=HTMLResponse, name="admin_api_tester")
async def admin_api_tester_page(request: Request, admin_user: User = Depends(require_admin_user)):
    context = get_template_context(request)
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/api_tester.html", context)

# --- EXO API Test Endpoints (No EXO Authentication Required) ---

@router.post("/api-tester/node-id", name="admin_test_node_id", dependencies=[Depends(validate_csrf_token)])
async def admin_test_node_id(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None)
):
    """Test GET /node_id endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        response = await client.get(f"{base_url}/node_id", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /node_id: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/models", name="admin_test_models", dependencies=[Depends(validate_csrf_token)])
async def admin_test_models(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None)
):
    """Test GET /models endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        response = await client.get(f"{base_url}/models", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /models: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/state", name="admin_test_state", dependencies=[Depends(validate_csrf_token)])
async def admin_test_state(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None)
):
    """Test GET /state endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        response = await client.get(f"{base_url}/state", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /state: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/events", name="admin_test_events", dependencies=[Depends(validate_csrf_token)])
async def admin_test_events(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None)
):
    """Test GET /events endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        response = await client.get(f"{base_url}/events", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /events: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/placement", name="admin_test_placement", dependencies=[Depends(validate_csrf_token)])
async def admin_test_placement(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None),
    model_id: str = Form(...),
    sharding: str = Form("Pipeline"),
    instance_meta: str = Form("MlxRing"),
    min_nodes: str = Form("1")
):
    """Test GET /instance/placement endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        url = f"{base_url}/instance/placement?model_id={model_id}&sharding={sharding}&instance_meta={instance_meta}&min_nodes={min_nodes}"
        response = await client.get(url, headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /instance/placement: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/previews", name="admin_test_previews", dependencies=[Depends(validate_csrf_token)])
async def admin_test_previews(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None)
):
    """Test GET /instance/previews endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        response = await client.get(f"{base_url}/instance/previews", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /instance/previews: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/create-instance", name="admin_test_create_instance", dependencies=[Depends(validate_csrf_token)])
async def admin_test_create_instance(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None),
    instance_json: str = Form(...)
):
    """Test POST /instance endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        import json
        payload = {"instance": json.loads(instance_json)}
        response = await client.post(f"{base_url}/instance", headers=headers, json=payload, timeout=30.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except json.JSONDecodeError as e:
        return JSONResponse(content={"detail": f"Invalid JSON: {str(e)}"}, status_code=400)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /instance: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/place-instance", name="admin_test_place_instance", dependencies=[Depends(validate_csrf_token)])
async def admin_test_place_instance(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None),
    model_id: str = Form(...),
    sharding: str = Form("Pipeline"),
    instance_meta: str = Form("MlxRing"),
    min_nodes: int = Form(1)
):
    """Test POST /place_instance endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        payload = {
            "model_id": model_id,
            "sharding": sharding,
            "instance_meta": instance_meta,
            "min_nodes": min_nodes
        }
        response = await client.post(f"{base_url}/place_instance", headers=headers, json=payload, timeout=30.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /place_instance: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/get-instance", name="admin_test_get_instance", dependencies=[Depends(validate_csrf_token)])
async def admin_test_get_instance(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None),
    instance_id: str = Form(...)
):
    """Test GET /instance/{instance_id} endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        import urllib.parse
        instance_id_encoded = urllib.parse.quote(instance_id, safe='')
        response = await client.get(f"{base_url}/instance/{instance_id_encoded}", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /instance/{{id}}: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/delete-instance", name="admin_test_delete_instance", dependencies=[Depends(validate_csrf_token)])
async def admin_test_delete_instance(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None),
    instance_id: str = Form(...)
):
    """Test DELETE /instance/{instance_id} endpoint"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        import urllib.parse
        instance_id_encoded = urllib.parse.quote(instance_id, safe='')
        response = await client.delete(f"{base_url}/instance/{instance_id_encoded}", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"detail": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing DELETE /instance/{{id}}: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@router.post("/api-tester/chat", name="admin_test_chat", dependencies=[Depends(validate_csrf_token)])
async def admin_test_chat(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Form(...),
    exo_api_key: Optional[str] = Form(None),
    model: str = Form(...),
    messages: str = Form(...),
    temperature: float = Form(0.7),
    max_tokens: int = Form(100),
    stream: bool = Form(False)
):
    """Test POST /v1/chat/completions endpoint - uses same format as playground"""
    client = request.app.state.http_client
    base_url = exo_base_url.rstrip('/')
    headers = {"Content-Type": "application/json"}
    if exo_api_key:
        headers["Authorization"] = f"Bearer {exo_api_key}"
    
    try:
        import json
        import time
        messages_list = json.loads(messages)
        payload = {
            "model": model,
            "messages": messages_list,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if stream:
            # For streaming, convert EXO SSE format to playground ndjson format
            async def event_stream():
                start_time = time.monotonic()
                total_content = ""
                buffer = ""
                has_received_data = False
                connection_error = None
                
                try:
                    logger.info(f"Starting stream to EXO API: {base_url}/v1/chat/completions")
                    logger.debug(f"Payload: model={model}, stream=True, messages_count={len(messages_list)}")
                    
                    async with client.stream(
                        "POST",
                        f"{base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=600.0
                    ) as response:
                        logger.info(f"EXO API response status: {response.status_code}")
                        
                        if response.status_code != 200:
                            try:
                                error_body = await response.aread()
                                error_text = error_body.decode('utf-8', errors='replace')
                                logger.error(f"EXO API returned error {response.status_code}: {error_text}")
                                
                                # Try to parse as JSON for better error message
                                try:
                                    error_json = json.loads(error_text)
                                    error_msg = error_json.get("error", {}).get("message", error_text)
                                    if not error_msg:
                                        error_msg = str(error_json.get("error", error_text))
                                except:
                                    error_msg = error_text
                                
                                error_payload = {
                                    "error": f"EXO API error (HTTP {response.status_code}): {error_msg}",
                                    "status_code": response.status_code
                                }
                                yield (json.dumps(error_payload) + '\n').encode('utf-8')
                                return
                            except Exception as e:
                                logger.error(f"Error reading error response: {e}")
                                error_payload = {
                                    "error": f"EXO API returned status {response.status_code}",
                                    "status_code": response.status_code,
                                    "details": str(e)
                                }
                                yield (json.dumps(error_payload) + '\n').encode('utf-8')
                                return
                        
                        # Stream the response
                        try:
                            async for chunk_bytes in response.aiter_bytes():
                                if not chunk_bytes:
                                    continue
                                
                                has_received_data = True
                                chunk_text = chunk_bytes.decode('utf-8', errors='replace')
                                buffer += chunk_text
                                
                                # SSE format uses \n\n as delimiter, but we also handle \n
                                # Split by both to handle all cases
                                parts = buffer.split('\n\n')
                                if len(parts) > 1:
                                    # Process complete SSE messages
                                    for part in parts[:-1]:
                                        part = part.strip()
                                        if not part:
                                            continue
                                        
                                        # Handle multiple lines in one SSE message
                                        for line in part.split('\n'):
                                            line = line.strip()
                                            if not line:
                                                continue
                                            
                                            # Parse SSE format: "data: {...}" or "data: [DONE]"
                                            if line.startswith('data: '):
                                                data_str = line[6:].strip()  # Remove "data: " prefix
                                                
                                                if data_str == '[DONE]':
                                                    # Send final chunk with stats
                                                    end_time = time.monotonic()
                                                    final_chunk = {
                                                        "model": model,
                                                        "done": True,
                                                        "eval_count": len(total_content) // 4,  # Rough estimate
                                                        "eval_duration": int((end_time - start_time) * 1_000_000_000)
                                                    }
                                                    yield (json.dumps(final_chunk) + '\n').encode('utf-8')
                                                    continue
                                                
                                                try:
                                                    chunk_data = json.loads(data_str)
                                                    
                                                    # Check for errors in chunk
                                                    if "error" in chunk_data:
                                                        error_info = chunk_data["error"]
                                                        if isinstance(error_info, dict):
                                                            error_msg = error_info.get("message", str(error_info))
                                                        else:
                                                            error_msg = str(error_info)
                                                        logger.error(f"Error in EXO stream chunk: {error_msg}")
                                                        error_payload = {
                                                            "error": f"EXO API error: {error_msg}",
                                                            "type": "api_error"
                                                        }
                                                        yield (json.dumps(error_payload) + '\n').encode('utf-8')
                                                        continue
                                                    
                                                    # Extract content from choices[0].delta.content
                                                    content = ""
                                                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                                        choice = chunk_data["choices"][0]
                                                        delta = choice.get("delta", {})
                                                        content = delta.get("content", "")
                                                        
                                                        # Check for finish_reason (indicates end of stream)
                                                        finish_reason = choice.get("finish_reason")
                                                        if finish_reason:
                                                            end_time = time.monotonic()
                                                            usage = chunk_data.get("usage", {})
                                                            final_chunk = {
                                                                "model": chunk_data.get("model", model),
                                                                "done": True,
                                                                "eval_count": usage.get("completion_tokens", len(total_content) // 4),
                                                                "eval_duration": int((end_time - start_time) * 1_000_000_000)
                                                            }
                                                            yield (json.dumps(final_chunk) + '\n').encode('utf-8')
                                                        
                                                        # Also check for error in choice
                                                        if "error" in choice:
                                                            error_msg = choice["error"].get("message", str(choice["error"]))
                                                            error_payload = {
                                                                "error": f"EXO API error: {error_msg}",
                                                                "type": "api_error"
                                                            }
                                                            yield (json.dumps(error_payload) + '\n').encode('utf-8')
                                                            continue
                                                    
                                                    if content:
                                                        total_content += content
                                                        # Format as playground expects
                                                        formatted_chunk = {
                                                            "model": chunk_data.get("model", model),
                                                            "message": {
                                                                "role": "assistant",
                                                                "content": content
                                                            },
                                                            "done": False
                                                        }
                                                        yield (json.dumps(formatted_chunk) + '\n').encode('utf-8')
                                                
                                                except json.JSONDecodeError as e:
                                                    logger.warning(f"Could not parse EXO stream chunk (length: {len(data_str)}): {data_str[:200]}... Error: {e}")
                                                    # Don't yield error for parse failures, just log and continue
                                                    continue
                                    
                                    # Keep the last incomplete part
                                    buffer = parts[-1]
                                else:
                                    # No complete SSE message yet, keep buffering
                                    pass
                            
                            # Handle any remaining buffer after stream ends
                            if buffer.strip():
                                buffer = buffer.strip()
                                for line in buffer.split('\n'):
                                    line = line.strip()
                                    if line.startswith('data: '):
                                        data_str = line[6:].strip()
                                        if data_str and data_str != '[DONE]':
                                            try:
                                                chunk_data = json.loads(data_str)
                                                if "error" in chunk_data:
                                                    error_info = chunk_data["error"]
                                                    error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                                                    error_payload = {
                                                        "error": f"EXO API error: {error_msg}",
                                                        "type": "api_error"
                                                    }
                                                    yield (json.dumps(error_payload) + '\n').encode('utf-8')
                                            except json.JSONDecodeError:
                                                pass
                            
                            # If we received data but no content, send a warning
                            if has_received_data and not total_content:
                                logger.warning("Received stream data but no content was extracted")
                                end_time = time.monotonic()
                                final_chunk = {
                                    "model": model,
                                    "done": True,
                                    "eval_count": 0,
                                    "eval_duration": int((end_time - start_time) * 1_000_000_000)
                                }
                                yield (json.dumps(final_chunk) + '\n').encode('utf-8')
                        
                        except httpx.StreamError as e:
                            logger.error(f"Stream error from EXO API: {e}", exc_info=True)
                            connection_error = f"Stream connection error: {str(e)}"
                            error_payload = {
                                "error": connection_error,
                                "type": "stream_error",
                                "details": "The connection to EXO API was interrupted during streaming."
                            }
                            yield (json.dumps(error_payload) + '\n').encode('utf-8')
                
                except httpx.ConnectError as e:
                    logger.error(f"Connection error to EXO API: {e}")
                    error_payload = {
                        "error": f"Cannot connect to EXO API at {base_url}",
                        "type": "connection_error",
                        "details": f"Please check that the EXO instance is running and accessible. Error: {str(e)}"
                    }
                    yield (json.dumps(error_payload) + '\n').encode('utf-8')
                except httpx.TimeoutException as e:
                    logger.error(f"Timeout error to EXO API: {e}")
                    error_payload = {
                        "error": f"Request to EXO API timed out",
                        "type": "timeout_error",
                        "details": f"The request took longer than 600 seconds. Error: {str(e)}"
                    }
                    yield (json.dumps(error_payload) + '\n').encode('utf-8')
                except httpx.RequestError as e:
                    logger.error(f"Request error to EXO API: {e}", exc_info=True)
                    error_payload = {
                        "error": f"Request failed: {str(e)}",
                        "type": "request_error",
                        "details": "Failed to make request to EXO API. Check network connectivity and EXO instance status."
                    }
                    yield (json.dumps(error_payload) + '\n').encode('utf-8')
                except Exception as e:
                    logger.error(f"Unexpected error streaming from EXO API: {e}", exc_info=True)
                    error_payload = {
                        "error": f"Unexpected error: {str(e)}",
                        "type": "unexpected_error",
                        "details": "An unexpected error occurred while streaming. Check server logs for details."
                    }
                    yield (json.dumps(error_payload) + '\n').encode('utf-8')
            
            return StreamingResponse(event_stream(), media_type="application/x-ndjson")
        else:
            # Non-streaming mode
            # EXO always returns SSE format, so we need to collect all chunks and assemble them
            try:
                async with client.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0
                ) as response:
                    # Check for errors
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        error_str = error_text.decode()
                        logger.error(f"EXO API error {response.status_code}: {error_str}")
                        
                        # Try to parse JSON error response
                        try:
                            error_json = json.loads(error_str)
                            error_detail = error_json.get('detail', error_str)
                        except:
                            error_detail = error_str
                        
                        return JSONResponse(content={"error": error_detail}, status_code=response.status_code)
                    
                    # Client wants a single response - collect all chunks from SSE stream
                    chunks = []
                    full_content = ""
                    last_response = None
                    buffer = ""
                    
                    async for chunk in response.aiter_bytes():
                        chunk_text = chunk.decode('utf-8', errors='replace')
                        buffer += chunk_text
                        
                        # Process complete SSE messages (separated by \n\n or \n)
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                            
                            # Parse SSE format: "data: {...}" or "data: [DONE]"
                            if line.startswith('data: '):
                                json_str = line[6:].strip()  # Remove "data: " prefix
                                
                                if json_str == '[DONE]':
                                    # Stream is complete
                                    continue
                                
                                try:
                                    chunk_data = json.loads(json_str)
                                    chunks.append(chunk_data)
                                    
                                    # Extract content from delta chunks
                                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                        choice = chunk_data['choices'][0]
                                        
                                        # Check for content in delta
                                        if 'delta' in choice and 'content' in choice['delta']:
                                            content = choice['delta'].get('content', '')
                                            if content:
                                                full_content += content
                                        
                                        # Check for finish_reason (indicates final chunk)
                                        if 'finish_reason' in choice:
                                            # This is the final chunk with metadata
                                            last_response = chunk_data
                                        elif not last_response:
                                            # Store as potential last response if no finish_reason seen yet
                                            last_response = chunk_data
                                    
                                    # Also check if this chunk has usage info (might be in a separate chunk)
                                    if 'usage' in chunk_data and not last_response:
                                        last_response = chunk_data
                                        
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse SSE chunk: {json_str[:100]}... Error: {e}")
                                    continue
                    
                    # Process any remaining buffer
                    if buffer.strip():
                        for line in buffer.strip().split('\n'):
                            line = line.strip()
                            if line.startswith('data: '):
                                json_str = line[6:].strip()
                                if json_str and json_str != '[DONE]':
                                    try:
                                        chunk_data = json.loads(json_str)
                                        chunks.append(chunk_data)
                                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                            choice = chunk_data['choices'][0]
                                            if 'delta' in choice and 'content' in choice['delta']:
                                                content = choice['delta'].get('content', '')
                                                if content:
                                                    full_content += content
                                            if not last_response or 'finish_reason' in choice:
                                                last_response = chunk_data
                                    except json.JSONDecodeError:
                                        pass
                    
                    # Build a non-streaming response using OpenAI format
                    if last_response:
                        # Get finish_reason from the chunk that has it, or default to "stop"
                        finish_reason = "stop"
                        if 'choices' in last_response and len(last_response['choices']) > 0:
                            finish_reason = last_response['choices'][0].get('finish_reason', 'stop')
                        
                        response_data = {
                            "id": last_response.get("id", f"chatcmpl-{int(time.time())}"),
                            "object": "chat.completion",
                            "created": last_response.get("created", int(time.time())),
                            "model": last_response.get("model", payload.get("model", "")),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": full_content
                                    },
                                    "finish_reason": finish_reason
                                }
                            ],
                            "usage": last_response.get("usage", {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            })
                        }
                        return JSONResponse(content=response_data, status_code=200)
                    else:
                        # No chunks received - return error
                        return JSONResponse(
                            content={"error": "No response chunks received from EXO API"},
                            status_code=500
                        )
            except httpx.RequestError as e:
                logger.error(f"Request error to EXO API: {e}")
                return JSONResponse(
                    content={"error": f"Failed to connect to EXO API: {str(e)}"},
                    status_code=503
                )
    
    except json.JSONDecodeError as e:
        return JSONResponse(content={"error": f"Invalid JSON in messages: {str(e)}"}, status_code=400)
    except httpx.RequestError as e:
        return JSONResponse(
            content={"error": f"Failed to connect to EXO API: {str(e)}"},
            status_code=503
        )
    except Exception as e:
        logger.error(f"Error testing /v1/chat/completions: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/servers", response_class=HTMLResponse, name="admin_servers")
async def admin_server_management(request: Request, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user)):
    context = get_template_context(request)
    context["servers"] = await server_crud.get_servers(db)
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/servers.html", context)

@router.post("/servers/add", name="admin_add_server", dependencies=[Depends(validate_csrf_token)])
async def admin_add_server(
    request: Request, 
    db: AsyncIOMotorDatabase = Depends(get_db), 
    admin_user: User = Depends(require_admin_user), 
    server_name: str = Form(...), 
    server_url: str = Form(...), 
    server_type: str = Form(...),
    api_key: Optional[str] = Form(None)
):
    existing_server = await server_crud.get_server_by_url(db, url=server_url)
    if existing_server:
        flash(request, f"Server with URL '{server_url}' already exists.", "error")
    else:
        try:
            server_in = ServerCreate(name=server_name, url=server_url, server_type=server_type, api_key=api_key)
            new_server = await server_crud.create_server(db, server=server_in)
            
            # For EXO servers, fetch state and models immediately to validate connection
            if server_type == 'exo':
                http_client: httpx.AsyncClient = request.app.state.http_client
                try:
                    # Fetch state to validate connection
                    state_result = await server_crud.fetch_exo_server_state(http_client, new_server)
                    if state_result["success"]:
                        logger.info(f"Successfully connected to EXO server '{server_name}' and fetched state")
                    else:
                        logger.warning(f"Could not fetch state from EXO server '{server_name}': {state_result.get('error', 'Unknown error')}")
                    
                    # Fetch available models
                    models_result = await server_crud.fetch_and_update_models(db, str(new_server.id))
                    if models_result["success"]:
                        model_count = len(models_result["models"])
                        flash(request, f"Server '{server_name}' ({server_type}) added successfully. Fetched {model_count} model(s).", "success")
                    else:
                        flash(request, f"Server '{server_name}' ({server_type}) added successfully, but could not fetch models: {models_result.get('error', 'Unknown error')}", "warning")
                except Exception as e:
                    logger.error(f"Error fetching EXO details for server '{server_name}': {e}")
                    flash(request, f"Server '{server_name}' ({server_type}) added successfully, but could not fetch EXO details: {str(e)}", "warning")
            else:
                # For other server types, try to fetch models
                try:
                    models_result = await server_crud.fetch_and_update_models(db, str(new_server.id))
                    if models_result["success"]:
                        model_count = len(models_result["models"])
                        flash(request, f"Server '{server_name}' ({server_type}) added successfully. Fetched {model_count} model(s).", "success")
                    else:
                        flash(request, f"Server '{server_name}' ({server_type}) added successfully, but could not fetch models: {models_result.get('error', 'Unknown error')}", "warning")
                except Exception as e:
                    logger.error(f"Error fetching models for server '{server_name}': {e}")
                    flash(request, f"Server '{server_name}' ({server_type}) added successfully.", "success")
        except Exception as e:
            logger.error(f"Error adding server: {e}")
            flash(request, "Invalid URL format or server type.", "error")
    return RedirectResponse(url=request.url_for("admin_servers"), status_code=status.HTTP_303_SEE_OTHER)

@router.post("/servers/{server_id}/delete", name="admin_delete_server", dependencies=[Depends(validate_csrf_token)])
async def admin_delete_server(request: Request, server_id: str, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user)):
    await server_crud.delete_server(db, server_id=server_id)
    flash(request, "Server deleted successfully.", "success")
    return RedirectResponse(url=request.url_for("admin_servers"), status_code=status.HTTP_303_SEE_OTHER)

@router.post("/servers/{server_id}/refresh-models", name="admin_refresh_models", dependencies=[Depends(validate_csrf_token)])
async def admin_refresh_models(request: Request, server_id: str, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user)):
    result = await server_crud.fetch_and_update_models(db, server_id=server_id)
    if result["success"]:
        model_count = len(result["models"])
        flash(request, f"Successfully fetched {model_count} model(s) from server.", "success")
    else:
        flash(request, f"Failed to fetch models: {result['error']}", "error")
    return RedirectResponse(url=request.url_for("admin_servers"), status_code=status.HTTP_303_SEE_OTHER)

@router.get("/servers/{server_id}/edit", response_class=HTMLResponse, name="admin_edit_server_form")
async def admin_edit_server_form(request: Request, server_id: str, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user)):
    server = await server_crud.get_server_by_id(db, server_id=server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    context = get_template_context(request)
    context["server"] = server
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/edit_server.html", context)

@router.post("/servers/{server_id}/edit", name="admin_edit_server_post", dependencies=[Depends(validate_csrf_token)])
async def admin_edit_server_post(
    request: Request,
    server_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    name: str = Form(...),
    url: str = Form(...),
    server_type: str = Form(...),
    api_key: Optional[str] = Form(None),
    remove_api_key: Optional[bool] = Form(False)
):
    update_data = {"name": name, "url": url, "server_type": server_type}

    if remove_api_key:
        update_data["api_key"] = ""
    elif api_key is not None and api_key != "":
        update_data["api_key"] = api_key

    server_update = ServerUpdate(**update_data)
    
    updated_server = await server_crud.update_server(db, server_id=server_id, server_update=server_update)
    if not updated_server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # For EXO servers, fetch state and models after update to refresh connection
    if server_type == 'exo':
        http_client: httpx.AsyncClient = request.app.state.http_client
        try:
            # Fetch state to validate connection
            state_result = await server_crud.fetch_exo_server_state(http_client, updated_server)
            if state_result["success"]:
                logger.info(f"Successfully connected to EXO server '{name}' and fetched state after update")
            else:
                logger.warning(f"Could not fetch state from EXO server '{name}': {state_result.get('error', 'Unknown error')}")
            
            # Fetch available models
            models_result = await server_crud.fetch_and_update_models(db, server_id)
            if models_result["success"]:
                model_count = len(models_result["models"])
                flash(request, f"Server '{name}' updated successfully. Fetched {model_count} model(s).", "success")
            else:
                flash(request, f"Server '{name}' updated successfully, but could not fetch models: {models_result.get('error', 'Unknown error')}", "warning")
        except Exception as e:
            logger.error(f"Error fetching EXO details for server '{name}': {e}")
            flash(request, f"Server '{name}' updated successfully, but could not fetch EXO details: {str(e)}", "warning")
    else:
        # For other server types, try to fetch models
        try:
            models_result = await server_crud.fetch_and_update_models(db, server_id)
            if models_result["success"]:
                model_count = len(models_result["models"])
                flash(request, f"Server '{name}' updated successfully. Fetched {model_count} model(s).", "success")
            else:
                flash(request, f"Server '{name}' updated successfully, but could not fetch models: {models_result.get('error', 'Unknown error')}", "warning")
        except Exception as e:
            logger.error(f"Error fetching models for server '{name}': {e}")
            flash(request, f"Server '{name}' updated successfully.", "success")
    
    return RedirectResponse(url=request.url_for("admin_servers"), status_code=status.HTTP_303_SEE_OTHER)


# --- NEW SERVER MODEL MANAGEMENT ROUTES ---

@router.get("/servers/{server_id}/manage", response_class=HTMLResponse, name="admin_manage_server_models")
async def admin_manage_server_models(
    request: Request,
    server_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user)
):
    server = await server_crud.get_server_by_id(db, server_id=server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    context = get_template_context(request)
    context["server"] = server
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/manage_server.html", context)

@router.post("/servers/{server_id}/pull", name="admin_pull_model", dependencies=[Depends(validate_csrf_token)])
async def admin_pull_model(
    request: Request,
    server_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    model_name: str = Form(...)
):
    server = await server_crud.get_server_by_id(db, server_id=server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    flash(request, f"Pull initiated for '{model_name}'. This may take several minutes...", "info")
    
    http_client: httpx.AsyncClient = request.app.state.http_client
    result = await server_crud.pull_model_on_server(http_client, server, model_name)

    if result["success"]:
        flash(request, result["message"], "success")
        # Refresh the model list in the proxy's database after a successful pull
        await server_crud.fetch_and_update_models(db, server_id=server_id)
    else:
        flash(request, result["message"], "error")
        
    return RedirectResponse(url=request.url_for("admin_manage_server_models", server_id=server_id), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/servers/{server_id}/delete-model", name="admin_delete_model", dependencies=[Depends(validate_csrf_token)])
async def admin_delete_model(
    request: Request,
    server_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    model_name: str = Form(...)
):
    server = await server_crud.get_server_by_id(db, server_id=server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    http_client: httpx.AsyncClient = request.app.state.http_client
    result = await server_crud.delete_model_on_server(http_client, server, model_name)

    if result["success"]:
        flash(request, result["message"], "success")
        # Refresh the model list in the proxy's database after a successful delete
        await server_crud.fetch_and_update_models(db, server_id=server_id)
    else:
        flash(request, result["message"], "error")

    return RedirectResponse(url=request.url_for("admin_manage_server_models", server_id=server_id), status_code=status.HTTP_303_SEE_OTHER)

@router.post("/servers/{server_id}/load-model", name="admin_load_model", dependencies=[Depends(validate_csrf_token)])
async def admin_load_model(
    request: Request,
    server_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    model_name: str = Form(...)
):
    server = await server_crud.get_server_by_id(db, server_id=server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    http_client: httpx.AsyncClient = request.app.state.http_client
    result = await server_crud.load_model_on_server(http_client, server, model_name)

    flash(request, result["message"], "success" if result["success"] else "error")
    
    return RedirectResponse(url=request.url_for("admin_dashboard"), status_code=status.HTTP_303_SEE_OTHER)

@router.post("/servers/{server_id}/unload-model", name="admin_unload_model", dependencies=[Depends(validate_csrf_token)])
async def admin_unload_model(
    request: Request,
    server_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    model_name: str = Form(...)
):
    server = await server_crud.get_server_by_id(db, server_id=server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    http_client: httpx.AsyncClient = request.app.state.http_client
    result = await server_crud.unload_model_on_server(http_client, server, model_name)

    flash(request, result["message"], "success" if result["success"] else "error")
    
    return RedirectResponse(url=request.url_for("admin_dashboard"), status_code=status.HTTP_303_SEE_OTHER)

# --- NEW: Unload model from Dashboard ---
@router.post("/models/unload", name="admin_unload_model_dashboard", dependencies=[Depends(validate_csrf_token)])
async def admin_unload_model_dashboard(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    model_name: str = Form(...),
    server_name: str = Form(...)
):
    server = await server_crud.get_server_by_name(db, name=server_name)
    if not server:
        flash(request, f"Server '{server_name}' not found.", "error")
        return RedirectResponse(url=request.url_for("admin_dashboard"), status_code=status.HTTP_303_SEE_OTHER)

    http_client: httpx.AsyncClient = request.app.state.http_client
    result = await server_crud.unload_model_on_server(http_client, server, model_name)

    flash(request, result["message"], "success" if result["success"] else "error")
    
    await asyncio.sleep(1) # Give backend a moment to update state before reloading
    
    return RedirectResponse(url=request.url_for("admin_dashboard"), status_code=status.HTTP_303_SEE_OTHER)

# --- MODELS MANAGER ROUTES (NEW) ---
@router.get("/models-manager", response_class=HTMLResponse, name="admin_models_manager")
async def admin_models_manager_page(
    request: Request, 
    db: AsyncIOMotorDatabase = Depends(get_db), 
    admin_user: User = Depends(require_admin_user)
):
    context = get_template_context(request)
    
    # Ensure metadata exists for all discovered models
    all_model_names = await server_crud.get_all_available_model_names(db)
    for model_name in all_model_names:
        await model_metadata_crud.get_or_create_metadata(db, model_name=model_name)
        
    context["metadata_list"] = await model_metadata_crud.get_all_metadata(db)
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/models_manager.html", context)

@router.post("/models-manager/update", name="admin_update_model_metadata", dependencies=[Depends(validate_csrf_token)])
async def admin_update_model_metadata(
    request: Request, 
    db: AsyncIOMotorDatabase = Depends(get_db), 
    admin_user: User = Depends(require_admin_user)
):
    form_data = await request.form()
    
    # A set to keep track of which models were in the form
    updated_model_ids = set()
    
    # Loop through form data to find metadata fields
    for key, value in form_data.items():
        if key.startswith("description_"):
            meta_id = int(key.split("_")[1])
            updated_model_ids.add(meta_id)
            
    # Now process each model found in the form
    for meta_id in updated_model_ids:
        metadata = await db.get(model_metadata_crud.ModelMetadata, meta_id)
        if metadata:
            update_data = {
                "description": form_data.get(f"description_{meta_id}", "").strip(),
                "supports_images": f"supports_images_{meta_id}" in form_data,
                "is_code_model": f"is_code_model_{meta_id}" in form_data,
                "is_fast_model": f"is_fast_model_{meta_id}" in form_data,
                "priority": int(form_data.get(f"priority_{meta_id}", 10)),
            }
            await model_metadata_crud.update_metadata(db, model_name=metadata.model_name, **update_data)

    flash(request, "Model metadata updated successfully.", "success")
    return RedirectResponse(url=request.url_for("admin_models_manager"), status_code=status.HTTP_303_SEE_OTHER)


@router.get("/settings", response_class=HTMLResponse, name="admin_settings")
async def admin_settings_form(request: Request, admin_user: User = Depends(require_admin_user)):
    context = get_template_context(request)
    app_settings: AppSettingsModel = request.app.state.settings
    context["settings"] = app_settings
    context["themes"] = app_settings.available_themes # Pass themes to template
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/settings.html", context)


@router.post("/settings", name="admin_settings_post", dependencies=[Depends(validate_csrf_token)])
async def admin_settings_post(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    logo_file: UploadFile = File(None),
    ssl_key_file: UploadFile = File(None),
    ssl_cert_file: UploadFile = File(None)
):
    current_settings: AppSettingsModel = request.app.state.settings
    form_data = await request.form()
    
    # --- Create a dictionary to hold the final updated values ---
    update_data = {}

    # --- Handle Logo Logic ---
    final_logo_url = current_settings.branding_logo_url
    is_uploaded_logo = final_logo_url and final_logo_url.startswith("/static/uploads/")

    if form_data.get("remove_logo"):
        if is_uploaded_logo:
            logo_to_remove = Path("app" + final_logo_url)
            if logo_to_remove.exists(): os.remove(logo_to_remove)
        final_logo_url = None
        flash(request, "Logo removed successfully.", "success")
    elif logo_file and logo_file.filename:
        # (Validation logic for logo file remains the same)
        file_ext = Path(logo_file.filename).suffix
        secure_filename = f"{secrets.token_hex(16)}{file_ext}"
        save_path = UPLOADS_DIR / secure_filename
        try:
            with open(save_path, "wb") as buffer: shutil.copyfileobj(logo_file.file, buffer)
            if is_uploaded_logo:
                old_logo_path = Path("app" + current_settings.branding_logo_url)
                if old_logo_path.exists(): os.remove(old_logo_path)
            final_logo_url = f"/static/uploads/{secure_filename}"
            flash(request, "New logo uploaded successfully.", "success")
        except Exception as e:
            logger.error(f"Failed to save uploaded logo: {e}")
            flash(request, f"Error saving logo: {e}", "error")
    else:
        final_logo_url = form_data.get("branding_logo_url")
    update_data["branding_logo_url"] = final_logo_url

    # --- Handle SSL File Logic ---
    # Helper function to process SSL file uploads
    async def process_ssl_file(
        file_upload: UploadFile, 
        current_path: Optional[str],
        current_content: Optional[str],
        remove_flag: bool,
        file_type: str # 'key' or 'cert'
    ) -> (Optional[str], Optional[str]):
        
        managed_filename = f"uploaded_{file_type}.pem"
        managed_path = SSL_DIR / managed_filename

        # Priority 1: Removal
        if remove_flag:
            if managed_path.exists():
                os.remove(managed_path)
            flash(request, f"Uploaded SSL {file_type} file removed.", "success")
            return None, None

        # Priority 2: New Upload
        if file_upload and file_upload.filename:
            try:
                content_bytes = await file_upload.read()
                content_str = content_bytes.decode('utf-8')
                with open(managed_path, "w") as f:
                    f.write(content_str)
                flash(request, f"New SSL {file_type} file uploaded successfully.", "success")
                return str(managed_path), content_str
            except Exception as e:
                logger.error(f"Failed to save uploaded SSL {file_type} file: {e}")
                flash(request, f"Error saving SSL {file_type} file: {e}", "error")
                return current_path, current_content # Revert on error

        # Priority 3: Path from form input
        form_path = form_data.get(f"ssl_{file_type}file")
        if form_path != current_path:
             # If a path is specified, it overrides any uploaded file
            if managed_path.exists():
                os.remove(managed_path)
            return form_path, None

        # No changes
        return current_path, current_content

    update_data["ssl_keyfile"], update_data["ssl_keyfile_content"] = await process_ssl_file(
        ssl_key_file, current_settings.ssl_keyfile, current_settings.ssl_keyfile_content,
        bool(form_data.get("remove_ssl_key")), "key"
    )
    update_data["ssl_certfile"], update_data["ssl_certfile_content"] = await process_ssl_file(
        ssl_cert_file, current_settings.ssl_certfile, current_settings.ssl_certfile_content,
        bool(form_data.get("remove_ssl_cert")), "cert"
    )
    
    # --- Update other settings ---
    # (This logic remains the same)
    selected_theme = form_data.get("selected_theme", current_settings.selected_theme)
    ui_style = form_data.get("ui_style", current_settings.ui_style)
    new_redis_password = form_data.get("redis_password")
    
    update_data.update({
        "branding_title": form_data.get("branding_title"),
        "ui_style": ui_style,
        "selected_theme": selected_theme,
        "redis_host": form_data.get("redis_host"),
        "redis_port": int(form_data.get("redis_port", 6379)),
        "redis_username": form_data.get("redis_username") or None,
        "model_update_interval_minutes": int(form_data.get("model_update_interval_minutes", 10)),
        "allowed_ips": form_data.get("allowed_ips", ""),
        "denied_ips": form_data.get("denied_ips", ""),
        "blocked_exo_endpoints": form_data.get("blocked_exo_endpoints", ""),
    })
    if new_redis_password:
        update_data["redis_password"] = new_redis_password
        
    try:
        updated_settings_data = current_settings.model_copy(update=update_data)
        await settings_crud.update_app_settings(db, settings_data=updated_settings_data)
        request.app.state.settings = updated_settings_data
        flash(request, "Settings updated successfully. A restart is required for some changes (like HTTPS) to take effect.", "success")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid form data for settings: {e}")
        flash(request, "Error: Invalid data provided for a setting (e.g., a port number was not a number).", "error")
    except Exception as e:
        logger.error(f"Failed to update settings: {e}", exc_info=True)
        flash(request, "An unexpected error occurred while saving settings.", "error")

    return RedirectResponse(url=request.url_for("admin_settings"), status_code=status.HTTP_303_SEE_OTHER)

# --- USER MANAGEMENT ROUTES ---

@router.get("/users", response_class=HTMLResponse, name="admin_users")
async def admin_user_management(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    sort_by: str = Query("username"),
    sort_order: str = Query("asc"),
):
    context = get_template_context(request)
    context["users"] = await user_crud.get_users(db, sort_by=sort_by, sort_order=sort_order)
    context["csrf_token"] = await get_csrf_token(request)
    context["sort_by"] = sort_by
    context["sort_order"] = sort_order
    return templates.TemplateResponse("admin/users.html", context)

@router.post("/users", name="create_new_user", dependencies=[Depends(validate_csrf_token)])
async def create_new_user(request: Request, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user), username: str = Form(...), password: str = Form(...)):
    try:
        existing_user = await user_crud.get_user_by_username(db, username=username)
        if existing_user:
            flash(request, f"User '{username}' already exists.", "error")
        else:
            user_in = UserCreate(username=username, password=password)
            await user_crud.create_user(db, user=user_in)
            flash(request, f"User '{username}' created successfully.", "success")
    except ValueError as e:
        flash(request, str(e), "error")
    except Exception as e:
        flash(request, f"Failed to create user: {str(e)}", "error")

    return RedirectResponse(url=request.url_for("admin_users"), status_code=status.HTTP_303_SEE_OTHER)

@router.get("/users/{user_id}/edit", response_class=HTMLResponse, name="admin_edit_user_form")
async def admin_edit_user_form(request: Request, user_id: str, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user)):
    user = await user_crud.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    context = get_template_context(request)
    context["user"] = user
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/edit_user.html", context)

@router.post("/users/{user_id}/edit", name="admin_edit_user_post", dependencies=[Depends(validate_csrf_token)])
async def admin_edit_user_post(
    request: Request,
    user_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    username: str = Form(...),
    password: Optional[str] = Form(None),
    role: str = Form(...)
):
    # Check if the new username is already taken by another user
    existing_user = await user_crud.get_user_by_username(db, username=username)
    if existing_user and existing_user.id != user_id:
        flash(request, f"Username '{username}' is already taken.", "error")
        return RedirectResponse(url=request.url_for("admin_edit_user_form", user_id=user_id), status_code=status.HTTP_303_SEE_OTHER)

    # Convert role string to UserRole enum
    try:
        user_role = UserRole(role)
    except ValueError:
        flash(request, f"Invalid role '{role}'.", "error")
        return RedirectResponse(url=request.url_for("admin_edit_user_form", user_id=user_id), status_code=status.HTTP_303_SEE_OTHER)

    updated_user = await user_crud.update_user(db, user_id=user_id, username=username, password=password, role=user_role)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    flash(request, f"User '{username}' updated successfully.", "success")
    return RedirectResponse(url=request.url_for("admin_users"), status_code=status.HTTP_303_SEE_OTHER)

@router.get("/users/{user_id}", response_class=HTMLResponse, name="get_user_details")
async def get_user_details(request: Request, user_id: str, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user)):
    context = get_template_context(request)
    user = await user_crud.get_user_by_id(db, user_id=user_id)
    if not user: 
        raise HTTPException(status_code=404, detail="User not found")
    
    context["user"] = user
    api_keys = await apikey_crud.get_api_keys_for_user(db, user_id=user_id)
    
    logger.info(f"User {user.username} ({user_id}): Found {len(api_keys)} API keys")
    logger.info(f"API keys object type: {type(api_keys)}")
    
    if api_keys:
        logger.info(f"First key details: ID={api_keys[0].id}, Name={api_keys[0].key_name}, Type={type(api_keys[0])}")
        for key in api_keys:
            logger.info(f"  - Key: {key.key_name} ({key.key_prefix}), ID: {key.id}, Active: {key.is_active}, Revoked: {key.is_revoked}")
    else:
        logger.info("No API keys found in result")
    
    context["api_keys"] = api_keys
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/user_details.html", context)

@router.get("/users/{user_id}/stats", response_class=HTMLResponse, name="admin_user_stats")
async def admin_user_stats(
    request: Request,
    user_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    sort_by: str = Query("request_count"),
    sort_order: str = Query("desc"),
):
    user = await user_crud.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    context = get_template_context(request)
    
    # Fetch all statistics including key usage
    key_usage_stats = await log_crud.get_usage_statistics(db, sort_by=sort_by, sort_order=sort_order, user_id=user_id)
    daily_stats = await log_crud.get_daily_usage_stats_for_user(db, user_id=user_id, days=30)
    hourly_stats = await log_crud.get_hourly_usage_stats_for_user(db, user_id=user_id)
    server_stats = await log_crud.get_server_load_stats_for_user(db, user_id=user_id)
    model_stats = await log_crud.get_model_usage_stats_for_user(db, user_id=user_id)

    context.update({
        "user": user,
        "key_usage_stats": key_usage_stats,
        "daily_labels": [row.date.strftime('%Y-%m-%d') for row in daily_stats],
        "daily_data": [row.request_count for row in daily_stats],
        "hourly_labels": [row['hour'] for row in hourly_stats],
        "hourly_data": [row['request_count'] for row in hourly_stats],
        "server_labels": [row.server_name for row in server_stats if row.server_name],
        "server_data": [row.request_count for row in server_stats if row.server_name],
        "model_labels": [row.model_name for row in model_stats],
        "model_data": [row.request_count for row in model_stats],
        "sort_by": sort_by,
        "sort_order": sort_order,
    })
    
    # Create a new template for this
    return templates.TemplateResponse("admin/user_statistics.html", context)

@router.post("/users/{user_id}/keys/create", name="admin_create_key", dependencies=[Depends(validate_csrf_token)])
async def create_user_api_key(
    request: Request,
    user_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    key_name: str = Form(...),
    rate_limit_requests: Optional[int] = Form(None),
    rate_limit_window_minutes: Optional[int] = Form(None),
):
    try:
        logger.info(f"Creating API key '{key_name}' for user {user_id}")
        
        # --- FIX: Check for existing key with the same name for this user ---
        existing_key = await apikey_crud.get_api_key_by_name_and_user_id(db, key_name=key_name, user_id=user_id)
        if existing_key:
            flash(request, f"An API key with the name '{key_name}' already exists for this user.", "error")
            return RedirectResponse(url=request.url_for("get_user_details", user_id=user_id), status_code=status.HTTP_303_SEE_OTHER)
        # --- END FIX ---
        
        plain_key, db_api_key = await apikey_crud.create_api_key(
            db, 
            user_id=user_id, 
            key_name=key_name,
            rate_limit_requests=rate_limit_requests,
            rate_limit_window_minutes=rate_limit_window_minutes
        )
        
        logger.info(f"API key created successfully: {db_api_key.key_prefix} (ID: {db_api_key.id})")
        
        # Verify the key was saved by fetching it back
        saved_key = await apikey_crud.get_api_key_by_id(db, str(db_api_key.id))
        if not saved_key:
            logger.error(f"Failed to retrieve newly created key {db_api_key.id}")
            flash(request, "Error: API key was created but could not be verified. Please check the database.", "error")
        else:
            logger.info(f"Verified key exists in database: {saved_key.key_prefix}")
        
        context = get_template_context(request)
        context["plain_key"] = plain_key
        context["user_id"] = user_id
        request.state.user = admin_user  # Admin user is already loaded
        return templates.TemplateResponse("admin/key_created.html", context)
        
    except ValueError as e:
        logger.error(f"ValueError creating API key: {e}")
        flash(request, str(e), "error")
        return RedirectResponse(url=request.url_for("get_user_details", user_id=user_id), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        logger.error(f"Unexpected error creating API key: {e}", exc_info=True)
        flash(request, f"Failed to create API key: {str(e)}", "error")
        return RedirectResponse(url=request.url_for("get_user_details", user_id=user_id), status_code=status.HTTP_303_SEE_OTHER)

@router.post("/keys/{key_id}/toggle-active", name="admin_toggle_key_active", dependencies=[Depends(validate_csrf_token)])
async def toggle_key_active_status(
    request: Request,
    key_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
):
    key = await apikey_crud.toggle_api_key_active(db, key_id=key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API Key not found or already revoked")
    
    # Fetch the user to get the ID for redirect
    await key.fetch_link(APIKey.user)
    user_id = str(key.user.id)
    
    new_status = "enabled" if key.is_active else "disabled"
    flash(request, f"API Key '{key.key_name}' has been {new_status}.", "success")
    return RedirectResponse(url=request.url_for("get_user_details", user_id=user_id), status_code=status.HTTP_303_SEE_OTHER)

@router.post("/keys/{key_id}/revoke", name="admin_revoke_key", dependencies=[Depends(validate_csrf_token)])
async def revoke_user_api_key(
    request: Request,
    key_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
):
    key = await apikey_crud.get_api_key_by_id(db, key_id=key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API Key not found")
    
    # Fetch the user to get the ID for redirect
    await key.fetch_link(APIKey.user)
    user_id = str(key.user.id)
    
    await apikey_crud.revoke_api_key(db, key_id=key_id)
    flash(request, f"API Key '{key.key_name}' has been revoked.", "success")
    return RedirectResponse(url=request.url_for("get_user_details", user_id=user_id), status_code=status.HTTP_303_SEE_OTHER)

@router.post("/users/{user_id}/delete", name="delete_user_account", dependencies=[Depends(validate_csrf_token)])
async def delete_user_account(request: Request, user_id: str, db: AsyncIOMotorDatabase = Depends(get_db), admin_user: User = Depends(require_admin_user)):
    user = await user_crud.get_user_by_id(db, user_id=user_id)
    if not user: raise HTTPException(status_code=404, detail="User not found")
    if user.is_admin:
        flash(request, "Cannot delete an admin account.", "error")
        return RedirectResponse(url=request.url_for("admin_users"), status_code=status.HTTP_303_SEE_OTHER)
    
    await user_crud.delete_user(db, user_id=user_id)
    flash(request, f"User '{user.username}' has been deleted.", "success")
    return RedirectResponse(url=request.url_for("admin_users"), status_code=status.HTTP_303_SEE_OTHER)