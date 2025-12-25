import asyncio
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from fastapi import APIRouter, Depends, Request, Response, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from httpx import AsyncClient
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.session import get_db
from app.api.v1.dependencies import get_valid_api_key, rate_limiter, ip_filter, get_settings
from app.database.models import APIKey, ExoServer
from app.crud import log_crud, server_crud
from app.core.retry import retry_with_backoff, RetryConfig
from app.schema.settings import AppSettingsModel
from app.core.encryption import decrypt_data

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(ip_filter), Depends(rate_limiter)])

# --- Dependency to get active EXO servers ---
async def get_active_servers(db: AsyncIOMotorDatabase = Depends(get_db)) -> List[ExoServer]:
    servers = await server_crud.get_servers(db)
    active_servers = [s for s in servers if s.is_active and s.server_type == 'exo']
    if not active_servers:
        logger.error("No active EXO servers are configured in the database.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active EXO servers available."
        )
    return active_servers


async def _send_exo_request(
    http_client: AsyncClient,
    server: ExoServer,
    path: str,
    method: str,
    headers: dict,
    query_params,
    body_bytes: bytes
):
    """
    Send a request to an EXO server.
    EXO uses OpenAI-compatible endpoints with /v1/ prefix.
    """
    normalized_url = server.url.rstrip('/')
    
    # Map common endpoint paths to EXO format
    if path in ['chat', 'chat/completion', 'chat/completions']:
        backend_url = f"{normalized_url}/v1/chat/completions"
    elif path == 'embeddings':
        backend_url = f"{normalized_url}/v1/embeddings"
    elif path in ['models', 'tags']:
        backend_url = f"{normalized_url}/v1/models"
    elif path == 'state':
        backend_url = f"{normalized_url}/state"
    elif path == 'events':
        backend_url = f"{normalized_url}/events"
    elif path.startswith('instance'):
        backend_url = f"{normalized_url}/{path}"
    elif path.startswith('v1/'):
        # Already has v1 prefix
        backend_url = f"{normalized_url}/{path}"
    else:
        # Default: assume v1 prefix
        backend_url = f"{normalized_url}/v1/{path}"

    request_headers = headers.copy()
    if server.encrypted_api_key:
        api_key = decrypt_data(server.encrypted_api_key)
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"

    backend_request = http_client.build_request(
        method=method,
        url=backend_url,
        headers=request_headers,
        params=query_params,
        content=body_bytes
    )

    try:
        backend_response = await http_client.send(backend_request, stream=True)

        # Consider 5xx errors as failures that should be retried
        if backend_response.status_code >= 500:
            await backend_response.aclose()
            raise Exception(
                f"EXO server returned {backend_response.status_code}: "
                f"{backend_response.reason_phrase}"
            )

        return backend_response

    except Exception as e:
        logger.debug(f"Request to EXO server {server.url} failed: {type(e).__name__}: {str(e)}")
        raise


async def _reverse_proxy(
    request: Request, 
    path: str, 
    servers: List[ExoServer], 
    body_bytes: bytes = b""
) -> Tuple[Response, ExoServer]:
    """
    Core reverse proxy logic with retry support. Forwards requests to EXO servers
    and streams the response back. Returns the response and the chosen server.
    """
    http_client: AsyncClient = request.app.state.http_client
    app_settings: AppSettingsModel = request.app.state.settings

    # Get retry configuration from app settings
    retry_config = RetryConfig(
        max_retries=app_settings.max_retries,
        total_timeout_seconds=app_settings.retry_total_timeout_seconds,
        base_delay_ms=app_settings.retry_base_delay_ms
    )

    if not hasattr(request.app.state, 'backend_server_index'):
        request.app.state.backend_server_index = 0

    # Simple in-memory cooldown to avoid hammering a failing backend
    if not hasattr(request.app.state, 'backend_cooldowns'):
        request.app.state.backend_cooldowns = {}
    cooldowns: Dict[str, float] = request.app.state.backend_cooldowns
    cooldown_seconds = getattr(app_settings, "backend_failure_cooldown_seconds", 10)

    # Prepare request headers (exclude 'host' header)
    headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}

    # Try each server in round-robin fashion (skipping servers on cooldown when possible)
    now = asyncio.get_event_loop().time()
    available_servers = [s for s in servers if cooldowns.get(str(s.id), 0) <= now]
    if not available_servers:
        # If everything is cooling down, fall back to trying all servers
        available_servers = servers

    num_servers = len(available_servers)
    servers_tried = []

    for server_attempt in range(num_servers):
        # Select next server using round-robin
        index = request.app.state.backend_server_index
        chosen_server = available_servers[index % len(available_servers)]
        request.app.state.backend_server_index = (index + 1) % len(available_servers)

        servers_tried.append(chosen_server.name)

        logger.info(
            f"Attempting request to EXO server '{chosen_server.name}' "
            f"({server_attempt + 1}/{num_servers})"
        )

        # Retry logic for EXO server
        retry_result = await retry_with_backoff(
            _send_exo_request,
            http_client=http_client,
            server=chosen_server,
            path=path,
            method=request.method,
            headers=headers,
            query_params=request.query_params,
            body_bytes=body_bytes,
            config=retry_config,
            retry_on_exceptions=(Exception,),
            operation_name=f"Request to EXO server {chosen_server.name}"
        )

        if retry_result.success:
            # Success! Create streaming response
            backend_response = retry_result.result

            logger.info(
                f"Successfully proxied to EXO server '{chosen_server.name}' "
                f"after {retry_result.attempts} attempt(s) "
                f"in {retry_result.total_duration_ms:.1f}ms"
            )

            response = StreamingResponse(
                backend_response.aiter_raw(),
                status_code=backend_response.status_code,
                headers=backend_response.headers,
            )
            return response, chosen_server
        else:
            # This server failed after all retries, try next server
            logger.warning(
                f"EXO server '{chosen_server.name}' failed after {retry_result.attempts} "
                f"attempts. Trying next server if available."
            )
            cooldowns[str(chosen_server.id)] = asyncio.get_event_loop().time() + float(cooldown_seconds)

    # All servers exhausted
    logger.error(
        f"All {num_servers} EXO server(s) failed after retries. "
        f"Servers tried: {', '.join(servers_tried)}"
    )
    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=f"All EXO servers unavailable. Tried: {', '.join(servers_tried)}"
    )


# --- Chat Completions Endpoints ---
@router.post("/chat/completions")
@router.post("/chat/completion")
async def proxy_chat_completion(
    request: Request,
    api_key: APIKey = Depends(get_valid_api_key),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: AppSettingsModel = Depends(get_settings),
    servers: List[ExoServer] = Depends(get_active_servers),
):
    """
    Proxy chat completion requests to EXO servers.
    Compatible with OpenAI API format.
    """
    logger.info(f"POST /api/chat/completions endpoint called")
    
    body_bytes = await request.body()
    model_name = None
    
    if body_bytes:
        try:
            body = json.loads(body_bytes)
            model_name = body.get("model")
        except json.JSONDecodeError:
            pass

    # Smart routing: filter servers by model availability
    candidate_servers = servers
    if model_name:
        servers_with_model = await server_crud.get_servers_with_model(db, model_name)
        if servers_with_model:
            candidate_servers = [s for s in servers_with_model if s.server_type == 'exo']
            logger.info(f"Smart routing: Found {len(candidate_servers)} EXO server(s) with model '{model_name}'")
        else:
            logger.warning(
                f"Model '{model_name}' not found in any EXO server's catalog. "
                f"Falling back to round-robin across all {len(servers)} active EXO server(s)."
            )

    # Proxy to EXO server
    response, chosen_server = await _reverse_proxy(request, "chat/completions", candidate_servers, body_bytes)

    # Log usage (don't let logging errors break the API response)
    try:
        await log_crud.create_usage_log(
            db=db,
            api_key_id=api_key.id,
            endpoint="/api/chat/completions",
            status_code=response.status_code,
            server_id=chosen_server.id,
            model=model_name
        )
    except Exception as e:
        logger.error(f"Failed to create usage log: {type(e).__name__}: {str(e)}")

    return response


# --- Models/Tags Endpoints ---
@router.get("/models")
@router.get("/tags")
async def proxy_models(
    request: Request,
    api_key: APIKey = Depends(get_valid_api_key),
    db: AsyncIOMotorDatabase = Depends(get_db),
    servers: List[ExoServer] = Depends(get_active_servers),
):
    """Get available models from EXO servers."""
    logger.info("GET /api/models endpoint called")
    
    response, chosen_server = await _reverse_proxy(request, "models", servers)
    
    # Log usage (don't let logging errors break the API response)
    try:
        await log_crud.create_usage_log(
            db=db,
            api_key_id=api_key.id,
            endpoint="/api/models",
            status_code=response.status_code,
            server_id=chosen_server.id,
            model=None
        )
    except Exception as e:
        logger.error(f"Failed to create usage log: {type(e).__name__}: {str(e)}")
    
    return response


# --- Embeddings Endpoint ---
@router.post("/embeddings")
async def proxy_embeddings(
    request: Request,
    api_key: APIKey = Depends(get_valid_api_key),
    db: AsyncIOMotorDatabase = Depends(get_db),
    servers: List[ExoServer] = Depends(get_active_servers),
):
    """Proxy embeddings requests to EXO servers."""
    logger.info("POST /api/embeddings endpoint called")
    
    body_bytes = await request.body()
    model_name = None
    
    if body_bytes:
        try:
            body = json.loads(body_bytes)
            model_name = body.get("model")
        except json.JSONDecodeError:
            pass
    
    response, chosen_server = await _reverse_proxy(request, "embeddings", servers, body_bytes)
    
    # Log usage (don't let logging errors break the API response)
    try:
        await log_crud.create_usage_log(
            db=db,
            api_key_id=api_key.id,
            endpoint="/api/embeddings",
            status_code=response.status_code,
            server_id=chosen_server.id,
            model=model_name
        )
    except Exception as e:
        logger.error(f"Failed to create usage log: {type(e).__name__}: {str(e)}")
    
    return response


# --- Catch-all route for other EXO endpoints ---
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_exo_catchall(
    request: Request,
    path: str,
    api_key: APIKey = Depends(get_valid_api_key),
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: AppSettingsModel = Depends(get_settings),
    servers: List[ExoServer] = Depends(get_active_servers),
):
    """
    Catch-all route that proxies other requests to EXO servers.
    Supports EXO-specific endpoints like /state, /events, /instance, etc.
    """
    # Endpoint security check
    blocked_paths = {p.strip().lstrip('/') for p in settings.blocked_exo_endpoints.split(',') if p.strip()}
    request_path = path.strip().lstrip('/')

    if request_path in blocked_paths:
        logger.warning(
            f"Blocked attempt to access endpoint '/api/{request_path}' by API key {api_key.key_prefix}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access to the endpoint '/api/{request_path}' is disabled."
        )

    body_bytes = await request.body()
    model_name = None
    
    if body_bytes:
        try:
            body = json.loads(body_bytes)
            if isinstance(body, dict) and "model" in body:
                model_name = body["model"]
        except (json.JSONDecodeError, Exception):
            pass

    # Smart routing if model specified
    candidate_servers = servers
    if model_name:
        servers_with_model = await server_crud.get_servers_with_model(db, model_name)
        if servers_with_model:
            candidate_servers = [s for s in servers_with_model if s.server_type == 'exo']
            if candidate_servers:
                logger.info(f"Smart routing: Found {len(candidate_servers)} EXO server(s) with model '{model_name}'")

    response, chosen_server = await _reverse_proxy(request, path, candidate_servers, body_bytes)

    # Log usage (don't let logging errors break the API response)
    try:
        await log_crud.create_usage_log(
            db=db,
            api_key_id=api_key.id,
            endpoint=f"/api/{path}",
            status_code=response.status_code,
            server_id=chosen_server.id,
            model=model_name
        )
    except Exception as e:
        logger.error(f"Failed to create usage log: {type(e).__name__}: {str(e)}")

    return response
