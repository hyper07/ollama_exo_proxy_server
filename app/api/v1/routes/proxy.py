import asyncio
import json
import logging
import time
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
    For non-streaming requests, collects SSE chunks and assembles into single JSON response.
    """
    logger.info(f"POST /api/chat/completions endpoint called")
    
    body_bytes = await request.body()
    model_name = None
    is_streaming = False
    
    if body_bytes:
        try:
            body = json.loads(body_bytes)
            model_name = body.get("model")
            is_streaming = body.get("stream", False)
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

    # For streaming requests, return StreamingResponse directly
    if is_streaming:
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
    
    # For non-streaming requests, collect SSE chunks and assemble into single JSON response
    # EXO always returns SSE format, so we need to collect and assemble
    http_client: AsyncClient = request.app.state.http_client
    
    # Try each server in round-robin fashion
    if not hasattr(request.app.state, 'backend_server_index'):
        request.app.state.backend_server_index = 0
    
    num_servers = len(candidate_servers)
    servers_tried = []
    
    for server_attempt in range(num_servers):
        index = request.app.state.backend_server_index
        chosen_server = candidate_servers[index % len(candidate_servers)]
        request.app.state.backend_server_index = (index + 1) % len(candidate_servers)
        
        servers_tried.append(chosen_server.name)
        
        logger.info(
            f"Attempting non-streaming request to EXO server '{chosen_server.name}' "
            f"({server_attempt + 1}/{num_servers})"
        )
        
        try:
            normalized_url = chosen_server.url.rstrip('/')
            backend_url = f"{normalized_url}/v1/chat/completions"
            logger.info(f"Preparing request to EXO server: {backend_url}")
            
            # Prepare headers
            headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}
            # Ensure Content-Type is set
            if 'content-type' not in headers:
                headers["Content-Type"] = "application/json"
            if chosen_server.encrypted_api_key:
                api_key_decrypted = decrypt_data(chosen_server.encrypted_api_key)
                if api_key_decrypted:
                    headers["Authorization"] = f"Bearer {api_key_decrypted}"
            
            logger.info(f"Making request to EXO server: {backend_url}")
            logger.debug(f"Request headers: {dict(headers)}")
            logger.debug(f"Request body length: {len(body_bytes)} bytes")
            
            # Make request to EXO server with timeout
            try:
                logger.info(f"Opening connection to {backend_url}...")
                logger.debug(f"Request body: {body_bytes.decode('utf-8')[:200]}...")
                
                # Use explicit timeout configuration - increased for large models
                timeout_config = httpx.Timeout(300.0, connect=10.0, read=290.0, write=10.0, pool=10.0)
                logger.info(f"Using timeout: connect=10s, read=290s, total=300s")
                
                # Wrap the entire operation with asyncio timeout to ensure it's enforced
                async def process_request():
                    async with http_client.stream(
                        "POST",
                        backend_url,
                        headers=headers,
                        content=body_bytes,
                        timeout=timeout_config
                    ) as response:
                        logger.info(f"Connection established! Received response from EXO server: {response.status_code}")
                        
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
                            
                            if response.status_code == 404:
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"Model not found or no instance running. {error_detail}"
                                )
                            else:
                                raise HTTPException(
                                    status_code=response.status_code,
                                    detail=f"EXO API error: {error_detail}"
                                )
                        
                        # Collect all SSE chunks
                        full_content = ""
                        last_response = None
                        buffer = ""
                        chunk_count = 0
                        
                        logger.info("Starting to collect SSE chunks from EXO server")
                        async for chunk in response.aiter_bytes():
                            chunk_count += 1
                            if chunk_count == 1:
                                logger.info(f"Received first chunk: {len(chunk)} bytes")
                            chunk_text = chunk.decode('utf-8', errors='replace')
                            buffer += chunk_text
                            
                            # Process complete SSE messages (separated by \n)
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                
                                if not line:
                                    continue
                                
                                # Parse SSE format: "data: {...}" or "data: [DONE]"
                                if line.startswith('data: '):
                                    json_str = line[6:].strip()  # Remove "data: " prefix
                                    
                                    if json_str == '[DONE]':
                                        continue
                                    
                                    try:
                                        chunk_data = json.loads(json_str)
                                        
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
                                                last_response = chunk_data
                                            elif not last_response:
                                                last_response = chunk_data
                                        
                                        # Also check if this chunk has usage info
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
                        
                        # Build non-streaming response in OpenAI format
                        if last_response:
                            finish_reason = "stop"
                            if 'choices' in last_response and len(last_response['choices']) > 0:
                                finish_reason = last_response['choices'][0].get('finish_reason', 'stop')
                            
                            response_data = {
                                "id": last_response.get("id", f"chatcmpl-{int(time.time())}"),
                                "object": "chat.completion",
                                "created": last_response.get("created", int(time.time())),
                                "model": last_response.get("model", model_name or ""),
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
                            
                            logger.info(f"Assembled response with {len(full_content)} characters of content from {chunk_count} chunks")
                            json_response = JSONResponse(content=response_data, status_code=200)
                            
                            # Log usage
                            try:
                                await log_crud.create_usage_log(
                                    db=db,
                                    api_key_id=api_key.id,
                                    endpoint="/api/chat/completions",
                                    status_code=200,
                                    server_id=chosen_server.id,
                                    model=model_name
                                )
                            except Exception as e:
                                logger.error(f"Failed to create usage log: {type(e).__name__}: {str(e)}")
                            
                            return json_response
                        else:
                            logger.error("No response chunks received from EXO API")
                            raise HTTPException(
                                status_code=500,
                                detail="No response chunks received from EXO API"
                            )
                
                # Execute the request with a hard timeout - increased for large models
                try:
                    return await asyncio.wait_for(process_request(), timeout=300.0)
                except asyncio.TimeoutError:
                    logger.error(f"Request to EXO server '{chosen_server.name}' timed out after 300 seconds")
                    raise httpx.TimeoutException("Request timed out after 300 seconds")
                    
            except httpx.TimeoutException as e:
                logger.error(f"Timeout connecting to EXO server '{chosen_server.name}': {e}")
                if server_attempt == num_servers - 1:
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=f"Timeout connecting to EXO server after 300 seconds. The server may be overloaded or the model may not be available. Tried: {', '.join(servers_tried)}"
                    )
                continue
            except httpx.RequestError as e:
                logger.error(f"Request error connecting to EXO server '{chosen_server.name}': {e}")
                if server_attempt == num_servers - 1:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to connect to EXO server. Tried: {', '.join(servers_tried)}"
                    )
                continue
            except httpx.TimeoutException as e:
                logger.error(f"Timeout connecting to EXO server '{chosen_server.name}': {e}")
                if server_attempt == num_servers - 1:
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=f"Timeout connecting to EXO server after 30 seconds. The server may be overloaded or the model may not be available. Tried: {', '.join(servers_tried)}"
                    )
                continue
            except httpx.RequestError as e:
                logger.error(f"Request error connecting to EXO server '{chosen_server.name}': {e}")
                if server_attempt == num_servers - 1:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to connect to EXO server. Tried: {', '.join(servers_tried)}"
                    )
                continue
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"EXO server '{chosen_server.name}' failed: {type(e).__name__}: {str(e)}", exc_info=True)
            if server_attempt == num_servers - 1:
                # Last server, raise error
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"All EXO servers unavailable. Tried: {', '.join(servers_tried)}"
                )
            # Try next server
            continue
    
    # Should not reach here, but just in case
    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=f"All EXO servers unavailable. Tried: {', '.join(servers_tried)}"
    )


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
    
    # Try to get from cache first
    cache = getattr(request.app.state, 'cache', None)
    if cache and cache.enabled:
        try:
            # Use first server's ID for cache key (models are usually the same across servers)
            cache_key = f"models:{servers[0].id if servers else 'default'}"
            cached_response = await cache.get(cache_key)
            if cached_response:
                logger.info("Returning cached models response")
                return JSONResponse(content=cached_response)
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
    
    response, chosen_server = await _reverse_proxy(request, "models", servers)
    
    # Cache the response if it's successful
    if cache and cache.enabled and response.status_code == 200:
        try:
            # Read response body (for non-streaming responses)
            if hasattr(response, 'body'):
                body = response.body
            else:
                # For streaming responses, we can't cache easily
                pass
        except Exception as e:
            logger.warning(f"Failed to cache models response: {e}")
    
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
