# app/api/v1/routes/exo_proxy.py
"""
Proxy endpoints for EXO Master API
These endpoints forward requests from the admin panel to the EXO API instance,
avoiding CORS issues in the browser.
"""
import json
import logging
import time
import traceback
from typing import Any, Dict, Optional
import httpx

from fastapi import APIRouter, Depends, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.database.models import User
from app.api.v1.routes.admin import require_admin_user

logger = logging.getLogger(__name__)
# Admin-only endpoints for testing EXO APIs - no proxy API key required
router = APIRouter(prefix="/exo-api-test", tags=["exo-api-test"])


def normalize_base_url(base_url: str) -> str:
    """Normalize the base URL by removing trailing slashes"""
    return base_url.rstrip('/')


def get_exo_auth_headers(request: Request) -> Dict[str, str]:
    """
    Extract EXO API key from request headers and format as Authorization header.
    The frontend sends the key via X-EXO-API-Key header.
    """
    headers = {}
    api_key = request.headers.get("X-EXO-API-Key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def get_exo_client(request: Request) -> httpx.AsyncClient:
    """Get or create an httpx client for EXO API requests"""
    return request.app.state.http_client


@router.get("/node_id")
async def proxy_node_id(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance")
):
    """Proxy GET /node_id"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.get(f"{base_url}/node_id", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying /node_id to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def proxy_models(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance")
):
    """Proxy GET /models"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.get(f"{base_url}/models", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying /models to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state")
async def proxy_state(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance")
):
    """Proxy GET /state"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.get(f"{base_url}/state", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying /state to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def proxy_events(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance")
):
    """Proxy GET /events"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.get(f"{base_url}/events", headers=headers, timeout=10.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying /events to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instance/placement")
async def proxy_instance_placement(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance"),
    model_id: str = Query(...),
    sharding: str = Query(default="Pipeline"),
    instance_meta: str = Query(default="MlxRing"),
    min_nodes: int = Query(default=1)
):
    """Proxy GET /instance/placement"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        url = f"{base_url}/instance/placement?model_id={model_id}&sharding={sharding}&instance_meta={instance_meta}&min_nodes={min_nodes}"
        response = await client.get(url, headers=headers, timeout=30.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying /instance/placement to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instance/previews")
async def proxy_instance_previews(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance")
):
    """Proxy GET /instance/previews"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.get(f"{base_url}/instance/previews", headers=headers, timeout=30.0)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying /instance/previews to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/instance")
async def proxy_create_instance(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance"),
    payload: Dict[str, Any] = Body(...)
):
    """Proxy POST /instance"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.post(
            f"{base_url}/instance",
            json=payload,
            headers=headers,
            timeout=30.0
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying POST /instance to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/place_instance")
async def proxy_place_instance(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance"),
    payload: Dict[str, Any] = Body(...)
):
    """Proxy POST /place_instance"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.post(
            f"{base_url}/place_instance",
            json=payload,
            headers=headers,
            timeout=30.0
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying POST /place_instance to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instance/{instance_id}")
async def proxy_get_instance(
    request: Request,
    instance_id: str,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance")
):
    """Proxy GET /instance/{instance_id}"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.get(
            f"{base_url}/instance/{instance_id}",
            headers=headers,
            timeout=10.0
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying GET /instance/{instance_id} to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/instance/{instance_id}")
async def proxy_delete_instance(
    request: Request,
    instance_id: str,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance")
):
    """Proxy DELETE /instance/{instance_id}"""
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    try:
        response = await client.delete(
            f"{base_url}/instance/{instance_id}",
            headers=headers,
            timeout=10.0
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.RequestError as e:
        logger.error(f"Error proxying DELETE /instance/{instance_id} to {base_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check_model")
async def proxy_check_model(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance"),
    model: str = Query(..., description="Model name to check")
):
    """
    Check if a model has an active instance and is ready for chat completions.
    This helps diagnose issues before attempting a chat completion.
    """
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    
    try:
        # Get state
        state_response = await client.get(f"{base_url}/state", headers=headers, timeout=10.0)
        state_data = state_response.json()
        
        # Get models
        models_response = await client.get(f"{base_url}/models", headers=headers, timeout=10.0)
        models_data = models_response.json()
        
        # Build model mapping (short ID -> full ID)
        model_mapping = []
        for m in models_data.get('data', []):
            short_id = m.get('id', '')
            full_id = m.get('hugging_face_id', '')
            if short_id and full_id:
                model_mapping.append({
                    'short_id': short_id,
                    'full_id': full_id,
                    'name': m.get('name', '')
                })
        
        # Check if model exists (as short ID or full ID)
        available_short_ids = [m['short_id'] for m in model_mapping]
        available_full_ids = [m['full_id'] for m in model_mapping]
        model_exists = model in available_short_ids or model in available_full_ids
        
        # Find the corresponding full model ID if user provided short ID
        resolved_full_id = model
        for mapping in model_mapping:
            if model == mapping['short_id']:
                resolved_full_id = mapping['full_id']
                break
        
        # Check if instance exists for this model
        instances = state_data.get('instances', {})
        instance_found = False
        instance_info = None
        
        for instance_id, instance_data in instances.items():
            instance_type = list(instance_data.keys())[0]
            instance = instance_data[instance_type]
            
            if 'shardAssignments' in instance:
                shard_model = instance['shardAssignments'].get('modelId', '')
                # Match against full ID or if either contains the other
                if (shard_model == resolved_full_id or 
                    model in shard_model or 
                    shard_model in model or
                    resolved_full_id in shard_model):
                    instance_found = True
                    instance_info = {
                        'instance_id': instance.get('instanceId', instance_id),
                        'type': instance_type,
                        'model_id': shard_model
                    }
                    break
        
        # Build helpful message
        if model_exists and instance_found:
            message = f'✅ Model is ready for chat completions! Use this model ID: {instance_info["model_id"]}'
        elif model_exists:
            matching_model = next((m for m in model_mapping if model == m['short_id'] or model == m['full_id']), None)
            if matching_model:
                message = f'⚠️ Model exists but no instance is placed. Use POST /place_instance with short ID "{matching_model["short_id"]}", then use full ID "{matching_model["full_id"]}" for chat.'
            else:
                message = 'Model exists but no instance is placed - use POST /place_instance'
        else:
            message = f'❌ Model not found. Available models: {", ".join(available_short_ids)}'
        
        return JSONResponse(content={
            'model': model,
            'resolved_full_id': resolved_full_id,
            'model_exists': model_exists,
            'model_mapping': model_mapping,
            'instance_found': instance_found,
            'instance_info': instance_info,
            'ready': model_exists and instance_found,
            'message': message
        })
    except httpx.RequestError as e:
        logger.error(f"Error checking model status: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/completions")
async def proxy_chat_completions(
    request: Request,
    admin_user: User = Depends(require_admin_user),
    exo_base_url: str = Query(..., description="Base URL of the EXO instance"),
    payload: Dict[str, Any] = Body(...)
):
    """
    Proxy POST /v1/chat/completions (OpenAI-compatible endpoint)
    
    Response Formats:
    - stream=true: Returns Server-Sent Events (SSE) format
      Example: data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}\n\n
               data: [DONE]\n\n
      
    - stream=false: Returns single JSON response (OpenAI-compatible)
      Example: {"id":"...","choices":[{"message":{"content":"Hello!"}}],"usage":{...}}
    
    Note: EXO API always returns streaming responses. When stream=false, this proxy
    collects all SSE chunks and assembles them into a single JSON response.
    """
    client = await get_exo_client(request)
    base_url = normalize_base_url(exo_base_url)
    headers = get_exo_auth_headers(request)
    
    # Check if streaming is requested by the client
    is_streaming = payload.get("stream", False)
    
    # Log the request for debugging
    logger.info(f"Chat completion request to {base_url}/v1/chat/completions")
    logger.info(f"Model: {payload.get('model')}, Stream: {is_streaming}")
    logger.debug(f"Full payload: {payload}")
    
    if is_streaming:
        # For streaming responses, we need to keep the connection open
        # Create a generator that manages the stream lifecycle
        async def stream_generator():
            try:
                async with client.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=120.0  # Longer timeout for streaming
                ) as response:
                    # Check for errors
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        error_str = error_text.decode()
                        logger.error(f"EXO API streaming error {response.status_code}: {error_str}")
                        
                        # For streaming errors, yield an error message in SSE format
                        error_msg = {
                            "error": {
                                "message": f"EXO API error: {error_str}",
                                "type": "api_error",
                                "code": response.status_code
                            }
                        }
                        yield f"data: {json.dumps(error_msg)}\n\n".encode()
                        return
                    
                    # Stream the response chunks
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        
            except httpx.RequestError as e:
                logger.error(f"Stream connection error: {e}")
                error_msg = {
                    "error": {
                        "message": f"Connection error: {str(e)}",
                        "type": "connection_error"
                    }
                }
                yield f"data: {json.dumps(error_msg)}\n\n".encode()
            except Exception as e:
                logger.error(f"Unexpected streaming error: {e}")
                logger.error(traceback.format_exc())
                error_msg = {
                    "error": {
                        "message": f"Streaming error: {str(e)}",
                        "type": "internal_error"
                    }
                }
                yield f"data: {json.dumps(error_msg)}\n\n".encode()
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    else:
        # Non-streaming mode
        try:
            # For non-streaming, we can use the async with safely
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
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
                    
                    # Provide helpful error messages based on status code
                    if response.status_code == 404:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Model not found or no instance running. {error_detail}. Please check: 1) Model exists in /models, 2) Instance is placed for this model (check /state endpoint)"
                        )
                    elif response.status_code == 500:
                        raise HTTPException(
                            status_code=500,
                            detail=f"EXO internal error: {error_detail}. Check EXO server logs for details."
                        )
                    else:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"EXO API error: {error_detail}"
                        )
                
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
                else:
                    # No chunks received - return error
                    raise HTTPException(
                        status_code=500,
                        detail="No response chunks received from EXO API"
                    )
                
                return JSONResponse(content=response_data, status_code=200)
        
        except httpx.RequestError as e:
            logger.error(f"Error proxying POST /v1/chat/completions to {base_url}: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to connect to EXO API at {base_url}: {str(e)}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat completions proxy: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

