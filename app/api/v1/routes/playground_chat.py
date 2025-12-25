# app/api/v1/routes/playground_chat.py
import logging
import json
import time
from typing import Optional, Union

from fastapi import APIRouter, Depends, Request, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, Response
from motor.motor_asyncio import AsyncIOMotorDatabase
import httpx

from app.database.session import get_db
from app.database.models import User
from app.crud import server_crud
from app.api.v1.dependencies import validate_csrf_token_header
from app.api.v1.routes.admin import require_admin_user, get_template_context, templates
from app.core.test_prompts import PREBUILT_TEST_PROMPTS


logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/playground", response_class=HTMLResponse, name="admin_playground")
async def admin_playground_ui(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    model: Optional[str] = Query(None)
):
    from app.api.v1.dependencies import get_csrf_token
    context = get_template_context(request)
    context["model_groups"] = await server_crud.get_all_models_grouped_by_server(db, filter_type='chat')
    context["selected_model"] = model
    context["csrf_token"] = await get_csrf_token(request)
    return templates.TemplateResponse("admin/model_playground.html", context)


@router.post("/playground-stream", name="admin_playground_stream", dependencies=[Depends(validate_csrf_token_header)])
async def admin_playground_stream(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user)
):
    try:
        data = await request.json()
        model_name = data.get("model")
        messages = data.get("messages")
        think_option = data.get("think_option") # Can be True, "low", "medium", "high"
        
        if not model_name or not messages:
            return JSONResponse({"error": "Model and messages are required."}, status_code=400)

        http_client: httpx.AsyncClient = request.app.state.http_client
        
        servers_with_model = await server_crud.get_servers_with_model(db, model_name)
        if not servers_with_model:
            active_servers = [s for s in await server_crud.get_servers(db) if s.is_active]
            if not active_servers:
                error_payload = {"error": "No active backend servers available."}
                return Response(json.dumps(error_payload), media_type="application/x-ndjson", status_code=503)
            target_server = active_servers[0]
        else:
            target_server = servers_with_model[0]
        
        # Handle base64 images - format depends on server type
        # EXO/vLLM use OpenAI format (keep as-is), Ollama needs conversion
        if target_server.server_type != 'exo' and target_server.server_type != 'vllm':
            # Convert to Ollama format
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    new_content = []
                    images_list = []
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            new_content.append(item["text"])
                        elif item.get("type") == "image_url":
                            base64_str = item["image_url"]["url"].split(",")[-1]
                            images_list.append(base64_str)
                    
                    msg["content"] = " ".join(new_content)
                    if images_list:
                        msg["images"] = images_list
        
        if target_server.server_type == 'exo':
            # EXO server - uses OpenAI-compatible /v1/chat/completions endpoint
            chat_url = f"{target_server.url.rstrip('/')}/v1/chat/completions"
            
            # EXO expects OpenAI-compatible format with messages array
            # Keep messages in OpenAI format (already compatible)
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
            }
            
            # EXO doesn't support think_option in the same way, but we can pass it if needed
            if think_option:
                # Some EXO models might support thinking, but it's not standard
                # For now, we'll just log it
                logger.info(f"Think option requested for EXO model: {think_option}")
            
            from app.crud.server_crud import _get_auth_headers
            headers = _get_auth_headers(target_server)

            async def event_stream_exo():
                final_chunk = None
                total_content = ""
                start_time = time.monotonic()
                buffer = ""
                try:
                    async with http_client.stream("POST", chat_url, json=payload, timeout=600.0, headers=headers) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            error_text = error_body.decode('utf-8')
                            logger.error(f"EXO backend returned error {response.status_code}: {error_text}")
                            error_payload = {"error": f"EXO server error: {error_text}"}
                            yield (json.dumps(error_payload) + '\n').encode('utf-8')
                            return
                        
                        # EXO returns SSE format: "data: {...}\n\n" or "data: [DONE]\n\n"
                        # SSE messages are separated by \n\n, but we also handle single \n for robustness
                        async for chunk_bytes in response.aiter_bytes():
                            buffer += chunk_bytes.decode('utf-8', errors='replace')
                            
                            # Process complete SSE messages (separated by \n\n)
                            while '\n\n' in buffer:
                                sse_message, buffer = buffer.split('\n\n', 1)
                                sse_message = sse_message.strip()
                                
                                if not sse_message:
                                    continue
                                
                                # Handle multiple lines in one SSE message (shouldn't happen but be safe)
                                for line in sse_message.split('\n'):
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
                                            
                                            # Convert EXO format to playground NDJSON format
                                            # EXO format: {"id":"...","choices":[{"delta":{"content":"..."}}]}
                                            # Playground format: {"message": {"role": "assistant", "content": "..."}}
                                            
                                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                                choice = chunk_data['choices'][0]
                                                
                                                # Check for content in delta
                                                if 'delta' in choice and 'content' in choice['delta']:
                                                    content = choice['delta'].get('content', '')
                                                    if content:
                                                        total_content += content
                                                        # Convert to playground format
                                                        playground_chunk = {
                                                            "message": {
                                                                "role": "assistant",
                                                                "content": content
                                                            }
                                                        }
                                                        yield (json.dumps(playground_chunk) + '\n').encode('utf-8')
                                                
                                                # Check for finish_reason (indicates final chunk)
                                                if 'finish_reason' in choice:
                                                    final_chunk = chunk_data
                                            
                                            # Also check for usage info
                                            if 'usage' in chunk_data:
                                                final_chunk = chunk_data
                                                
                                        except json.JSONDecodeError as e:
                                            logger.warning(f"Failed to parse EXO SSE chunk: {json_str[:100]}... Error: {e}")
                                            continue
                        
                        # Process any remaining buffer (handle incomplete SSE messages at end of stream)
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
                                                        total_content += content
                                                        playground_chunk = {
                                                            "message": {
                                                                "role": "assistant",
                                                                "content": content
                                                            }
                                                        }
                                                        yield (json.dumps(playground_chunk) + '\n').encode('utf-8')
                                                if 'finish_reason' in choice:
                                                    final_chunk = chunk_data
                                            if 'usage' in chunk_data:
                                                final_chunk = chunk_data
                                        except json.JSONDecodeError:
                                            pass
                        
                        # Send final chunk with stats
                        if final_chunk:
                            end_time = time.monotonic()
                            duration_ns = int((end_time - start_time) * 1_000_000_000)
                            
                            # Calculate approximate token count (rough estimate: 4 chars per token)
                            eval_count = len(total_content) // 4 if total_content else 0
                            
                            final_playground_chunk = {
                                "done": True,
                                "eval_count": eval_count,
                                "eval_duration": duration_ns
                            }
                            
                            # Include usage info if available
                            if 'usage' in final_chunk:
                                final_playground_chunk["usage"] = final_chunk['usage']
                            
                            yield (json.dumps(final_playground_chunk) + '\n').encode('utf-8')
                        else:
                            # No final chunk - send a done message anyway
                            end_time = time.monotonic()
                            duration_ns = int((end_time - start_time) * 1_000_000_000)
                            eval_count = len(total_content) // 4 if total_content else 0
                            final_playground_chunk = {
                                "done": True,
                                "eval_count": eval_count,
                                "eval_duration": duration_ns
                            }
                            yield (json.dumps(final_playground_chunk) + '\n').encode('utf-8')
                            
                except Exception as e:
                    logger.error(f"Error streaming from EXO backend: {e}", exc_info=True)
                    error_payload = {"error": "Failed to stream from EXO server.", "details": str(e)}
                    yield (json.dumps(error_payload) + '\n').encode('utf-8')
            
            return StreamingResponse(event_stream_exo(), media_type="application/x-ndjson")

        elif target_server.server_type == 'vllm':
            from app.core.vllm_translator import translate_exo_to_vllm_chat, vllm_stream_to_exo_stream
            
            chat_url = f"{target_server.url.rstrip('/')}/v1/chat/completions"
            
            exo_payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
            }
            if think_option is True:
                exo_payload["think"] = True
            
            payload = translate_exo_to_vllm_chat(exo_payload)
            
            from app.crud.server_crud import _get_auth_headers
            headers = _get_auth_headers(target_server)

            async def event_stream_vllm():
                try:
                    async with http_client.stream("POST", chat_url, json=payload, timeout=600.0, headers=headers) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            error_text = error_body.decode('utf-8')
                            logger.error(f"vLLM backend returned error {response.status_code}: {error_text}")
                            error_payload = {"error": f"vLLM server error: {error_text}"}
                            yield json.dumps(error_payload).encode('utf-8')
                            return
                        
                        async for chunk in vllm_stream_to_exo_stream(response.aiter_text(), model_name):
                            yield chunk
                except Exception as e:
                    logger.error(f"Error streaming from vLLM backend: {e}", exc_info=True)
                    error_payload = {"error": "Failed to stream from backend server.", "details": str(e)}
                    yield json.dumps(error_payload).encode('utf-8')
            
            return StreamingResponse(event_stream_vllm(), media_type="application/x-ndjson")

        else: # Ollama server
            chat_url = f"{target_server.url.rstrip('/')}/api/chat"
            payload = {"model": model_name, "messages": messages, "stream": True}

            if think_option:
                model_name_lower = model_name.lower()
                supported_think_models = ["qwen", "gpt-oss", "deepseek"]

                if any(keyword in model_name_lower for keyword in supported_think_models):
                    payload["think"] = think_option
                else:
                    logger.warning(f"Frontend requested thinking for '{model_name}', but it's not in the known support list. Ignoring 'think' parameter.")

            from app.crud.server_crud import _get_auth_headers
            headers = _get_auth_headers(target_server)

            async def event_stream_ollama():
                final_chunk_from_ollama = None
                total_eval_text = ""
                start_time = time.monotonic()
                thinking_block_open = False
                try:
                    async with http_client.stream("POST", chat_url, json=payload, timeout=600.0, headers=headers) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            error_text = error_body.decode('utf-8')
                            logger.error(f"Ollama backend returned error {response.status_code}: {error_text}")
                            error_payload = {"error": f"Ollama server error: {error_text}"}
                            yield (json.dumps(error_payload) + '\n').encode('utf-8')
                            return

                        buffer = ""
                        async for chunk_str in response.aiter_text():
                            buffer += chunk_str
                            lines = buffer.split('\n')
                            buffer = lines.pop()

                            for line in lines:
                                if not line.strip(): continue
                                try:
                                    data = json.loads(line)
                                    message = data.get("message", {})
                                    
                                    # Handle thinking tokens
                                    if "thinking" in message and message["thinking"]:
                                        think_content = message["thinking"]
                                        if not thinking_block_open:
                                            thinking_block_open = True
                                            think_content = "<think>" + think_content
                                        
                                        # Modify chunk to look like a content chunk for the UI
                                        data["message"]["content"] = think_content
                                        del data["message"]["thinking"]
                                        total_eval_text += think_content
                                        yield (json.dumps(data) + '\n').encode('utf-8')
                                        continue

                                    # Handle content tokens
                                    if "content" in message and message["content"]:
                                        if thinking_block_open:
                                            thinking_block_open = False
                                            # Send closing tag as its own chunk first
                                            closing_chunk = data.copy()
                                            closing_chunk["message"] = {"role": "assistant", "content": "</think>"}
                                            if "done" in closing_chunk: del closing_chunk["done"]
                                            yield (json.dumps(closing_chunk) + '\n').encode('utf-8')
                                        
                                        total_eval_text += message["content"]
                                        # Yield original content chunk
                                        yield (line + '\n').encode('utf-8')
                                        continue
                                    
                                    if data.get("done"):
                                        final_chunk_from_ollama = data
                                    else:
                                        # Forward any other non-content/thinking chunks
                                        yield (line + '\n').encode('utf-8')

                                except json.JSONDecodeError:
                                    logger.warning(f"Could not parse Ollama stream chunk: {line}, forwarding as is.")
                                    yield (line + '\n').encode('utf-8')
                        
                        if buffer.strip():
                            try:
                                data = json.loads(buffer.strip())
                                if data.get("done"):
                                    final_chunk_from_ollama = data
                                else:
                                    yield (buffer.strip() + '\n').encode('utf-8')
                            except json.JSONDecodeError:
                                 yield (buffer.strip() + '\n').encode('utf-8')

                    if thinking_block_open:
                        closing_chunk = {"model": model_name, "created_at": "...", "message": {"role": "assistant", "content": "</think>"}, "done": False}
                        yield (json.dumps(closing_chunk) + '\n').encode('utf-8')

                    if final_chunk_from_ollama:
                        if "eval_count" not in final_chunk_from_ollama or "eval_duration" not in final_chunk_from_ollama:
                            logger.warning("Ollama response did not include stats, calculating manually.")
                            end_time = time.monotonic()
                            final_chunk_from_ollama["eval_count"] = len(total_eval_text) // 4
                            final_chunk_from_ollama["eval_duration"] = int((end_time - start_time) * 1_000_000_000)
                        
                        yield (json.dumps(final_chunk_from_ollama) + '\n').encode('utf-8')
                    else:
                        logger.error("No 'done' chunk received from Ollama stream.")

                except Exception as e:
                    logger.error(f"Error streaming from Ollama backend: {e}", exc_info=True)
                    error_payload = {"error": "Failed to stream from backend server.", "details": str(e)}
                    yield json.dumps(error_payload).encode('utf-8')
            
            return StreamingResponse(event_stream_ollama(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {e}", exc_info=True)
        return JSONResponse({"error": "An internal error occurred."}, status_code=500)

@router.get("/playground/test-prompts", name="admin_get_test_prompts")
async def admin_get_test_prompts(admin_user: User = Depends(require_admin_user)):
    return JSONResponse(PREBUILT_TEST_PROMPTS)