from motor.motor_asyncio import AsyncIOMotorDatabase
from app.database.models import ExoServer
from app.schema.server import ServerCreate, ServerUpdate
from app.core.encryption import encrypt_data, decrypt_data
import httpx
import logging
import datetime
from typing import Optional, List, Dict, Any
import asyncio
import json

logger = logging.getLogger(__name__)

def _get_auth_headers(server: ExoServer) -> Dict[str, str]:
    headers = {}
    if server.encrypted_api_key:
        api_key = decrypt_data(server.encrypted_api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers

async def get_server_by_id(db: AsyncIOMotorDatabase, server_id: str) -> ExoServer | None:
    return await ExoServer.get(server_id)

async def get_server_by_url(db: AsyncIOMotorDatabase, url: str) -> ExoServer | None:
    return await ExoServer.find_one(ExoServer.url == url)

async def get_server_by_name(db: AsyncIOMotorDatabase, name: str) -> ExoServer | None:
    return await ExoServer.find_one(ExoServer.name == name)

async def get_servers(db: AsyncIOMotorDatabase, skip: int = 0, limit: Optional[int] = None) -> list[ExoServer]:
    query = ExoServer.find().sort(-ExoServer.created_at).skip(skip)
    if limit is not None:
        query = query.limit(limit)
    return await query.to_list()

async def create_server(db: AsyncIOMotorDatabase, server: ServerCreate) -> ExoServer:
    encrypted_key = encrypt_data(server.api_key) if server.api_key else None
    db_server = ExoServer(
        name=server.name, 
        url=str(server.url), 
        server_type=server.server_type,
        encrypted_api_key=encrypted_key
    )
    await db_server.insert()
    return db_server

async def update_server(db: AsyncIOMotorDatabase, server_id: str, server_update: ServerUpdate) -> ExoServer | None:
    db_server = await get_server_by_id(db, server_id)
    if not db_server:
        return None

    update_data = server_update.model_dump(exclude_unset=True)
    
    if "api_key" in update_data:
        api_key = update_data.pop("api_key")
        # A non-None value in api_key means we are intentionally setting/updating/clearing it.
        # An empty string will clear it.
        if api_key is not None:
            db_server.encrypted_api_key = encrypt_data(api_key) if api_key else None
        
    for key, value in update_data.items():
        if value is not None:
            # FIX: Convert Pydantic URL object to a string before setting.
            if key == 'url':
                setattr(db_server, key, str(value))
            else:
                setattr(db_server, key, value)
    
    await db_server.save()
    return db_server


async def delete_server(db: AsyncIOMotorDatabase, server_id: str) -> ExoServer | None:
    server = await get_server_by_id(db, server_id)
    if server:
        await server.delete()
    return server

async def fetch_and_update_models(db: AsyncIOMotorDatabase, server_id: str) -> dict:
    """
    Fetches the list of available models from an EXO server and updates the database.

    Returns a dict with 'success' (bool), 'models' (list), and optionally 'error' (str)
    """
    server = await get_server_by_id(db, server_id)
    if not server:
        return {"success": False, "error": "Server not found", "models": []}
    
    if server.server_type != "exo":
        return {"success": False, "error": "Only EXO servers are supported", "models": []}
    
    headers = _get_auth_headers(server)

    try:
        models = []
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            # EXO uses /v1/models endpoint with OpenAI-compatible format
            endpoint_url = f"{server.url.rstrip('/')}/v1/models"
            response = await client.get(endpoint_url)
            response.raise_for_status()
            data = response.json()
            models_data = data.get("data", [])
            
            for model in models_data:
                model_id = model.get("id")  # short_id like "llama-3.2-1b"
                model_name = model.get("name", model_id)
                hf_id = model.get("hugging_face_id", model_id)
                if not model_id: 
                    continue
                
                family = model_id.split(':')[0].split('-')[0] if model_id else "unknown"

                models.append({
                    "name": model_id,
                    "size": model.get("storage_size_megabytes", 0) * 1024 * 1024,  # Convert MB to bytes
                    "modified_at": datetime.datetime.fromtimestamp(
                        model.get("created", 0), tz=datetime.timezone.utc
                    ).isoformat(),
                    "digest": hf_id,  # Use HuggingFace ID as digest
                    "details": {
                        "parent_model": "",
                        "format": "exo",
                        "family": family,
                        "families": [family] if family else None,
                        "parameter_size": model.get("context_length", "N/A"),
                        "quantization_level": "N/A"
                    }
                })
        
        server.available_models = models
        server.models_last_updated = datetime.datetime.utcnow()
        server.last_error = None
        await server.save()

        logger.info(f"Successfully fetched {len(models)} models from {server.server_type} server '{server.name}'")
        return {"success": True, "models": models, "error": None}

    except httpx.HTTPError as e:
        error_msg = f"HTTP error: {str(e)}"
        logger.error(f"Failed to fetch models from server '{server.name}': {error_msg}")
        server.last_error = error_msg
        server.available_models = None
        await server.save()
        return {"success": False, "error": error_msg, "models": []}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Failed to fetch models from server '{server.name}': {error_msg}")
        server.last_error = error_msg
        server.available_models = None
        await server.save()
        return {"success": False, "error": error_msg, "models": []}


async def pull_model_on_server(http_client: httpx.AsyncClient, server: ExoServer, model_name: str) -> dict:
    """Pulls a model on a specific Ollama server."""
    if server.server_type == 'vllm':
        return {"success": False, "message": "Pulling models is not supported for vLLM servers."}
        
    headers = _get_auth_headers(server)
    pull_url = f"{server.url.rstrip('/')}/api/pull"
    payload = {"name": model_name, "stream": False}
    try:
        # Use a long timeout as pulling can take a significant amount of time
        async with http_client.stream("POST", pull_url, json=payload, timeout=1800.0, headers=headers) as response:
            async for chunk in response.aiter_text():
                try:
                    line = json.loads(chunk)
                    # You could process status updates here if needed in the future
                    logger.debug(f"Pull status for {model_name} on {server.name}: {line.get('status')}")
                except json.JSONDecodeError:
                    continue # Ignore non-json chunks
        
        response.raise_for_status() # Will raise an exception for 4xx/5xx responses
        logger.info(f"Successfully pulled/updated model '{model_name}' on server '{server.name}'")
        return {"success": True, "message": f"Model '{model_name}' pulled/updated successfully."}
    except httpx.HTTPStatusError as e:
        error_msg = f"Failed to pull model '{model_name}': Server returned status {e.response.status_code}"
        logger.error(f"{error_msg} on server '{server.name}'")
        return {"success": False, "message": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred while pulling model '{model_name}': {e}"
        logger.error(f"{error_msg} on server '{server.name}'")
        return {"success": False, "message": error_msg}

async def delete_model_on_server(http_client: httpx.AsyncClient, server: ExoServer, model_name: str) -> dict:
    """Deletes a model from a specific Ollama server."""
    if server.server_type == 'vllm':
        return {"success": False, "message": "Deleting models is not supported for vLLM servers."}

    headers = _get_auth_headers(server)
    delete_url = f"{server.url.rstrip('/')}/api/delete"
    payload = {"name": model_name}
    try:
        # FIX: Use the more robust .request() method to send a JSON body with DELETE.
        # This is compatible with a wider range of httpx versions.
        response = await http_client.request("DELETE", delete_url, json=payload, timeout=120.0, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully deleted model '{model_name}' from server '{server.name}'")
        return {"success": True, "message": f"Model '{model_name}' deleted successfully."}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            message = f"Model '{model_name}' not found on server."
            logger.warning(message)
            return {"success": True, "message": message} # Treat not found as a success
        error_msg = f"Failed to delete model '{model_name}': Server returned status {e.response.status_code}"
        logger.error(f"{error_msg} on server '{server.name}'")
        return {"success": False, "message": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred while deleting model '{model_name}': {e}"
        logger.error(f"{error_msg} on server '{server.name}'")
        return {"success": False, "message": error_msg}

async def load_model_on_server(http_client: httpx.AsyncClient, server: ExoServer, model_name: str) -> dict:
    """Sends a dummy request to a server to load a model into memory."""
    if server.server_type == 'vllm':
        return {"success": False, "message": "Explicit model loading is not applicable for vLLM servers."}

    headers = _get_auth_headers(server)
    generate_url = f"{server.url.rstrip('/')}/api/generate"
    payload = {"model": model_name, "prompt": " ", "stream": False}
    try:
        # Use a timeout sufficient for model loading
        response = await http_client.post(generate_url, json=payload, timeout=300.0, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully triggered load for model '{model_name}' on server '{server.name}'")
        return {"success": True, "message": f"Model '{model_name}' is being loaded into memory."}
    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json().get('error', e.response.text)
        except json.JSONDecodeError:
            error_detail = e.response.text
        error_msg = f"Failed to load model '{model_name}': Server returned status {e.response.status_code}: {error_detail}"
        logger.error(f"{error_msg} on server '{server.name}'")
        return {"success": False, "message": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred while loading model '{model_name}': {e}"
        logger.error(f"{error_msg} on server '{server.name}'")
        return {"success": False, "message": error_msg}

async def unload_model_on_server(http_client: httpx.AsyncClient, server: ExoServer, model_name: str) -> dict:
    """Sends a request to a server to unload a model from memory."""
    if server.server_type == 'vllm':
        return {"success": False, "message": "Explicit model unloading is not applicable for vLLM servers."}

    headers = _get_auth_headers(server)
    generate_url = f"{server.url.rstrip('/')}/api/generate"
    # Setting keep_alive to 0s tells Ollama to unload the model after this request.
    payload = {"model": model_name, "prompt": " ", "keep_alive": "0s"}
    try:
        response = await http_client.post(generate_url, json=payload, timeout=60.0, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully triggered unload for model '{model_name}' on server '{server.name}'")
        return {"success": True, "message": f"Unload signal sent for model '{model_name}'. It will be removed from memory shortly."}
    except httpx.HTTPStatusError as e:
        # If the model isn't found (which can happen if it's not loaded), treat as success.
        if e.response.status_code == 404:
             return {"success": True, "message": f"Model '{model_name}' was not loaded in memory."}
        try:
            error_detail = e.response.json().get('error', e.response.text)
        except json.JSONDecodeError:
            error_detail = e.response.text
        error_msg = f"Failed to unload model '{model_name}': Server returned status {e.response.status_code}: {error_detail}"
        logger.error(f"{error_msg} on server '{server.name}'")
        return {"success": False, "message": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred while unloading model '{model_name}': {e}"
        logger.error(f"{error_msg} on server '{server.name}'")
        return {"success": False, "message": error_msg}

async def get_servers_with_model(db: AsyncIOMotorDatabase, model_name: str) -> list[ExoServer]:
    """
    Get all active servers that have the specified model available, using flexible matching.
    """
    servers = await get_servers(db)
    active_servers = [s for s in servers if s.is_active]

    servers_with_model = []
    for server in active_servers:
        if server.available_models:
            for model_data in server.available_models:
                if isinstance(model_data, dict) and "name" in model_data:
                    available_model_name = model_data["name"]
                    # Flexible matching:
                    # 1. Exact match (e.g., "llama3:8b" == "llama3:8b")
                    # 2. Prefix match (e.g., "llama3" matches "llama3:8b")
                    # 3. Substring match for vLLM (e.g., "Llama-2-7b" matches "models--meta-llama--Llama-2-7b-chat-hf")
                    if (available_model_name == model_name or 
                        available_model_name.startswith(f"{model_name}:") or
                        (server.server_type == 'vllm' and model_name in available_model_name)):
                        servers_with_model.append(server)
                        break  # Found on this server, move to the next
    return servers_with_model

def is_embedding_model(model_name: str) -> bool:
    """Heuristically determines if a model is for embeddings."""
    name_lower = model_name.lower()
    
    # Direct embedding indicators
    if "embed" in name_lower or "embedding" in name_lower:
        return True
    
    # Common embedding model families and patterns
    embedding_patterns = [
        "bge-",           # BAAI General Embedding
        "e5-",            # E5 models
        "instructor-",    # Instructor models
        "all-minilm",     # Sentence transformers
        "all-mpnet",      # Sentence transformers
        "sentence-transformers",
        "text-embedding-",
        "multilingual-e5-",
        "gte-",           # General Text Embeddings
        "jina-",          # Jina embeddings
        "nomic-embed",    # Nomic embeddings
        "mxbai-embed",    # MixedBread embeddings
        "text2vec-",      # Text2Vec models
        "paraphrase-",    # Paraphrase models (often used for embeddings)
    ]
    
    return any(name_lower.startswith(pattern) or pattern in name_lower for pattern in embedding_patterns)

async def get_all_available_model_names(db: AsyncIOMotorDatabase, filter_type: Optional[str] = None) -> List[str]:
    """
    Gets a unique, sorted list of all model names across all active servers.
    Can be filtered by type ('chat' or 'embedding').
    """
    servers = await get_servers(db)
    active_servers = [s for s in servers if s.is_active]

    all_models = set()
    for server in active_servers:
        if not server.available_models:
            continue
        
        models_list = server.available_models
        if isinstance(models_list, str):
            try:
                models_list = json.loads(models_list)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse available_models JSON for server {server.name} in get_all_available_model_names")
                continue

        for model in models_list:
            if isinstance(model, dict) and "name" in model:
                model_name = model["name"]
                is_embed = is_embedding_model(model_name)
                
                if filter_type == 'embedding' and is_embed:
                    all_models.add(model_name)
                elif filter_type == 'chat' and not is_embed:
                    all_models.add(model_name)
                elif filter_type is None:
                    all_models.add(model_name)
    
    return sorted(list(all_models))

async def get_all_models_grouped_by_server(db: AsyncIOMotorDatabase, filter_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Gets all available model names, grouped by their server, and includes proxy-native models.
    """
    servers = await get_servers(db)
    active_servers = [s for s in servers if s.is_active]

    grouped_models = {}
    for server in active_servers:
        server_models = []
        if server.available_models:
            models_list = server.available_models
            if isinstance(models_list, str):
                try:
                    models_list = json.loads(models_list)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse available_models JSON for server {server.name} in get_all_models_grouped_by_server")
                    continue
                    
            for model in models_list:
                if isinstance(model, dict) and "name" in model:
                    model_name = model["name"]
                    is_embed = is_embedding_model(model_name)
                    
                    should_add = False
                    if filter_type == 'embedding' and is_embed:
                        should_add = True
                    elif filter_type == 'chat' and not is_embed:
                        should_add = True
                    elif filter_type is None:
                        should_add = True
                    
                    if should_add:
                        server_models.append(model_name)
        
        if server_models:
            grouped_models[server.name] = sorted(server_models)

    # Create a new dictionary to control order
    final_grouped_models = {}
    
    # Merge the server-specific models
    final_grouped_models.update(grouped_models)
            
    return final_grouped_models


async def fetch_exo_server_state(http_client: httpx.AsyncClient, server: ExoServer) -> Dict[str, Any]:
    """
    Fetches the state from an EXO server, including running instances, topology, and tasks.
    Uses the GET /state endpoint to retrieve instances.
    Returns the state dict or an error dict.
    """
    if server.server_type != 'exo':
        return {"success": False, "error": "Server is not an EXO server"}
    
    headers = _get_auth_headers(server)
    # Get the base URL from the configured server and call /state endpoint
    base_url = server.url.rstrip('/')
    state_url = f"{base_url}/state"
    
    logger.info(f"Calling EXO /state endpoint for server '{server.name}' at: {state_url}")
    
    try:
        response = await http_client.get(state_url, timeout=10.0, headers=headers)
        response.raise_for_status()
        state_data = response.json()
        
        # Validate that we got the expected structure
        if not isinstance(state_data, dict):
            error_msg = f"Invalid response format: expected dict, got {type(state_data)}"
            logger.error(f"Invalid state response from EXO server '{server.name}': {error_msg}")
            return {"success": False, "error": error_msg, "state": None}
        
        # Log the structure we received
        instances = state_data.get("instances", {})
        logger.info(f"Received state from '{server.name}': {len(instances)} instances found")
        if instances:
            # Log first instance structure for debugging
            first_instance_id = list(instances.keys())[0]
            first_instance = instances[first_instance_id]
            logger.debug(f"Sample instance structure from '{server.name}': instance_id={first_instance_id}, keys={list(first_instance.keys()) if isinstance(first_instance, dict) else type(first_instance)}")
        
        return {"success": True, "state": state_data, "error": None}
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error fetching EXO state from {state_url}: {e.response.text[:200]}"
        logger.error(f"Failed to fetch state from EXO server '{server.name}': {error_msg}")
        return {"success": False, "error": error_msg, "state": None}
    except httpx.HTTPError as e:
        error_msg = f"HTTP error fetching EXO state: {str(e)}"
        logger.error(f"Failed to fetch state from EXO server '{server.name}': {error_msg}")
        return {"success": False, "error": error_msg, "state": None}
    except Exception as e:
        error_msg = f"Unexpected error fetching EXO state: {str(e)}"
        logger.error(f"Failed to fetch state from EXO server '{server.name}': {error_msg}", exc_info=True)
        return {"success": False, "error": error_msg, "state": None}


async def get_active_models_all_servers(db: AsyncIOMotorDatabase, http_client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    Fetches running instances from active EXO servers using the `/state` endpoint.
    """
    servers = await get_servers(db)
    exo_servers = [s for s in servers if s.is_active and s.server_type == 'exo']
    
    all_models = []

    # Fetch running instances from EXO servers
    if exo_servers:
        async def fetch_exo_instances(server: ExoServer):
            try:
                state_result = await fetch_exo_server_state(http_client, server)
                if not state_result["success"]:
                    return []
                
                state = state_result["state"]
                instances = state.get("instances", {})
                node_profiles = state.get("nodeProfiles", {})
                
                # Convert EXO instances to a format similar to Ollama models
                exo_models = []
                for instance_id, instance in instances.items():
                    # Handle MlxRingInstance structure
                    mlx_instance = instance.get("MlxRingInstance", {})
                    if not mlx_instance:
                        continue
                    
                    shard_assignments = mlx_instance.get("shardAssignments", {})
                    model_id = shard_assignments.get("modelId", "unknown")
                    runner_to_shard = shard_assignments.get("runnerToShard", {})
                    
                    # Extract detailed information from first shard (all shards have same model info)
                    model_size = 0
                    pretty_name = model_id
                    sharding_type = "Unknown"
                    world_size = 1
                    n_layers = 0
                    
                    if runner_to_shard:
                        first_shard_key = next(iter(runner_to_shard.keys()))
                        first_shard = runner_to_shard[first_shard_key]
                        
                        # Check for PipelineShardMetadata or TensorShardMetadata
                        if "PipelineShardMetadata" in first_shard:
                            shard_meta = first_shard["PipelineShardMetadata"]
                            sharding_type = "Pipeline"
                            model_meta = shard_meta.get("modelMeta", {})
                            model_size = model_meta.get("storageSize", {}).get("inBytes", 0)
                            pretty_name = model_meta.get("prettyName", model_id)
                            world_size = shard_meta.get("worldSize", 1)
                            n_layers = model_meta.get("nLayers", 0)
                        elif "TensorShardMetadata" in first_shard:
                            shard_meta = first_shard["TensorShardMetadata"]
                            sharding_type = "Tensor"
                            model_meta = shard_meta.get("modelMeta", {})
                            model_size = model_meta.get("storageSize", {}).get("inBytes", 0)
                            pretty_name = model_meta.get("prettyName", model_id)
                            world_size = shard_meta.get("worldSize", 1)
                            n_layers = model_meta.get("nLayers", 0)
                    
                    # Calculate total memory usage from nodes
                    total_ram = 0
                    available_ram = 0
                    total_swap = 0
                    available_swap = 0
                    
                    node_to_runner = shard_assignments.get("nodeToRunner", {})
                    for node_id in node_to_runner.keys():
                        node_profile = node_profiles.get(node_id, {})
                        memory = node_profile.get("memory", {})
                        if memory:
                            ram_total = memory.get("ramTotal", {}).get("inBytes", 0)
                            ram_avail = memory.get("ramAvailable", {}).get("inBytes", 0)
                            swap_total = memory.get("swapTotal", {}).get("inBytes", 0)
                            swap_avail = memory.get("swapAvailable", {}).get("inBytes", 0)
                            total_ram += ram_total
                            available_ram += ram_avail
                            total_swap += swap_total
                            available_swap += swap_avail
                    
                    exo_models.append({
                        "name": model_id,
                        "pretty_name": pretty_name,
                        "server_name": server.name,
                        "size": model_size,
                        "size_vram": 1,
                        "expires_at": "Running (EXO Instance)",
                        "instance_id": instance_id,
                        "sharding": sharding_type,
                        "world_size": world_size,
                        "n_layers": n_layers,
                        "n_shards": len(runner_to_shard),
                        "total_ram": total_ram,
                        "available_ram": available_ram,
                        "total_swap": total_swap,
                        "available_swap": available_swap,
                        "hosts": mlx_instance.get("hosts", [])
                    })
                return exo_models
            except Exception as e:
                logger.error(f"Failed to fetch instances from EXO server '{server.name}': {e}")
                import traceback
                logger.error(traceback.format_exc())
                return []

        tasks = [fetch_exo_instances(server) for server in exo_servers]
        results = await asyncio.gather(*tasks)
        exo_instances = [model for sublist in results for model in sublist]
        all_models.extend(exo_instances)

    return all_models

async def refresh_all_server_models(db: AsyncIOMotorDatabase) -> dict:
    """
    Refreshes model lists for all active servers.

    Returns:
        dict with 'total', 'success', 'failed' counts
    """
    # Get all servers and extract their IDs/names before any async operations
    servers = await get_servers(db)
    active_servers = [(s.id, s.name, s.is_active) for s in servers]
    active_servers = [(sid, sname) for sid, sname, is_active in active_servers if is_active]

    results = {
        "total": len(active_servers),
        "success": 0,
        "failed": 0,
        "errors": []
    }

    for server_id, server_name in active_servers:
        result = await fetch_and_update_models(db, server_id)
        if result["success"]:
            results["success"] += 1
        else:
            results["failed"] += 1
            results["errors"].append({
                "server_id": server_id,
                "server_name": server_name,
                "error": result["error"]
            })

    return results

async def check_server_health(http_client: httpx.AsyncClient, server: ExoServer) -> Dict[str, Any]:
    """Performs a quick health check on a single backend server."""
    headers = _get_auth_headers(server)
    try:
        base_url = server.url.rstrip('/')
        # Prefer a cheap endpoint that reliably returns 200 when the backend is ready.
        # EXO exposes an OpenAI-compatible /v1/models endpoint
        ping_url = f"{base_url}/v1/models"
            
        response = await http_client.get(ping_url, timeout=3.0, headers=headers)
        
        if response.status_code == 200:
            return {"server_id": server.id, "name": server.name, "url": server.url, "status": "Online", "reason": None}
        else:
            return {"server_id": server.id, "name": server.name, "url": server.url, "status": "Offline", "reason": f"Status {response.status_code}"}
    
    except httpx.RequestError as e:
        logger.warning(f"Health check failed for server '{server.name}': {e}")
        return {"server_id": server.id, "name": server.name, "url": server.url, "status": "Offline", "reason": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error during health check for server '{server.name}': {e}")
        return {"server_id": server.id, "name": server.name, "url": server.url, "status": "Offline", "reason": "Unexpected error"}

async def check_all_servers_health(db: AsyncIOMotorDatabase, http_client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Checks the health of all configured servers."""
    servers = await get_servers(db)
    if not servers:
        return []

    tasks = [check_server_health(http_client, server) for server in servers]
    results = await asyncio.gather(*tasks)
    return results
