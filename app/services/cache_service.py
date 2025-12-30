# app/services/cache_service.py
"""
KV Cache Service using Redis for caching various data types.
Supports response caching, embedding caching, and metadata caching.
"""
import logging
import json
import hashlib
from typing import Optional, Any, Dict, List
from datetime import timedelta
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheService:
    """Service for managing KV cache using Redis."""
    
    # Cache key prefixes
    PREFIX_RESPONSE = "cache:response:"
    PREFIX_EMBEDDING = "cache:embedding:"
    PREFIX_MODEL_METADATA = "cache:model:"
    PREFIX_RAG_QUERY = "cache:rag:query:"
    PREFIX_SERVER_MODELS = "cache:server:models:"
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.enabled = redis_client is not None
    
    def _make_key(self, prefix: str, identifier: str) -> str:
        """Create a cache key from prefix and identifier."""
        return f"{prefix}{identifier}"
    
    def _hash_key(self, data: Any) -> str:
        """Create a hash from data for use as cache key."""
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self.enabled:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        ttl_timedelta: Optional[timedelta] = None
    ) -> bool:
        """Set a value in cache with optional TTL."""
        if not self.enabled:
            return False
        
        try:
            value_str = json.dumps(value)
            if ttl_timedelta:
                ttl_seconds = int(ttl_timedelta.total_seconds())
            
            if ttl_seconds:
                await self.redis.setex(key, ttl_seconds, value_str)
            else:
                await self.redis.set(key, value_str)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.enabled:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        if not self.enabled:
            return 0
        
        try:
            count = 0
            async for key in self.redis.scan_iter(match=pattern):
                await self.redis.delete(key)
                count += 1
            return count
        except Exception as e:
            logger.warning(f"Cache delete pattern error for {pattern}: {e}")
            return 0
    
    # --- Response Cache Methods ---
    
    async def get_cached_response(
        self, 
        endpoint: str, 
        method: str, 
        payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached API response."""
        cache_key = self._make_key(
            self.PREFIX_RESPONSE,
            self._hash_key({"endpoint": endpoint, "method": method, "payload": payload})
        )
        return await self.get(cache_key)
    
    async def cache_response(
        self,
        endpoint: str,
        method: str,
        payload: Dict[str, Any],
        response: Dict[str, Any],
        ttl_seconds: int = 300  # 5 minutes default
    ) -> bool:
        """Cache an API response."""
        cache_key = self._make_key(
            self.PREFIX_RESPONSE,
            self._hash_key({"endpoint": endpoint, "method": method, "payload": payload})
        )
        return await self.set(cache_key, response, ttl_seconds=ttl_seconds)
    
    # --- Embedding Cache Methods ---
    
    async def get_cached_embedding(
        self,
        text: str,
        model: str
    ) -> Optional[List[float]]:
        """Get cached embedding for text and model."""
        cache_key = self._make_key(
            self.PREFIX_EMBEDDING,
            self._hash_key({"text": text, "model": model})
        )
        return await self.get(cache_key)
    
    async def cache_embedding(
        self,
        text: str,
        model: str,
        embedding: List[float],
        ttl_seconds: int = 86400 * 7  # 7 days default
    ) -> bool:
        """Cache an embedding."""
        cache_key = self._make_key(
            self.PREFIX_EMBEDDING,
            self._hash_key({"text": text, "model": model})
        )
        return await self.set(cache_key, embedding, ttl_seconds=ttl_seconds)
    
    async def cache_embeddings_batch(
        self,
        texts: List[str],
        model: str,
        embeddings: List[List[float]],
        ttl_seconds: int = 86400 * 7
    ) -> int:
        """Cache multiple embeddings at once."""
        if not self.enabled:
            return 0
        
        count = 0
        for text, embedding in zip(texts, embeddings):
            if await self.cache_embedding(text, model, embedding, ttl_seconds):
                count += 1
        return count
    
    # --- Model Metadata Cache Methods ---
    
    async def get_cached_model_metadata(
        self,
        server_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached model metadata for a server."""
        cache_key = self._make_key(self.PREFIX_SERVER_MODELS, server_id)
        return await self.get(cache_key)
    
    async def cache_model_metadata(
        self,
        server_id: str,
        models: List[Dict[str, Any]],
        ttl_seconds: int = 600  # 10 minutes default
    ) -> bool:
        """Cache model metadata for a server."""
        cache_key = self._make_key(self.PREFIX_SERVER_MODELS, server_id)
        return await self.set(cache_key, models, ttl_seconds=ttl_seconds)
    
    async def invalidate_model_cache(self, server_id: Optional[str] = None):
        """Invalidate model metadata cache."""
        if server_id:
            pattern = self._make_key(self.PREFIX_SERVER_MODELS, server_id)
            await self.delete(pattern)
        else:
            pattern = f"{self.PREFIX_SERVER_MODELS}*"
            await self.delete_pattern(pattern)
    
    # --- RAG Query Cache Methods ---
    
    async def get_cached_rag_query(
        self,
        query: str,
        kb_id: str,
        top_k: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached RAG query results."""
        cache_key = self._make_key(
            self.PREFIX_RAG_QUERY,
            self._hash_key({"query": query, "kb_id": kb_id, "top_k": top_k})
        )
        return await self.get(cache_key)
    
    async def cache_rag_query(
        self,
        query: str,
        kb_id: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        ttl_seconds: int = 3600  # 1 hour default
    ) -> bool:
        """Cache RAG query results."""
        cache_key = self._make_key(
            self.PREFIX_RAG_QUERY,
            self._hash_key({"query": query, "kb_id": kb_id, "top_k": top_k})
        )
        return await self.set(cache_key, results, ttl_seconds=ttl_seconds)
    
    async def invalidate_rag_cache(self, kb_id: Optional[str] = None):
        """Invalidate RAG cache for a knowledge base or all."""
        if kb_id:
            # Invalidate all queries for this KB
            pattern = f"{self.PREFIX_RAG_QUERY}*"
            # We need to check each key to see if it's for this KB
            # For simplicity, we'll invalidate all RAG cache
            await self.delete_pattern(pattern)
        else:
            pattern = f"{self.PREFIX_RAG_QUERY}*"
            await self.delete_pattern(pattern)
    
    # --- Cache Statistics ---
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            info = await self.redis.info("stats")
            keyspace = await self.redis.info("keyspace")
            
            # Count keys by prefix
            response_count = len([k async for k in self.redis.scan_iter(match=f"{self.PREFIX_RESPONSE}*")])
            embedding_count = len([k async for k in self.redis.scan_iter(match=f"{self.PREFIX_EMBEDDING}*")])
            model_count = len([k async for k in self.redis.scan_iter(match=f"{self.PREFIX_SERVER_MODELS}*")])
            rag_count = len([k async for k in self.redis.scan_iter(match=f"{self.PREFIX_RAG_QUERY}*")])
            
            return {
                "enabled": True,
                "total_keys": info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                ) if (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)) > 0 else 0,
                "by_type": {
                    "responses": response_count,
                    "embeddings": embedding_count,
                    "model_metadata": model_count,
                    "rag_queries": rag_count
                }
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    async def clear_all_cache(self) -> int:
        """Clear all cache entries."""
        if not self.enabled:
            return 0
        
        patterns = [
            f"{self.PREFIX_RESPONSE}*",
            f"{self.PREFIX_EMBEDDING}*",
            f"{self.PREFIX_MODEL_METADATA}*",
            f"{self.PREFIX_RAG_QUERY}*",
            f"{self.PREFIX_SERVER_MODELS}*"
        ]
        
        total = 0
        for pattern in patterns:
            total += await self.delete_pattern(pattern)
        return total

