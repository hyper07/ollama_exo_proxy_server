# app/services/rag_service.py
"""
RAG (Retrieval-Augmented Generation) Service
Handles document chunking, embedding generation, and retrieval.
"""
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import httpx
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from app.database.models import KnowledgeBase, RAGDocument
from app.crud import server_crud

logger = logging.getLogger(__name__)

# Global ChromaDB client
_chroma_client: Optional[chromadb.ClientAPI] = None


def get_chroma_client(persist_directory: str = "./chroma_db") -> chromadb.ClientAPI:
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client


class DocumentChunker:
    """Handles document chunking with overlap, prioritizing sentence and paragraph boundaries."""
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences, preserving sentence boundaries."""
        # Split on sentence-ending punctuation followed by whitespace
        # This pattern matches: . ! ? followed by whitespace or end of string
        # Uses positive lookbehind to keep the punctuation with the sentence
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        # Filter out empty strings and strip whitespace
        result = [s.strip() for s in sentences if s.strip()]
        # If no sentences found (no punctuation), return the whole text as one sentence
        return result if result else [text.strip()]
    
    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs (double newlines)."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    @staticmethod
    def _get_overlap_sentences(text: str, overlap_size: int) -> str:
        """Get overlap text by including complete sentences from the end."""
        sentences = DocumentChunker._split_sentences(text)
        if not sentences:
            # Fallback to character-based overlap if no sentences found
            return text[-overlap_size:] if len(text) > overlap_size else text
        
        overlap_text = ""
        # Start from the last sentence and work backwards, including complete sentences
        for sentence in reversed(sentences):
            # Check if adding this sentence would exceed overlap_size
            potential_text = sentence + " " + overlap_text if overlap_text else sentence
            if len(potential_text) <= overlap_size or not overlap_text:
                overlap_text = potential_text
            else:
                break
        
        # If we still need more overlap, take characters from the end of the text
        # (this handles cases where we need more than just sentences)
        if len(overlap_text) < overlap_size:
            remaining = overlap_size - len(overlap_text)
            # Take remaining characters from the end of the original text
            char_overlap = text[-remaining:].strip()
            # Try to start at a word boundary
            if char_overlap and not char_overlap[0].isspace():
                # Find the first space in the char_overlap
                first_space = char_overlap.find(' ')
                if first_space > 0:
                    char_overlap = char_overlap[first_space:].strip()
            overlap_text = char_overlap + " " + overlap_text if char_overlap else overlap_text
        
        return overlap_text.strip()
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ) -> List[str]:
        """
        Split text into chunks with overlap, prioritizing paragraph and sentence boundaries.
        
        Args:
            text: The text to chunk
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            separator: Preferred separator for splitting (unused, kept for compatibility)
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        
        # Step 1: Split by paragraphs first
        paragraphs = DocumentChunker._split_paragraphs(text)
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph alone exceeds chunk_size, we need to split it by sentences
            if len(paragraph) > chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap to next chunk
                    overlap_text = DocumentChunker._get_overlap_sentences(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + "\n\n" if overlap_text else ""
                
                # Split large paragraph by sentences
                sentences = DocumentChunker._split_sentences(paragraph)
                for sentence in sentences:
                    # If sentence alone is too large, we have to split it (last resort)
                    if len(sentence) > chunk_size:
                        # Save current chunk
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            overlap_text = DocumentChunker._get_overlap_sentences(current_chunk, chunk_overlap)
                            current_chunk = overlap_text + " " if overlap_text else ""
                        
                        # Split long sentence by words (last resort)
                        words = sentence.split()
                        for word in words:
                            if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
                                chunks.append(current_chunk.strip())
                                # Character-based overlap for word-level splits
                                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                                current_chunk = overlap_text + " " + word if overlap_text else word
                            else:
                                current_chunk += " " + word if current_chunk else word
                    else:
                        # Normal sentence processing
                        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                            chunks.append(current_chunk.strip())
                            # Get overlap with complete sentences
                            overlap_text = DocumentChunker._get_overlap_sentences(current_chunk, chunk_overlap)
                            current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
            else:
                # Normal paragraph processing
                if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Get overlap with complete sentences
                    overlap_text = DocumentChunker._get_overlap_sentences(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Final pass: ensure no chunk exceeds chunk_size (safety check)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                # This shouldn't happen often, but handle it as a last resort
                # Split by sentences
                sentences = DocumentChunker._split_sentences(chunk)
                temp_chunk = ""
                for sentence in sentences:
                    if len(sentence) > chunk_size:
                        # Even sentence is too large, split by words
                        words = sentence.split()
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 > chunk_size and temp_chunk:
                                final_chunks.append(temp_chunk.strip())
                                temp_chunk = word
                            else:
                                temp_chunk += " " + word if temp_chunk else word
                    else:
                        if len(temp_chunk) + len(sentence) + 1 > chunk_size and temp_chunk:
                            final_chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                        else:
                            temp_chunk += " " + sentence if temp_chunk else sentence
                if temp_chunk:
                    final_chunks.append(temp_chunk.strip())
        
        return final_chunks


class EmbeddingService:
    """Handles embedding generation using Exo servers."""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
    
    async def get_embedding(
        self,
        text: str,
        model: str,
        db,
        servers: Optional[List] = None,
        cache_service = None
    ) -> List[float]:
        """
        Get embedding for a text using an embedding model.
        Uses cache if available to avoid regenerating embeddings.
        
        Args:
            text: Text to embed
            model: Embedding model name
            db: Database connection
            servers: Optional list of servers to try
            cache_service: Optional cache service for caching embeddings
            
        Returns:
            Embedding vector as list of floats
        """
        # Try cache first
        if cache_service and cache_service.enabled:
            try:
                cached_embedding = await cache_service.get_cached_embedding(text, model)
                if cached_embedding:
                    logger.debug(f"Cache hit for embedding: {model} (text length: {len(text)})")
                    return cached_embedding
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
        
        if servers is None:
            servers = await server_crud.get_active_servers(db)
        
        # Try to find a server with the embedding model
        for server in servers:
            try:
                # Check if server has the model
                if server.available_models:
                    model_names = [m.get("name") for m in server.available_models if m.get("name")]
                    if model not in model_names:
                        continue
                
                # Try to get embedding via v1/embeddings endpoint (OpenAI-compatible)
                url = f"{server.url.rstrip('/')}/v1/embeddings"
                payload = {
                    "model": model,
                    "input": text
                }
                
                headers = {"Content-Type": "application/json"}
                if server.encrypted_api_key:
                    from app.core.encryption import decrypt_api_key
                    api_key = decrypt_api_key(server.encrypted_api_key)
                    headers["Authorization"] = f"Bearer {api_key}"
                
                response = await self.http_client.post(url, json=payload, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        return data["data"][0].get("embedding", [])
                
            except Exception as e:
                logger.warning(f"Failed to get embedding from server {server.name}: {e}")
                continue
        
        raise ValueError(f"Could not get embedding for model '{model}' from any available server")
    
    async def get_embedding_with_cache(
        self,
        text: str,
        model: str,
        db,
        cache_service,
        servers: Optional[List] = None
    ) -> List[float]:
        """
        Get embedding with automatic caching.
        """
        embedding = await self.get_embedding(text, model, db, servers, cache_service)
        
        # Cache the result
        if cache_service and cache_service.enabled:
            try:
                await cache_service.cache_embedding(text, model, embedding)
                logger.debug(f"Cached embedding: {model} (text length: {len(text)})")
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")
        
        return embedding
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
        model: str,
        db,
        batch_size: int = 10,
        cache_service = None
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batches.
        Uses cache to avoid regenerating embeddings for cached texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
            db: Database connection
            batch_size: Number of texts to process per batch
            cache_service: Optional cache service for caching embeddings
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        texts_to_fetch = []
        text_indices = []
        
        # Check cache for all texts first
        if cache_service and cache_service.enabled:
            for idx, text in enumerate(texts):
                try:
                    cached = await cache_service.get_cached_embedding(text, model)
                    if cached:
                        embeddings.append((idx, cached))
                    else:
                        texts_to_fetch.append((idx, text))
                except Exception:
                    texts_to_fetch.append((idx, text))
        else:
            texts_to_fetch = [(idx, text) for idx, text in enumerate(texts)]
        
        # Fetch missing embeddings
        if texts_to_fetch:
            for i in range(0, len(texts_to_fetch), batch_size):
                batch = texts_to_fetch[i:i + batch_size]
                batch_embeddings = await asyncio.gather(*[
                    self.get_embedding(text, model, db, cache_service=cache_service) 
                    for _, text in batch
                ])
                
                # Cache the new embeddings
                if cache_service and cache_service.enabled:
                    for (idx, text), embedding in zip(batch, batch_embeddings):
                        try:
                            await cache_service.cache_embedding(text, model, embedding)
                        except Exception:
                            pass
                        embeddings.append((idx, embedding))
                else:
                    for (idx, _), embedding in zip(batch, batch_embeddings):
                        embeddings.append((idx, embedding))
        
        # Sort by original index and return just the embeddings
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]


class RAGService:
    """Main RAG service for document indexing and retrieval."""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService(http_client)
    
    async def index_document(
        self,
        document: RAGDocument,
        knowledge_base: KnowledgeBase,
        db
    ) -> int:
        """
        Index a document: chunk it, generate embeddings, and store in vector DB.
        
        Args:
            document: Document to index
            knowledge_base: Knowledge base this document belongs to
            db: Database connection
            
        Returns:
            Number of chunks created
        """
        # Get ChromaDB collection for this knowledge base
        chroma_client = get_chroma_client()
        collection_name = f"kb_{knowledge_base.id}"
        
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            # Create collection if it doesn't exist
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"kb_id": str(knowledge_base.id), "kb_name": knowledge_base.name}
            )
        
        # Chunk the document
        chunks = self.chunker.chunk_text(
            document.content,
            chunk_size=knowledge_base.chunk_size,
            chunk_overlap=knowledge_base.chunk_overlap
        )
        
        if not chunks:
            logger.warning(f"No chunks created for document {document.id}")
            return 0
        
        # Generate embeddings for chunks (with caching)
        logger.info(f"Generating embeddings for {len(chunks)} chunks using model {knowledge_base.embedding_model}")
        cache_service = getattr(db, 'cache_service', None) if hasattr(db, 'cache_service') else None
        embeddings = await self.embedding_service.get_embeddings_batch(
            chunks,
            knowledge_base.embedding_model,
            db,
            cache_service=cache_service
        )
        
        # Store in ChromaDB
        chunk_ids = [f"{document.id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": str(document.id),
                "document_name": document.filename,
                "chunk_index": i,
                "kb_id": str(knowledge_base.id)
            }
            for i in range(len(chunks))
        ]
        
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        # Update document
        document.chunk_count = len(chunks)
        document.is_indexed = True
        await document.save()
        
        logger.info(f"Indexed document {document.id} with {len(chunks)} chunks")
        return len(chunks)
    
    async def retrieve_relevant_chunks(
        self,
        query: str,
        knowledge_base: KnowledgeBase,
        db,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            knowledge_base: Knowledge base to search in
            db: Database connection
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        # Get ChromaDB collection
        chroma_client = get_chroma_client()
        collection_name = f"kb_{knowledge_base.id}"
        
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except:
            logger.warning(f"Collection {collection_name} not found")
            return []
        
        # Generate query embedding (with caching)
        cache_service = getattr(db, 'cache_service', None) if hasattr(db, 'cache_service') else None
        query_embedding = await self.embedding_service.get_embedding(
            query,
            knowledge_base.embedding_model,
            db,
            cache_service=cache_service
        )
        
        # Search for similar chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        chunks = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                chunks.append({
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        
        return chunks
    
    async def delete_document_from_index(
        self,
        document: RAGDocument,
        knowledge_base: KnowledgeBase
    ):
        """Remove a document's chunks from the vector database."""
        chroma_client = get_chroma_client()
        collection_name = f"kb_{knowledge_base.id}"
        
        try:
            collection = chroma_client.get_collection(name=collection_name)
            # Delete all chunks for this document
            chunk_ids = [f"{document.id}_chunk_{i}" for i in range(document.chunk_count)]
            collection.delete(ids=chunk_ids)
            
            # Update document
            document.is_indexed = False
            document.chunk_count = 0
            await document.save()
            
            logger.info(f"Deleted document {document.id} from index")
        except Exception as e:
            logger.error(f"Error deleting document from index: {e}")

