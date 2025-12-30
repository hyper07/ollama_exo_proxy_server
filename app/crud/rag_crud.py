# app/crud/rag_crud.py
"""
CRUD operations for RAG (Knowledge Bases and Documents).
"""
import logging
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from beanie import PydanticObjectId

from app.database.models import KnowledgeBase, RAGDocument, User

logger = logging.getLogger(__name__)


async def create_knowledge_base(
    db: AsyncIOMotorDatabase,
    name: str,
    description: Optional[str],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    user_id: str
) -> KnowledgeBase:
    """Create a new knowledge base."""
    user = await User.get(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    
    # Check if name already exists
    existing = await KnowledgeBase.find_one(KnowledgeBase.name == name)
    if existing:
        raise ValueError(f"Knowledge base with name '{name}' already exists")
    
    kb = KnowledgeBase(
        name=name,
        description=description,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        created_by=user
    )
    await kb.insert()
    logger.info(f"Created knowledge base: {kb.name} (ID: {kb.id})")
    return kb


async def get_knowledge_base_by_id(db: AsyncIOMotorDatabase, kb_id: str) -> Optional[KnowledgeBase]:
    """Get a knowledge base by ID."""
    return await KnowledgeBase.get(kb_id)


async def get_knowledge_base_by_name(db: AsyncIOMotorDatabase, name: str) -> Optional[KnowledgeBase]:
    """Get a knowledge base by name."""
    return await KnowledgeBase.find_one(KnowledgeBase.name == name)


async def get_all_knowledge_bases(db: AsyncIOMotorDatabase) -> List[KnowledgeBase]:
    """Get all knowledge bases."""
    return await KnowledgeBase.find_all().to_list()


async def update_knowledge_base(
    db: AsyncIOMotorDatabase,
    kb_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    is_active: Optional[bool] = None
) -> Optional[KnowledgeBase]:
    """Update a knowledge base."""
    kb = await KnowledgeBase.get(kb_id)
    if not kb:
        return None
    
    if name is not None:
        # Check if new name conflicts
        existing = await KnowledgeBase.find_one(KnowledgeBase.name == name)
        if existing and existing.id != kb_id:
            raise ValueError(f"Knowledge base with name '{name}' already exists")
        kb.name = name
    
    if description is not None:
        kb.description = description
    if embedding_model is not None:
        kb.embedding_model = embedding_model
    if chunk_size is not None:
        kb.chunk_size = chunk_size
    if chunk_overlap is not None:
        kb.chunk_overlap = chunk_overlap
    if is_active is not None:
        kb.is_active = is_active
    
    from datetime import datetime
    kb.updated_at = datetime.utcnow()
    await kb.save()
    return kb


async def delete_knowledge_base(db: AsyncIOMotorDatabase, kb_id: str) -> bool:
    """Delete a knowledge base and all its documents."""
    kb = await KnowledgeBase.get(kb_id)
    if not kb:
        return False
    
    # Delete all documents in this knowledge base
    documents = await RAGDocument.find(RAGDocument.knowledge_base.id == kb_id).to_list()
    for doc in documents:
        await RAGDocument.delete(doc)
    
    # Delete ChromaDB collection
    from app.services.rag_service import get_chroma_client
    chroma_client = get_chroma_client()
    collection_name = f"kb_{kb_id}"
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass  # Collection might not exist
    
    await KnowledgeBase.delete(kb)
    logger.info(f"Deleted knowledge base: {kb.name}")
    return True


async def create_document(
    db: AsyncIOMotorDatabase,
    kb_id: str,
    filename: str,
    file_type: str,
    file_size: int,
    content: str,
    metadata: dict,
    user_id: str
) -> RAGDocument:
    """Create a new document in a knowledge base."""
    kb = await KnowledgeBase.get(kb_id)
    if not kb:
        raise ValueError(f"Knowledge base {kb_id} not found")
    
    user = await User.get(user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    
    doc = RAGDocument(
        knowledge_base=kb,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        content=content,
        metadata=metadata,
        uploaded_by=user
    )
    await doc.insert()
    logger.info(f"Created document: {doc.filename} in KB {kb.name}")
    return doc


async def get_document_by_id(db: AsyncIOMotorDatabase, doc_id: str) -> Optional[RAGDocument]:
    """Get a document by ID."""
    return await RAGDocument.get(doc_id)


async def get_documents_by_kb(db: AsyncIOMotorDatabase, kb_id: str) -> List[RAGDocument]:
    """Get all documents in a knowledge base."""
    return await RAGDocument.find(RAGDocument.knowledge_base.id == kb_id).to_list()


async def delete_document(db: AsyncIOMotorDatabase, doc_id: str) -> bool:
    """Delete a document."""
    doc = await RAGDocument.get(doc_id)
    if not doc:
        return False
    
    # Remove from vector index if indexed
    if doc.is_indexed:
        await doc.fetch_link(RAGDocument.knowledge_base)
        from app.services.rag_service import RAGService
        from app.core.config import settings
        import httpx
        async with httpx.AsyncClient() as client:
            rag_service = RAGService(client)
            await rag_service.delete_document_from_index(doc, doc.knowledge_base)
    
    await RAGDocument.delete(doc)
    logger.info(f"Deleted document: {doc.filename}")
    return True

