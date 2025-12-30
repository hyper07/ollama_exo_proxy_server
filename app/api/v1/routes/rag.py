# app/api/v1/routes/rag.py
"""
RAG (Retrieval-Augmented Generation) API routes.
"""
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Request, HTTPException, status, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorDatabase
import httpx

from app.database.models import User
from app.database.session import get_db
from app.api.v1.routes.admin import get_template_context, require_admin_user
from app.crud import rag_crud, server_crud
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Register get_flashed_messages function for templates
def get_flashed_messages(request: Request):
    """Get flashed messages from session."""
    return request.session.pop("_messages", [])

templates.env.globals["get_flashed_messages"] = get_flashed_messages


@router.get("/knowledge-bases", response_class=HTMLResponse, name="admin_knowledge_bases")
async def admin_knowledge_bases(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user)
):
    """Knowledge base management page."""
    try:
        context = get_template_context(request)
        knowledge_bases = await rag_crud.get_all_knowledge_bases(db)
        
        # Calculate chunk_count for each knowledge base
        # Create a simple wrapper class to add chunk_count without modifying the model
        class KBWithCount:
            def __init__(self, kb, chunk_count):
                self.kb = kb
                self.chunk_count = chunk_count
                # Proxy all other attributes to the original kb object
                for attr in ['id', 'name', 'description', 'embedding_model', 'chunk_size', 
                            'chunk_overlap', 'is_active', 'created_at', 'updated_at']:
                    setattr(self, attr, getattr(kb, attr))
        
        knowledge_bases_with_counts = []
        for kb in knowledge_bases:
            try:
                documents = await rag_crud.get_documents_by_kb(db, str(kb.id))
                chunk_count = sum(doc.chunk_count for doc in documents if hasattr(doc, 'chunk_count'))
            except Exception as e:
                logger.warning(f"Error calculating chunk_count for KB {kb.id}: {e}")
                chunk_count = 0
            
            knowledge_bases_with_counts.append(KBWithCount(kb, chunk_count))
        
        context["knowledge_bases"] = knowledge_bases_with_counts
        return templates.TemplateResponse("admin/knowledge_bases.html", context)
    except Exception as e:
        logger.error(f"Error loading knowledge bases page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/knowledge-bases/create", name="admin_create_knowledge_base")
async def create_knowledge_base(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    embedding_model: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Create a new knowledge base."""
    try:
        kb = await rag_crud.create_knowledge_base(
            db,
            name=name,
            description=description,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            user_id=str(admin_user.id)
        )
        return JSONResponse(content={"success": True, "kb_id": str(kb.id), "message": f"Knowledge base '{name}' created successfully"})
    except ValueError as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Error creating knowledge base: {e}", exc_info=True)
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@router.get("/knowledge-bases/{kb_id}", response_class=HTMLResponse, name="admin_knowledge_base_details")
async def knowledge_base_details(
    request: Request,
    kb_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user)
):
    """Knowledge base details page with documents."""
    context = get_template_context(request)
    kb = await rag_crud.get_knowledge_base_by_id(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    documents = await rag_crud.get_documents_by_kb(db, kb_id)
    context["knowledge_base"] = kb
    context["documents"] = documents
    
    # Get available embedding models
    all_models = await server_crud.get_all_available_model_names(db, filter_type='embedding')
    context["embedding_models"] = all_models
    
    return templates.TemplateResponse("admin/knowledge_base_details.html", context)


@router.post("/knowledge-bases/{kb_id}/upload", name="admin_upload_document")
async def upload_document(
    request: Request,
    kb_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user),
    file: UploadFile = File(...)
):
    """Upload and index a document."""
    kb = await rag_crud.get_knowledge_base_by_id(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8', errors='replace')
        
        # Create document
        doc = await rag_crud.create_document(
            db,
            kb_id=kb_id,
            filename=file.filename,
            file_type=file.filename.split('.')[-1] if '.' in file.filename else 'txt',
            file_size=len(content),
            content=content_str,
            metadata={},
            user_id=str(admin_user.id)
        )
        
        # Index the document
        http_client: httpx.AsyncClient = request.app.state.http_client
        cache_service = getattr(request.app.state, 'cache', None)
        rag_service = RAGService(http_client)
        # Pass cache service to RAG service via db (workaround for method signature)
        if cache_service:
            db.cache_service = cache_service
        chunk_count = await rag_service.index_document(doc, kb, db)
        # Clean up
        if hasattr(db, 'cache_service'):
            delattr(db, 'cache_service')
        
        return JSONResponse(content={
            "success": True,
            "message": f"Document uploaded and indexed with {chunk_count} chunks",
            "document_id": str(doc.id)
        })
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@router.post("/knowledge-bases/{kb_id}/search", name="admin_search_knowledge_base")
async def search_knowledge_base(
    request: Request,
    kb_id: str,
    query: str = Form(...),
    top_k: int = Form(5),
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user)
):
    """Search a knowledge base for relevant chunks."""
    kb = await rag_crud.get_knowledge_base_by_id(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        http_client: httpx.AsyncClient = request.app.state.http_client
        cache_service = getattr(request.app.state, 'cache', None)
        rag_service = RAGService(http_client)
        
        # Pass cache service to RAG service
        if cache_service:
            db.cache_service = cache_service
        
        # Try cache first
        if cache_service and cache_service.enabled:
            try:
                cached_results = await cache_service.get_cached_rag_query(query, str(kb.id), top_k)
                if cached_results:
                    logger.info(f"Cache hit for RAG query: {query[:50]}...")
                    if hasattr(db, 'cache_service'):
                        delattr(db, 'cache_service')
                    return JSONResponse(content={
                        "success": True,
                        "chunks": cached_results,
                        "count": len(cached_results),
                        "cached": True
                    })
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
        
        chunks = await rag_service.retrieve_relevant_chunks(query, kb, db, top_k=top_k)
        
        # Cache the results
        if cache_service and cache_service.enabled and chunks:
            try:
                await cache_service.cache_rag_query(query, str(kb.id), chunks, top_k)
            except Exception as e:
                logger.warning(f"Failed to cache RAG query: {e}")
        
        # Clean up
        if hasattr(db, 'cache_service'):
            delattr(db, 'cache_service')
        
        return JSONResponse(content={
            "success": True,
            "chunks": chunks,
            "count": len(chunks)
        })
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@router.post("/chat/rag-context", name="admin_get_rag_context")
async def get_rag_context(
    request: Request,
    query: str = Form(...),
    kb_ids: List[str] = Form(...),
    top_k: int = Form(5),
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user)
):
    """Get RAG context for a chat query from multiple knowledge bases."""
    try:
        http_client: httpx.AsyncClient = request.app.state.http_client
        cache_service = getattr(request.app.state, 'cache', None)
        rag_service = RAGService(http_client)
        
        # Pass cache service to RAG service
        if cache_service:
            db.cache_service = cache_service
        
        all_chunks = []
        for kb_id in kb_ids:
            kb = await rag_crud.get_knowledge_base_by_id(db, kb_id)
            if kb and kb.is_active:
                chunks = await rag_service.retrieve_relevant_chunks(query, kb, db, top_k=top_k)
                all_chunks.extend(chunks)
        
        # Clean up
        if hasattr(db, 'cache_service'):
            delattr(db, 'cache_service')
        
        # Sort by relevance (distance) and take top_k
        all_chunks.sort(key=lambda x: x.get("distance", 1.0))
        top_chunks = all_chunks[:top_k]
        
        # Format context
        context_text = "\n\n".join([
            f"[From: {chunk['metadata'].get('document_name', 'Unknown')}]\n{chunk['content']}"
            for chunk in top_chunks
        ])
        
        return JSONResponse(content={
            "success": True,
            "context": context_text,
            "chunks": top_chunks,
            "count": len(top_chunks)
        })
    except Exception as e:
        logger.error(f"Error getting RAG context: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@router.delete("/knowledge-bases/{kb_id}", name="admin_delete_knowledge_base")
async def delete_knowledge_base(
    request: Request,
    kb_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user)
):
    """Delete a knowledge base."""
    success = await rag_crud.delete_knowledge_base(db, kb_id)
    if success:
        return JSONResponse(content={"success": True, "message": "Knowledge base deleted"})
    else:
        return JSONResponse(content={"success": False, "error": "Knowledge base not found"}, status_code=404)


@router.delete("/documents/{doc_id}", name="admin_delete_document")
async def delete_document(
    request: Request,
    doc_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    admin_user: User = Depends(require_admin_user)
):
    """Delete a document."""
    success = await rag_crud.delete_document(db, doc_id)
    if success:
        return JSONResponse(content={"success": True, "message": "Document deleted"})
    else:
        return JSONResponse(content={"success": False, "error": "Document not found"}, status_code=404)

