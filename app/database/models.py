import datetime
from beanie import Document, Link
from pydantic import Field
from typing import Optional, List, Dict, Any
from enum import Enum


class UserRole(str, Enum):
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"


class User(Document):
    username: str = Field(unique=True, index=True)
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False  # Keep for backward compatibility during migration
    role: UserRole = UserRole.USER
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    @property
    def is_admin_role(self) -> bool:
        """Check if user has admin role."""
        return self.role == UserRole.ADMIN


class APIKey(Document):
    key_name: str
    hashed_key: str = Field(unique=True, index=True)
    key_prefix: str = Field(unique=True)
    user: Link[User]
    expires_at: Optional[datetime.datetime] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    
    is_active: bool = True
    is_revoked: bool = False
    
    rate_limit_requests: Optional[int] = None
    rate_limit_window_minutes: Optional[int] = None


class UsageLog(Document):
    api_key: Link[APIKey]
    endpoint: str
    status_code: int
    request_timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    server: Optional[Link["ExoServer"]] = None
    model: Optional[str] = Field(index=True)


class ExoServer(Document):
    name: str
    url: str = Field(unique=True)
    server_type: str = "exo"
    encrypted_api_key: Optional[str] = None
    is_active: bool = True
    available_models: Optional[List[Dict[str, Any]]] = None
    models_last_updated: Optional[datetime.datetime] = None
    last_error: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    @property
    def has_api_key(self) -> bool:
        return bool(self.encrypted_api_key)


class AppSettings(Document):
    id: str = Field(default="main", alias="_id")
    settings_data: dict


class ModelMetadata(Document):
    model_name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    supports_images: bool = False
    is_code_model: bool = False
    is_chat_model: bool = True
    is_fast_model: bool = False
    priority: int = 10


class KnowledgeBase(Document):
    """A knowledge base collection for RAG."""
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    embedding_model: str  # Model to use for generating embeddings
    chunk_size: int = Field(default=1000)  # Characters per chunk
    chunk_overlap: int = Field(default=200)  # Overlap between chunks
    created_by: Link[User]
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    is_active: bool = True


class RAGDocument(Document):
    """A document in a knowledge base."""
    knowledge_base: Link[KnowledgeBase]
    filename: str
    file_type: str  # e.g., 'pdf', 'txt', 'md', 'docx'
    file_size: int  # Size in bytes
    content: str  # Full document content
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata
    uploaded_by: Link[User]
    uploaded_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    chunk_count: int = Field(default=0)  # Number of chunks created
    is_indexed: bool = Field(default=False)  # Whether chunks are in vector DB
