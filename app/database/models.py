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

    api_keys: List["APIKey"] = Field(default_factory=list)

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

    usage_logs: List["UsageLog"] = Field(default_factory=list)


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
