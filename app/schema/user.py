from pydantic import BaseModel
from app.database.models import UserRole


class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    role: UserRole

    class Config:
        from_attributes = True