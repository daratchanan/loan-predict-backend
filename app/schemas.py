# app/schemas.py

from pydantic import BaseModel
from typing import List, Optional

# --- Schemas for Roles (from RBAC) ---
class RoleBase(BaseModel):
    name: str
    description: Optional[str] = None

class Role(RoleBase):
    id: int

    class Config:
        from_attributes = True

# --- Schemas for Users ---
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    roles: List[Role] = [] # แสดง Roles ของ User

    class Config:
        from_attributes = True

# --- Schemas for Token ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None