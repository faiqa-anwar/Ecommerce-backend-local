import uuid
from datetime import datetime
from typing import Literal

from fastapi_users import schemas
from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------- Auth / users (fastapi-users) ----------
class UserRead(schemas.BaseUser[uuid.UUID]):
    store_id: int | None = None


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass


# ---------- Products ----------
class ProductCreate(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    category: str = ""
    price: float = Field(ge=0)
    stock_quantity: int = Field(ge=0, default=0)


class ProductUpdateStock(BaseModel):
    stock_quantity: int = Field(ge=0)


class ProductOut(BaseModel):
    id: int
    sku: str
    name: str
    description: str
    category: str
    price: float
    stock_quantity: int

    model_config = {"from_attributes": True}


# ---------- Orders (admin-side, for seeding/testing) ----------
class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int = Field(ge=1)


class OrderCreate(BaseModel):
    customer_email: EmailStr
    customer_name: str = ""
    items: list[OrderItemCreate] = Field(min_length=1)


class OrderUpdate(BaseModel):
    status: Literal["pending", "paid", "fulfilled", "shipped", "delivered", "cancelled", "return_requested", "refunded"] | None = None
    tracking_number: str | None = Field(default=None, max_length=120)


class OrderOut(BaseModel):
    id: int
    customer_email: str
    customer_name: str
    status: str
    total_amount: float
    tracking_number: str | None
    return_reason: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------- Knowledge base ----------
class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)


class KnowledgeChunkOut(BaseModel):
    id: int
    source_type: str
    source_id: str | None
    title: str
    content: str

    model_config = {"from_attributes": True}


class KnowledgeSearchIn(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class KnowledgeSearchResult(BaseModel):
    chunk: KnowledgeChunkOut
    score: float


# ---------- Chat ----------
class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None
    customer_email: EmailStr | None = None
    customer_identifier: str | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be blank")
        return v


class ChatOut(BaseModel):
    conversation_id: int
    reply: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: int
    customer_identifier: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
