import enum
import secrets
import uuid
from datetime import datetime

from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import (
    ARRAY,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _gen_public_key() -> str:
    return secrets.token_urlsafe(24)


class Store(Base):
    """A tenant. Every product/order/conversation/knowledge-base row belongs
    to exactly one store, so tenants can never see each other's data."""

    __tablename__ = "stores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Public, low-privilege key used by the customer-facing /chat widget.
    # Distinct from admin JWT auth on purpose: a storefront visitor should
    # never need to log in just to talk to the support bot.
    public_api_key: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, default=_gen_public_key
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list["User"]] = relationship(back_populates="store")
    products: Mapped[list["Product"]] = relationship(back_populates="store")


class User(SQLAlchemyBaseUserTableUUID, Base):
    """Store-owner / admin account. Provides fastapi-users' standard fields:
    id, email, hashed_password, is_active, is_superuser, is_verified."""

    __tablename__ = "users"

    store_id: Mapped[int | None] = mapped_column(ForeignKey("stores.id"), nullable=True)
    store: Mapped["Store | None"] = relationship(back_populates="users")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), nullable=False, index=True)
    sku: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(120), default="")
    price: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    stock_quantity: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    store: Mapped["Store"] = relationship(back_populates="products")

    __table_args__ = ()


class OrderStatus(str, enum.Enum):
    pending = "pending"
    paid = "paid"
    fulfilled = "fulfilled"
    shipped = "shipped"
    delivered = "delivered"
    cancelled = "cancelled"
    return_requested = "return_requested"
    refunded = "refunded"


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), nullable=False, index=True)
    customer_email: Mapped[str] = mapped_column(String(255), index=True)
    customer_name: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus), default=OrderStatus.pending)
    total_amount: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    tracking_number: Mapped[str | None] = mapped_column(String(120), nullable=True)
    return_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    items: Mapped[list["OrderItem"]] = relationship(back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2), default=0)

    order: Mapped["Order"] = relationship(back_populates="items")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), nullable=False, index=True)
    customer_identifier: Mapped[str] = mapped_column(String(255), default="anonymous")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    tool = "tool"
    system = "system"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), nullable=False, index=True)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole))
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class KnowledgeChunk(Base):
    """A single embedded chunk of text used for retrieval-augmented answers.
    Embeddings are stored as a plain float array column (no pgvector
    extension needed) and compared with cosine similarity in Python — see
    app/rag.py. That's plenty fast for a knowledge base of a few thousand
    chunks; swap in pgvector later if it ever needs to scale further."""

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50))  # "document" | "product"
    source_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
