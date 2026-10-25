from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_active_user
from app.database import get_async_session
from app.models import Product, User
from app.schemas import ProductCreate, ProductOut, ProductUpdateStock

router = APIRouter(prefix="/products", tags=["products"])


def _require_store(user: User) -> int:
    if user.store_id is None:
        # Defensive: should not happen given on_after_register, but guards
        # against manually-created users or future changes to that flow.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This account has no associated store.")
    return user.store_id


@router.post("", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
async def create_product(
    payload: ProductCreate,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Product:
    store_id = _require_store(user)
    existing = await session.execute(
        select(Product).where(Product.store_id == store_id, Product.sku == payload.sku)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"SKU '{payload.sku}' already exists.")

    product = Product(store_id=store_id, **payload.model_dump())
    session.add(product)
    await session.commit()
    await session.refresh(product)
    return product


@router.get("", response_model=list[ProductOut])
async def list_products(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[Product]:
    store_id = _require_store(user)
    result = await session.execute(select(Product).where(Product.store_id == store_id))
    return list(result.scalars().all())


@router.get("/{product_id}", response_model=ProductOut)
async def get_product(
    product_id: int,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Product:
    store_id = _require_store(user)
    result = await session.execute(
        select(Product).where(Product.id == product_id, Product.store_id == store_id)
    )
    product = result.scalar_one_or_none()
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found.")
    return product


@router.patch("/{product_id}/stock", response_model=ProductOut)
async def update_stock(
    product_id: int,
    payload: ProductUpdateStock,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Product:
    store_id = _require_store(user)
    result = await session.execute(
        select(Product).where(Product.id == product_id, Product.store_id == store_id)
    )
    product = result.scalar_one_or_none()
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found.")
    product.stock_quantity = payload.stock_quantity
    session.add(product)
    await session.commit()
    await session.refresh(product)
    return product
