from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_active_user
from app.database import get_async_session
from app.models import Order, OrderItem, OrderStatus, Product, User
from app.routers.products import _require_store
from app.schemas import OrderCreate, OrderOut, OrderUpdate

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post("", response_model=OrderOut, status_code=status.HTTP_201_CREATED)
async def create_order(
    payload: OrderCreate,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Order:
    store_id = _require_store(user)

    product_ids = [item.product_id for item in payload.items]
    result = await session.execute(
        select(Product).where(Product.store_id == store_id, Product.id.in_(product_ids))
    )
    products_by_id = {p.id: p for p in result.scalars().all()}

    missing = set(product_ids) - set(products_by_id)
    if missing:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown product id(s): {sorted(missing)}")

    order = Order(store_id=store_id, customer_email=payload.customer_email, customer_name=payload.customer_name)
    total = 0.0
    for item in payload.items:
        product = products_by_id[item.product_id]
        unit_price = float(product.price)
        total += unit_price * item.quantity
        order.items.append(OrderItem(product_id=product.id, quantity=item.quantity, unit_price=unit_price))
    order.total_amount = total

    session.add(order)
    await session.commit()
    await session.refresh(order)
    return order


@router.get("", response_model=list[OrderOut])
async def list_orders(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[Order]:
    store_id = _require_store(user)
    result = await session.execute(select(Order).where(Order.store_id == store_id))
    return list(result.scalars().all())


@router.get("/{order_id}", response_model=OrderOut)
async def get_order(
    order_id: int,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Order:
    store_id = _require_store(user)
    result = await session.execute(select(Order).where(Order.id == order_id, Order.store_id == store_id))
    order = result.scalar_one_or_none()
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Order not found.")
    return order


@router.patch("/{order_id}", response_model=OrderOut)
async def update_order(
    order_id: int,
    payload: OrderUpdate,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> Order:
    """Admin-side: move an order through its lifecycle (e.g. mark it shipped
    with a tracking number) so order-status and return flows can be demoed."""
    store_id = _require_store(user)
    result = await session.execute(select(Order).where(Order.id == order_id, Order.store_id == store_id))
    order = result.scalar_one_or_none()
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Order not found.")
    if payload.status is not None:
        order.status = OrderStatus(payload.status)
    if payload.tracking_number is not None:
        order.tracking_number = payload.tracking_number.strip() or None
    session.add(order)
    await session.commit()
    await session.refresh(order)
    return order
